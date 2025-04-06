from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
import asyncpg
from datetime import datetime, timedelta
from typing import List, Dict
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

db_pool = None
SECRET_KEY = "your-secret-key-here-1234567890"  # Replace with a secure key in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
clients: Dict[int, WebSocket] = {}
online_users: Dict[int, str] = {}

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL not set in .env")
    raise ValueError("DATABASE_URL not set in .env")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL)
        async with db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS friends (
                    user_id INTEGER REFERENCES users(id),
                    friend_id INTEGER REFERENCES users(id),
                    PRIMARY KEY (user_id, friend_id)
                );
                CREATE TABLE IF NOT EXISTS friend_requests (
                    id SERIAL PRIMARY KEY,
                    sender_id INTEGER REFERENCES users(id),
                    recipient_id INTEGER REFERENCES users(id),
                    status VARCHAR(10) DEFAULT 'pending', -- 'pending', 'accepted', 'rejected'
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (sender_id, recipient_id)
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    sender_id INTEGER REFERENCES users(id),
                    recipient_id INTEGER REFERENCES users(id),
                    content TEXT NOT NULL,
                    type VARCHAR(20) NOT NULL DEFAULT 'text', -- Adjusted to VARCHAR(20)
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_read BOOLEAN DEFAULT FALSE
                );
            """)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise
    yield
    if db_pool:
        await db_pool.close()
        logger.info("Database connection closed")

app = FastAPI(lifespan=lifespan)

# Updated CORS to include frontend URL (replace with your actual frontend Render URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",              # Local dev
        "https://your-frontend.onrender.com", # Replace with your frontend Render URL
        "https://chitchat-f4e6.onrender.com"  # Backend itself (optional)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Added root endpoint to avoid 404
@app.get("/")
async def root():
    return {"message": "ChitChat Backend is Live!"}

class UserCreate(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class MessageCreate(BaseModel):
    recipient_username: str
    content: str
    type: str = "text"

class MessageUpdate(BaseModel):
    content: str

class FriendRequest(BaseModel):
    recipient_username: str

class FriendRequestResponse(BaseModel):
    request_id: int
    accept: bool

class MessageOut(BaseModel):
    id: int
    sender_username: str
    recipient_username: str
    content: str
    type: str
    timestamp: str
    is_read: bool

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }

class Conversation(BaseModel):
    username: str
    messages: List[MessageOut]

class FriendRequestOut(BaseModel):
    id: int
    sender_username: str
    recipient_username: str
    status: str
    timestamp: str

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }

class UserOut(BaseModel):
    id: int
    username: str

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        async with db_pool.acquire() as conn:
            user = await conn.fetchrow("SELECT id, username FROM users WHERE id = $1", int(user_id))
            if not user:
                raise HTTPException(status_code=401, detail="User not found")
        return {"id": user["id"], "username": user["username"]}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/register")
async def register(user: UserCreate):
    async with db_pool.acquire() as conn:
        try:
            hashed_password = pwd_context.hash(user.password)
            await conn.execute(
                "INSERT INTO users (username, password_hash) VALUES ($1, $2)",
                user.username, hashed_password
            )
            logger.info(f"User registered: {user.username}")
            return {"msg": "User created"}
        except asyncpg.exceptions.UniqueViolationError:
            logger.warning(f"Username already exists: {user.username}")
            raise HTTPException(status_code=400, detail="Username already exists")

@app.post("/login")
async def login(form_data: LoginRequest):
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT id, username, password_hash FROM users WHERE username = $1",
            form_data.username
        )
        if not user or not pwd_context.verify(form_data.password, user["password_hash"]):
            logger.warning(f"Login failed: Invalid credentials for '{form_data.username}'")
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_access_token({"sub": str(user["id"])})
        logger.info(f"User logged in: {user['username']}")
        return {"access_token": token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return {"username": current_user["username"]}

@app.get("/users")
async def search_users(search: str = "", current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        if search:
            users = await conn.fetch(
                "SELECT id, username FROM users WHERE username ILIKE $1 AND id != $2",
                f"%{search}%", current_user["id"]
            )
        else:
            users = await conn.fetch("SELECT id, username FROM users WHERE id != $1", current_user["id"])
        logger.debug(f"Search returned {len(users)} users for query: '{search}'")
        return [{"id": str(u["id"]), "username": u["username"]} for u in users]

@app.get("/users/suggestions", response_model=List[UserOut])
async def get_suggested_users(current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        suggestions = await conn.fetch(
            """
            SELECT id, username
            FROM users u
            WHERE u.id != $1
            AND NOT EXISTS (
                SELECT 1 FROM friends f 
                WHERE (f.user_id = u.id AND f.friend_id = $1) 
                OR (f.user_id = $1 AND f.friend_id = u.id)
            )
            AND NOT EXISTS (
                SELECT 1 FROM friend_requests fr 
                WHERE (fr.sender_id = u.id AND fr.recipient_id = $1 AND fr.status = 'pending')
                OR (fr.sender_id = $1 AND fr.recipient_id = u.id AND fr.status = 'pending')
            )
            ORDER BY RANDOM()
            LIMIT 10
            """, current_user["id"]
        )
        logger.debug(f"Fetched {len(suggestions)} suggested users for {current_user['username']}")
        return [{"id": s["id"], "username": s["username"]} for s in suggestions]

async def fetch_message_row(conn, query, *args):
    row = await conn.fetchrow(query, *args)
    if row:
        return dict(row) | {"timestamp": row["timestamp"].isoformat() + "Z"}
    return None

async def are_friends(conn, user_id: int, friend_id: int):
    result = await conn.fetchrow(
        "SELECT 1 FROM friends WHERE (user_id = $1 AND friend_id = $2) OR (user_id = $2 AND friend_id = $1)",
        user_id, friend_id
    )
    return bool(result)

@app.post("/friend-request")
async def send_friend_request(request: FriendRequest, current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        recipient = await conn.fetchrow(
            "SELECT id, username FROM users WHERE username = $1", request.recipient_username
        )
        if not recipient:
            raise HTTPException(status_code=404, detail="Recipient not found")
        if recipient["id"] == current_user["id"]:
            raise HTTPException(status_code=400, detail="Cannot send friend request to yourself")
        if await are_friends(conn, current_user["id"], recipient["id"]):
            raise HTTPException(status_code=400, detail="Already friends")
        existing_request = await conn.fetchrow(
            "SELECT id FROM friend_requests WHERE sender_id = $1 AND recipient_id = $2 AND status = 'pending'",
            current_user["id"], recipient["id"]
        )
        if existing_request:
            raise HTTPException(status_code=400, detail="Friend request already sent")
        msg = await fetch_message_row(
            conn,
            "INSERT INTO messages (sender_id, recipient_id, content, type) VALUES ($1, $2, $3, $4) RETURNING id, sender_id, recipient_id, content, type, timestamp, is_read",
            current_user["id"], recipient["id"], f"Friend request from {current_user['username']}", "friend_request"
        )
        request_data = await conn.fetchrow(
            "INSERT INTO friend_requests (sender_id, recipient_id) VALUES ($1, $2) RETURNING id, sender_id, recipient_id, status, timestamp",
            current_user["id"], recipient["id"]
        )
        message_data = {
            "id": msg["id"],
            "sender_username": current_user["username"],
            "recipient_username": recipient["username"],
            "content": msg["content"],
            "type": msg["type"],
            "timestamp": msg["timestamp"],
            "is_read": msg["is_read"]
        }
        if recipient["id"] in clients:
            await clients[recipient["id"]].send_json({"type": "message", "data": message_data})
        logger.info(f"Friend request sent from {current_user['username']} to {request.recipient_username}")
        return {"msg": "Friend request sent", "request_id": request_data["id"]}

@app.post("/friend-request/respond")
async def respond_friend_request(response: FriendRequestResponse, current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        request = await conn.fetchrow(
            "SELECT id, sender_id, recipient_id, status FROM friend_requests WHERE id = $1 AND recipient_id = $2",
            response.request_id, current_user["id"]
        )
        if not request or request["status"] != "pending":
            raise HTTPException(status_code=404, detail="Friend request not found or already processed")
        sender = await conn.fetchrow("SELECT username FROM users WHERE id = $1", request["sender_id"])
        if response.accept:
            await conn.execute(
                "UPDATE friend_requests SET status = 'accepted' WHERE id = $1", response.request_id
            )
            await conn.execute(
                "INSERT INTO friends (user_id, friend_id) VALUES ($1, $2), ($2, $1)",
                current_user["id"], request["sender_id"]
            )
            logger.info(f"Friend request accepted: {current_user['username']} and {sender['username']}")
            if request["sender_id"] in clients:
                await clients[request["sender_id"]].send_json({
                    "type": "friend_accepted",
                    "data": {"username": current_user["username"]}
                })
            return {"msg": "Friend request accepted"}
        else:
            await conn.execute(
                "UPDATE friend_requests SET status = 'rejected' WHERE id = $1", response.request_id
            )
            logger.info(f"Friend request rejected: {current_user['username']} from {sender['username']}")
            return {"msg": "Friend request rejected"}

@app.get("/friend-requests", response_model=List[FriendRequestOut])
async def get_friend_requests(current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        requests = await conn.fetch(
            """
            SELECT fr.id, s.username as sender_username, r.username as recipient_username, fr.status, fr.timestamp
            FROM friend_requests fr
            JOIN users s ON fr.sender_id = s.id
            JOIN users r ON fr.recipient_id = r.id
            WHERE fr.recipient_id = $1 AND fr.status = 'pending'
            """, current_user["id"]
        )
        return [{
            "id": r["id"],
            "sender_username": r["sender_username"],
            "recipient_username": r["recipient_username"],
            "status": r["status"],
            "timestamp": r["timestamp"].isoformat() + "Z"
        } for r in requests]

@app.post("/messages")
async def send_message(message: MessageCreate, current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        recipient = await conn.fetchrow(
            "SELECT id, username FROM users WHERE username = $1", message.recipient_username
        )
        if not recipient:
            raise HTTPException(status_code=404, detail="Recipient not found")
        if message.type not in ["text", "image", "audio"]:
            raise HTTPException(status_code=400, detail="Invalid message type")
        if not await are_friends(conn, current_user["id"], recipient["id"]):
            raise HTTPException(status_code=403, detail="You must be friends to send messages")
        msg = await fetch_message_row(
            conn,
            "INSERT INTO messages (sender_id, recipient_id, content, type) VALUES ($1, $2, $3, $4) RETURNING id, sender_id, recipient_id, content, type, timestamp, is_read",
            current_user["id"], recipient["id"], message.content, message.type
        )
        message_data = {
            "id": msg["id"],
            "sender_username": current_user["username"],
            "recipient_username": recipient["username"],
            "content": msg["content"],
            "type": msg["type"],
            "timestamp": msg["timestamp"],
            "is_read": msg["is_read"]
        }
        for user_id in [current_user["id"], recipient["id"]]:
            if user_id in clients:
                await clients[user_id].send_json({"type": "message", "data": message_data})
        logger.info(f"Message sent from {current_user['username']} to {message.recipient_username} (Type: {message.type})")
        return {"msg": "Message sent", "message_id": msg["id"]}

@app.put("/messages/{message_id}")
async def edit_message(message_id: int, message: MessageUpdate, current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        msg = await conn.fetchrow(
            "SELECT sender_id, recipient_id, type FROM messages WHERE id = $1", message_id
        )
        if not msg or msg["sender_id"] != current_user["id"]:
            raise HTTPException(status_code=403, detail="Not authorized to edit this message")
        if msg["type"] in ["image", "audio", "friend_request"]:
            raise HTTPException(status_code=400, detail="Cannot edit this message type")
        updated_msg = await fetch_message_row(
            conn,
            "UPDATE messages SET content = $1 WHERE id = $2 RETURNING id, sender_id, recipient_id, content, type, timestamp, is_read",
            message.content, message_id
        )
        message_data = {
            "id": updated_msg["id"],
            "sender_username": current_user["username"],
            "recipient_username": (await conn.fetchval("SELECT username FROM users WHERE id = $1", updated_msg["recipient_id"])),
            "content": updated_msg["content"],
            "type": updated_msg["type"],
            "timestamp": updated_msg["timestamp"],
            "is_read": updated_msg["is_read"]
        }
        for user_id in [msg["sender_id"], msg["recipient_id"]]:
            if user_id in clients:
                await clients[user_id].send_json({"type": "edit", "data": message_data})
        logger.info(f"Message {message_id} edited by {current_user['username']}")
        return {"msg": "Message updated"}

@app.delete("/messages/{message_id}")
async def delete_message(message_id: int, current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        msg = await conn.fetchrow(
            "SELECT sender_id, recipient_id FROM messages WHERE id = $1", message_id
        )
        if not msg or (msg["sender_id"] != current_user["id"] and msg["recipient_id"] != current_user["id"]):
            raise HTTPException(status_code=403, detail="Not authorized to delete this message")
        await conn.execute("DELETE FROM messages WHERE id = $1", message_id)
        for user_id in [msg["sender_id"], msg["recipient_id"]]:
            if user_id in clients:
                await clients[user_id].send_json({"type": "delete", "data": {"id": message_id}})
        logger.info(f"Message {message_id} deleted by {current_user['username']}")
        return {"msg": "Message deleted"}

@app.delete("/conversations/{username}")
async def delete_conversation(username: str, current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        recipient = await conn.fetchrow("SELECT id FROM users WHERE username = $1", username)
        if not recipient:
            logger.warning(f"User not found for deletion: {username}")
            raise HTTPException(status_code=404, detail="User not found")
        await conn.execute(
            "DELETE FROM messages WHERE (sender_id = $1 AND recipient_id = $2) OR (sender_id = $2 AND recipient_id = $1)",
            current_user["id"], recipient["id"]
        )
        logger.info(f"Conversation with {username} deleted by {current_user['username']}")
        return {"msg": f"Conversation with {username} deleted"}

@app.get("/messages", response_model=List[Conversation])
async def get_conversations(current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        partners = await conn.fetch(
            """
            SELECT DISTINCT u.id, u.username
            FROM users u
            JOIN messages m ON u.id = m.sender_id OR u.id = m.recipient_id
            WHERE (m.sender_id = $1 OR m.recipient_id = $1) AND u.id != $1
            """, current_user["id"]
        )
        conversations = []
        for partner in partners:
            messages = await conn.fetch(
                """
                SELECT m.id, s.username as sender_username, r.username as recipient_username,
                       m.content, m.type, m.timestamp, m.is_read
                FROM messages m
                JOIN users s ON m.sender_id = s.id
                JOIN users r ON m.recipient_id = r.id
                WHERE (m.sender_id = $1 AND m.recipient_id = $2) OR (m.sender_id = $2 AND m.recipient_id = $1)
                ORDER BY m.timestamp ASC
                """, current_user["id"], partner["id"]
            )
            conversations.append({
                "username": partner["username"],
                "messages": [
                    {
                        "id": m["id"],
                        "sender_username": m["sender_username"],
                        "recipient_username": m["recipient_username"],
                        "content": m["content"],
                        "type": m["type"],
                        "timestamp": m["timestamp"].isoformat() + "Z",
                        "is_read": m["is_read"]
                    } for m in messages
                ]
            })
        logger.debug(f"Fetched {len(conversations)} conversations for {current_user['username']}")
        return conversations

@app.post("/messages/mark_read/{message_id}")
async def mark_message_read(message_id: int, current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        msg = await fetch_message_row(
            conn,
            """
            UPDATE messages SET is_read = TRUE
            WHERE id = $1 AND recipient_id = $2 AND is_read = FALSE
            RETURNING id, sender_id, recipient_id, content, type, timestamp, is_read
            """, message_id, current_user["id"]
        )
        if not msg:
            logger.warning(f"Message {message_id} not found or already read for {current_user['username']}")
            return {"msg": "Message not found or already read"}
        sender_id = msg["sender_id"]
        if sender_id in clients and sender_id != current_user["id"]:
            await clients[sender_id].send_json({
                "type": "read",
                "data": {"id": msg["id"]}
            })
            logger.debug(f"Notified sender {sender_id} of read message {msg['id']}")
        return {"msg": "Message marked as read"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str):
    await websocket.accept()
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
        async with db_pool.acquire() as conn:
            user = await conn.fetchrow("SELECT id, username FROM users WHERE id = $1", user_id)
            if not user:
                await websocket.close(code=1008)
                return
        clients[user_id] = websocket
        online_users[user_id] = user["username"]
        for client_id, client_ws in clients.items():
            await client_ws.send_json({"type": "status", "data": {"username": user["username"], "online": True}})
        logger.info(f"WebSocket connected: {user['username']} (ID: {user_id})")
        try:
            while True:
                data = await websocket.receive_json()
                if data["type"] == "typing":
                    for client_id, client_ws in clients.items():
                        if client_id != user_id:
                            await client_ws.send_json({"type": "typing", "data": {"username": user["username"], "isTyping": data["data"]["isTyping"]}})
        except WebSocketDisconnect:
            del clients[user_id]
            del online_users[user_id]
            for client_id, client_ws in clients.items():
                await client_ws.send_json({"type": "status", "data": {"username": user["username"], "online": False}})
            logger.info(f"WebSocket disconnected: {user['username']} (ID: {user_id})")
    except JWTError:
        await websocket.close(code=1008)
        logger.error("WebSocket connection failed: Invalid token")
