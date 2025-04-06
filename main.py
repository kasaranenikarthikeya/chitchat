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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
db_pool = None
SECRET_KEY = "your-secret-key-here-1234567890"  # Replace with a secure key in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
connections: Dict[str, WebSocket] = {}  # WebSocket connections by username

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL not set in .env")
    raise ValueError("DATABASE_URL not set in .env")

# Lifespan context manager for database connection
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
                    status VARCHAR(10) DEFAULT 'pending',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (sender_id, recipient_id)
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    sender_id INTEGER REFERENCES users(id),
                    recipient_id INTEGER REFERENCES users(id),
                    content TEXT NOT NULL,
                    type VARCHAR(20) NOT NULL DEFAULT 'text',
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

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://chitfront.onrender.com",  # Your frontend URL
        "https://chitchat-f4e6.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "ChitChat Backend is Live!"}

# Pydantic models
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

# Token creation
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Dependency to get current user
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

# Register endpoint
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

# Login endpoint
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

# Get current user endpoint
@app.get("/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return {"username": current_user["username"]}

# Search users endpoint
@app.get("/users", response_model=List[UserOut])
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
        return [{"id": u["id"], "username": u["username"]} for u in users]

# Get suggested users endpoint
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

# Helper function to fetch message row with formatted timestamp
async def fetch_message_row(conn, query, *args):
    row = await conn.fetchrow(query, *args)
    if row:
        return dict(row) | {"timestamp": row["timestamp"].isoformat() + "Z"}
    return None

# Helper function to check if users are friends
async def are_friends(conn, user_id: int, friend_id: int):
    result = await conn.fetchrow(
        "SELECT 1 FROM friends WHERE (user_id = $1 AND friend_id = $2) OR (user_id = $2 AND friend_id = $1)",
        user_id, friend_id
    )
    return bool(result)

# Send friend request endpoint
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
        if recipient["username"] in connections:
            await connections[recipient["username"]].send_json({"type": "message", "data": message_data})
        logger.info(f"Friend request sent from {current_user['username']} to {request.recipient_username}")
        return {"msg": "Friend request sent", "request_id": request_data["id"]}

# Respond to friend request endpoint
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
                "INSERT INTO friends (user_id, friend_id) VALUES ($1, $2), ($2, $1) ON CONFLICT DO NOTHING",
                current_user["id"], request["sender_id"]
            )
            logger.info(f"Friend request accepted: {current_user['username']} and {sender['username']}")
            if sender["username"] in connections:
                await connections[sender["username"]].send_json({
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

# Get friend requests endpoint
@app.get("/friend-requests", response_model=List[FriendRequestOut])
async def get_friend_requests(current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        requests = await conn.fetch(
            """
            SELECT fr.id, s.username AS sender_username, r.username AS recipient_username, fr.status, fr.timestamp
            FROM friend_requests fr
            JOIN users s ON fr.sender_id = s.id
            JOIN users r ON fr.recipient_id = r.id
            WHERE fr.recipient_id = $1 AND fr.status = 'pending'
            ORDER BY fr.timestamp DESC
            """, current_user["id"]
        )
        return [{
            "id": r["id"],
            "sender_username": r["sender_username"],
            "recipient_username": r["recipient_username"],
            "status": r["status"],
            "timestamp": r["timestamp"].isoformat() + "Z"
        } for r in requests]

# Send message endpoint
@app.post("/messages", response_model=MessageOut)
async def send_message(message: MessageCreate, current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        recipient = await conn.fetchrow(
            "SELECT id, username FROM users WHERE username = $1", message.recipient_username
        )
        if not recipient:
            raise HTTPException(status_code=404, detail="Recipient not found")
        if message.type != "friend_request" and not await are_friends(conn, current_user["id"], recipient["id"]):
            raise HTTPException(status_code=403, detail="You are not friends with this user")
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
        if recipient["username"] in connections:
            await connections[recipient["username"]].send_json({"type": "message", "data": message_data})
        if current_user["username"] in connections:
            await connections[current_user["username"]].send_json({"type": "message", "data": message_data})
        logger.info(f"Message sent from {current_user['username']} to {recipient['username']}")
        return message_data

# Get conversations endpoint
@app.get("/messages", response_model=List[Conversation])
async def get_conversations(current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        friends = await conn.fetch(
            """
            SELECT DISTINCT u.username
            FROM friends f
            JOIN users u ON (f.friend_id = u.id OR f.user_id = u.id)
            WHERE (f.user_id = $1 OR f.friend_id = $1) AND u.id != $1
            """, current_user["id"]
        )
        conversations = []
        for friend in friends:
            messages = await conn.fetch(
                """
                SELECT m.id, s.username AS sender_username, r.username AS recipient_username,
                       m.content, m.type, m.timestamp, m.is_read
                FROM messages m
                JOIN users s ON m.sender_id = s.id
                JOIN users r ON m.recipient_id = r.id
                WHERE (m.sender_id = $1 AND m.recipient_id = (SELECT id FROM users WHERE username = $2))
                   OR (m.recipient_id = $1 AND m.sender_id = (SELECT id FROM users WHERE username = $2))
                ORDER BY m.timestamp ASC
                """, current_user["id"], friend["username"]
            )
            conversations.append({
                "username": friend["username"],
                "messages": [{
                    "id": m["id"],
                    "sender_username": m["sender_username"],
                    "recipient_username": m["recipient_username"],
                    "content": m["content"],
                    "type": m["type"],
                    "timestamp": m["timestamp"].isoformat() + "Z",
                    "is_read": m["is_read"]
                } for m in messages]
            })
        logger.info(f"Fetched conversations for {current_user['username']}")
        return conversations

# Edit message endpoint
@app.put("/messages/{message_id}", response_model=MessageOut)
async def edit_message(message_id: int, update: MessageUpdate, current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        msg = await conn.fetchrow(
            "SELECT id, sender_id, recipient_id, content, type, timestamp, is_read FROM messages WHERE id = $1 AND sender_id = $2",
            message_id, current_user["id"]
        )
        if not msg:
            raise HTTPException(status_code=404, detail="Message not found or not authorized")
        recipient = await conn.fetchrow("SELECT username FROM users WHERE id = $1", msg["recipient_id"])
        await conn.execute(
            "UPDATE messages SET content = $1 WHERE id = $2", update.content, message_id
        )
        message_data = {
            "id": msg["id"],
            "sender_username": current_user["username"],
            "recipient_username": recipient["username"],
            "content": update.content,
            "type": msg["type"],
            "timestamp": msg["timestamp"].isoformat() + "Z",
            "is_read": msg["is_read"]
        }
        if recipient["username"] in connections:
            await connections[recipient["username"]].send_json({"type": "edit", "data": message_data})
        if current_user["username"] in connections:
            await connections[current_user["username"]].send_json({"type": "edit", "data": message_data})
        logger.info(f"Message {message_id} edited by {current_user['username']}")
        return message_data

# Delete message endpoint
@app.delete("/messages/{message_id}")
async def delete_message(message_id: int, current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        msg = await conn.fetchrow(
            "SELECT id, recipient_id FROM messages WHERE id = $1 AND sender_id = $2",
            message_id, current_user["id"]
        )
        if not msg:
            raise HTTPException(status_code=404, detail="Message not found or not authorized")
        recipient = await conn.fetchrow("SELECT username FROM users WHERE id = $1", msg["recipient_id"])
        await conn.execute("DELETE FROM messages WHERE id = $1", message_id)
        delete_data = {"id": message_id}
        if recipient["username"] in connections:
            await connections[recipient["username"]].send_json({"type": "delete", "data": delete_data})
        if current_user["username"] in connections:
            await connections[current_user["username"]].send_json({"type": "delete", "data": delete_data})
        logger.info(f"Message {message_id} deleted by {current_user['username']}")
        return {"msg": "Message deleted"}

# Mark message as read endpoint
@app.post("/messages/mark_read/{message_id}")
async def mark_message_as_read(message_id: int, current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        msg = await conn.fetchrow(
            "SELECT id, sender_id FROM messages WHERE id = $1 AND recipient_id = $2 AND is_read = FALSE",
            message_id, current_user["id"]
        )
        if not msg:
            raise HTTPException(status_code=404, detail="Message not found or already read")
        await conn.execute("UPDATE messages SET is_read = TRUE WHERE id = $1", message_id)
        sender = await conn.fetchrow("SELECT username FROM users WHERE id = $1", msg["sender_id"])
        read_data = {"id": message_id}
        if sender["username"] in connections:
            await connections[sender["username"]].send_json({"type": "read", "data": read_data})
        logger.info(f"Message {message_id} marked as read by {current_user['username']}")
        return {"msg": "Message marked as read"}

# Delete conversation endpoint
@app.delete("/conversations/{username}")
async def delete_conversation(username: str, current_user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        recipient = await conn.fetchrow("SELECT id FROM users WHERE username = $1", username)
        if not recipient:
            raise HTTPException(status_code=404, detail="User not found")
        if not await are_friends(conn, current_user["id"], recipient["id"]):
            raise HTTPException(status_code=403, detail="Not friends with this user")
        await conn.execute(
            """
            DELETE FROM messages
            WHERE (sender_id = $1 AND recipient_id = $2) OR (sender_id = $2 AND recipient_id = $1)
            """, current_user["id"], recipient["id"]
        )
        logger.info(f"Conversation with {username} deleted by {current_user['username']}")
        return {"msg": "Conversation deleted"}

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    token = websocket.query_params.get("token")
    logger.info(f"WebSocket connection attempt with token: {token[:10]}...")  # Log partial token for security
    if not token:
        logger.error("No token provided")
        await websocket.close(code=1008, reason="No token provided")
        return
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            logger.error("No sub claim in token")
            await websocket.close(code=1008, reason="Invalid token")
            return
        async with db_pool.acquire() as conn:
            user = await conn.fetchrow("SELECT id, username FROM users WHERE id = $1", int(user_id))
            if not user:
                logger.error(f"User with ID {user_id} not found")
                await websocket.close(code=1008, reason="Invalid user")
                return
        username = user["username"]
    except jwt.ExpiredSignatureError:
        logger.error("Token expired")
        await websocket.close(code=1008, reason="Token expired")
        return
    except (jwt.InvalidTokenError, ValueError) as e:
        logger.error(f"Invalid token: {str(e)}")
        await websocket.close(code=1008, reason="Invalid token")
        return

    connections[username] = websocket
    logger.info(f"User {username} connected via WebSocket")
    await websocket.send_json({"type": "status", "data": {"username": username, "online": True}})

    # Notify all connected users of online status
    for conn_username, conn in connections.items():
        if conn_username != username:
            await conn.send_json({"type": "status", "data": {"username": username, "online": True}})

    try:
        while True:
            data = await websocket.receive_json()
            logger.info(f"Received WebSocket data from {username}: {data}")
            if data.get("type") == "message":
                async with db_pool.acquire() as conn:
                    recipient = await conn.fetchrow(
                        "SELECT id, username FROM users WHERE username = $1",
                        data["data"]["recipient_username"]
                    )
                    if not recipient:
                        logger.error(f"Recipient {data['data']['recipient_username']} not found")
                        continue
                    if data["data"]["type"] != "friend_request" and not await are_friends(conn, user["id"], recipient["id"]):
                        logger.error(f"{username} not friends with {recipient['username']}")
                        continue
                    msg = await fetch_message_row(
                        conn,
                        "INSERT INTO messages (sender_id, recipient_id, content, type) VALUES ($1, $2, $3, $4) RETURNING id, sender_id, recipient_id, content, type, timestamp, is_read",
                        user["id"], recipient["id"], data["data"]["content"], data["data"]["type"]
                    )
                    msg_data = {
                        "id": msg["id"],
                        "sender_username": username,
                        "recipient_username": recipient["username"],
                        "content": msg["content"],
                        "type": msg["type"],
                        "timestamp": msg["timestamp"],
                        "is_read": msg["is_read"]
                    }
                    if recipient["username"] in connections:
                        await connections[recipient["username"]].send_json({"type": "message", "data": msg_data})
                    await websocket.send_json({"type": "message", "data": msg_data})
            elif data.get("type") == "typing":
                if data["data"]["username"] in connections:
                    await connections[data["data"]["username"]].send_json({
                        "type": "typing",
                        "data": {"username": username, "isTyping": data["data"]["isTyping"]}
                    })
    except WebSocketDisconnect:
        logger.info(f"User {username} disconnected from WebSocket")
        if username in connections:
            del connections[username]
        for conn in connections.values():
            await conn.send_json({"type": "status", "data": {"username": username, "online": False}})
    except Exception as e:
        logger.error(f"WebSocket error for {username}: {str(e)}")
        if username in connections:
            del connections[username]
        await websocket.close(code=1011, reason="Internal error")
