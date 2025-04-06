// import React, { useState, useEffect, useRef } from 'react';
// import { HubConnectionBuilder, HubConnectionState } from '@microsoft/signalr';

// const ChatWindow = ({ currentUser, selectedContact }) => {
//   const [messages, setMessages] = useState([]);
//   const [message, setMessage] = useState('');
//   const [connection, setConnection] = useState(null);
//   const [connectionStatus, setConnectionStatus] = useState('Disconnected');
//   const messagesEndRef = useRef(null);

//   // Fetch messages from FastAPI
//   useEffect(() => {
//     const fetchMessages = async () => {
//       try {
//         const token = localStorage.getItem('token');
//         const response = await fetch(
//           `http://localhost:8000/messages/${currentUser.id}/${selectedContact.id}`,
//           { headers: { 'Authorization': `Bearer ${token}` } }
//         );
//         const data = await response.json();
//         setMessages(data);
//       } catch (e) {
//         console.error('Failed to fetch messages:', e);
//       }
//     };
//     fetchMessages();
//   }, [currentUser.id, selectedContact.id]);

//   // Set up SignalR connection
//   useEffect(() => {
//     const token = localStorage.getItem('token');
//     const newConnection = new HubConnectionBuilder()
//       .withUrl('http://localhost:5273/messagingHub', {
//         accessTokenFactory: () => token
//       })
//       .withAutomaticReconnect()
//       .build();
//     setConnection(newConnection);
//   }, []);
//   // In sendMessage:
//   // await connection.send('SendMessage', selectedContact.username, message);

//   useEffect(() => {
//     if (connection) {
//       const startConnection = async () => {
//         try {
//           await connection.start();
//           setConnectionStatus('Connected');
//           connection.on('ReceiveMessage', (senderId, message) => {
//             setMessages(prev => [...prev, { senderId, message }]);
//           });
//         } catch (e) {
//           console.log('Connection failed: ', e);
//           setConnectionStatus('Failed');
//         }
//       };
//       startConnection();
//       return () => {
//         if (connection) connection.stop();
//       };
//     }
//   }, [connection]);

//   // Scroll to latest message
//   useEffect(() => {
//     messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
//   }, [messages]);

//   // Send message via SignalR and FastAPI
//   const sendMessage = async () => {
//     if (connection && message.trim()) {
//       if (connection.state === HubConnectionState.Connected) {
//         try {
//           await connection.send('SendMessage', selectedContact.username, message);
//           const token = localStorage.getItem('token');
//           const payload = {
//             sender_id: currentUser.id,
//             receiver_id: selectedContact.id,
//             message: message
//           };
//           console.log('Sending payload:', payload);
//           const response = await fetch('http://localhost:8000/messages/', {
//             method: 'POST',
//             headers: {
//               'Content-Type': 'application/json',
//               'Authorization': `Bearer ${token}`
//             },
//             body: JSON.stringify(payload)
//           });
//           if (!response.ok) throw new Error('Failed to save message');
//           setMessages(prev => [...prev, { senderId: currentUser.id, message }]);
//           setMessage('');
//         } catch (e) {
//           console.error('Failed to send message:', e);
//         }
//       } else {
//         alert('Cannot send message: Not connected to the server.');
//       }
//     }
//   };

//   const handleKeyPress = (e) => {
//     if (e.key === 'Enter') sendMessage();
//   };

//   return (
//     <div className="h-screen flex flex-col">
//       <div className="p-4 bg-blue-600 text-white flex items-center justify-between">
//         <h2 className="text-lg font-semibold">{selectedContact?.username || 'Select a contact'}</h2>
//         <span className={`text-sm ${connectionStatus === 'Connected' ? 'text-green-200' : 'text-red-200'}`}>
//           {connectionStatus}
//         </span>
//       </div>
//       <div className="flex-1 overflow-y-auto p-4 bg-gray-50 space-y-3">
//         {messages.map((msg, index) => (
//           <div
//             key={index}
//             className={`p-3 rounded-lg max-w-[70%] ${
//               msg.senderId === currentUser.id
//                 ? 'ml-auto bg-blue-500 text-white'
//                 : 'mr-auto bg-white text-gray-800 shadow'
//             }`}
//           >
//             {msg.message}
//           </div>
//         ))}
//         <div ref={messagesEndRef} />
//       </div>
//       <div className="p-4 bg-white border-t flex items-center">
//         <input
//           value={message}
//           onChange={(e) => setMessage(e.target.value)}
//           onKeyPress={handleKeyPress}
//           placeholder="Type a message..."
//           className="flex-1 p-2 border rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
//           disabled={connectionStatus !== 'Connected'}
//         />
//         <button
//           onClick={sendMessage}
//           className="px-4 py-2 bg-blue-500 text-white rounded-r-lg hover:bg-blue-600 transition-colors disabled:bg-gray-400"
//           disabled={connectionStatus !== 'Connected'}
//         >
//           Send
//         </button>
//       </div>
//     </div>
//   );
// };

// export default ChatWindow;