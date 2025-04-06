// import React, { useState } from 'react';

// const ContactList = ({ contacts, onSelectContact, currentUser }) => {
//   const [search, setSearch] = useState('');
//   const filteredContacts = contacts.filter(contact =>
//     contact.username.toLowerCase().includes(search.toLowerCase())
//   );

//   return (
//     <div className="h-screen overflow-y-auto bg-white">
//       <div className="p-4 bg-green-600 text-white shadow-md">
//         <div className="flex items-center space-x-3">
//           <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center text-green-600 font-semibold text-xl shadow-inner">
//             {currentUser.username[0].toUpperCase()}
//           </div>
//           <div>
//             <h2 className="text-lg font-semibold">{currentUser.username}</h2>
//             <p className="text-sm text-green-200">Online</p>
//           </div>
//         </div>
//       </div>
//       <div className="p-4">
//         <div className="relative">
//           <input
//             type="text"
//             placeholder="Search or start new chat"
//             value={search}
//             onChange={(e) => setSearch(e.target.value)}
//             className="w-full pl-10 pr-4 py-2 bg-gray-100 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 text-base"
//           />
//           <svg
//             className="absolute left-3 top-2.5 h-6 w-6 text-gray-400"
//             fill="none"
//             stroke="currentColor"
//             viewBox="0 0 24 24"
//           >
//             <path
//               strokeLinecap="round"
//               strokeLinejoin="round"
//               strokeWidth={2}
//               d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
//             />
//           </svg>
//         </div>
//       </div>
//       <ul className="divide-y divide-gray-200">
//         {filteredContacts.map(contact => (
//           <li
//             key={contact.id}
//             onClick={() => onSelectContact(contact)}
//             role="button"
//             tabIndex={0}
//             onKeyPress={(e) => e.key === 'Enter' && onSelectContact(contact)}
//             className="p-4 hover:bg-gray-50 cursor-pointer transition-colors duration-150 shadow-sm hover:shadow-md rounded-lg"
//           >
//             <div className="flex items-center space-x-3">
//               <div className="relative">
//                 <div className="w-12 h-12 bg-gray-300 rounded-full flex items-center justify-center text-white font-semibold text-lg">
//                   {contact.username[0].toUpperCase()}
//                 </div>
//                 <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-600 rounded-full border-2 border-white animate-pulse"></div>
//               </div>
//               <div className="flex-1">
//                 <div className="flex items-center justify-between">
//                   <h3 className="font-semibold text-gray-900">{contact.username}</h3>
//                   <span className="text-xs text-green-600">12:30 PM</span>
//                 </div>
//                 <div className="flex items-center justify-between mt-1">
//                   <p className="text-sm text-gray-600 truncate">Hey, how are you?</p>
//                   <div className="flex items-center">
//                     <span className="bg-green-600 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
//                       2
//                     </span>
//                   </div>
//                 </div>
//               </div>
//             </div>
//           </li>
//         ))}
//       </ul>
//     </div>
//   );
// };

// export default ContactList;