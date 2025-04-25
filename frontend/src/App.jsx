import React, { useState, useEffect } from 'react';
import ChatBox from '../components/ChatBox';
import SideBar from '../components/SideBar';
import './App.css';
import axios from 'axios';

// Remove trailing slash from backend URL if it exists
const BASE_URL = import.meta.env.VITE_BACKEND_URL.replace(/\/$/, '');

console.log("Using backend URL:", BASE_URL);

function App() {
  const [selectedHistory, setSelectedHistory] = useState(null);
  const [history, setHistory] = useState([]);
  const [showSidebar, setShowSidebar] = useState(true);

  const handleHistoryClick = (query) => {
    setSelectedHistory(query);
    setShowSidebar(false); // Hide sidebar after first selection
  };

  const updateHistory = (prompt) => {
    setHistory((prev) => (prev.includes(prompt) ? prev : [...prev, prompt]));
  };

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await axios.get(`${BASE_URL}/get_history/user123`);
        const prompts = res.data.map(item => item.prompt);
        setHistory(prompts);
      } catch (err) {
        console.error('Error fetching history:', err);
      }
    };
    fetchHistory();
  }, []);

  return (
    <div className="app-wrapper">
      <div className="app-box">
        {showSidebar && (
          <SideBar onHistoryClick={handleHistoryClick} history={history} />
        )}
        <ChatBox selectedHistory={selectedHistory} onSendQuery={updateHistory} />
      </div>
    </div>
  );
}

export default App;