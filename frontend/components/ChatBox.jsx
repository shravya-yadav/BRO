import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { formatGeminiResponse } from '../src/utils/formatGeminiResponse.jsx';
import HistoryPanel from './HistoryPanel';
import './ChatBox.css';

// Normalize backend URL to avoid double slashes
const BASE_URL = import.meta.env.VITE_BACKEND_URL.replace(/\/$/, '');
console.log("Using backend URL:", BASE_URL);

function ChatBox({ selectedHistory }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const messagesEndRef = useRef(null);

  const userId = 'user123';

  useEffect(() => {
    if (selectedHistory) {
      setInput(selectedHistory);
      sendMessage(selectedHistory);
    }
  }, [selectedHistory]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const extractPrompt = (text) => {
    const periodIndex = text.indexOf('.');
    return periodIndex !== -1 ? text.substring(0, periodIndex + 1).trim() : text.trim();
  };

  const sendMessage = async (customInput = null) => {
    const finalInput = customInput || input;
    if (!finalInput.trim()) return;

    const userMsg = { sender: 'You', text: finalInput };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    const prompt = extractPrompt(finalInput);
    setHistory(prev => (prev.includes(prompt) ? prev : [...prev, prompt]));

    try {
      await axios.post(`${BASE_URL}/save_history`, {
        user_id: userId,
        query: prompt,
        response: ''
      });
    } catch (err) {
      console.error("Failed to save history:", err);
    }

    try {
      const res = await axios.post(`${BASE_URL}/chat`, {
        user_id: userId,
        message: finalInput
      });

      const aiMsg = { sender: 'BRO', text: res.data.response };
      setMessages(prev => [...prev, aiMsg]);
    } catch (error) {
      console.error(error);
    }

    setLoading(false);
  };

  const resetChat = async () => {
    setMessages([]);
    setInput('');
    try {
      await axios.post(`${BASE_URL}/start_session/${userId}`);
    } catch (err) {
      console.error("Failed to reset session:", err);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') sendMessage();
  };

  const toggleHistory = () => {
    setShowHistory(!showHistory);
  };

  return (
    <div className="chat-wrapper">
      <div className="chat-container">
        <div className="chat-section">
          <div className="chat-header">
            <h3>🤖 Chat with <span className="bro-text">BRO</span></h3>
            <div className="chat-actions">
              <button onClick={resetChat}>Clear Chat</button>
              {messages.length > 0 && (
                <button onClick={toggleHistory} className="history-toggle">
                  {showHistory ? 'Hide History' : 'Show History'}
                </button>
              )}
            </div>
          </div>

          <div className="chat-messages">
            {messages.map((msg, idx) => (
              <div key={idx} className={`msg ${msg.sender === 'You' ? 'you' : 'bot'}`}>
                <div>
                  <strong>{msg.sender}:</strong>
                  <div>{msg.sender === 'BRO' ? formatGeminiResponse(msg.text) : msg.text}</div>
                </div>
              </div>
            ))}
            {loading && <div className="loading">Thinking...🤔</div>}
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input">
            <input
              type="text"
              placeholder="Type your message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <button onClick={() => sendMessage()}>Send</button>
          </div>
        </div>

        {/* History panel component - now togglable */}
        {showHistory && <HistoryPanel history={history} onSelect={sendMessage} />}
      </div>
    </div>
  );
}

export default ChatBox;