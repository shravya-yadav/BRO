import React from 'react';

function SideBar({ onHistoryClick }) {
  const historyItems = [
    'How to prepare for interviews?',
    'What is LangChain?',
    'Summarize Gemini API',
    'What is the capital of France?',
    'Based on our last conversation, how does quantum computing work?',
    'Can you explain the concept of blockchain, as we discussed earlier?',
    "How do today's climate change statistics compare to what we researched before?",
    "What are the latest advancements in renewable energy, as mentioned previously?",
  ];

  return (
    <div className="sidebar">
      <h5 className="heading">Chat Recommendations 💡</h5>
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {historyItems.map((item, index) => (
          <li
            key={index}
            style={{ marginBottom: '1rem', cursor: 'pointer', color: '#FFFFFF' }}
            onClick={() => onHistoryClick(item)}
          >
            🔹 {item}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default SideBar;
