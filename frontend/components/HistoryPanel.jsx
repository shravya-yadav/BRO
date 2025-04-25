import React from 'react';
import './HistoryPanel.css';

function HistoryPanel({ history, onSelect }) {
  return (
    <div className="history-panel">
      <h4 className="history-title">🕘 Chat History</h4>
      <div className="history-list">
        {history.length === 0 ? (
          <p className="muted">No history yet.</p>
        ) : (
          history.map((item, idx) => {
            // Handle both string and object formats
            const query = typeof item === 'string' ? item : item.query;
            const timestamp = typeof item === 'string' ? null : item.timestamp;

            return (
              <div
                key={idx}
                className="history-item"
                onClick={() => onSelect(query)}
                title={query}
              >
                <span className="history-text">
                  {query.length > 40 ? query.slice(0, 40) + '...' : query}
                </span>
                {timestamp && (
                  <span className="timestamp">
                    {new Date(timestamp).toLocaleTimeString()}
                  </span>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

export default HistoryPanel;
