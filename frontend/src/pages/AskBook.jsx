import React, { useState, useEffect } from 'react';
import { MessageBox } from 'react-chat-elements';
import 'react-chat-elements/dist/main.css';

const AskBookPage = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [selectedText, setSelectedText] = useState('');
  const [useSelection, setUseSelection] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Function to handle text selection
  useEffect(() => {
    const handleSelection = () => {
      const selectedText = window.getSelection().toString();
      if (selectedText) {
        setSelectedText(selectedText);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage = {
      id: Date.now(),
      message: inputValue,
      sender: 'user',
      position: 'right',
      date: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Prepare the request to backend
      const requestData = {
        query: inputValue,
        selected_text: useSelection ? selectedText : null,
        session_id: localStorage.getItem('session_id') || Date.now().toString()
      };

      // Send request to backend API
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      });

      const data = await response.json();

      if (response.ok) {
        const botMessage = {
          id: Date.now() + 1,
          message: formatCitations(data.response),
          sender: 'bot',
          position: 'left',
          date: new Date(),
          quote: true
        };

        setMessages(prev => [...prev, botMessage]);
      } else {
        const errorMessage = {
          id: Date.now() + 1,
          message: 'Sorry, I encountered an error processing your request.',
          sender: 'bot',
          position: 'left',
          date: new Date()
        };

        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: Date.now() + 1,
        message: 'Sorry, I encountered a connection error. Please try again.',
        sender: 'bot',
        position: 'left',
        date: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Function to format citations in the response
  const formatCitations = (text) => {
    // This is a simplified example - in a real implementation, 
    // you'd parse the text for citation markers and format them as links
    return text.replace(/\[(\d+)\]/g, (match, p1) => 
      `<a href="/docs/reference#${p1}" target="_blank" rel="noopener">[${p1}]</a>`
    );
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="container margin-vert--lg">
      <div className="row">
        <div className="col col--12">
          <div style={{ marginBottom: '20px' }}>
            <h1>Ask the Physical AI & Humanoid Robotics Book</h1>
            <p>
              Ask questions about the Physical AI & Humanoid Robotics textbook. 
              {selectedText && (
                <span>
                  {' '}Currently using selected text: <em>"{selectedText.substring(0, 100)}{selectedText.length > 100 ? '...' : ''}"</em>
                  <button 
                    onClick={() => setUseSelection(!useSelection)}
                    style={{ marginLeft: '10px', padding: '4px 8px', fontSize: '12px' }}
                  >
                    {useSelection ? 'Disable Selection Mode' : 'Use Selection Mode'}
                  </button>
                </span>
              )}
            </p>
          </div>

          <div style={{ 
            height: '60vh', 
            overflowY: 'auto', 
            border: '1px solid #ccc', 
            borderRadius: '4px', 
            padding: '10px',
            marginBottom: '10px'
          }}>
            {messages.map((msg) => (
              <div key={msg.id} className="margin-bottom--md">
                <MessageBox
                  key={msg.id}
                  text={msg.message}
                  position={msg.position}
                  date={msg.date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  dateString={true}
                />
              </div>
            ))}
            {isLoading && (
              <div className="margin-bottom--md">
                <MessageBox
                  text="Thinking..."
                  position="left"
                  date={new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  dateString={true}
                />
              </div>
            )}
          </div>

          <div style={{ display: 'flex', alignItems: 'center' }}>
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about the book..."
              style={{ 
                flex: 1, 
                padding: '10px', 
                border: '1px solid #ccc', 
                borderRadius: '4px',
                minHeight: '60px',
                resize: 'vertical'
              }}
              disabled={isLoading}
            />
            <button 
              onClick={handleSendMessage} 
              style={{ 
                marginLeft: '10px', 
                padding: '10px 20px', 
                backgroundColor: '#007cba', 
                color: 'white', 
                border: 'none', 
                borderRadius: '4px',
                cursor: isLoading ? 'not-allowed' : 'pointer'
              }}
              disabled={isLoading || !inputValue.trim()}
            >
              Send
            </button>
          </div>

          <div style={{ marginTop: '10px', fontSize: 'small', color: '#666' }}>
            <p>All responses include citations back to the Physical AI & Humanoid Robotics book.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AskBookPage;