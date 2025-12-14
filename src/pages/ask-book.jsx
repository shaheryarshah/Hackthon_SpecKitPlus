import React, { useState, useEffect, useRef } from 'react';

const AskBookPage = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [selectedText, setSelectedText] = useState('');
  const [useSelection, setUseSelection] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef(null);

  /* -------------------- Auto Scroll -------------------- */
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  /* -------------------- Text Selection -------------------- */
  useEffect(() => {
    const handleSelection = () => {
      const text = window.getSelection().toString();
      if (text && text.length > 3) {
        setSelectedText(text);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => document.removeEventListener('mouseup', handleSelection);
  }, []);

  /* -------------------- Send Message -------------------- */
  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const sessionId =
      localStorage.getItem('session_id') ||
      (() => {
        const id = Date.now().toString();
        localStorage.setItem('session_id', id);
        return id;
      })();

    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userMessage.text,
          selected_text: useSelection ? selectedText : null,
          session_id: sessionId
        })
      });

      const data = await response.json();

      if (!response.ok) throw new Error('API Error');

      const botMessage = {
        id: Date.now() + 1,
        text: formatCitations(data.response),
        sender: 'bot'
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          id: Date.now() + 2,
          text: '⚠️ Sorry, something went wrong. Please try again.',
          sender: 'bot'
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  /* -------------------- Citation Formatter -------------------- */
  const formatCitations = (text = '') => {
    return text.replace(/\[(\d+)\]/g, (m, p1) =>
      `<a href="/docs/reference#${p1}" target="_blank" rel="noopener noreferrer">[${p1}]</a>`
    );
  };

  /* -------------------- Enter Key -------------------- */
  const handleKeyDown = e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  /* -------------------- UI -------------------- */
  return (
    <div className="container margin-vert--lg">
      <h1>Ask the Physical AI & Humanoid Robotics Book</h1>

      <p>
        Ask questions about the textbook.
        {selectedText && (
          <>
            <br />
            <strong>Selected text:</strong>{' '}
            <em>{selectedText.slice(0, 120)}...</em>
            <button
              style={{ marginLeft: 10 }}
              onClick={() => setUseSelection(v => !v)}
            >
              {useSelection ? 'Disable' : 'Use'} Selection
            </button>
          </>
        )}
      </p>

      {/* -------------------- Chat Window -------------------- */}
      <div
        style={{
          height: '60vh',
          overflowY: 'auto',
          border: '1px solid #ddd',
          padding: 12,
          borderRadius: 6
        }}
      >
        {messages.map(msg => (
          <div
            key={msg.id}
            style={{
              maxWidth: '75%',
              marginBottom: 10,
              padding: 10,
              borderRadius: 8,
              background: msg.sender === 'user' ? '#007cba' : '#f2f2f2',
              color: msg.sender === 'user' ? '#fff' : '#000',
              marginLeft: msg.sender === 'user' ? 'auto' : 0
            }}
            dangerouslySetInnerHTML={{ __html: msg.text }}
          />
        ))}

        {isLoading && (
          <div style={{ fontStyle: 'italic', color: '#666' }}>
            Thinking...
          </div>
        )}

        <div ref={chatEndRef} />
      </div>

      {/* -------------------- Input -------------------- */}
      <div style={{ marginTop: 10 }}>
        <textarea
          value={inputValue}
          onChange={e => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about the book..."
          style={{
            width: '100%',
            minHeight: 70,
            padding: 10,
            borderRadius: 6
          }}
          disabled={isLoading}
        />
        <button
          onClick={handleSendMessage}
          disabled={isLoading || !inputValue.trim()}
          style={{
            marginTop: 8,
            padding: '10px 20px',
            background: '#007cba',
            color: '#fff',
            border: 'none',
            borderRadius: 6
          }}
        >
          Send
        </button>
      </div>

      <p style={{ marginTop: 10, fontSize: 12, color: '#666' }}>
        All responses include citations from the book.
      </p>
    </div>
  );
};

export default AskBookPage;
