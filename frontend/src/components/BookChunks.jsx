import React, { useState, useEffect } from 'react';
import api from '../services/api';

const BookChunks = ({ bookVersion }) => {
  const [chunks, setChunks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedChapter, setSelectedChapter] = useState('');
  const [chapters, setChapters] = useState([]);

  useEffect(() => {
    if (bookVersion) {
      fetchChunks();
    }
  }, [bookVersion]);

  useEffect(() => {
    if (chunks.length > 0) {
      // Extract unique chapters from chunks
      const uniqueChapters = [...new Set(chunks.map(chunk => chunk.chapter))];
      setChapters(uniqueChapters);
    }
  }, [chunks]);

  const fetchChunks = async () => {
    try {
      setLoading(true);
      const data = await api.getBookChunks(bookVersion);
      setChunks(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const filteredChunks = selectedChapter
    ? chunks.filter(chunk => chunk.chapter === selectedChapter)
    : chunks;

  if (!bookVersion) {
    return (
      <div className="container">
        <p>Please select a book version to view its chunks.</p>
      </div>
    );
  }

  if (loading) return <div className="container">Loading chunks...</div>;
  if (error) return <div className="container">Error: {error}</div>;

  return (
    <div className="container margin-vert--lg">
      <div className="row">
        <div className="col col--12">
          <div style={{ marginBottom: '20px' }}>
            <h1>Book Chunks for {bookVersion}</h1>
            <p>Content chunks for the selected book version</p>
          </div>

          {/* Chapter filter */}
          <div style={{ marginBottom: '20px' }}>
            <label htmlFor="chapter-filter">Filter by Chapter:</label>
            <select
              id="chapter-filter"
              value={selectedChapter}
              onChange={(e) => setSelectedChapter(e.target.value)}
              style={{ marginLeft: '10px', padding: '5px' }}
            >
              <option value="">All Chapters</option>
              {chapters.map((chapter, index) => (
                <option key={index} value={chapter}>{chapter}</option>
              ))}
            </select>
          </div>

          {/* Chunks table */}
          <div>
            <h2>Content Chunks</h2>
            {filteredChunks.length === 0 ? (
              <p>No chunks found.</p>
            ) : (
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={{ border: '1px solid #ddd', padding: '8px' }}>Chunk ID</th>
                    <th style={{ border: '1px solid #ddd', padding: '8px' }}>Chapter</th>
                    <th style={{ border: '1px solid #ddd', padding: '8px' }}>Section</th>
                    <th style={{ border: '1px solid #ddd', padding: '8px' }}>Start</th>
                    <th style={{ border: '1px solid #ddd', padding: '8px' }}>End</th>
                    <th style={{ border: '1px solid #ddd', padding: '8px' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredChunks.map((chunk, index) => (
                    <tr key={index}>
                      <td style={{ border: '1px solid #ddd', padding: '8px' }}>{chunk.chunk_id}</td>
                      <td style={{ border: '1px solid #ddd', padding: '8px' }}>{chunk.chapter}</td>
                      <td style={{ border: '1px solid #ddd', padding: '8px' }}>{chunk.section}</td>
                      <td style={{ border: '1px solid #ddd', padding: '8px' }}>{chunk.char_start}</td>
                      <td style={{ border: '1px solid #ddd', padding: '8px' }}>{chunk.char_end}</td>
                      <td style={{ border: '1px solid #ddd', padding: '8px' }}>
                        <button
                          onClick={() => {
                            // In a real app, this might open a modal to view the full content
                            alert(`Chunk ${chunk.chunk_id} content would be displayed here`);
                          }}
                          style={{
                            padding: '5px 10px',
                            backgroundColor: '#007cba',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: 'pointer'
                          }}
                        >
                          View
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default BookChunks;