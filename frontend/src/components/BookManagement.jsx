import React, { useState, useEffect } from 'react';
import api from '../services/api';

const BookManagement = () => {
  const [books, setBooks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [newBook, setNewBook] = useState({
    version: '',
    title: '',
    published_at: new Date().toISOString().split('T')[0] + 'T00:00:00',
    is_active: false
  });
  const [showForm, setShowForm] = useState(false);

  useEffect(() => {
    fetchBooks();
  }, []);

  const fetchBooks = async () => {
    try {
      setLoading(true);
      const data = await api.getBooks(false); // Get all books, not just active
      setBooks(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateBook = async (e) => {
    e.preventDefault();
    try {
      const createdBook = await api.createBook(newBook);
      setBooks([...books, createdBook]);
      setNewBook({
        version: '',
        title: '',
        published_at: new Date().toISOString().split('T')[0] + 'T00:00:00',
        is_active: false
      });
      setShowForm(false);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleActivateBook = async (version) => {
    try {
      const activatedBook = await api.activateBook(version);
      setBooks(books.map(book => 
        book.version === version 
          ? { ...book, is_active: true } 
          : { ...book, is_active: false }
      ));
    } catch (err) {
      setError(err.message);
    }
  };

  const handleDeactivateBook = async (version) => {
    try {
      await api.deactivateBook(version);
      setBooks(books.map(book => 
        book.version === version 
          ? { ...book, is_active: false } 
          : book
      ));
    } catch (err) {
      setError(err.message);
    }
  };

  if (loading) return <div className="container">Loading books...</div>;
  if (error) return <div className="container">Error: {error}</div>;

  return (
    <div className="container margin-vert--lg">
      <div className="row">
        <div className="col col--12">
          <div style={{ marginBottom: '20px' }}>
            <h1>Book Management</h1>
            <p>Manage different versions of the Physical AI & Humanoid Robotics book</p>
          </div>

          <div style={{ marginBottom: '20px' }}>
            <button 
              onClick={() => setShowForm(!showForm)}
              style={{
                padding: '10px 20px',
                backgroundColor: '#007cba',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              {showForm ? 'Cancel' : 'Add New Book Version'}
            </button>
            
            {showForm && (
              <form onSubmit={handleCreateBook} style={{ marginTop: '20px', padding: '20px', border: '1px solid #ccc', borderRadius: '4px' }}>
                <h3>Add New Book Version</h3>
                <div style={{ marginBottom: '10px' }}>
                  <label>Version:</label>
                  <input
                    type="text"
                    value={newBook.version}
                    onChange={(e) => setNewBook({...newBook, version: e.target.value})}
                    required
                    style={{ width: '100%', padding: '8px', marginTop: '4px' }}
                  />
                </div>
                <div style={{ marginBottom: '10px' }}>
                  <label>Title:</label>
                  <input
                    type="text"
                    value={newBook.title}
                    onChange={(e) => setNewBook({...newBook, title: e.target.value})}
                    required
                    style={{ width: '100%', padding: '8px', marginTop: '4px' }}
                  />
                </div>
                <div style={{ marginBottom: '10px' }}>
                  <label>Published Date:</label>
                  <input
                    type="datetime-local"
                    value={newBook.published_at.slice(0, 16)}
                    onChange={(e) => setNewBook({...newBook, published_at: e.target.value + ':00'})}
                    required
                    style={{ width: '100%', padding: '8px', marginTop: '4px' }}
                  />
                </div>
                <button 
                  type="submit"
                  style={{
                    padding: '10px 20px',
                    backgroundColor: '#28a745',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  Create Book
                </button>
              </form>
            )}
          </div>

          <div>
            <h2>Book Versions</h2>
            {books.length === 0 ? (
              <p>No books found.</p>
            ) : (
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={{ border: '1px solid #ddd', padding: '8px' }}>Version</th>
                    <th style={{ border: '1px solid #ddd', padding: '8px' }}>Title</th>
                    <th style={{ border: '1px solid #ddd', padding: '8px' }}>Published At</th>
                    <th style={{ border: '1px solid #ddd', padding: '8px' }}>Active</th>
                    <th style={{ border: '1px solid #ddd', padding: '8px' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {books.map((book) => (
                    <tr key={book.version}>
                      <td style={{ border: '1px solid #ddd', padding: '8px' }}>{book.version}</td>
                      <td style={{ border: '1px solid #ddd', padding: '8px' }}>{book.title}</td>
                      <td style={{ border: '1px solid #ddd', padding: '8px' }}>{new Date(book.published_at).toLocaleString()}</td>
                      <td style={{ border: '1px solid #ddd', padding: '8px' }}>
                        {book.is_active ? (
                          <span style={{ color: 'green', fontWeight: 'bold' }}>Active</span>
                        ) : (
                          <span style={{ color: 'red' }}>Inactive</span>
                        )}
                      </td>
                      <td style={{ border: '1px solid #ddd', padding: '8px' }}>
                        {!book.is_active ? (
                          <button
                            onClick={() => handleActivateBook(book.version)}
                            style={{
                              padding: '5px 10px',
                              backgroundColor: '#28a745',
                              color: 'white',
                              border: 'none',
                              borderRadius: '4px',
                              cursor: 'pointer',
                              marginRight: '5px'
                            }}
                          >
                            Activate
                          </button>
                        ) : (
                          <button
                            onClick={() => handleDeactivateBook(book.version)}
                            style={{
                              padding: '5px 10px',
                              backgroundColor: '#dc3545',
                              color: 'white',
                              border: 'none',
                              borderRadius: '4px',
                              cursor: 'pointer',
                              marginRight: '5px'
                            }}
                          >
                            Deactivate
                          </button>
                        )}
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

export default BookManagement;