// API service for book management
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api';

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  // Generic request method
  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `API error: ${response.status}`);
      }

      return data;
    } catch (error) {
      console.error(`API request error for ${url}:`, error);
      throw error;
    }
  }

  // Book version methods
  async getBooks(activeOnly = true) {
    const params = new URLSearchParams({ active_only: activeOnly });
    return this.request(`/books/?${params}`);
  }

  async getBookByVersion(version) {
    return this.request(`/books/${version}`);
  }

  async createBook(bookData) {
    return this.request('/books/', {
      method: 'POST',
      body: JSON.stringify(bookData),
    });
  }

  async activateBook(version) {
    return this.request(`/books/${version}/activate`, {
      method: 'PUT',
    });
  }

  async deactivateBook(version) {
    return this.request(`/books/${version}`, {
      method: 'DELETE',
    });
  }

  async getBookChunks(version, chapter = null, section = null) {
    const params = new URLSearchParams();
    if (chapter) params.append('chapter', chapter);
    if (section) params.append('section', section);
    
    return this.request(`/books/${version}/chunks?${params}`);
  }

  // Query methods
  async query(requestData) {
    return this.request('/query', {
      method: 'POST',
      body: JSON.stringify(requestData),
    });
  }

  // Feedback methods
  async submitFeedback(feedbackData) {
    return this.request('/feedback/submit', {
      method: 'POST',
      body: JSON.stringify(feedbackData),
    });
  }

  // Health check
  async getHealthStatus() {
    return this.request('/health/status');
  }
}

export default new ApiService();