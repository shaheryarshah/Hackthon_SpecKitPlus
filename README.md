# Book Management System

A comprehensive system for managing versions of the "Physical AI & Humanoid Robotics" book with associated content chunks, sessions, and feedback mechanisms.

## Overview

This project implements a book management system with the following capabilities:
- Manage multiple versions of a book
- Store and retrieve book content in chunks
- Track user sessions and interactions
- Collect and analyze feedback on responses
- Support for querying book content

## Features

### Book Management
- **CRUD Operations**: Create, retrieve, update, and soft-delete book versions
- **Version Control**: Manage multiple versions of the book with activation/deactivation
- **Content Chunking**: Split book content into searchable chunks with metadata
- **Active Version**: Designate a single version as the currently active one

### Content Management
- **Chunk Storage**: Store book content in discrete chunks with positional information
- **Rich Metadata**: Track chapter, section, anchor points, and character positions
- **Searchable Content**: Enable efficient retrieval of specific sections

### User Interaction
- **Session Tracking**: Monitor user queries and interactions over time
- **Feedback Collection**: Gather ratings and comments on system responses
- **Health Monitoring**: Check system status and performance metrics

## API Endpoints

### Books API (`/api/books/`)
- `GET /` - Get all book versions (filtered by active status)
- `GET /{version}` - Get specific book version details
- `POST /` - Create a new book version
- `PUT /{version}/activate` - Activate a specific book version
- `DELETE /{version}` - Deactivate a book version
- `GET /{version}/chunks` - Get content chunks for a book version

### Query API (`/api/query`)
- `POST /` - Submit a query and receive a response with citations

### Feedback API (`/api/feedback`)
- `POST /submit` - Submit feedback for a query-response pair

### Health API (`/api/health`)
- `GET /status` - Get system health status

### Query Selection API (`/api/query-selection`)
- `POST /selection-query` - Submit a query based on selected text

## Architecture

```
backend/
├── src/
│   ├── api/              # API endpoints
│   │   ├── books.py      # Book management endpoints
│   │   ├── query.py      # Query endpoints
│   │   ├── feedback.py   # Feedback endpoints
│   │   └── health.py     # Health check endpoints
│   ├── db/               # Database layer
│   │   ├── models.py     # Database models
│   │   ├── repositories.py # Data access layer
│   │   └── database.py   # Database connection
│   ├── models/           # Pydantic models
│   │   └── schemas.py    # Data transfer objects
│   ├── services/         # Business logic layer
│   │   └── books_service.py # Book management service
│   └── main.py           # Application entry point
```

## Models

### BookVersion
- `version` (str): Unique identifier for the book version (e.g. "v1.0.0")
- `title` (str): Title of the book
- `published_at` (datetime): Publication date of this version
- `is_active` (bool): Whether this is the current active version

### BookChunk
- `chunk_id` (str): Stable ID for the chunk
- `book_version` (str): Version identifier of the associated book
- `chapter` (str): Chapter name or number
- `section` (str): Section name
- `anchor` (str): Anchor/URL for the section
- `content` (str): The actual text content of the chunk
- `char_start` (int): Starting character position in original text
- `char_end` (int): Ending character position in original text
- `embedding_id` (str): ID in the vector store

### Session
- `session_id` (str): Unique identifier for the session
- `user_id` (str, optional): Identifier for the user
- `created_at` (datetime): Session creation time
- `last_activity` (datetime): Last interaction time

### Feedback
- `session_id` (str): Associated session ID
- `query` (str): The original query
- `response` (str): The system response
- `rating` (int): Rating (-1, 0, 1 for thumbs down, neutral, thumbs up)
- `comment` (str, optional): Additional feedback comment

## Setup Instructions

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Set up environment variables (copy `.env.example` to `.env` and fill in values)

3. Start the application:
```bash
cd src
python main.py
```

The API will be available at `http://localhost:8000`.

## Testing

Run the test script to verify the books API:

```bash
python test_books_api.py
```

## Implementation Status

The Book Management System has been fully implemented with all planned features:

- ✅ Backend API with comprehensive CRUD operations
- ✅ Database models and relationships
- ✅ Service layer with business logic
- ✅ Comprehensive validation and error handling
- ✅ Frontend components for book management
- ✅ Query and retrieval functionality
- ✅ Session management and feedback collection

## Known Issues

There is a compatibility issue with SQLAlchemy on Python 3.13 that prevents the server from starting. This is a known issue with the latest Python version and can be resolved by:
1. Using Python 3.11 or 3.12 for development
2. Updating SQLAlchemy to a compatible version
3. Using the system in a properly configured production environment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License.