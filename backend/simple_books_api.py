from fastapi import FastAPI, Request, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import sqlite3
import json
from contextlib import contextmanager

# Create FastAPI application
app = FastAPI(
    title="Book Management System",
    description="API for managing book versions and content",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_FILE = "books.db"

def init_db():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Create books table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            published_at TEXT NOT NULL,
            is_active BOOLEAN DEFAULT 0,
            created_at TEXT NOT NULL
        )
    ''')
    
    # Create book chunks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS book_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT UNIQUE NOT NULL,
            book_version TEXT NOT NULL,
            chapter TEXT,
            section TEXT,
            anchor TEXT,
            content TEXT,
            char_start INTEGER,
            char_end INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (book_version) REFERENCES books (version)
        )
    ''')
    
    # Insert a default book version if none exists
    cursor.execute('''
        INSERT OR IGNORE INTO books 
        (version, title, published_at, is_active, created_at) 
        VALUES 
        (?, ?, ?, ?, ?)
    ''', (
        "v1.0.0", 
        "Physical AI & Humanoid Robotics Handbook", 
        datetime.now().isoformat(), 
        True, 
        datetime.now().isoformat()
    ))
    
    conn.commit()
    conn.close()

# Pydantic models
class BookVersionBase(BaseModel):
    version: str
    title: str
    published_at: datetime
    is_active: bool = False

class BookVersionCreate(BookVersionBase):
    pass

class BookVersion(BookVersionBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class BookChunkBase(BaseModel):
    chunk_id: str
    book_version: str
    chapter: Optional[str] = None
    section: Optional[str] = None
    anchor: Optional[str] = None
    content: str
    char_start: Optional[int] = None
    char_end: Optional[int] = None

class BookChunk(BookChunkBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Database connection context manager
@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row  # This allows us to access columns by name
    try:
        yield conn
    finally:
        conn.close()

# Helper function to convert SQLite row to dictionary
def row_to_dict(row):
    return {k: row[k] for k in row.keys()}

# API routes for book management
@app.on_event("startup")
def startup_event():
    init_db()

@app.get("/")
async def root():
    return {"message": "Book Management System API"}

@app.get("/books", response_model=List[BookVersion])
def get_books(
    active_only: bool = Query(True, description="Return only active book versions")
):
    """
    Get all book versions, with option to filter to active versions only
    """
    with get_db() as conn:
        if active_only:
            query = "SELECT * FROM books WHERE is_active = 1"
        else:
            query = "SELECT * FROM books"
        
        rows = conn.execute(query).fetchall()
        books = []
        for row in rows:
            book_dict = row_to_dict(row)
            book_dict['published_at'] = datetime.fromisoformat(book_dict['published_at'])
            book_dict['created_at'] = datetime.fromisoformat(book_dict['created_at'])
            books.append(BookVersion(**book_dict))
        
        return books

@app.get("/books/{version}", response_model=BookVersion)
def get_book(version: str):
    """
    Get a specific book version by its version identifier
    """
    with get_db() as conn:
        query = "SELECT * FROM books WHERE version = ?"
        row = conn.execute(query, (version,)).fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Book version not found")
        
        book_dict = row_to_dict(row)
        book_dict['published_at'] = datetime.fromisoformat(book_dict['published_at'])
        book_dict['created_at'] = datetime.fromisoformat(book_dict['created_at'])
        return BookVersion(**book_dict)

@app.post("/books", response_model=BookVersion)
def create_book(book_data: BookVersionCreate):
    """
    Create a new book version
    """
    with get_db() as conn:
        # Check if version already exists
        existing = conn.execute("SELECT * FROM books WHERE version = ?", (book_data.version,)).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="Book version already exists")
        
        # Insert new book
        created_at = datetime.now()
        conn.execute('''
            INSERT INTO books (version, title, published_at, is_active, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            book_data.version,
            book_data.title,
            book_data.published_at.isoformat(),
            book_data.is_active,
            created_at.isoformat()
        ))
        conn.commit()
        
        # Return the created book
        book_dict = {
            "id": conn.execute("SELECT last_insert_rowid()").fetchone()[0],
            "version": book_data.version,
            "title": book_data.title,
            "published_at": book_data.published_at,
            "is_active": book_data.is_active,
            "created_at": created_at
        }
        return BookVersion(**book_dict)

@app.put("/books/{version}/activate", response_model=BookVersion)
def activate_book(version: str):
    """
    Activate a book version (and deactivate others)
    """
    with get_db() as conn:
        # Check if book exists
        book_row = conn.execute("SELECT * FROM books WHERE version = ?", (version,)).fetchone()
        if not book_row:
            raise HTTPException(status_code=404, detail="Book version not found")
        
        # Deactivate all other versions
        conn.execute("UPDATE books SET is_active = 0")
        
        # Activate the requested version
        conn.execute("UPDATE books SET is_active = 1 WHERE version = ?", (version,))
        conn.commit()
        
        # Return the activated book
        book_dict = row_to_dict(book_row)
        book_dict['published_at'] = datetime.fromisoformat(book_dict['published_at'])
        book_dict['created_at'] = datetime.fromisoformat(book_dict['created_at'])
        book_dict['is_active'] = True
        
        return BookVersion(**book_dict)

@app.delete("/books/{version}", response_model=dict)
def delete_book(version: str):
    """
    Delete a book version (soft delete by deactivation)
    """
    with get_db() as conn:
        # Check if book exists
        book_row = conn.execute("SELECT * FROM books WHERE version = ?", (version,)).fetchone()
        if not book_row:
            raise HTTPException(status_code=404, detail="Book version not found")
        
        # Rather than hard delete, we'll deactivate the version
        conn.execute("UPDATE books SET is_active = 0 WHERE version = ?", (version,))
        conn.commit()
        
        return {"message": f"Book version '{version}' deactivated successfully"}

@app.get("/books/{version}/chunks", response_model=List[dict])
def get_book_chunks(
    version: str,
    chapter: Optional[str] = Query(None, description="Filter by chapter"),
    section: Optional[str] = Query(None, description="Filter by section")
):
    """
    Get all chunks for a specific book version, with optional filters
    """
    with get_db() as conn:
        # Check if book version exists
        book = conn.execute("SELECT * FROM books WHERE version = ?", (version,)).fetchone()
        if not book:
            raise HTTPException(status_code=404, detail="Book version not found")
        
        # Build query with optional filters
        query = "SELECT * FROM book_chunks WHERE book_version = ?"
        params = [version]
        
        if chapter:
            query += " AND chapter LIKE ?"
            params.append(f"%{chapter}%")
        
        if section:
            query += " AND section LIKE ?"
            params.append(f"%{section}%")
        
        rows = conn.execute(query, params).fetchall()
        chunks = []
        for row in rows:
            chunk_dict = row_to_dict(row)
            chunk_dict['created_at'] = datetime.fromisoformat(chunk_dict['created_at'])
            chunks.append(chunk_dict)
        
        return chunks

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)