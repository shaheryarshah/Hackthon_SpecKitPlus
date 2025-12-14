from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class BookChunk(Base):
    __tablename__ = "book_chunks"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(String, unique=True, index=True)  # Stable ID for the chunk
    book_version = Column(String)  # Version of the book
    chapter = Column(String)  # Chapter name/number
    section = Column(String)  # Section name
    anchor = Column(String)  # Anchor/URL for the section
    content = Column(Text)  # The actual content of the chunk
    char_start = Column(Integer)  # Character start position in original text
    char_end = Column(Integer)  # Character end position in original text
    embedding_id = Column(String)  # ID in vector store
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)  # Unique session identifier
    user_id = Column(String, index=True)  # User identifier (anonymous)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)  # Reference to session
    query = Column(Text)  # The original query
    response = Column(Text)  # The response that was rated
    rating = Column(Integer)  # 1 for thumbs up, -1 for thumbs down, 0 for neutral
    comment = Column(Text)  # Optional comment from user
    created_at = Column(DateTime, default=datetime.utcnow)


class BookVersion(Base):
    __tablename__ = "book_versions"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True, index=True)  # Book version identifier
    title = Column(String)  # Book title
    published_at = Column(DateTime)  # When this version was published
    is_active = Column(Boolean, default=True)  # Whether this is the current version
    created_at = Column(DateTime, default=datetime.utcnow)