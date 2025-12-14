from pydantic import BaseModel, field_validator
from typing import List, Optional
from datetime import datetime, date
from enum import Enum
import re


class BookChunkBase(BaseModel):
    chunk_id: str
    book_version: str
    chapter: str
    section: str
    anchor: str
    content: str
    char_start: int
    char_end: int
    embedding_id: str

    @field_validator('chunk_id')
    @classmethod
    def validate_chunk_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Chunk ID cannot be empty')
        return v

    @field_validator('book_version')
    @classmethod
    def validate_book_version(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Book version cannot be empty')
        return v

    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Content cannot be empty')
        return v

    @field_validator('char_start', 'char_end')
    @classmethod
    def validate_char_positions(cls, v):
        if v < 0:
            raise ValueError('Character positions must be non-negative')
        return v

    @field_validator('char_end')
    @classmethod
    def validate_char_order(cls, v, info):
        # Accessing other field values through the field validation info
        # Since this validation happens per field, we can't access other fields in the same instance during validation
        # So this validation will need to be handled in the service layer after object creation
        if v < info.data.get('char_start', 0):
            raise ValueError('char_end must be greater than or equal to char_start')
        return v


class BookChunkCreate(BookChunkBase):
    pass


class BookChunk(BookChunkBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SessionBase(BaseModel):
    session_id: str
    user_id: Optional[str] = None

    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Session ID cannot be empty')
        return v


class SessionCreate(SessionBase):
    pass


class Session(SessionBase):
    id: int
    created_at: datetime
    updated_at: datetime
    last_activity: datetime

    class Config:
        from_attributes = True


class FeedbackBase(BaseModel):
    session_id: str
    query: str
    response: str
    rating: int  # 1 for thumbs up, -1 for thumbs down, 0 for neutral
    comment: Optional[str] = None

    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Session ID cannot be empty')
        return v

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Query cannot be empty')
        return v

    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v):
        if v not in [-1, 0, 1]:
            raise ValueError('Rating must be -1, 0, or 1')
        return v


class FeedbackCreate(FeedbackBase):
    pass


class Feedback(FeedbackBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class BookVersionBase(BaseModel):
    version: str
    title: str
    published_at: datetime
    is_active: bool = True

    @field_validator('version')
    @classmethod
    def validate_version(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Version cannot be empty')
        # Check that version follows common patterns like v1.0.0, 1.0.0, etc.
        if not re.match(r'^v?(\d+\.){2}\d+.*$', v) and not re.match(r'^\d+\.\d+\.\d+.*$', v):
            raise ValueError('Version must follow semantic versioning pattern (e.g., v1.0.0, 1.0.0)')
        return v

    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Title cannot be empty')
        if len(v) > 500:
            raise ValueError('Title must be 500 characters or less')
        return v


class BookVersionCreate(BookVersionBase):
    pass


class BookVersion(BookVersionBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class Citation(BaseModel):
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    chapter: Optional[str] = None
    url: Optional[str] = None

    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Citation text cannot be empty')
        return v


class QueryRequest(BaseModel):
    query: str
    selected_text: Optional[str] = None  # For selection-based queries
    session_id: Optional[str] = None

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Query cannot be empty')
        if len(v) > 1000:  # Limit query length
            raise ValueError('Query must be 1000 characters or less')
        return v


class QueryResponse(BaseModel):
    response: str
    citations: List[Citation]
    session_id: str
    query_time: datetime = datetime.now()


class FeedbackRequest(BaseModel):
    session_id: str
    query: str
    response: str
    rating: int  # 1 for thumbs up, -1 for thumbs down, 0 for neutral
    comment: Optional[str] = None

    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Session ID cannot be empty')
        return v

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Query cannot be empty')
        return v

    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v):
        if v not in [-1, 0, 1]:
            raise ValueError('Rating must be -1, 0, or 1')
        return v


class HealthResponse(BaseModel):
    status: str
    message: str