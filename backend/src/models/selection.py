from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class SelectionBase(BaseModel):
    session_id: str
    book_version: str
    chapter: str
    section: str
    anchor: str
    content: str  # The selected text
    char_start: int  # Character start position in original text
    char_end: int   # Character end position in original text
    query: str  # The query asked about this selection


class SelectionCreate(SelectionBase):
    pass


class Selection(SelectionBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True