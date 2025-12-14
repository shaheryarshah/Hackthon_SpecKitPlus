from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from sqlalchemy.orm import Session
from ..db.database import get_db
from ..models.schemas import BookVersionCreate, BookVersion
from ..services.books_service import BookService

router = APIRouter()

@router.get("/", response_model=List[BookVersion])
def get_books(
    active_only: bool = Query(True, description="Return only active book versions"),
    db: Session = Depends(get_db)
):
    """
    Get all book versions, with option to filter to active versions only
    """
    try:
        book_service = BookService(db)
        books = book_service.get_all_books(active_only=active_only)
        return books
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving books: {str(e)}")


@router.get("/{version}", response_model=BookVersion)
def get_book(version: str, db: Session = Depends(get_db)):
    """
    Get a specific book version by its version identifier
    """
    try:
        book_service = BookService(db)
        book = book_service.get_book_by_version(version)
        if not book:
            raise HTTPException(status_code=404, detail="Book version not found")
        return book
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving book: {str(e)}")


@router.post("/", response_model=BookVersion)
def create_book(book_data: BookVersionCreate, db: Session = Depends(get_db)):
    """
    Create a new book version
    """
    try:
        book_service = BookService(db)
        book = book_service.create_book(book_data)
        return book
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating book: {str(e)}")


@router.put("/{version}/activate", response_model=BookVersion)
def activate_book(version: str, db: Session = Depends(get_db)):
    """
    Activate a book version (and deactivate others)
    """
    try:
        book_service = BookService(db)
        book = book_service.activate_book(version)
        return book
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error activating book: {str(e)}")


@router.delete("/{version}", response_model=dict)
def delete_book(version: str, db: Session = Depends(get_db)):
    """
    Delete a book version (soft delete by deactivation)
    """
    try:
        book_service = BookService(db)
        deleted = book_service.delete_book(version)
        if not deleted:
            raise HTTPException(status_code=404, detail="Book version not found")

        return {"message": f"Book version '{version}' deactivated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting book: {str(e)}")


@router.get("/{version}/chunks", response_model=List[dict])
def get_book_chunks(
    version: str,
    chapter: Optional[str] = Query(None, description="Filter by chapter"),
    section: Optional[str] = Query(None, description="Filter by section"),
    db: Session = Depends(get_db)
):
    """
    Get all chunks for a specific book version, with optional filters
    """
    try:
        book_service = BookService(db)
        chunks = book_service.get_book_chunks(version, chapter, section)

        # For now, return chunks with a simplified representation
        # In a real app, we'd want to return a proper schema
        return [{
            "chunk_id": chunk.chunk_id,
            "chapter": chunk.chapter,
            "section": chunk.section,
            "anchor": chunk.anchor,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end
        } for chunk in chunks]
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving book chunks: {str(e)}")