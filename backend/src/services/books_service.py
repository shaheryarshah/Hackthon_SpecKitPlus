from typing import List, Optional
from sqlalchemy.orm import Session
from ..db.models import BookVersion as BookVersionModel
from ..models.schemas import BookVersionCreate
from ..db.repositories import BookChunkRepository


class BookService:
    """
    Service layer for book-related operations
    """

    def __init__(self, db: Session):
        self.db = db
        self.chunk_repository = BookChunkRepository(db)

    def get_all_books(self, active_only: bool = True) -> List[BookVersionModel]:
        """
        Retrieve all book versions from the database
        """
        query = self.db.query(BookVersionModel)
        if active_only:
            query = query.filter(BookVersionModel.is_active == True)
        
        return query.all()

    def get_book_by_version(self, version: str) -> Optional[BookVersionModel]:
        """
        Retrieve a specific book version by its version identifier
        """
        return self.db.query(BookVersionModel).filter(BookVersionModel.version == version).first()

    def create_book(self, book_data: BookVersionCreate) -> BookVersionModel:
        """
        Create a new book version in the database
        """
        # Validate data before creating
        if not book_data.version or not book_data.title:
            raise ValueError("Version and title are required fields")

        # Check if version already exists
        existing_book = self.db.query(BookVersionModel).filter(BookVersionModel.version == book_data.version).first()
        if existing_book:
            raise ValueError(f"Book version '{book_data.version}' already exists")

        book = BookVersionModel(**book_data.model_dump())
        self.db.add(book)
        self.db.commit()
        self.db.refresh(book)
        return book

    def activate_book(self, version: str) -> BookVersionModel:
        """
        Activate a book version (and deactivate others)
        """
        # Find the version to activate
        book_to_activate = self.db.query(BookVersionModel).filter(BookVersionModel.version == version).first()
        if not book_to_activate:
            raise ValueError(f"Book version '{version}' not found")
        
        # Deactivate all other versions
        self.db.query(BookVersionModel).filter(BookVersionModel.version != version).update({"is_active": False})
        
        # Activate the requested version
        book_to_activate.is_active = True
        self.db.commit()
        self.db.refresh(book_to_activate)
        
        return book_to_activate

    def deactivate_book(self, version: str) -> bool:
        """
        Deactivate a book version
        """
        book = self.db.query(BookVersionModel).filter(BookVersionModel.version == version).first()
        if not book:
            return False
        
        book.is_active = False
        self.db.commit()
        
        return True

    def delete_book(self, version: str) -> bool:
        """
        Delete a book version (by deactivating it)
        """
        return self.deactivate_book(version)

    def get_book_chunks(self, version: str, chapter: Optional[str] = None, section: Optional[str] = None):
        """
        Get all chunks for a specific book version, with optional filters
        """
        # Check if book version exists
        book = self.get_book_by_version(version)
        if not book:
            raise ValueError(f"Book version '{version}' not found")
        
        # Use repository to get chunks
        if chapter and section:
            chunks = self.chunk_repository.get_chunks_by_section(version, section)
        elif chapter:
            # This is a simplified filter - would need more complex logic for proper chapter filtering
            chunks = self.chunk_repository.get_chunks_by_book_version(version)
            chunks = [c for c in chunks if chapter.lower() in c.chapter.lower()]
        else:
            chunks = self.chunk_repository.get_chunks_by_book_version(version)
        
        return chunks