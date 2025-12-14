from sqlalchemy.orm import Session
from typing import List, Optional
from backend.src.db.models import Session as SessionModel, Feedback as FeedbackModel, BookChunk as BookChunkModel
from backend.src.models.schemas import SessionCreate, FeedbackCreate
import logging

logger = logging.getLogger(__name__)

class SessionRepository:
    """
    Repository for handling session-related database operations
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_session(self, session_data: SessionCreate) -> SessionModel:
        """
        Create a new session in the database
        """
        try:
            db_session = SessionModel(**session_data.model_dump())
            self.db.add(db_session)
            self.db.commit()
            self.db.refresh(db_session)
            logger.info(f"Created session with ID: {db_session.session_id}")
            return db_session
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            self.db.rollback()
            raise
    
    def get_session(self, session_id: str) -> Optional[SessionModel]:
        """
        Get a session by its ID
        """
        try:
            return self.db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            raise
    
    def update_session_activity(self, session_id: str):
        """
        Update the last activity timestamp for a session
        """
        try:
            session = self.get_session(session_id)
            if session:
                session.last_activity = SessionModel.updated_at.default.arg
                self.db.commit()
                logger.info(f"Updated activity for session {session_id}")
        except Exception as e:
            logger.error(f"Error updating session activity {session_id}: {e}")
            self.db.rollback()
            raise


class FeedbackRepository:
    """
    Repository for handling feedback-related database operations
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_feedback(self, feedback_data: FeedbackCreate) -> FeedbackModel:
        """
        Create a new feedback entry in the database
        """
        try:
            db_feedback = FeedbackModel(**feedback_data.model_dump())
            self.db.add(db_feedback)
            self.db.commit()
            self.db.refresh(db_feedback)
            logger.info(f"Created feedback for session {feedback_data.session_id}")
            return db_feedback
        except Exception as e:
            logger.error(f"Error creating feedback: {e}")
            self.db.rollback()
            raise
    
    def get_feedback_by_session(self, session_id: str) -> List[FeedbackModel]:
        """
        Get all feedback entries for a specific session
        """
        try:
            return self.db.query(FeedbackModel).filter(FeedbackModel.session_id == session_id).all()
        except Exception as e:
            logger.error(f"Error retrieving feedback for session {session_id}: {e}")
            raise
    
    def get_feedback_stats(self) -> dict:
        """
        Get feedback statistics (like thumbs up/down ratios)
        """
        try:
            total_feedback = self.db.query(FeedbackModel).count()
            positive_feedback = self.db.query(FeedbackModel).filter(FeedbackModel.rating == 1).count()
            negative_feedback = self.db.query(FeedbackModel).filter(FeedbackModel.rating == -1).count()
            
            stats = {
                "total": total_feedback,
                "positive": positive_feedback,
                "negative": negative_feedback,
                "positive_ratio": positive_feedback / total_feedback if total_feedback > 0 else 0,
                "negative_ratio": negative_feedback / total_feedback if total_feedback > 0 else 0
            }
            
            logger.info(f"Retrieved feedback stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error retrieving feedback stats: {e}")
            raise


class BookChunkRepository:
    """
    Repository for handling book chunk-related database operations
    """

    def __init__(self, db: Session):
        self.db = db

    def get_chunk_by_id(self, chunk_id: str) -> Optional[BookChunkModel]:
        """
        Get a book chunk by its ID
        """
        try:
            if not chunk_id or not chunk_id.strip():
                raise ValueError("Chunk ID cannot be empty")

            return self.db.query(BookChunkModel).filter(BookChunkModel.chunk_id == chunk_id.strip()).first()
        except ValueError as ve:
            logger.error(f"Invalid input for chunk ID: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            raise

    def get_chunks_by_book_version(self, book_version: str) -> List[BookChunkModel]:
        """
        Get all chunks for a specific book version
        """
        try:
            if not book_version or not book_version.strip():
                raise ValueError("Book version cannot be empty")

            return self.db.query(BookChunkModel).filter(BookChunkModel.book_version == book_version.strip()).all()
        except ValueError as ve:
            logger.error(f"Invalid input for book version: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error retrieving chunks for book version {book_version}: {e}")
            raise

    def get_chunks_by_section(self, book_version: str, section: str) -> List[BookChunkModel]:
        """
        Get chunks for a specific section of a book version
        """
        try:
            if not book_version or not book_version.strip():
                raise ValueError("Book version cannot be empty")
            if not section or not section.strip():
                raise ValueError("Section cannot be empty")

            return self.db.query(BookChunkModel).filter(
                BookChunkModel.book_version == book_version.strip(),
                BookChunkModel.section == section.strip()
            ).all()
        except ValueError as ve:
            logger.error(f"Invalid input for book version or section: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error retrieving chunks for section {section} in book {book_version}: {e}")
            raise

    def create_chunk(self, chunk_data) -> BookChunkModel:
        """
        Create a new book chunk in the database
        """
        try:
            # Validate the data before creating
            if chunk_data.char_end < chunk_data.char_start:
                raise ValueError(f"char_end ({chunk_data.char_end}) must be greater than or equal to char_start ({chunk_data.char_start})")

            # Check if chunk with this ID already exists
            existing_chunk = self.get_chunk_by_id(chunk_data.chunk_id)
            if existing_chunk:
                raise ValueError(f"Chunk with ID '{chunk_data.chunk_id}' already exists")

            # Create new chunk
            db_chunk = BookChunkModel(**chunk_data.model_dump())
            self.db.add(db_chunk)
            self.db.commit()
            self.db.refresh(db_chunk)
            logger.info(f"Created chunk with ID: {db_chunk.chunk_id}")
            return db_chunk
        except ValueError as ve:
            logger.error(f"Validation error when creating chunk: {ve}")
            self.db.rollback()
            raise
        except Exception as e:
            logger.error(f"Error creating chunk: {e}")
            self.db.rollback()
            raise