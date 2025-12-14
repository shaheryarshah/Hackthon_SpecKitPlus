from typing import List, Dict, Any, Optional
from .qdrant_service import QdrantService
from ..db.repositories import BookChunkRepository
from ..db.database import get_db
from .. import config
import logging
import openai
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class RetrievalService:
    """
    Service for handling document retrieval from vector store and database
    """

    def __init__(self):
        try:
            self.qdrant_service = QdrantService()
            self.openai_client = openai.OpenAI(api_key=config.settings.OPENAI_API_KEY)
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize RetrievalService: {e}")
            self._initialized = False
    
    def retrieve_chunks(self,
                       query: str,
                       top_k: int = 5,
                       book_version: Optional[str] = None,
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks based on the query

        Args:
            query: The query to search for
            top_k: Number of results to return
            book_version: Specific book version to search in
            filters: Additional filters to apply

        Returns:
            List of retrieved chunks with metadata
        """
        if not self._initialized:
            logger.error("RetrievalService not properly initialized")
            # Return empty results instead of raising an exception
            return []

        try:
            # Generate embedding for the query using OpenAI
            response = self.openai_client.embeddings.create(
                input=query,
                model="text-embedding-ada-002"  # Using OpenAI's embedding model
            )
            query_embedding = response.data[0].embedding

            # Prepare filters
            search_filters = filters or {}
            if book_version:
                search_filters["book_version"] = book_version

            # Search in Qdrant
            results = self.qdrant_service.search_vectors(
                query_vector=query_embedding,
                top_k=top_k,
                filters=search_filters
            )

            logger.info(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Error retrieving chunks for query '{query}': {e}")
            # Return empty results instead of raising an exception to avoid breaking the UI
            return []
    
    def retrieve_chunks_by_selection(self,
                                   selected_text: str,
                                   top_k: int = 5,
                                   book_version: Optional[str] = None,
                                   char_start: Optional[int] = None,
                                   char_end: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve chunks specifically related to the selected text

        Args:
            selected_text: The selected text to search around
            top_k: Number of results to return
            book_version: Specific book version to search in
            char_start: Character start position (for additional filtering)
            char_end: Character end position (for additional filtering)

        Returns:
            List of retrieved chunks with metadata
        """
        try:
            # Generate embedding for the selected text
            response = self.openai_client.embeddings.create(
                input=selected_text,
                model="text-embedding-ada-002"
            )
            query_embedding = response.data[0].embedding

            # Prepare filters
            filters = {}
            if book_version:
                filters["book_version"] = book_version

            # Search in Qdrant
            results = self.qdrant_service.search_vectors(
                query_vector=query_embedding,
                top_k=top_k,
                filters=filters
            )

            # Apply character bounds filtering if provided
            if char_start is not None and char_end is not None:
                results = self._filter_by_character_bounds(results, char_start, char_end)

            logger.info(f"Retrieved {len(results)} chunks for selection: {selected_text[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Error retrieving chunks for selection '{selected_text}': {e}")
            raise

    def _filter_by_character_bounds(self,
                                  results: List[Dict[str, Any]],
                                  char_start: int,
                                  char_end: int) -> List[Dict[str, Any]]:
        """
        Filter results based on character position bounds

        Args:
            results: List of search results to filter
            char_start: Starting character position
            char_end: Ending character position

        Returns:
            Filtered list of results
        """
        try:
            filtered_results = []

            for result in results:
                payload = result.get("payload", {})

                # Get character bounds from the payload
                result_start = payload.get("char_start", 0)
                result_end = payload.get("char_end", float('inf'))

                # Check if the result chunk overlaps with the selected bounds
                if (result_start <= char_end and result_end >= char_start):
                    filtered_results.append(result)

            logger.info(f"Filtered results by character bounds: {len(results)} -> {len(filtered_results)}")
            return filtered_results
        except Exception as e:
            logger.error(f"Error filtering results by character bounds: {e}")
            # Return original results if filtering fails
            return results
    
    def get_chunk_content(self, chunk_id: str) -> Optional[str]:
        """
        Get the content of a specific chunk by ID
        
        Args:
            chunk_id: The ID of the chunk to retrieve
            
        Returns:
            Content of the chunk if found, None otherwise
        """
        try:
            # Get the chunk from Qdrant
            chunk_data = self.qdrant_service.get_point(chunk_id)
            if chunk_data:
                return chunk_data["payload"].get("content", "")
            
            return None
        except Exception as e:
            logger.error(f"Error retrieving chunk content for ID {chunk_id}: {e}")
            raise
    
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