from typing import List, Dict, Any, Optional
from backend.src.services.retrieval_service import RetrievalService
from backend.src.services.qdrant_service import QdrantService
from backend.src import config
import logging
import openai

logger = logging.getLogger(__name__)

class SelectionService:
    """
    Service for handling text selection-based queries and operations
    """
    
    def __init__(self):
        self.retrieval_service = RetrievalService()
        self.qdrant_service = QdrantService()
        self.openai_client = openai.OpenAI(api_key=config.settings.OPENAI_API_KEY)
    
    def process_selection_query(self,
                              query: str,
                              selected_text: str,
                              session_id: str,
                              book_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a query based on selected text, constraining responses to the selection
        
        Args:
            query: The user's query about the selected text
            selected_text: The text that was selected by the user
            session_id: Session identifier
            book_version: Specific book version to search in (optional)
            
        Returns:
            Dictionary with response and citations
        """
        try:
            # Retrieve chunks related to the selected text
            retrieved_chunks = self.retrieval_service.retrieve_chunks_by_selection(
                selected_text=selected_text,
                top_k=5,
                book_version=book_version
            )
            
            # Ensure retrieved chunks are relevant to the selection
            # This could involve additional filtering to ensure content is within
            # the selected text or closely related
            
            # Generate response using the agent service
            from backend.src.services.agent_service import AgentService
            agent_service = AgentService()
            
            result = agent_service.generate_selection_based_response(
                query=query,
                selected_text=selected_text,
                retrieved_chunks=retrieved_chunks,
                session_id=session_id
            )
            
            return result
        except Exception as e:
            logger.error(f"Error processing selection query: {e}")
            raise
    
    def validate_selection_context(self, 
                                 selected_text: str, 
                                 char_start: int, 
                                 char_end: int,
                                 original_content: str) -> bool:
        """
        Validate that the selected text matches the specified character bounds
        
        Args:
            selected_text: The selected text
            char_start: Start character position
            char_end: End character position
            original_content: The original content to validate against
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Extract the text from the original content using the character bounds
            extracted_text = original_content[char_start:char_end]
            
            # Compare the extracted text with the selected text
            # (ignoring minor whitespace differences)
            return extracted_text.strip() == selected_text.strip()
        except Exception as e:
            logger.error(f"Error validating selection context: {e}")
            return False
    
    def get_context_around_selection(self,
                                   selected_text: str,
                                   char_start: int,
                                   char_end: int,
                                   original_content: str,
                                   context_chars: int = 100) -> Dict[str, Any]:
        """
        Get additional context around the selected text
        
        Args:
            selected_text: The selected text
            char_start: Start character position
            char_end: End character position
            original_content: The original content
            context_chars: Number of characters of context to include before and after
            
        Returns:
            Dictionary with the selected text and surrounding context
        """
        try:
            # Calculate context bounds
            context_start = max(0, char_start - context_chars)
            context_end = min(len(original_content), char_end + context_chars)
            
            # Extract the context
            before_context = original_content[context_start:char_start]
            after_context = original_content[char_end:context_end]
            
            return {
                "selected_text": selected_text,
                "before_context": before_context,
                "after_context": after_context,
                "full_context": before_context + selected_text + after_context
            }
        except Exception as e:
            logger.error(f"Error getting context around selection: {e}")
            raise
    
    def enforce_selection_only_mode(self,
                                  query: str,
                                  selected_text: str,
                                  retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enforce that only chunks related to the selected text are used
        
        Args:
            query: The original query
            selected_text: The selected text
            retrieved_chunks: List of retrieved chunks
            
        Returns:
            Filtered list of chunks that are relevant to the selection
        """
        try:
            # This would involve more sophisticated filtering to ensure
            # that only information directly related to the selection is used
            
            # For now, return the retrieved chunks with a relevance check
            # In a full implementation, we would verify that the chunks
            # actually relate to the selected text
            
            # This is a simplified approach - a production system might use
            # semantic similarity between the selection and chunks,
            # or character position matching
            filtered_chunks = []
            selection_lower = selected_text.lower()
            
            for chunk in retrieved_chunks:
                payload = chunk.get("payload", {})
                content = payload.get("content", "").lower()
                
                # Check if the chunk content is related to the selected text
                if selection_lower in content or content in selection_lower:
                    filtered_chunks.append(chunk)
            
            logger.info(f"Filtered {len(retrieved_chunks)} to {len(filtered_chunks)} chunks for selection-only mode")
            return filtered_chunks
        except Exception as e:
            logger.error(f"Error enforcing selection-only mode: {e}")
            # Return original chunks if filtering fails
            return retrieved_chunks