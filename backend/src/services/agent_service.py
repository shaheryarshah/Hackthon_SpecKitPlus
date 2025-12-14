from typing import List, Dict, Any, Optional
import openai
from .. import config
from ..models.schemas import Citation
from .retrieval_service import RetrievalService
from .qdrant_service import QdrantService
import logging
import re

logger = logging.getLogger(__name__)

class AgentService:
    """
    Service for handling OpenAI agent interactions with citation enforcement
    """

    def __init__(self):
        try:
            self.openai_client = openai.OpenAI(api_key=config.settings.OPENAI_API_KEY)
            self.retrieval_service = RetrievalService()
            self.qdrant_service = QdrantService()
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize AgentService: {e}")
            self._initialized = False
    
    def generate_response(self,
                         query: str,
                         retrieved_chunks: List[Dict[str, Any]],
                         session_id: str,
                         require_citations: bool = True) -> Dict[str, Any]:
        """
        Generate a response to the query using retrieved context

        Args:
            query: The user's query
            retrieved_chunks: Chunks retrieved from the vector store
            session_id: Session identifier
            require_citations: Whether to enforce citation generation

        Returns:
            Dictionary with response text and citations
        """
        if not self._initialized:
            logger.error("AgentService not properly initialized")
            # Return a default response instead of raising an exception
            return {
                "response": "The system is not properly configured. Please contact the administrator to set up the required API keys.",
                "citations": []
            }

        try:
            # Format the retrieved context for the prompt
            context = self._format_context_for_prompt(retrieved_chunks)

            # Build the prompt with citation requirements if needed
            if require_citations:
                system_prompt = self._get_citations_system_prompt()
            else:
                system_prompt = "You are a helpful assistant."

            # Construct the full prompt with context and query
            full_prompt = f"""
            Context information:
            {context}

            User query: {query}

            Please provide a detailed and accurate response based on the context provided.
            """

            # Add citation requirement to the prompt if needed
            if require_citations:
                full_prompt += """

                IMPORTANT: Every factual statement or claim in your response MUST be accompanied by a citation to the relevant source. Use the format [source_id] where source_id corresponds to the IDs in the context information above.
                """

            # Call the OpenAI API to generate the response
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # You can change this to gpt-4 if preferred
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent, factual responses
                max_tokens=1000
            )

            # Extract the response text
            response_text = response.choices[0].message.content

            # Extract citations from the response
            citations = self._extract_citations_from_response(response_text, retrieved_chunks)

            # Log the retrieval and response for observability
            from backend.src.utils import logging as log_utils
            log_utils.log_citation_event(citations, session_id)

            logger.info(f"Generated response with {len(citations)} citations for session {session_id}")

            return {
                "response": response_text,
                "citations": citations
            }
        except Exception as e:
            logger.error(f"Error generating response for query '{query}': {e}")
            # Return a more user-friendly error response
            return {
                "response": "Sorry, I encountered an error processing your request. The system may not be properly configured with API keys.",
                "citations": []
            }
    
    def _format_context_for_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format the retrieved chunks for inclusion in the prompt
        
        Args:
            chunks: List of retrieved chunks with metadata
            
        Returns:
            Formatted context string
        """
        formatted_context = ""
        for i, chunk in enumerate(chunks):
            payload = chunk.get("payload", {})
            formatted_context += f"""
            [Source {i+1} - ID: {chunk.get('id', 'unknown')}]
            Book Version: {payload.get('book_version', 'unknown')}
            Chapter: {payload.get('chapter', 'unknown')}
            Section: {payload.get('section', 'unknown')}
            Anchor: {payload.get('anchor', 'unknown')}
            Content: {payload.get('content', '')[:500]}...  # Limit content length
            """
        return formatted_context.strip()
    
    def _get_citations_system_prompt(self) -> str:
        """
        Get the system prompt that enforces citation requirements
        """
        return """
        You are an AI assistant that helps users with questions about the Physical AI & Humanoid Robotics book.
        Your responses must be grounded strictly in the provided context.
        Every factual statement must be accompanied by a citation to the relevant source using the format [source_id].
        Be concise but thorough in your answers.
        Do not hallucinate information that is not present in the context.
        """
    
    def _extract_citations_from_response(self, response_text: str, chunks: List[Dict[str, Any]]) -> List[Citation]:
        """
        Extract citations from the response text and match them to the original chunks
        
        Args:
            response_text: The response text that may contain citations
            chunks: The original chunks that were used to generate the response
            
        Returns:
            List of Citation objects
        """
        citations = []
        
        # Find citation references in the response (e.g., [Source 1], [1], etc.)
        # This is a simplified version - you might need a more sophisticated approach
        citation_pattern = r'\[Source (\d+)\]'  # Pattern to match citations like [Source 1]
        matches = re.findall(citation_pattern, response_text)
        
        # Create Citation objects for each match
        for match in matches:
            try:
                idx = int(match) - 1  # Convert to 0-based index
                if 0 <= idx < len(chunks):
                    chunk = chunks[idx]
                    payload = chunk.get("payload", {})
                    
                    citation = Citation(
                        text=f"Chapter {payload.get('chapter', 'N/A')}, Section {payload.get('section', 'N/A')}",
                        page=None,  # Could be added if page info is available
                        section=payload.get('section', 'N/A'),
                        chapter=payload.get('chapter', 'N/A'),
                        url=payload.get('anchor', 'N/A')
                    )
                    citations.append(citation)
            except (ValueError, IndexError):
                continue  # Skip invalid citation references
        
        # Also try to extract citations more broadly by matching chunk IDs
        for chunk in chunks:
            chunk_id = str(chunk.get('id', ''))
            if chunk_id in response_text:
                payload = chunk.get("payload", {})
                citation = Citation(
                    text=f"Chapter {payload.get('chapter', 'N/A')}, Section {payload.get('section', 'N/A')}",
                    page=None,
                    section=payload.get('section', 'N/A'),
                    chapter=payload.get('chapter', 'N/A'),
                    url=payload.get('anchor', 'N/A')
                )
                citations.append(citation)
        
        # Remove duplicates while preserving order
        unique_citations = []
        seen = set()
        for citation in citations:
            citation_key = (citation.chapter, citation.section, citation.url)
            if citation_key not in seen:
                seen.add(citation_key)
                unique_citations.append(citation)
        
        return unique_citations
    
    def generate_selection_based_response(self,
                                        query: str,
                                        selected_text: str,
                                        retrieved_chunks: List[Dict[str, Any]],
                                        session_id: str) -> Dict[str, Any]:
        """
        Generate a response based on the selected text context

        Args:
            query: The user's query
            selected_text: The text selected by the user
            retrieved_chunks: Chunks related to the selected text
            session_id: Session identifier

        Returns:
            Dictionary with response text and citations
        """
        if not self._initialized:
            logger.error("AgentService not properly initialized")
            # Return a default response instead of raising an exception
            return {
                "response": "The system is not properly configured. Please contact the administrator to set up the required API keys.",
                "citations": []
            }

        try:
            # Format the context from the selected text and related chunks
            context = f"""
            Selected Text: {selected_text}

            Related Context:
            {self._format_context_for_prompt(retrieved_chunks)}
            """

            # Build the prompt with the constraint to only use selected context
            system_prompt = """
            You are an AI assistant that helps users with questions about the Physical AI & Humanoid Robotics book.
            Your responses must be grounded strictly in the provided selected text and related context.
            Every factual statement must be accompanied by a citation to the relevant source.
            Do not introduce information outside of the provided context.
            """

            full_prompt = f"""
            Context information:
            {context}

            User query: {query}

            Please provide a detailed and accurate response based on ONLY the context provided.
            Do not use any external knowledge or make assumptions beyond what's in the context.

            IMPORTANT: Every factual statement in your response MUST be accompanied by a citation to the relevant source.
            """

            # Call the OpenAI API to generate the response
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            # Extract the response text
            response_text = response.choices[0].message.content

            # Extract citations from the response
            citations = self._extract_citations_from_response(response_text, retrieved_chunks)

            # Log the retrieval and response for observability
            from backend.src.utils import logging as log_utils
            log_utils.log_citation_event(citations, session_id)

            logger.info(f"Generated selection-based response with {len(citations)} citations for session {session_id}")

            return {
                "response": response_text,
                "citations": citations
            }
        except Exception as e:
            logger.error(f"Error generating selection-based response for query '{query}': {e}")
            # Return a more user-friendly error response
            return {
                "response": "Sorry, I encountered an error processing your request. The system may not be properly configured with API keys.",
                "citations": []
            }