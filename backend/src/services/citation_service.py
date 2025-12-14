from typing import List, Dict, Any, Optional
from backend.src.models.schemas import Citation
from backend.src import config
import logging

logger = logging.getLogger(__name__)

class CitationService:
    """
    Service for handling citation generation, validation, and formatting
    """
    
    def __init__(self):
        pass
    
    def validate_citation_format(self, citation_text: str) -> bool:
        """
        Validate if a citation is in the expected format
        
        Args:
            citation_text: The citation text to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic validation - check if it has expected elements
            # This is a simplified validation - in a real app, you might have more complex rules
            required_elements = ['chapter', 'section']  # Based on our citation format
            return all(element in citation_text.lower() for element in required_elements)
        except Exception as e:
            logger.error(f"Error validating citation format '{citation_text}': {e}")
            return False
    
    def format_citations(self, citations: List[Citation]) -> List[Dict[str, Any]]:
        """
        Format citations for API response or frontend display
        
        Args:
            citations: List of Citation objects
            
        Returns:
            List of formatted citation dictionaries
        """
        try:
            formatted_citations = []
            for citation in citations:
                formatted_citations.append({
                    "text": citation.text,
                    "page": citation.page,
                    "section": citation.section,
                    "chapter": citation.chapter,
                    "url": citation.url,
                    "formatted": self._format_single_citation(citation)
                })
            
            return formatted_citations
        except Exception as e:
            logger.error(f"Error formatting citations: {e}")
            raise
    
    def _format_single_citation(self, citation: Citation) -> str:
        """
        Format a single citation according to academic standards
        
        Args:
            citation: A single Citation object
            
        Returns:
            Formatted citation string
        """
        try:
            parts = []
            
            if citation.chapter:
                parts.append(f"Chapter {citation.chapter}")
            if citation.section:
                parts.append(f"Section {citation.section}")
            if citation.page:
                parts.append(f"Page {citation.page}")
            if citation.url:
                parts.append(f"URL: {citation.url}")
            
            return ", ".join(parts) if parts else "No citation details available"
        except Exception as e:
            logger.error(f"Error formatting single citation: {e}")
            return "Error formatting citation"
    
    def resolve_citation_urls(self, citations: List[Citation], book_version: str = None) -> List[Citation]:
        """
        Resolve citation URLs to actual document locations
        
        Args:
            citations: List of Citation objects to resolve
            book_version: Book version to use for URL resolution
            
        Returns:
            List of citations with resolved URLs
        """
        try:
            resolved_citations = []
            for citation in citations:
                # Create a proper URL based on the citation attributes
                if citation.url:
                    # If it's already a full URL, use as is
                    resolved_url = citation.url
                else:
                    # Construct a URL based on chapter and section
                    if citation.chapter and citation.section:
                        resolved_url = f"/docs/{book_version or 'latest'}/chapter-{citation.chapter}#section-{citation.section.replace('.', '-')}"
                    else:
                        resolved_url = f"/docs/{book_version or 'latest'}"
                
                # Create a new Citation with the resolved URL
                resolved_citation = Citation(
                    text=citation.text,
                    page=citation.page,
                    section=citation.section,
                    chapter=citation.chapter,
                    url=resolved_url
                )
                resolved_citations.append(resolved_citation)
            
            return resolved_citations
        except Exception as e:
            logger.error(f"Error resolving citation URLs: {e}")
            return citations  # Return original citations if resolution fails
    
    def extract_citation_references(self, text: str) -> List[str]:
        """
        Extract citation references from text (e.g., [1], [2], [Smith2020])
        
        Args:
            text: Text to extract citations from
            
        Returns:
            List of citation reference strings found in the text
        """
        try:
            import re
            # Pattern to match different citation formats like [1], [2, 3], [Smith et al., 2020]
            pattern = r'\[([^]]+)\]'
            matches = re.findall(pattern, text)
            return matches
        except Exception as e:
            logger.error(f"Error extracting citation references from text: {e}")
            return []
    
    def create_citation_from_metadata(self, 
                                    content_snippet: str, 
                                    chapter: str, 
                                    section: str, 
                                    anchor: str,
                                    page: Optional[int] = None) -> Citation:
        """
        Create a Citation object from content metadata
        
        Args:
            content_snippet: Snippet of the content being cited
            chapter: Chapter reference
            section: Section reference
            anchor: Anchor/URL reference
            page: Page number (optional)
            
        Returns:
            Citation object
        """
        try:
            citation_text = f"Chapter {chapter}, Section {section}"
            if page:
                citation_text += f", Page {page}"
            
            return Citation(
                text=citation_text,
                page=page,
                section=section,
                chapter=chapter,
                url=anchor
            )
        except Exception as e:
            logger.error(f"Error creating citation from metadata: {e}")
            # Return a basic citation in case of error
            return Citation(
                text="Error creating citation",
                page=None,
                section=None,
                chapter=None,
                url=None
            )