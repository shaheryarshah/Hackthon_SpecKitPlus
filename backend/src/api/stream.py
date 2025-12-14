from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from backend.src.security.auth import get_current_session, check_rate_limit
from backend.src.models.schemas import QueryRequest, QueryResponse, Citation
from backend.src.services.retrieval_service import RetrievalService
from backend.src.services.agent_service import AgentService
from backend.src.utils import logging as log_utils
from fastapi.responses import StreamingResponse
import json
import asyncio

router = APIRouter()
retrieval_service = RetrievalService()
agent_service = AgentService()


@router.post("/stream")
async def stream_endpoint(
    request: QueryRequest, 
    session_id: str = Depends(get_current_session)
):
    # Check rate limits
    if not check_rate_limit(session_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    async def generate_stream():
        try:
            # Log the API call
            log_utils.log_api_call(
                endpoint="/stream",
                method="POST", 
                session_id=session_id,
                query=request.query
            )
            
            # Retrieve relevant chunks for the query
            retrieved_chunks = retrieval_service.retrieve_chunks(
                query=request.query,
                top_k=5
            )
            
            # Use the agent to generate a streaming response
            # First, get the completion using OpenAI's streaming API
            import openai
            from backend.src import config
            client = openai.OpenAI(api_key=config.settings.OPENAI_API_KEY)
            
            # Format the retrieved context for the prompt
            context = _format_context_for_prompt(retrieved_chunks)
            
            # Build the full prompt with context and query
            full_prompt = f"""
            Context information:
            {context}
            
            User query: {request.query}
            
            Please provide a detailed and accurate response based on the context provided.
            Every factual statement or claim in your response MUST be accompanied by a citation to the relevant source. Use the format [source_id] where source_id corresponds to the IDs in the context information above.
            """
            
            # Call the OpenAI API with streaming
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # You can change this to gpt-4 if preferred
                messages=[
                    {"role": "system", "content": "You are an AI assistant that helps users with questions about the Physical AI & Humanoid Robotics book. Your responses must be grounded strictly in the provided context. Every factual statement must be accompanied by a citation to the relevant source using the format [source_id]. Be concise but thorough in your answers. Do not hallucinate information that is not present in the context."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                stream=True  # Enable streaming
            )
            
            # Stream the response
            full_response = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    
                    # Yield the content as JSON
                    yield f"data: {json.dumps({'content': content})}\n\n"
            
            # After streaming is complete, generate and return citations
            # This would happen after the full response is received
            from backend.src.services.citation_service import CitationService
            citation_service = CitationService()
            
            # Extract citations from the complete response
            citations = agent_service._extract_citations_from_response(full_response, retrieved_chunks)
            
            # Format citations
            formatted_citations = citation_service.format_citations(citations)
            
            # Send the final message with citations
            yield f"data: {json.dumps({'type': 'citations', 'citations': formatted_citations})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            # Log the error
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error processing stream query: {e}", exc_info=True)
            
            yield f"data: {json.dumps({'error': 'Error processing your query'})}\n\n"

    def _format_context_for_prompt(chunks):
        """Helper function to format context for the prompt"""
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
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")