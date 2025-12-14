from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from ..security.auth import get_current_session, check_rate_limit
from ..models.schemas import QueryRequest, QueryResponse, Citation
from ..services.retrieval_service import RetrievalService
from ..services.agent_service import AgentService
from ..db.database import get_db
from sqlalchemy.orm import Session
from ..utils import logging as log_utils
import os

router = APIRouter()

# Initialize services globally but allow for graceful failure
try:
    retrieval_service = RetrievalService()
    agent_service = AgentService()
except Exception as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to initialize services: {e}")
    retrieval_service = None
    agent_service = None


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    session_id: str = Depends(get_current_session)
):
    # Check rate limits
    if not check_rate_limit(session_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )

    # Check if services are properly initialized
    if retrieval_service is None or agent_service is None:
        import logging
        logger = logging.getLogger(__name__)
        logger.error("Query services not properly initialized")
        return QueryResponse(
            response="The system is not properly configured. Please contact the administrator to set up the required API keys.",
            citations=[],
            session_id=session_id
        )

    try:
        # Log the API call
        log_utils.log_api_call(
            endpoint="/query",
            method="POST",
            session_id=session_id,
            query=request.query
        )

        # If selected_text is provided, use selection-based retrieval
        if request.selected_text:
            # Retrieve chunks relevant to the selected text
            retrieved_chunks = retrieval_service.retrieve_chunks_by_selection(
                selected_text=request.selected_text,
                top_k=5
            )

            # Generate response based on selected text
            result = agent_service.generate_selection_based_response(
                query=request.query,
                selected_text=request.selected_text,
                retrieved_chunks=retrieved_chunks,
                session_id=session_id
            )
        else:
            # Retrieve relevant chunks for the query
            retrieved_chunks = retrieval_service.retrieve_chunks(
                query=request.query,
                top_k=5
            )

            # Generate response using the agent
            result = agent_service.generate_response(
                query=request.query,
                retrieved_chunks=retrieved_chunks,
                session_id=session_id
            )

        return QueryResponse(
            response=result["response"],
            citations=result["citations"],
            session_id=session_id
        )
    except Exception as e:
        # Log the error
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error processing query: {e}", exc_info=True)

        # Return a user-friendly error instead of raising an HTTP exception
        return QueryResponse(
            response="Sorry, I encountered an error processing your request. The system may not be properly configured with API keys.",
            citations=[],
            session_id=session_id
        )