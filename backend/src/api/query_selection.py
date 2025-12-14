from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from ..security.auth import get_current_session, check_rate_limit
from ..models.schemas import QueryRequest, QueryResponse, Citation
from ..services.retrieval_service import RetrievalService
from ..services.agent_service import AgentService
from ..utils import logging as log_utils

router = APIRouter()
retrieval_service = RetrievalService()
agent_service = AgentService()


@router.post("/query-selection", response_model=QueryResponse)
async def query_selection_endpoint(request: QueryRequest, session_id: str = Depends(get_current_session)):
    # Check rate limits
    if not check_rate_limit(session_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )

    try:
        # Log the API call
        log_utils.log_api_call(
            endpoint="/query-selection",
            method="POST",
            session_id=session_id,
            query=request.query
        )

        # Validate that selected_text is provided
        if not request.selected_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Selected text is required for selection-based queries"
            )

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

        return QueryResponse(
            response=result["response"],
            citations=result["citations"],
            session_id=session_id
        )
    except Exception as e:
        # Log the error
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error processing selection query: {e}", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing your selection-based query"
        )