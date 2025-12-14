from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional
from ..security.auth import get_current_session, check_rate_limit
from ..models.schemas import FeedbackRequest as FeedbackRequestSchema, HealthResponse
from ..db.repositories import FeedbackRepository
from ..db.database import get_db
from sqlalchemy.orm import Session
from ..utils import logging as log_utils

router = APIRouter()


@router.post("/feedback", response_model=HealthResponse)
async def feedback_endpoint(
    request: FeedbackRequestSchema,
    session_id: str = Depends(get_current_session)
):
    # Check rate limits
    if not check_rate_limit(session_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )

    try:
        # Log the API call
        log_utils.log_api_call(
            endpoint="/feedback",
            method="POST",
            session_id=session_id
        )

        # Create feedback entry in the database
        db = next(get_db())
        feedback_repo = FeedbackRepository(db)

        # Map the request to the schema expected by the repository
        from backend.src.models.schemas import FeedbackCreate
        feedback_data = FeedbackCreate(
            session_id=request.session_id,
            query=request.query,
            response=request.response,
            rating=request.rating,
            comment=request.comment
        )

        feedback_repo.create_feedback(feedback_data)

        return HealthResponse(status="success", message="Feedback received successfully")
    except Exception as e:
        # Log the error
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error processing feedback: {e}", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing your feedback"
        )
    finally:
        db.close()