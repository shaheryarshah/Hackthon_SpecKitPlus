from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import os

# Add the backend/src directory to the path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from backend.src.api import query, query_selection, feedback, health, books
from backend.src.models import schemas
from backend.src.utils import logging as log_utils
from backend.src import config
import time


# Setup advanced logging
logger = log_utils.setup_logging(log_level="INFO", log_file="logs/app.log")

# Create FastAPI application instance
app = FastAPI(
    title=config.settings.PROJECT_NAME,
    description="API for RAG Chatbot Integration with Physical AI & Humanoid Robotics Book",
    version=config.settings.VERSION,
    debug=config.settings.DEBUG
)

# Add CORS middleware - configure origins as needed for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.settings.FRONTEND_URL],  # Use configured frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(query_selection.router, prefix="/api", tags=["query-selection"])
app.include_router(feedback.router, prefix="/api", tags=["feedback"])
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(books.router, prefix="/api/books", tags=["books"])


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    # Log API call with enhanced details
    log_utils.log_api_call(
        endpoint=request.url.path,
        method=request.method,
        query=request.query_params.get('query', None)
    )

    logger.info(f"{request.method} {request.url.path} - {process_time:.3f}s")
    return response


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Middleware to add security headers to responses"""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    logger.error(f"Request details: {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred"}
    )


# Root endpoint
@app.get("/")
async def root():
    return {"message": "RAG Chatbot API for Physical AI & Humanoid Robotics Book"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)