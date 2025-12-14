import logging
import sys
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler


class LoggerSetup:
    """
    Setup and configure logging for the application with different levels for 
    console and file outputs, supporting structured logging for observability
    """
    
    def __init__(self, name: str = "rag_chatbot", log_level: str = "INFO", 
                 log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Avoid adding handlers multiple times
        if self.logger.handlers:
            return
        
        # Create formatters for console and file
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (with rotation)
        if log_file:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB files, keep 5
            )
            file_handler.setLevel(logging.DEBUG)  # Log everything to file
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def get_logger(self):
        return self.logger


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Set up logging for the application"""
    logger_setup = LoggerSetup("rag_chatbot", log_level, log_file)
    return logger_setup.get_logger()


def log_api_call(endpoint: str, method: str, user_id: Optional[str] = None, 
                 session_id: Optional[str] = None, query: Optional[str] = None):
    """Log API calls for observability"""
    logger = logging.getLogger("rag_chatbot")
    extra_info = []
    if user_id:
        extra_info.append(f"user_id={user_id}")
    if session_id:
        extra_info.append(f"session_id={session_id}")
    if query:
        extra_info.append(f"query='{query[:50]}...'")  # Only log first 50 chars of query
    
    extra_str = ", ".join(extra_info) if extra_info else "no additional info"
    logger.info(f"API CALL: {method} {endpoint} ({extra_str})")


def log_retrieval_event(query: str, results_count: int, retrieval_time: float, 
                       session_id: str):
    """Log retrieval events for observability"""
    logger = logging.getLogger("rag_chatbot")
    logger.info(f"RETRIEVAL: Query='{query[:50]}...', Results={results_count}, "
                f"Time={retrieval_time:.3f}s, Session={session_id}")


def log_citation_event(citations: list, session_id: str):
    """Log citation events for observability"""
    logger = logging.getLogger("rag_chatbot")
    logger.info(f"CITATION: Generated {len(citations)} citations for session {session_id}")