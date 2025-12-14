from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
from contextlib import contextmanager
import logging
from backend.src import config
import os

logger = logging.getLogger(__name__)

# Use SQLite for local development or configured database for production
# For development, allow override with USE_SQLITE environment variable
if config.settings.ENVIRONMENT == "development" or os.getenv("USE_SQLITE", "true").lower() == "true":
    # Use SQLite for local development
    DATABASE_URL = "sqlite:///./book_management.db"
else:
    DATABASE_URL = config.settings.DATABASE_URL

# Create the database engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=300,    # Recycle connections every 5 minutes
    echo=False,          # Set to True for SQL query logging
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}  # Required for SQLite
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Initialize the database by creating all tables
    """
    try:
        # Import all models to ensure they're registered with SQLAlchemy
        from backend.src.db.models import Base
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def test_connection():
    """
    Test the database connection
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False