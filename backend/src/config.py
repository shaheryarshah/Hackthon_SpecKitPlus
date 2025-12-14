import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings loaded from environment variables"""

    # Environment setting
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Database settings
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://neondb_owner:npg_iMjtsfpeO2y4@ep-plain-dew-ahaoq9yl-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    )

    # Qdrant settings
    QDRANT_URL: str = os.getenv("QDRANT_URL", "https://e4898987-8eba-4a93-b45b-a614b7211419.us-east4-0.gcp.cloud.qdrant.io:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.eiEZMEy-OE3tO_C7Q2mis4mzIRQPhg1cHaF5eIROCJI")

    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-...")  # Use actual key in deployment

    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "fallback-secret-key-for-development")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    # Application settings
    PROJECT_NAME: str = "RAG Chatbot API"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Rate limiting
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # in seconds

    # Frontend URL for CORS
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")


# Create a single instance of settings
settings = Settings()