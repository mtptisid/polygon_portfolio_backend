"""
Configuration settings for the application.

This module defines all configuration parameters using Pydantic's BaseSettings
for environment variable management and validation.

Usage:
    from myapp.core.config import get_settings
    
    settings = get_settings()
    print(settings.DATABASE_URL)
    print(settings.EMBEDDING_MODEL)

Environment Variables:
    # RAG Configuration
    DATABASE_URL: PostgreSQL connection string (required)
    EMBEDDING_MODEL: HuggingFace model name (default: sentence-transformers/all-MiniLM-L6-v2)
    RETRIEVAL_TOP_K: Number of chunks to retrieve (default: 5, range: 1-20)
    RETRIEVAL_TIMEOUT_MS: Query timeout in milliseconds (default: 2000, min: 100)
    ENABLE_RAG: Enable RAG retrieval (default: true)
    
    # HR Assistant & Admin Panel
    ADMIN_PASSWORD: Admin panel password (required for admin access)
    SECRET_KEY: JWT signing key (required, minimum 32 characters)
    JWT_EXPIRATION_HOURS: JWT token expiration in hours (default: 24)
    HR_ASSISTANT_MODEL: Default LLM model for HR assistant (default: gemini)
    SESSION_STORAGE_PATH: Path for session JSON files (default: backend/app/data/recruiter_sessions/)

Example .env file:
    DATABASE_URL=postgresql://user:password@localhost:5432/dbname
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
    RETRIEVAL_TOP_K=5
    RETRIEVAL_TIMEOUT_MS=2000
    ENABLE_RAG=true
    
    ADMIN_PASSWORD=your_secure_password_here
    SECRET_KEY=your_secret_key_minimum_32_characters
    JWT_EXPIRATION_HOURS=24
    HR_ASSISTANT_MODEL=gemini
    SESSION_STORAGE_PATH=backend/app/data/recruiter_sessions/
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application configuration settings.
    
    All settings can be overridden using environment variables.
    """
    
    # RAG Configuration
    # -----------------
    
    # PostgreSQL connection string for vector store
    # Format: postgresql://user:password@host:port/database
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    
    # HuggingFace embedding model name
    # Default: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL",
        description="HuggingFace model for generating text embeddings"
    )
    
    # Number of top results to retrieve from vector store
    RETRIEVAL_TOP_K: int = Field(
        default=5,
        env="RETRIEVAL_TOP_K",
        ge=1,
        le=20,
        description="Number of most relevant chunks to retrieve (1-20)"
    )
    
    # Timeout for retrieval operations in milliseconds
    RETRIEVAL_TIMEOUT_MS: int = Field(
        default=2000,
        env="RETRIEVAL_TIMEOUT_MS",
        ge=100,
        description="Timeout for vector search queries in milliseconds"
    )
    
    # Enable/disable RAG retrieval (fallback to basic profile if disabled)
    ENABLE_RAG: bool = Field(
        default=True,
        env="ENABLE_RAG",
        description="Enable RAG retrieval (false uses fallback mode)"
    )
    
    # HR Assistant & Admin Panel Configuration
    # -----------------------------------------
    
    # Note: ADMIN_PASSWORD and SECRET_KEY are read directly from environment
    # by auth.py to avoid validation errors. They are not part of this Settings class.
    
    # Default LLM model for HR assistant
    HR_ASSISTANT_MODEL: str = Field(
        default="gemini",
        env="HR_ASSISTANT_MODEL",
        description="Default LLM model for HR assistant (gemini/groq)"
    )
    
    # Session storage path
    SESSION_STORAGE_PATH: str = Field(
        default="backend/app/data/recruiter_sessions/",
        env="SESSION_STORAGE_PATH",
        description="Directory path for storing recruiter session JSON files"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance - will be initialized when DATABASE_URL is available
# Usage: from myapp.core.config import get_settings
def get_settings() -> Settings:
    """
    Get the application settings instance.
    
    Returns:
        Settings: The application settings
        
    Raises:
        ValidationError: If required environment variables are missing
    """
    return Settings()
