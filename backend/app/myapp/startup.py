"""
Application Startup Module

Handles initialization tasks that should run when the application starts,
including automatic data ingestion for RAG profile retrieval.
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


async def initialize_rag_data():
    """
    Initialize RAG data on application startup.
    
    This function:
    1. Checks if the vector database has data
    2. If empty, logs a warning (manual ingestion required)
    3. If data exists, logs stats
    4. Handles errors gracefully (app continues even if check fails)
    
    Note: Automatic ingestion is disabled to prevent startup timeouts.
    Run manual ingestion after deployment: python scripts/ingest_profile_data.py
    """
    try:
        from myapp.services.vector_store import VectorStoreManager
        from myapp.core.config import get_settings
        
        # Get settings
        settings = get_settings()
        
        # Check if DATABASE_URL is set
        if not settings.DATABASE_URL:
            logger.warning("DATABASE_URL not set - skipping RAG data check")
            return
        
        logger.info("Checking RAG data status...")
        
        # Connect to vector store
        vector_store = VectorStoreManager(settings.DATABASE_URL)
        
        # Check if data already exists
        try:
            stats = vector_store.get_stats()
            total_embeddings = stats.get('total_embeddings', 0)
            
            if total_embeddings > 0:
                logger.info(f"✓ RAG data found: {total_embeddings} embeddings")
                logger.info(f"✓ Categories: {stats.get('categories', {})}")
            else:
                logger.warning("⚠ No RAG data found - RAG will use fallback mode")
                logger.info("To ingest data: python scripts/ingest_profile_data.py")
        except Exception as e:
            logger.warning(f"Could not check RAG data: {e}")
        finally:
            vector_store.close()
        
    except Exception as e:
        logger.error(f"Error during RAG data check: {e}")
        logger.warning("Application will continue with RAG in fallback mode")


async def startup_tasks():
    """
    Run all startup tasks.
    
    This function is called when the FastAPI application starts.
    Add any additional startup tasks here.
    """
    logger.info("Running application startup tasks...")
    
    # Initialize RAG data
    await initialize_rag_data()
    
    logger.info("Startup tasks completed")
