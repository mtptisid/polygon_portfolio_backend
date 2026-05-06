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
    2. If empty, automatically runs data ingestion
    3. If data exists, skips ingestion
    4. Handles errors gracefully (app continues even if ingestion fails)
    
    This ensures data is automatically populated on first deployment to Render
    without requiring manual script execution.
    """
    try:
        from myapp.services.vector_store import VectorStoreManager
        from myapp.core.config import get_settings
        
        # Get settings
        settings = get_settings()
        
        # Check if DATABASE_URL is set
        if not settings.DATABASE_URL:
            logger.warning("DATABASE_URL not set - skipping RAG data initialization")
            return
        
        logger.info("Checking RAG data initialization status...")
        
        # Connect to vector store
        vector_store = VectorStoreManager(settings.DATABASE_URL)
        
        # Check if data already exists
        stats = vector_store.get_stats()
        total_embeddings = stats.get('total_embeddings', 0)
        
        if total_embeddings > 0:
            logger.info(f"RAG data already initialized ({total_embeddings} embeddings found)")
            logger.info(f"Categories: {stats.get('categories', {})}")
            vector_store.close()
            return
        
        logger.info("No RAG data found - starting automatic ingestion...")
        
        # Import ingestion module
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.ingest_profile_data import ProfileDataIngestion
        from myapp.services.embedding import EmbeddingService
        
        # Determine data directory
        data_dir = Path(__file__).parent.parent / 'data'
        
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            vector_store.close()
            return
        
        # Initialize services
        logger.info("Initializing embedding service...")
        embedding_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL)
        
        # Create ingestion handler
        logger.info("Creating ingestion handler...")
        ingestion = ProfileDataIngestion(
            data_dir=str(data_dir),
            embedding_service=embedding_service,
            vector_store=vector_store
        )
        
        # Run ingestion
        logger.info("Running data ingestion...")
        result = ingestion.ingest(dry_run=False)
        
        if result['success']:
            logger.info("=" * 60)
            logger.info("RAG Data Initialization Complete!")
            logger.info("=" * 60)
            logger.info(f"Total chunks: {result['total_chunks']}")
            logger.info(f"Categories: {', '.join(result['categories'])}")
            logger.info(f"Rows inserted: {result['rows_inserted']}")
            logger.info(f"Duration: {result['duration_seconds']:.2f} seconds")
            logger.info("=" * 60)
        else:
            logger.error(f"RAG data ingestion failed: {result.get('error', 'Unknown error')}")
        
        # Close vector store
        vector_store.close()
        
    except Exception as e:
        logger.error(f"Error during RAG data initialization: {e}")
        logger.warning("Application will continue with RAG in fallback mode")
        # Don't raise - allow application to start even if ingestion fails


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
