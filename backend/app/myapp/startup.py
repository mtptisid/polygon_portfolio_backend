"""
Application Startup Module

Handles initialization tasks that should run when the application starts,
including background RAG initialization.
"""

import logging
import asyncio

logger = logging.getLogger(__name__)

# Global state for RAG initialization
_rag_initialized = False
_rag_init_lock = asyncio.Lock()


async def initialize_rag_background():
    """
    Initialize RAG components in the background after server starts.
    
    This runs AFTER uvicorn binds to the port, so it doesn't block startup.
    Uses the same lazy initialization pattern as the routers.
    """
    global _rag_initialized
    
    async with _rag_init_lock:
        if _rag_initialized:
            logger.info("RAG already initialized, skipping background init")
            return
        
        try:
            logger.info("=" * 60)
            logger.info("Starting background RAG initialization...")
            logger.info("=" * 60)
            
            from myapp.services.embedding import EmbeddingService
            from myapp.services.vector_store import VectorStoreManager
            from myapp.services.rag_retriever import RAGRetriever
            from myapp.core.config import get_settings
            
            settings = get_settings()
            
            # Check if DATABASE_URL is set
            if not settings.DATABASE_URL:
                logger.warning("DATABASE_URL not set - RAG will use fallback mode")
                _rag_initialized = True
                return
            
            # Initialize embedding service (this downloads the model)
            logger.info("Loading embedding model (this may take 2-5 minutes)...")
            embedding_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL)
            
            # Initialize vector store
            logger.info("Connecting to vector store...")
            vector_store = VectorStoreManager(settings.DATABASE_URL)
            
            # Create RAG retriever
            logger.info("Creating RAG retriever...")
            rag_retriever = RAGRetriever(vector_store, embedding_service)
            
            # Check data status
            try:
                stats = vector_store.get_stats()
                total_embeddings = stats.get('total_embeddings', 0)
                
                if total_embeddings > 0:
                    logger.info(f"✓ RAG data found: {total_embeddings} embeddings")
                    logger.info(f"✓ Categories: {stats.get('categories', {})}")
                else:
                    logger.warning("⚠ No RAG data found - run: python scripts/ingest_profile_data.py")
            except Exception as e:
                logger.warning(f"Could not check RAG data: {e}")
            
            _rag_initialized = True
            
            logger.info("=" * 60)
            logger.info("✓ Background RAG initialization complete")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"✗ Background RAG initialization failed: {e}")
            logger.warning("RAG endpoints will use fallback mode")
            _rag_initialized = True  # Mark as attempted to avoid retries
