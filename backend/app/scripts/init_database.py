#!/usr/bin/env python3
"""
Database Initialization Helper Script

This script initializes the PostgreSQL database schema for RAG profile retrieval.
It reads and executes the init_db_schema.sql file to set up:
- pgvector extension
- profile_embeddings table
- IVFFlat index for vector similarity search
- Indexes on category and technologies

Usage:
    python backend/app/scripts/init_database.py
    
Environment Variables:
    DATABASE_URL: PostgreSQL connection string (required)
                  Format: postgresql://user:pass@host:5432/dbname

Example:
    export DATABASE_URL="postgresql://user:pass@localhost:5432/mydb"
    python backend/app/scripts/init_database.py
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path to import myapp modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from myapp.services.vector_store import VectorStoreManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to initialize the database schema.
    """
    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        logger.error("DATABASE_URL environment variable is not set!")
        logger.error("Please set it using: export DATABASE_URL='postgresql://user:pass@host:5432/dbname'")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("PostgreSQL + pgvector Schema Initialization")
    logger.info("=" * 60)
    logger.info(f"Database URL: {database_url.split('@')[1] if '@' in database_url else 'hidden'}")
    
    try:
        # Create vector store manager
        logger.info("Creating VectorStoreManager...")
        vector_store = VectorStoreManager(database_url)
        
        # Initialize schema
        logger.info("Initializing database schema...")
        vector_store.initialize_schema()
        
        # Get and display stats
        logger.info("Retrieving database statistics...")
        stats = vector_store.get_stats()
        
        logger.info("=" * 60)
        logger.info("Schema Initialization Complete!")
        logger.info("=" * 60)
        logger.info(f"Total embeddings: {stats['total_embeddings']}")
        logger.info(f"Database size: {stats['database_size']}")
        
        if stats['categories']:
            logger.info("Categories:")
            for category, count in stats['categories'].items():
                logger.info(f"  - {category}: {count} embeddings")
        else:
            logger.info("No embeddings found (database is empty)")
        
        logger.info("=" * 60)
        logger.info("Next steps:")
        logger.info("1. Run the data ingestion script to populate the database:")
        logger.info("   python backend/app/scripts/ingest_profile_data.py")
        logger.info("2. Verify the data was ingested correctly")
        logger.info("3. Start the FastAPI application")
        logger.info("=" * 60)
        
        # Close connection pool
        vector_store.close()
        
        return 0
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("Schema Initialization Failed!")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        logger.error("")
        logger.error("Troubleshooting:")
        logger.error("1. Verify DATABASE_URL is correct")
        logger.error("2. Ensure PostgreSQL is running and accessible")
        logger.error("3. Check that you have CREATE privileges on the database")
        logger.error("4. Verify pgvector extension is available (version 0.5.0+)")
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
