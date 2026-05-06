"""
Vector Store Manager for PostgreSQL + pgvector operations.

This module provides a high-level interface for managing vector embeddings
in PostgreSQL with pgvector extension. It handles schema creation, insertion,
similarity search, and connection pooling.
"""

import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import execute_values
from typing import List, Optional, Dict, Any, Tuple
import logging
import json
from dataclasses import dataclass
from datetime import datetime
import os

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingData:
    """Data class for embedding insertion."""
    content: str
    embedding: List[float]
    category: str
    subcategory: Optional[str] = None
    technologies: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Data class for search results."""
    id: int
    content: str
    category: str
    subcategory: Optional[str]
    technologies: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]
    similarity_score: float
    created_at: datetime


class VectorStoreManager:
    """
    Manager for PostgreSQL + pgvector vector store operations.
    
    Features:
    - Connection pooling (max 5 connections)
    - Schema initialization with IVFFlat indexes
    - Batch insertion for efficiency
    - Similarity search with metadata filtering
    - Error handling and logging
    """
    
    def __init__(self, connection_string: str, min_connections: int = 1, max_connections: int = 5):
        """
        Initialize the vector store manager with connection pooling.
        
        Args:
            connection_string: PostgreSQL connection string (e.g., postgresql://user:pass@host:5432/dbname)
            min_connections: Minimum number of connections in pool
            max_connections: Maximum number of connections in pool (default 5 for Render free tier)
            
        Raises:
            Exception: If connection pool creation fails
        """
        self.connection_string = connection_string
        self.logger = logging.getLogger(__name__)
        
        try:
            self.logger.info("Creating PostgreSQL connection pool...")
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                min_connections,
                max_connections,
                connection_string
            )
            
            if self.connection_pool:
                self.logger.info(f"Connection pool created successfully (min={min_connections}, max={max_connections})")
            else:
                raise Exception("Failed to create connection pool")
                
        except Exception as e:
            self.logger.error(f"Failed to create connection pool: {e}")
            raise
    
    def _get_connection(self):
        """
        Get a connection from the pool.
        
        Returns:
            psycopg2 connection object
            
        Raises:
            Exception: If no connection available
        """
        try:
            conn = self.connection_pool.getconn()
            if conn:
                return conn
            else:
                raise Exception("No connection available from pool")
        except Exception as e:
            self.logger.error(f"Failed to get connection from pool: {e}")
            raise
    
    def _return_connection(self, conn):
        """
        Return a connection to the pool.
        
        Args:
            conn: psycopg2 connection object
        """
        try:
            self.connection_pool.putconn(conn)
        except Exception as e:
            self.logger.error(f"Failed to return connection to pool: {e}")
    
    def initialize_schema(self):
        """
        Initialize the database schema by executing the SQL initialization script.
        
        Creates:
        - pgvector extension
        - profile_embeddings table
        - IVFFlat index on embeddings
        - B-tree index on category
        - GIN index on technologies
        
        This method is idempotent (safe to run multiple times).
        
        Raises:
            Exception: If schema initialization fails
        """
        conn = None
        try:
            self.logger.info("Initializing database schema...")
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Read SQL script
            script_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'scripts',
                'init_db_schema.sql'
            )
            
            with open(script_path, 'r') as f:
                sql_script = f.read()
            
            # Execute schema initialization
            cursor.execute(sql_script)
            conn.commit()
            
            self.logger.info("Database schema initialized successfully")
            
            # Verify pgvector extension
            cursor.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
            result = cursor.fetchone()
            if result:
                self.logger.info(f"pgvector extension verified: version {result[1]}")
            else:
                self.logger.warning("pgvector extension not found!")
            
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize schema: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._return_connection(conn)
    
    def insert_embedding(
        self,
        content: str,
        embedding: List[float],
        category: str,
        subcategory: Optional[str] = None,
        technologies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Insert a single embedding into the database.
        
        Args:
            content: Text content of the chunk
            embedding: 384-dimensional embedding vector
            category: Category (e.g., "skills", "experience", "projects")
            subcategory: Optional subcategory
            technologies: Optional list of technologies
            metadata: Optional metadata as JSON
            
        Returns:
            ID of the inserted row
            
        Raises:
            Exception: If insertion fails
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Validate embedding dimension
            if len(embedding) != 384:
                raise ValueError(f"Expected 384-dimensional embedding, got {len(embedding)}")
            
            # Convert embedding to string format for pgvector
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            # Convert metadata to JSON string
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Insert query
            insert_query = """
                INSERT INTO profile_embeddings 
                (content, embedding, category, subcategory, technologies, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
            """
            
            cursor.execute(
                insert_query,
                (content, embedding_str, category, subcategory, technologies, metadata_json)
            )
            
            row_id = cursor.fetchone()[0]
            conn.commit()
            
            self.logger.debug(f"Inserted embedding with ID {row_id} (category: {category})")
            cursor.close()
            
            return row_id
            
        except Exception as e:
            self.logger.error(f"Failed to insert embedding: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._return_connection(conn)
    
    def insert_batch(self, embeddings: List[EmbeddingData]) -> int:
        """
        Insert multiple embeddings in a single batch operation.
        
        More efficient than individual inserts for large datasets.
        
        Args:
            embeddings: List of EmbeddingData objects to insert
            
        Returns:
            Number of rows inserted
            
        Raises:
            Exception: If batch insertion fails
        """
        conn = None
        try:
            if not embeddings:
                self.logger.warning("No embeddings to insert")
                return 0
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Prepare data for batch insert
            values = []
            for emb in embeddings:
                # Validate embedding dimension
                if len(emb.embedding) != 384:
                    raise ValueError(f"Expected 384-dimensional embedding, got {len(emb.embedding)}")
                
                embedding_str = '[' + ','.join(map(str, emb.embedding)) + ']'
                metadata_json = json.dumps(emb.metadata) if emb.metadata else None
                
                values.append((
                    emb.content,
                    embedding_str,
                    emb.category,
                    emb.subcategory,
                    emb.technologies,
                    metadata_json
                ))
            
            # Batch insert using execute_values for efficiency
            insert_query = """
                INSERT INTO profile_embeddings 
                (content, embedding, category, subcategory, technologies, metadata)
                VALUES %s
            """
            
            execute_values(cursor, insert_query, values)
            conn.commit()
            
            rows_inserted = len(values)
            self.logger.info(f"Batch inserted {rows_inserted} embeddings")
            cursor.close()
            
            return rows_inserted
            
        except Exception as e:
            self.logger.error(f"Failed to batch insert embeddings: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._return_connection(conn)
    
    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        category_filter: Optional[str] = None,
        tech_filter: Optional[List[str]] = None,
        similarity_threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for similar embeddings using cosine similarity.
        
        Args:
            query_embedding: 384-dimensional query embedding vector
            top_k: Number of results to return (default 5)
            category_filter: Optional category to filter by
            tech_filter: Optional list of technologies to filter by
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of SearchResult objects ordered by similarity (highest first)
            
        Raises:
            Exception: If search fails
        """
        conn = None
        try:
            # Validate embedding dimension
            if len(query_embedding) != 384:
                raise ValueError(f"Expected 384-dimensional embedding, got {len(query_embedding)}")
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Convert embedding to string format
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Build query with optional filters
            query_parts = ["""
                SELECT 
                    id,
                    content,
                    category,
                    subcategory,
                    technologies,
                    metadata,
                    created_at,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM profile_embeddings
                WHERE 1=1
            """]
            
            params = [embedding_str]
            
            # Add category filter
            if category_filter:
                query_parts.append("AND category = %s")
                params.append(category_filter)
            
            # Add technology filter
            if tech_filter:
                query_parts.append("AND technologies && %s")
                params.append(tech_filter)
            
            # Add similarity threshold
            if similarity_threshold > 0.0:
                query_parts.append("AND (1 - (embedding <=> %s::vector)) >= %s")
                params.extend([embedding_str, similarity_threshold])
            
            # Order by similarity and limit
            query_parts.append("""
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """)
            params.extend([embedding_str, top_k])
            
            query = ' '.join(query_parts)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to SearchResult objects
            results = []
            for row in rows:
                result = SearchResult(
                    id=row[0],
                    content=row[1],
                    category=row[2],
                    subcategory=row[3],
                    technologies=row[4],
                    metadata=row[5],
                    similarity_score=float(row[7]),
                    created_at=row[6]
                )
                results.append(result)
            
            self.logger.debug(f"Found {len(results)} similar embeddings (top_k={top_k})")
            cursor.close()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search similar embeddings: {e}")
            raise
        finally:
            if conn:
                self._return_connection(conn)
    
    def delete_by_category(self, category: str) -> int:
        """
        Delete all embeddings in a specific category.
        
        Useful for re-ingestion of updated data.
        
        Args:
            category: Category to delete (e.g., "skills", "experience")
            
        Returns:
            Number of rows deleted
            
        Raises:
            Exception: If deletion fails
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            delete_query = "DELETE FROM profile_embeddings WHERE category = %s"
            cursor.execute(delete_query, (category,))
            
            rows_deleted = cursor.rowcount
            conn.commit()
            
            self.logger.info(f"Deleted {rows_deleted} embeddings from category '{category}'")
            cursor.close()
            
            return rows_deleted
            
        except Exception as e:
            self.logger.error(f"Failed to delete embeddings by category: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._return_connection(conn)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary containing:
            - total_embeddings: Total number of embeddings
            - categories: Dict of category counts
            - database_size_mb: Approximate database size in MB
            - oldest_entry: Timestamp of oldest entry
            - newest_entry: Timestamp of newest entry
            
        Raises:
            Exception: If stats retrieval fails
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Total embeddings
            cursor.execute("SELECT COUNT(*) FROM profile_embeddings")
            total_embeddings = cursor.fetchone()[0]
            
            # Category counts
            cursor.execute("""
                SELECT category, COUNT(*) 
                FROM profile_embeddings 
                GROUP BY category 
                ORDER BY COUNT(*) DESC
            """)
            categories = dict(cursor.fetchall())
            
            # Database size (approximate)
            cursor.execute("""
                SELECT pg_size_pretty(pg_total_relation_size('profile_embeddings'))
            """)
            db_size = cursor.fetchone()[0]
            
            # Timestamp range
            cursor.execute("""
                SELECT MIN(created_at), MAX(created_at) 
                FROM profile_embeddings
            """)
            oldest, newest = cursor.fetchone()
            
            stats = {
                "total_embeddings": total_embeddings,
                "categories": categories,
                "database_size": db_size,
                "oldest_entry": oldest.isoformat() if oldest else None,
                "newest_entry": newest.isoformat() if newest else None
            }
            
            self.logger.info(f"Database stats: {total_embeddings} embeddings, {len(categories)} categories")
            cursor.close()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            raise
        finally:
            if conn:
                self._return_connection(conn)
    
    def close(self):
        """
        Close all connections in the pool.
        
        Should be called when shutting down the application.
        """
        try:
            if self.connection_pool:
                self.connection_pool.closeall()
                self.logger.info("Connection pool closed")
        except Exception as e:
            self.logger.error(f"Failed to close connection pool: {e}")
    
    def __del__(self):
        """Destructor to ensure connections are closed."""
        self.close()
