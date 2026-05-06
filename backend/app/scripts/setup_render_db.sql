-- =============================================================================
-- Render Database Setup Script
-- =============================================================================
-- Run this script ONCE after creating your Render PostgreSQL database
-- This sets up pgvector extension and creates the necessary tables
-- =============================================================================

-- Enable pgvector extension (required for RAG system)
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify pgvector is installed
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Create profile_embeddings table for RAG system
CREATE TABLE IF NOT EXISTS profile_embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),  -- 384 dimensions for all-MiniLM-L6-v2
    metadata JSONB,
    category VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster similarity search
CREATE INDEX IF NOT EXISTS profile_embeddings_embedding_idx 
ON profile_embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index on category for filtering
CREATE INDEX IF NOT EXISTS profile_embeddings_category_idx 
ON profile_embeddings(category);

-- Verify tables are created
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;

-- Check if data exists
SELECT 
    category,
    COUNT(*) as count
FROM profile_embeddings
GROUP BY category
ORDER BY category;

-- =============================================================================
-- NOTES:
-- =============================================================================
-- 1. This script is idempotent (safe to run multiple times)
-- 2. The application will automatically ingest profile data on first startup
-- 3. If you see 0 rows, that's normal - data will be ingested automatically
-- 4. The embedding dimension (384) matches the all-MiniLM-L6-v2 model
-- =============================================================================
