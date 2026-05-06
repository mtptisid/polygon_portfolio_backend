-- PostgreSQL + pgvector Schema Initialization Script
-- This script sets up the complete schema for RAG profile retrieval
-- Safe to run multiple times (idempotent)

-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create profile_embeddings table
-- Stores text chunks with their vector embeddings and metadata
CREATE TABLE IF NOT EXISTS profile_embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),  -- 384 dimensions for all-MiniLM-L6-v2 model
    category VARCHAR(50),
    subcategory VARCHAR(100),
    technologies TEXT[],
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create IVFFlat index for efficient vector similarity search
-- Uses cosine distance for similarity comparison
-- lists=100 is optimal for ~200-500 vectors
CREATE INDEX IF NOT EXISTS idx_embedding 
ON profile_embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create B-tree index on category for filtered queries
CREATE INDEX IF NOT EXISTS idx_category 
ON profile_embeddings (category);

-- Create GIN index on technologies array for technology-based filtering
CREATE INDEX IF NOT EXISTS idx_technologies 
ON profile_embeddings 
USING GIN (technologies);

-- Create index on created_at for time-based queries
CREATE INDEX IF NOT EXISTS idx_created_at 
ON profile_embeddings (created_at);

-- Display schema information
DO $$
BEGIN
    RAISE NOTICE 'Schema initialization complete!';
    RAISE NOTICE 'Table: profile_embeddings';
    RAISE NOTICE 'Indexes: idx_embedding (IVFFlat), idx_category (B-tree), idx_technologies (GIN)';
END $$;
