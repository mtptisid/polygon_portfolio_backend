#!/usr/bin/env python3
"""
RAG status diagnostic script.

Checks whether the profile_embeddings table actually has data, and runs a
live retrieval to see what context (if any) the HR assistant would get for
a sample query. Use this to diagnose "Gemini says it has no profile info".

Usage:
    export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
    python backend/app/scripts/check_rag_status.py
"""

import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from myapp.services.embedding import EmbeddingService
from myapp.services.vector_store import VectorStoreManager
from myapp.services.rag_retriever import RAGRetriever

logging.basicConfig(level=logging.WARNING)  # keep noisy library logs quiet
logger = logging.getLogger(__name__)


def main():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Set DATABASE_URL first, e.g.:")
        print('  export DATABASE_URL="postgresql://user:pass@host:5432/dbname"')
        sys.exit(1)

    print("Connecting to vector store...")
    vector_store = VectorStoreManager(database_url)

    print("\n=== profile_embeddings stats ===")
    stats = vector_store.get_stats()
    print(f"Total embeddings: {stats['total_embeddings']}")
    print(f"Categories: {stats['categories']}")
    print(f"Table size: {stats['database_size']}")
    print(f"Oldest entry: {stats['oldest_entry']}")
    print(f"Newest entry: {stats['newest_entry']}")

    if stats["total_embeddings"] == 0:
        print("\n>>> Table is EMPTY. This is why the HR assistant has no profile context.")
        print(">>> Fix: run backend/app/scripts/ingest_profile_data.py against this DATABASE_URL.")
        return

    print("\n=== Live retrieval test ===")
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"Loading embedding model: {embedding_model} (first run downloads it, can take a while)...")
    embedding_service = EmbeddingService(model_name=embedding_model)
    retriever = RAGRetriever(vector_store, embedding_service)

    query = "What is Siddharamayya's experience with AI engineering and software engineering?"
    context = retriever.retrieve(query=query, top_k=5, timeout=10.0)

    print(f"\nQuery: {query}")
    print(f"Returned context length: {len(context)} chars")
    print("--- context preview ---")
    print(context[:1500] if context else "(EMPTY — no chunks matched)")


if __name__ == "__main__":
    main()
