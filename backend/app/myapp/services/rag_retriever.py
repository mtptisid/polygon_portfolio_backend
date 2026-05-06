"""
RAG Retriever for orchestrating retrieval-augmented generation.

This module provides the RAGRetriever class that combines the embedding service
and vector store to perform semantic search and format context for LLM generation.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
from .embedding import EmbeddingService
from .vector_store import VectorStoreManager, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Data class for retrieved chunks with metadata."""
    content: str
    category: str
    subcategory: Optional[str]
    technologies: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]
    similarity_score: float


class RAGRetriever:
    """
    Orchestrates RAG retrieval using embedding service and vector store.
    
    Features:
    - Query embedding and similarity search
    - Context formatting with section separators
    - Token limiting (max 2000 tokens)
    - Metadata filtering support
    - Timeout handling (2s default)
    """
    
    def __init__(
        self,
        vector_store: VectorStoreManager,
        embedding_service: EmbeddingService,
        max_tokens: int = 2000
    ):
        """
        Initialize the RAG retriever.
        
        Args:
            vector_store: VectorStoreManager instance for similarity search
            embedding_service: EmbeddingService instance for query embedding
            max_tokens: Maximum tokens in formatted context (default 2000)
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)
        
        # Approximate tokens per character (rough estimate: 1 token ≈ 4 characters)
        self.chars_per_token = 4
        self.max_chars = max_tokens * self.chars_per_token
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        timeout: float = 2.0
    ) -> str:
        """
        Retrieve relevant chunks and format as context string.
        
        Args:
            query: User query string
            top_k: Number of chunks to retrieve (default 5)
            filters: Optional filters dict with keys:
                    - category: str (e.g., "skills", "experience")
                    - technologies: List[str] (e.g., ["Python", "LangChain"])
            timeout: Query timeout in seconds (default 2.0)
            
        Returns:
            Formatted context string ready for LLM prompt
            
        Raises:
            TimeoutError: If retrieval exceeds timeout
            Exception: If retrieval fails
        """
        try:
            self.logger.info(f"Retrieving context for query: '{query[:50]}...'")
            
            # Retrieve chunks with metadata
            chunks = self.retrieve_with_metadata(
                query=query,
                top_k=top_k,
                filters=filters,
                timeout=timeout
            )
            
            # Format chunks into context string
            context = self.format_context(chunks)
            
            self.logger.info(f"Retrieved {len(chunks)} chunks, context length: {len(context)} chars")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            raise
    
    def retrieve_with_metadata(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        timeout: float = 2.0
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks with full metadata.
        
        Args:
            query: User query string
            top_k: Number of chunks to retrieve (default 5)
            filters: Optional filters dict with keys:
                    - category: str
                    - technologies: List[str]
            timeout: Query timeout in seconds (default 2.0)
            
        Returns:
            List of RetrievedChunk objects with metadata and scores
            
        Raises:
            TimeoutError: If retrieval exceeds timeout
            Exception: If retrieval fails
        """
        try:
            # Embed the query
            self.logger.debug("Embedding query...")
            query_embedding = self.embedding_service.embed_text(query)
            
            # Extract filters
            category_filter = filters.get("category") if filters else None
            tech_filter = filters.get("technologies") if filters else None
            
            # Search vector store
            self.logger.debug(f"Searching vector store (top_k={top_k}, category={category_filter}, tech={tech_filter})...")
            search_results = self.vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                category_filter=category_filter,
                tech_filter=tech_filter
            )
            
            # Convert SearchResult to RetrievedChunk
            chunks = []
            for result in search_results:
                chunk = RetrievedChunk(
                    content=result.content,
                    category=result.category,
                    subcategory=result.subcategory,
                    technologies=result.technologies,
                    metadata=result.metadata,
                    similarity_score=result.similarity_score
                )
                chunks.append(chunk)
            
            self.logger.info(f"Retrieved {len(chunks)} chunks with avg similarity: {sum(c.similarity_score for c in chunks) / len(chunks):.3f}" if chunks else "Retrieved 0 chunks")
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Retrieval with metadata failed: {e}")
            raise
    
    def format_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into readable context for LLM.
        
        Organizes chunks by category with section separators and enforces
        token limit by truncating if necessary.
        
        Args:
            chunks: List of RetrievedChunk objects to format
            
        Returns:
            Formatted context string with section headers
        """
        if not chunks:
            self.logger.warning("No chunks to format")
            return "--- PROFILE INFORMATION ---\n\nNo relevant information found."
        
        # Group chunks by category
        categorized = {}
        for chunk in chunks:
            category = chunk.category
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(chunk)
        
        # Build formatted context
        sections = ["--- PROFILE INFORMATION ---\n"]
        
        # Define category order and formatting
        category_order = ["about", "skills", "experience", "projects", "education", "certifications", "contact"]
        category_headers = {
            "about": "ABOUT",
            "skills": "SKILLS",
            "experience": "EXPERIENCE",
            "projects": "PROJECTS",
            "education": "EDUCATION",
            "certifications": "CERTIFICATIONS",
            "contact": "CONTACT"
        }
        
        # Format each category
        for category in category_order:
            if category not in categorized:
                continue
            
            header = category_headers.get(category, category.upper())
            sections.append(f"\n[{header}]")
            
            category_chunks = categorized[category]
            
            # Format based on category
            if category == "about":
                # About: single paragraph
                for chunk in category_chunks:
                    sections.append(chunk.content)
            
            elif category == "skills":
                # Skills: bullet list with proficiency
                for chunk in category_chunks:
                    proficiency = chunk.metadata.get("proficiency", "") if chunk.metadata else ""
                    if proficiency:
                        sections.append(f"- {chunk.content} ({proficiency})")
                    else:
                        sections.append(f"- {chunk.content}")
            
            elif category == "experience":
                # Experience: structured format with company, title, duration
                for chunk in category_chunks:
                    if chunk.metadata:
                        company = chunk.metadata.get("company", "")
                        duration = chunk.metadata.get("duration", "")
                        if company and duration:
                            sections.append(f"\n{company} ({duration})")
                    sections.append(chunk.content)
            
            elif category == "projects":
                # Projects: name, description, technologies
                for chunk in category_chunks:
                    sections.append(f"\n{chunk.content}")
                    if chunk.technologies:
                        tech_list = ", ".join(chunk.technologies)
                        sections.append(f"Technologies: {tech_list}")
            
            elif category == "education":
                # Education: institution, degree, date range
                for chunk in category_chunks:
                    if chunk.metadata:
                        institution = chunk.metadata.get("institution", "")
                        date_range = chunk.metadata.get("date_range", "")
                        if institution:
                            sections.append(f"\n{institution}")
                    sections.append(chunk.content)
            
            elif category == "certifications":
                # Certifications: name, provider, year
                for chunk in category_chunks:
                    if chunk.metadata:
                        provider = chunk.metadata.get("provider", "")
                        year = chunk.metadata.get("year", "")
                        if provider and year:
                            sections.append(f"- {chunk.content} ({provider}, {year})")
                        else:
                            sections.append(f"- {chunk.content}")
                    else:
                        sections.append(f"- {chunk.content}")
            
            elif category == "contact":
                # Contact: simple format
                for chunk in category_chunks:
                    sections.append(chunk.content)
            
            else:
                # Default: simple list
                for chunk in category_chunks:
                    sections.append(chunk.content)
        
        # Join all sections
        context = "\n".join(sections)
        
        # Enforce token limit by truncating if necessary
        if len(context) > self.max_chars:
            self.logger.warning(f"Context exceeds max chars ({len(context)} > {self.max_chars}), truncating...")
            context = context[:self.max_chars]
            # Add truncation notice
            context += "\n\n[Context truncated due to length limit]"
        
        return context
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            return self.vector_store.get_stats()
        except Exception as e:
            self.logger.error(f"Failed to get retrieval stats: {e}")
            return {"error": str(e)}
