"""
Embedding Service for generating vector embeddings using HuggingFace models.

This service provides text-to-vector embedding functionality using the
sentence-transformers library with the all-MiniLM-L6-v2 model (384 dimensions).
Designed for CPU-only inference with memory constraints (<128MB).
"""

from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging
import os
from groq import Groq

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating 384-dimensional vector embeddings.
    
    Features:
    - Lazy loading of HuggingFace model
    - CPU-only inference
    - Batch processing support
    - Groq API fallback for embedding failures
    - Memory-efficient operation (<128MB)
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding service.
        
        Args:
            model_name: HuggingFace model identifier. Default is all-MiniLM-L6-v2
                       which produces 384-dimensional embeddings.
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self.logger = logging.getLogger(__name__)
        self._groq_client: Optional[Groq] = None
        
    def _load_model(self):
        """
        Lazy load the embedding model on first use.
        
        Raises:
            Exception: If model loading fails and no fallback is available.
        """
        if self._model is None:
            try:
                self.logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                self.logger.info(f"Successfully loaded embedding model: {self.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load model {self.model_name}: {e}")
                raise
    
    def _get_groq_client(self) -> Optional[Groq]:
        """
        Initialize Groq client for fallback embeddings.
        
        Returns:
            Groq client instance or None if API key not available.
        """
        if self._groq_client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self._groq_client = Groq(api_key=api_key)
                self.logger.info("Groq client initialized for fallback embeddings")
            else:
                self.logger.warning("GROQ_API_KEY not set, fallback unavailable")
        return self._groq_client
    
    def _embed_with_groq(self, text: str) -> List[float]:
        """
        Generate embedding using Groq API as fallback.
        
        Args:
            text: Input text to embed.
            
        Returns:
            List of floats representing the embedding vector.
            
        Raises:
            Exception: If Groq embedding fails.
        """
        groq_client = self._get_groq_client()
        if not groq_client:
            raise Exception("Groq fallback not available: GROQ_API_KEY not set")
        
        try:
            # Note: Groq doesn't have a direct embedding API like OpenAI
            # This is a placeholder for potential future Groq embedding support
            # For now, we'll raise an exception to indicate fallback is not fully implemented
            self.logger.error("Groq embedding fallback not yet implemented")
            raise NotImplementedError("Groq embedding API not available")
        except Exception as e:
            self.logger.error(f"Groq embedding failed: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Input text to embed.
            
        Returns:
            List of 384 floats representing the embedding vector.
            
        Raises:
            Exception: If both primary model and fallback fail.
        """
        try:
            self._load_model()
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Primary embedding failed for text: {e}")
            # Attempt Groq fallback
            try:
                self.logger.info("Attempting Groq fallback for embedding")
                return self._embed_with_groq(text)
            except Exception as fallback_error:
                self.logger.error(f"Fallback embedding also failed: {fallback_error}")
                raise Exception(f"All embedding methods failed. Primary: {e}, Fallback: {fallback_error}")
    
    def embed_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Batch processing is more memory-efficient than individual calls.
        Default batch size of 10 is optimized for Render free tier (256MB RAM).
        
        Args:
            texts: List of input texts to embed.
            batch_size: Number of texts to process in each batch. Default is 10.
            
        Returns:
            List of embedding vectors, each containing 384 floats.
            
        Raises:
            Exception: If embedding generation fails.
        """
        try:
            self._load_model()
            embeddings = self._model.encode(
                texts, 
                batch_size=batch_size, 
                convert_to_numpy=True,
                show_progress_bar=False  # Disable progress bar for cleaner logs
            )
            return embeddings.tolist()
        except Exception as e:
            self.logger.error(f"Batch embedding failed: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimensionality of embeddings produced by this service.
        
        Returns:
            Integer representing embedding dimension (384 for all-MiniLM-L6-v2).
        """
        self._load_model()
        return self._model.get_sentence_embedding_dimension()
    
    def is_model_loaded(self) -> bool:
        """
        Check if the embedding model is currently loaded in memory.
        
        Returns:
            True if model is loaded, False otherwise.
        """
        return self._model is not None
