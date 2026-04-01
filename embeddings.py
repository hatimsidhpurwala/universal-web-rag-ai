"""
Embeddings Module for Universal Web RAG AI System
Uses SentenceTransformers to convert text to embeddings
"""

import numpy as np
from typing import List, Optional
import streamlit as st


class EmbeddingGenerator:
    """Generates embeddings using SentenceTransformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        
    def load_model(self):
        """Lazy load the embedding model"""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            st.info(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            st.success(f"Embedding model loaded! Dimension: {self.embedding_dim}")
        return self.model
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        model = self.load_model()
        
        if not texts:
            return np.array([])
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and len(t.strip()) > 0]
        
        if not valid_texts:
            return np.array([])
        
        if show_progress:
            st.info(f"Generating embeddings for {len(valid_texts)} text chunks...")
        
        # Generate embeddings in batches
        embeddings = model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        if show_progress:
            st.success(f"Generated {len(embeddings)} embeddings!")
        
        return embeddings
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        model = self.load_model()
        embedding = model.encode([query], convert_to_numpy=True)
        return embedding[0]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if self.embedding_dim is None:
            self.load_model()
        return self.embedding_dim


# Singleton instance for reuse
_embedding_generator = None


def get_embedding_generator(model_name: str = 'all-MiniLM-L6-v2') -> EmbeddingGenerator:
    """Get or create singleton embedding generator"""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator(model_name)
    return _embedding_generator


def generate_embeddings(texts: List[str], show_progress: bool = True) -> np.ndarray:
    """Convenience function to generate embeddings"""
    generator = get_embedding_generator()
    return generator.generate_embeddings(texts, show_progress=show_progress)


def generate_query_embedding(query: str) -> np.ndarray:
    """Convenience function to generate query embedding"""
    generator = get_embedding_generator()
    return generator.generate_query_embedding(query)
