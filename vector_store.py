"""
Vector Database Module for Universal Web RAG AI System
Uses FAISS for efficient similarity search
"""

import numpy as np
import faiss
from typing import List, Tuple, Optional
import streamlit as st


class VectorStore:
    """FAISS-based vector store for embeddings"""
    
    def __init__(self, dimension: int = 384):  # 384 is the dimension for all-MiniLM-L6-v2
        self.dimension = dimension
        self.index = None
        self.texts = []
        self.embeddings = None
        
    def create_index(self, embeddings: np.ndarray, texts: List[str]):
        """Create FAISS index from embeddings"""
        if len(embeddings) == 0:
            raise ValueError("No embeddings provided")
        
        st.info(f"Creating vector database with {len(embeddings)} embeddings...")
        
        # Store texts
        self.texts = texts
        self.embeddings = embeddings
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        # Using IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        st.success(f"Vector database created! Indexed {self.index.ntotal} vectors")
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """Search for similar vectors"""
        if self.index is None:
            raise ValueError("Index not created. Call create_index first.")
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get corresponding texts
        results = []
        scores = []
        
        for idx, score in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self.texts):
                results.append(self.texts[idx])
                scores.append(float(score))
        
        return results, scores
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store"""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'texts_stored': len(self.texts)
        }
    
    def clear(self):
        """Clear the index and stored data"""
        self.index = None
        self.texts = []
        self.embeddings = None


# Singleton instance for reuse
_vector_store = None


def get_vector_store(dimension: int = 384) -> VectorStore:
    """Get or create singleton vector store"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(dimension)
    return _vector_store


def create_vector_store(embeddings: np.ndarray, texts: List[str], dimension: int = 384):
    """Convenience function to create vector store"""
    store = get_vector_store(dimension)
    store.create_index(embeddings, texts)
    return store


def search_similar(query_embedding: np.ndarray, top_k: int = 5) -> Tuple[List[str], List[float]]:
    """Convenience function to search similar texts"""
    store = get_vector_store()
    return store.search(query_embedding, top_k)
