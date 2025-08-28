from typing import List, Dict, Any, Tuple
import numpy as np

class SimpleVectorStore:
    def __init__(self):
        self.embeddings = []
        self.chunks = []
        self.metadata = []
    
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Add chunks and their embeddings to the store"""
        # Handle case where embeddings list is empty (basic mode)
        if not embeddings:
            for chunk in chunks:
                self.chunks.append(chunk)
                # Add a placeholder embedding for chunks without embeddings
                self.embeddings.append([0.0] * 1536)  # OpenAI embedding dimension
                self.metadata.append(chunk.get('metadata', {}))
        else:
            # Normal case with embeddings
            for chunk, embedding in zip(chunks, embeddings):
                self.chunks.append(chunk)
                if embedding:
                    self.embeddings.append(embedding)
                else:
                    # Add a placeholder embedding for chunks without embeddings
                    self.embeddings.append([0.0] * 1536)  # OpenAI embedding dimension
                self.metadata.append(chunk.get('metadata', {}))
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for most similar chunks"""
        if not self.embeddings:
            return []
        
        # Calculate similarities
        similarities = []
        for embedding in self.embeddings:
            sim = self._cosine_similarity(query_embedding, embedding)
            similarities.append(sim)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return chunks with their similarity scores
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only return relevant results
                results.append((self.chunks[idx], similarities[idx]))
        
        return results
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)