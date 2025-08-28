import os
from typing import List, Dict, Any
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class EmbeddingSystem:
    def __init__(self, api_key: str = None):
        api_key_to_use = api_key or os.getenv('OPENAI_API_KEY')
        print(f"Debug: Using API key: {api_key_to_use[:20] if api_key_to_use else 'None'}...")
        self.client = OpenAI(api_key=api_key_to_use)
        self.model = "text-embedding-ada-002"
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        embeddings = []
        
        for text in texts:
            embedding = self._generate_single_embedding(text)
            if embedding:
                embeddings.append(embedding)
        
        return embeddings
    
    def _generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)