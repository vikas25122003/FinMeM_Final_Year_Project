"""
Embeddings Module

Handles text embedding generation using sentence-transformers (local, free).
"""

from typing import List, Optional
import numpy as np


class EmbeddingModel:
    """Local embedding model using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformer model.
            device: Device to run on ('cpu' or 'cuda').
        """
        self.model_name = model_name
        self.device = device
        self._model = None
    
    @property
    def model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector as list of floats.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts.
        
        Args:
            text1: First text.
            text2: Second text.
            
        Returns:
            Cosine similarity score (0 to 1).
        """
        emb1 = np.array(self.embed(text1))
        emb2 = np.array(self.embed(text2))
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


# Global instance for convenience
_default_model: Optional[EmbeddingModel] = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    """Get or create the default embedding model.
    
    Args:
        model_name: Model name to use.
        
    Returns:
        EmbeddingModel instance.
    """
    global _default_model
    if _default_model is None or _default_model.model_name != model_name:
        _default_model = EmbeddingModel(model_name)
    return _default_model
