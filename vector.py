from sentence_transformers import SentenceTransformer
import torch
import numpy as np

class VectorEmbedder:
    """
    Generate embeddings for chunks
    """
    
    def __init__(self, model_name = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize the embedder
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use ('cpu', 'cuda', or 'mps'). Default 'cpu' to avoid memory issues.
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 512
        
    def encode_texts(self, texts: list[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show a progress bar
        
        Returns:
            List of embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Generate embeddings for a single text
        
        Args:
            text: The text to embed
        
        Returns:
            Embedding
        """
        return self.model.encode(text, convert_to_numpy=True)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            a: First embedding
            b: Second embedding
        
        Returns:
            Cosine similarity
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two embeddings
        
        Args:
            a: First embedding
            b: Second embedding
        
        Returns:
            Euclidean distance
        """
        return np.linalg.norm(a - b)