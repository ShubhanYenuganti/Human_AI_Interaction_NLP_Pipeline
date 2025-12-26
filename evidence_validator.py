import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Optional

class EvidenceValidator:
    """
    Validates the extracted excerpts actually exist in the source text.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", db_writer=None):
        """
        Initialize validator with sentence transformer model
        
        Args:
            model_name: Name of the sentence transformer model to use
            db_writer: Optional DatabaseWriter instance for document-level validation
        """
        self.model = SentenceTransformer(model_name)
        self.db_writer = db_writer
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for comparison
        """
        # removed all whitespace
        text = re.sub(r'\s+', '', text)
        
        # convert to lowercase
        text = text.lower()
        
        return text
    
    def validate_excerpt(self, source_text: str, excerpt: str) -> dict:
        """Validate if the excerpt exists in the source text using multiple approaches:
        1. Character-level matching (handles spacing issues)
        2. Semantic similarity with sliding windows
        
        Args:
            source_text: The source text to validate the excerpt against
            excerpt: The excerpt to validate
        
        Returns:
            Dictionary containing the validation result
        """
        
        if not excerpt or not excerpt.strip():
            return {
                "success": False,
                "method": "empty_excerpt",
                "confidence": 0.0,
                "location": "Empty or whitespace-only excerpt"
            }
        
        # First try character-level matching for spacing issues
        normalized_source = self.normalize_text(source_text)
        normalized_excerpt = self.normalize_text(excerpt)
        
        if normalized_excerpt in normalized_source:
            return {
                'success': True,
                'method': 'character_level_exact_match',
                'location': "Found normalized excerpt in source text",
                'confidence': 1.0
            }
        
        # If character-level fails, try semantic similarity with proper sliding windows
        try:
            excerpt_embedding = self.model.encode([excerpt])  # Make it 2D for sklearn
            
            best_similarity = 0.0
            best_window = ""
            best_location = ""
            
            # Use word-based sliding windows for semantic matching
            source_words = source_text.split()
            excerpt_words = excerpt.split()
            excerpt_word_count = len(excerpt_words)
            
            if excerpt_word_count == 0:
                return {
                    'success': False,
                    'method': 'semantic_similarity',
                    'location': 'No words in excerpt',
                    'confidence': 0.0
                }
            
            # Try different window sizes around the excerpt length
            window_sizes = [
                max(1, excerpt_word_count - 2),
                excerpt_word_count,
                excerpt_word_count + 2,
                min(len(source_words), excerpt_word_count * 2)
            ]
            
            for window_size in set(window_sizes):  # Remove duplicates
                if window_size > len(source_words):
                    continue
                    
                for i in range(len(source_words) - window_size + 1):
                    window_words = source_words[i:i + window_size]
                    window_text = ' '.join(window_words)
                    
                    if len(window_text.strip()) == 0:
                        continue
                        
                    window_embedding = self.model.encode([window_text])  # Make it 2D
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity(excerpt_embedding, window_embedding)[0][0]
                    similarity_float = float(similarity)
                    
                    if similarity_float > best_similarity:
                        best_similarity = similarity_float
                        best_window = window_text
                        best_location = f"Words {i} to {i + window_size} (window size: {window_size})"
            
            # Also try character-based windows if word-based didn't work well
            if best_similarity < 0.85:
                excerpt_char_count = len(excerpt.replace(' ', ''))
                step_size = max(5, excerpt_char_count // 10)  # Smaller steps for better coverage
                
                for i in range(0, len(source_text) - excerpt_char_count + 1, step_size):
                    window_text = source_text[i:i + excerpt_char_count * 2]  # Longer windows
                    
                    if len(window_text.strip()) < 10:  # Skip very short windows
                        continue
                        
                    try:
                        window_embedding = self.model.encode([window_text])
                        similarity = cosine_similarity(excerpt_embedding, window_embedding)[0][0]
                        similarity_float = float(similarity)
                        
                        if similarity_float > best_similarity:
                            best_similarity = similarity_float
                            best_window = window_text
                            best_location = f"Characters {i} to {i + len(window_text)}"
                    except Exception:
                        # Skip windows that cause encoding errors
                        continue
                        
            # Determine success based on similarity threshold
            if best_similarity > 0.85:  # High confidence threshold
                return {
                    'success': True,
                    'method': 'semantic_similarity',
                    'location': best_location,
                    'confidence': best_similarity,
                    'matched_text': best_window[:200]  # Limit output length
                }
            else:
                return {
                    'success': False,
                    'method': 'semantic_similarity', 
                    'location': f"Best match: {best_location}" if best_location else "No good matches found",
                    'confidence': best_similarity,
                    'matched_text': best_window[:200] if best_window else ""
                }
                
        except Exception as e:
            return {
                'success': False,
                'method': 'semantic_similarity_error',
                'location': f"Error during semantic validation: {str(e)}",
                'confidence': 0.0
            }