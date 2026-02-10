import re
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from vector import VectorEmbedder
from scipy.signal import find_peaks

class EmbeddingSemanticChunker:
    """
    Semantic chunking using sentence embeddings and similarity analysis
    
    Algorithm:
    1. Split text into sentences
    2. Generate embeddings for each sentence
    3. Calculate similarity between sentences
    4. Identify "valleys" in similarity scores
    5. Split at valleys to create chunks
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.5,
                 min_chunk_size: int = 4,
                 max_chunk_size: int = 24,
                 device: str = "cpu"
                 ):
        """
        Initialize the semantic chunker
        
        Args:
            model_name: Name of the sentence transformer model
            similarity_threshold: Threshold for similarity scores
            min_chunk_size: Minimum sentences per chunk
            max_chunk_size: Maximum sentences per chunk
            device: Device to use ('cpu', 'cuda', or 'mps'). Default 'cpu' to avoid memory issues.
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 512
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
    def split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using regex
        
        Args:
            text: The text to split into sentences
        
        Returns:
            List of sentences
        """
        
        # First, protect abbreviations
        text = re.sub(r'\b([A-Z][a-z]*\.)', r'\1<PROTECT>', text)
        text = re.sub(r'\b(et al|i\.e|e\.g|vs|cf|etc)\.', r'\1<PERIOD>', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore protected periods
        sentences = [s.replace('<PROTECT>', '').replace('<PERIOD>', '.') 
                    for s in sentences]
        
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
    
        return sentences
    
    def calculate_similarity(self, sentences: list[str]) -> np.ndarray:
        """
        Calculate cosine similarity between consecutive sentences
        
        Args:
            sentences: List of sentences to calculate similarity between
        
        Returns:
            Array of similarities: [sim(1,2), sim(2,3), ...]
        """
        
        embeddings = self.model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)
        similarities = []
        for i in range(len(embeddings) - 1):
            a = embeddings[i]
            b = embeddings[i + 1]
            similarities.append(VectorEmbedder.cosine_similarity(a, b))
        return np.array(similarities)
    
    def find_valleys(self, similarities: np.ndarray) -> list[int]:
        """
        Find optimal split points by detecting valleys in similarity scores
        
        Args:
            similarities: Array of similarities to find valleys in
        
        Returns:
            List of split points
        """
            
        split_points = []
        
        valleys, _ = find_peaks(-similarities, distance=self.min_chunk_size)
        
        current_chunk_start = 0
        
        for valley_idx in valleys:
            if similarities[valley_idx] < self.similarity_threshold:
                # Check chunk size constraints
                chunk_size = valley_idx - current_chunk_start + 1
                if chunk_size >= self.min_chunk_size:
                    split_points.append(valley_idx + 1)
                    current_chunk_start = valley_idx + 1
        
        return split_points
    
    def chunk_text(self, text: str, overlap_percentage: float = 0.5) -> list[dict]:
        """
        Main method: Perform semantic semantic chunking on text with overlap
        
        Args:
            text: The text to chunk
            overlap_percentage: Percentage of overlap between chunks (default 0.5 for 50%)
        
        Returns:
            List of chunks with metadata including overlap information
        """
        
        # Split into sentences
        sentences = self.split_into_sentences(text)
        
        if len(sentences) < self.min_chunk_size:
            return [
                {
                    'chunk_id': 0,
                    'text': text,
                    'sentences': sentences,
                    'sentence_count': len(sentences),
                    'start_sentence': 0,
                    'end_sentence': len(sentences) - 1,
                    'previous_chunk_id': None,
                    'next_chunk_id': None,
                    'overlap_start': None,
                    'overlap_end': None
                }
            ]
        
        # Calculate similarities
        similarities = self.calculate_similarity(sentences)
        
        # Find split points
        split_points = self.find_valleys(similarities)
        
        # Create chunks with overlap
        chunks = []
        start_idx = 0
        
        split_indices = split_points + [len(sentences)]
        
        for i, split_idx in enumerate(split_indices):
            # Calculate overlap region
            chunk_size = split_idx - start_idx
            overlap_size = max(1, int(chunk_size * overlap_percentage))
            
            # Determine the actual end index for this chunk
            end_idx = split_idx
            
            # Determine overlap boundaries for next chunk
            if i < len(split_indices) - 1:
                # This is not the last chunk, so include overlap
                overlap_start_next = max(start_idx, end_idx - overlap_size)
            else:
                overlap_start_next = None
            
            chunk_sentences = sentences[start_idx:end_idx]
            
            if chunk_sentences:
                # Determine if this chunk has overlap from previous chunk
                overlap_start = None
                overlap_end = None
                
                if i > 0:
                    # This chunk overlaps with the previous one
                    # The overlap region is from start_idx to the point where new content begins
                    prev_end = split_indices[i-1] if i > 0 else 0
                    prev_size = prev_end - (split_indices[i-2] if i > 1 else 0)
                    prev_overlap_size = max(1, int(prev_size * overlap_percentage))
                    overlap_start = start_idx
                    overlap_end = min(start_idx + prev_overlap_size, end_idx)
                
                chunks.append({
                    'chunk_id': i,
                    'text': ' '.join(chunk_sentences),
                    'sentences': chunk_sentences,
                    'sentence_count': len(chunk_sentences),
                    'start_sentence': start_idx,
                    'end_sentence': end_idx - 1,
                    'avg_similarity': np.mean(similarities[start_idx:end_idx-1]) if end_idx > start_idx + 1 else 1.0,
                    'previous_chunk_id': i - 1 if i > 0 else None,
                    'next_chunk_id': i + 1 if i < len(split_indices) - 1 else None,
                    'overlap_start': overlap_start,
                    'overlap_end': overlap_end
                })
                
            # Next chunk starts at overlap position
            if overlap_start_next is not None:
                start_idx = overlap_start_next
            else:
                start_idx = split_idx
                
        return chunks
        

class HybridSemanticChunker:
    """
    Hybrid semantic chunking that:
    1. First pass -- split by structural elements (headers, paragraphs, etc.)
    2. Second pass -- within each section, apply semantic chunking
    3. Third pass -- merge chunks or split chunks to a fixed token size
    
    Approach maintains high context within each embedding due to the structural element first pass while also maintaining a fixed token size for efficient retrieval of embeddings.
    """
    
    def __init__(self, 
                 target_chunk_size: int = 14, # Target sentences per chunk
                 min_chunk_size: int  = 4, # Minimum sentences per chunk
                 max_chunk_size: int = 24, # Maximum sentences per chunk
                 device: str = "cpu"
                 ):
        """
        Initialize the hybrid semantic chunker
        
        Args:
            target_chunk_size: Target sentences per chunk
            min_chunk_size: Minimum sentences per chunk
            max_chunk_size: Maximum sentences per chunk
            device: Device to use ('cpu', 'cuda', or 'mps'). Default 'cpu' to avoid memory issues.
        """
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.device = device
    
    def detect_structure(self, text: str):
        """
        Detect sections in the text and return them with their boundaries and types (header, paragraph, etc.)
        
        Args:
            text: The text to detect sections in
        
        Returns:
            List of sections with their boundaries and types
        """
        sections = []
        # Use regex to detect sections
        section_patterns = [
            # Abstract headers
            r"(?i)^(?:abstract)$",
            
            # Introduction headers
            r"(?i)^(?:\d+\.?\s+)?(?:introduction)$",
            
            # Background headers
            r"(?i)^(?:\d+\.?\s+)?(?:background)$",
            
            # Problem Statement headers
            r"(?i)^(?:\d+\.?\s+)?(?:problem\s+statement)$",
            
            # Related work / Literature Review headers
            r"(?i)^(?:\d+\.?\s+)?(?:related\s+work|literature\s+review)$",
            
            # Objectives / Hypothesis headers
            r"(?i)^(?:\d+\.?\s+)?(?:objectives?|hypothesis|hypotheses)$",
            
            # Method headers
            r"(?i)^(?:\d+\.?\s+)?(?:methods?|methodology|experimental\s+setup)$",
            
            # Results headers
            r"(?i)^(?:\d+\.?\s+)?(?:results|findings)$",
            
            # Discussion headers
            r"(?i)^(?:\d+\.?\s+)?(?:discussion)$",
            
            # Conclusion headers
            r"(?i)^(?:\d+\.?\s+)?(?:conclusions?|concluding\s+remarks)$",
            
            # References headers
            r"(?i)^(?:\d+\.?\s+)?(?:references|bibliography|works\s+cited)$",
            
            # Numbered headers (Generic pattern for any numbered section not listed above)
            r"^\d+(?:\.\d+)*\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
            
            # All Caps headers (Matches any line that is purely uppercase letters/spaces)
            r"^[A-Z\s]{3,25}$"
        ]
    
        lines = text.split('\n')
        current_section = {'start': 0, 'title': 'Preamble', 'lines': []}
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check if the line is a section header
            is_header = False
            
            for pattern in section_patterns:
                if re.match(pattern, line_stripped):
                    is_header = True
                
                    # Save previous section
                    if current_section['lines']:
                        current_section['text'] = '\n'.join(current_section['lines'])
                        sections.append(current_section)
                        
                    # Start new section
                    current_section = {
                        'start': i,
                        'title': line_stripped,
                        'lines': []
                    }
                    break
            
            if not is_header and line_stripped:
                current_section['lines'].append(line_stripped)
        
        # Save last section
        if current_section['lines']:
            current_section['text'] = '\n'.join(current_section['lines'])
            sections.append(current_section)
        
        return sections

    def chunk_section(self, section_text: str, overlap_percentage: float = 0.5):
        """
        Chunk a single section using semantic chunking with overlap
        
        Args:
            section_text: The text of the section to chunk
            overlap_percentage: Percentage of overlap between chunks (default 0.5 for 50%)
        
        Returns:
            List of chunks with metadata
        """
        
        semantic_chunker = EmbeddingSemanticChunker(
            similarity_threshold=0.5,
            min_chunk_size=self.min_chunk_size,
            max_chunk_size=self.max_chunk_size,
            device=self.device
        )
        
        return semantic_chunker.chunk_text(section_text, overlap_percentage=overlap_percentage)
    
    def merge_small_chunks(self, chunks: list[dict]):
        """
        Merge chunks that are too small with adjacent chunks while preserving overlap metadata
        
        Args:
            chunks: List of chunks to merge
        
        Returns:
            List of merged chunks
        """
        
        merged = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            
            # If chunk is too small and not the last one
            if (current['sentence_count'] < self.min_chunk_size) and (i < len(chunks) - 1):
                # Merge with next chunk
                next_chunk = chunks[i + 1]
                merged_chunk = {
                    'chunk_id': current['chunk_id'],
                    'text': current['text'] + ' ' + next_chunk['text'],
                    'sentences': current['sentences'] + next_chunk['sentences'],
                    'sentence_count': current['sentence_count'] + next_chunk['sentence_count'],
                    'section_title': current.get('section_title', ''),
                    'start_sentence': current.get('start_sentence', 0),
                    'end_sentence': next_chunk.get('end_sentence', 0),
                    'previous_chunk_id': current.get('previous_chunk_id'),
                    'next_chunk_id': next_chunk.get('next_chunk_id'),
                    'overlap_start': current.get('overlap_start'),
                    'overlap_end': next_chunk.get('overlap_end')
                }
                merged.append(merged_chunk)
                i += 2 # skip the next chunk
            else:
                merged.append(current)
                i += 1
                
        return merged
    
    def split_large_chunks(self, chunks: list[dict]):
        """
        Split large chunks into smaller chunks with overlap
        
        Args:
            chunks: List of chunks to split
        
        Returns:
            List of split chunks
        """
        
        result = []
        for chunk in chunks:
            if chunk['sentence_count'] <= self.max_chunk_size:
                result.append(chunk)
            else:
                sub_chunker = EmbeddingSemanticChunker(
                    similarity_threshold=0.5,
                    min_chunk_size=self.min_chunk_size,
                    max_chunk_size=self.max_chunk_size,
                    device=self.device
                )
                
                sub_chunks = sub_chunker.chunk_text(chunk['text'], overlap_percentage=0.5)
                
                for sub_chunk in sub_chunks:
                    sub_chunk['section_title'] = chunk.get('section_title', '')
                    result.append(sub_chunk)
                    
        # renumber chunks and update relationships
        for i, chunk in enumerate(result):
            chunk['chunk_id'] = i
            chunk['previous_chunk_id'] = i - 1 if i > 0 else None
            chunk['next_chunk_id'] = i + 1 if i < len(result) - 1 else None
            
        return result
    
    def chunk_text(self, text: str, overlap_percentage: float = 0.5):
        """
        Main method to chunk text into sections and then chunks within each section with overlap
        
        Args:
            text: The text to chunk
            overlap_percentage: Percentage of overlap between chunks (default 0.5 for 50%)
        
        Returns:
            List of chunks with metadata including section headers and chunk relationships
        """
        
        sections = self.detect_structure(text)
        
        if not sections:
            # fallback to pure semantic chunking
            return EmbeddingSemanticChunker(device=self.device).chunk_text(text, overlap_percentage=overlap_percentage)

        all_chunks = []
        chunk_id = 0
        
        for section in sections:
            section_chunks = self.chunk_section(section['text'], overlap_percentage=overlap_percentage)
            
            for chunk in section_chunks:
                chunk['chunk_id'] = chunk_id
                chunk['section_title'] = section['title']
                # Store section header as additional metadata
                chunk['section_header'] = section['title']
                all_chunks.append(chunk)
                chunk_id += 1
                
        # Step 3: Merge and split chunks as needed
        all_chunks = self.merge_small_chunks(all_chunks)
        all_chunks = self.split_large_chunks(all_chunks)
        
        # Final pass: Update chunk relationships to ensure consistency
        for i, chunk in enumerate(all_chunks):
            chunk['chunk_id'] = i
            chunk['previous_chunk_id'] = i - 1 if i > 0 else None
            chunk['next_chunk_id'] = i + 1 if i < len(all_chunks) - 1 else None
        
        return all_chunks
                