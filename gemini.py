from google import genai
import os
import sys
import argparse
import hashlib
from typing import Optional
from dotenv import load_dotenv

from vector import VectorEmbedder
from db_writer import DatabaseWriter

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = "gemini-3-flash-preview"


class RAGSystem:
    """
    Retrieval-Augmented Generation System
    """
    
    def __init__(self, top_k: int = 10, device: str = "cpu"):
        """
        Initialize RAG system
        
        Args:
            top_k: Number of top similar chunks to retrieve
            device: Device for embeddings ('cpu', 'cuda', 'mps')
        """
        print("🚀 Initializing RAG System...")
        
        # Initialize Gemini client
        print("  📡 Connecting to Gemini API...")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = GEMINI_MODEL
        
        # Initialize embedder
        print(f"  🧠 Loading embedding model on {device}...")
        self.embedder = VectorEmbedder(device=device)
        
        # Initialize database
        print("  💾 Connecting to database...")
        self.db = DatabaseWriter(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=int(os.getenv('DB_PORT'))
        )
        
        self.top_k = top_k
        print("✅ RAG System ready!\n")
    
    
    def format_context(self, results: list) -> str:
        """
        Format retrieved chunks into context string
        
        Args:
            results: List of chunk dictionaries
        
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}] {chunk['document_title'] or 'Document ' + str(chunk['document_id'])}\n"
                f"Section: {chunk['section_title'] or 'N/A'}\n"
                f"Similarity: {chunk['similarity']:.3f}\n"
                f"{chunk['text']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate response using Gemini with retrieved context
        
        Args:
            query: User's question
            context: Retrieved context from vector search
        
        Returns:
            Generated response
        """
        print("💬 Generating response with Gemini...\n")
        
        # Construct prompt with context
        prompt = f"""You are a helpful AI assistant with access to a knowledge base of research papers and documents.

Based on the following context retrieved from the knowledge base, please answer the user's question. 
If the context doesn't contain relevant information, say so. Always cite which source(s) you used.

CONTEXT:
{context}

USER QUESTION: {query}

Please provide a comprehensive, accurate answer based on the context above."""
        
        # Generate response
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        
        return response.text
    
    def save_query(self, 
                   question: str, 
                   answer: str, 
                   results: list,
                   top_k: int,
                   query_embedding) -> Optional[int]:
        """
        Save query to database for logging and analytics
        
        Args:
            question: User's question
            answer: Generated answer
            results: Retrieved chunks
            top_k: Number of results requested
            query_embedding: Query embedding vector
        
        Returns:
            Query ID if successful, None otherwise
        """
        try:
            # Create query hash for deduplication
            query_hash = hashlib.sha256(question.encode('utf-8')).hexdigest()
            
            # Extract document and chunk IDs
            document_ids = [r['document_id'] for r in results] if results else []
            chunk_ids = [r['id'] for r in results] if results else []
            similarity_scores = [r['similarity'] for r in results] if results else []
            
            # Calculate average similarity
            avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
            
            # Prepare LLM response data
            llm_response = {
                'output': answer,
                'model': self.model
            }
            
            # Prepare retrieval parameters
            retrieval_params = {
                'top_k': top_k,
                'max_results': top_k
            }
            
            # Prepare retrieval results
            retrieval_results = {
                'document_ids': document_ids,
                'chunk_ids': chunk_ids,
                'num_documents': len(set(document_ids)),  # Unique document count
                'num_chunks': len(chunk_ids),
                'similarity_scores': similarity_scores,
                'avg_similarity': avg_similarity
            }
            
            # Insert into database
            query_id = self.db.insert_query(
                query_text=question,
                query_hash=query_hash,
                llm_response=llm_response,
                retrieval_params=retrieval_params,
                retrieval_results=retrieval_results,
                query_embedding=query_embedding
            )
            
            return query_id
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save query to database: {e}")
            return None
    
    def query(self, question: str, top_k: int = None, show_sources: bool = True) -> str:
        """
        Main query method: search + generate
        
        Args:
            question: User's question
            top_k: Number of context chunks to retrieve
            show_sources: Whether to display source information
        
        Returns:
            Generated answer
        """
        print("="*60)
        print(f"QUERY: {question}")
        print("="*60 + "\n")
        
        # Generate query embedding (needed for both search and logging)
        query_embedding = self.embedder.encode_single(question)
        
        # Step 1: Retrieve relevant context
        k = top_k or self.top_k
        print(f"🔍 Searching for relevant context (top-{k})...")
        results = self.db.search_similar_chunks(
            query_embedding=query_embedding,
            top_k=k
        )
        print(f"✅ Found {len(results)} relevant chunks\n")
        
        if not results:
            return "❌ No relevant information found in the knowledge base."
        
        # Show sources if requested
        if show_sources:
            print("📚 Retrieved Sources:")
            for i, chunk in enumerate(results, 1):
                print(f"  [{i}] {chunk['document_title'] or 'Doc ' + str(chunk['document_id'])} "
                      f"(similarity: {chunk['similarity']:.3f})")
                if chunk['section_title']:
                    print(f"      Section: {chunk['section_title']}")
            print()
        
        # Step 2: Format context
        context = self.format_context(results)
        
        # Step 3: Generate response
        answer = self.generate_response(question, context)
        
        print("="*60)
        print("ANSWER:")
        print("="*60)
        print(answer)
        print("="*60 + "\n")
        
        # Step 4: Save query to database
        query_id = self.save_query(
            question=question,
            answer=answer,
            results=results,
            top_k=k,
            query_embedding=query_embedding
        )
        
        if query_id:
            print(f"💾 Response saved to database (Query ID: {query_id})\n")
        
        return answer


def interactive_mode(rag: RAGSystem):
    """
    Interactive terminal mode for continuous queries
    """
    print("\n" + "="*60)
    print("🤖 RAG SYSTEM - INTERACTIVE MODE")
    print("="*60)
    print("Type your questions below. Type 'quit' or 'exit' to stop.")
    print("="*60 + "\n")
    
    while True:
        try:
            # Get user input
            question = input("❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Process query
            rag.query(question)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(
        description="RAG System with Gemini - Ask questions about your document knowledge base"
    )
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Single query to process (non-interactive mode)'
    )
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=5,
        help='Number of context chunks to retrieve (default: 5)'
    )
    parser.add_argument(
        '--no-sources',
        action='store_true',
        help='Hide source information in output'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device for embeddings (default: cpu)'
    )
    
    args = parser.parse_args()
    
    # Initialize RAG system
    try:
        rag = RAGSystem(top_k=args.top_k, device=args.device)
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        sys.exit(1)
    
    # Single query mode or interactive mode
    if args.query:
        # Single query
        rag.query(args.query, show_sources=not args.no_sources)
    else:
        # Interactive mode
        interactive_mode(rag)


if __name__ == "__main__":
    main()
