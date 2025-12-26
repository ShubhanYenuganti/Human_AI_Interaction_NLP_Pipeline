"""
PDF Processing Pipeline
-----------------------
End-to-end pipeline that:
1. Reads PDFs from S3 bucket
2. Extracts text and metadata
3. Chunks text using hybrid semantic chunking
4. Generates embeddings for each chunk
5. Writes documents and chunks to PostgreSQL database
"""

import os
import logging
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

from read_files import S3Reader
from chunk import HybridSemanticChunker
from vector import VectorEmbedder
from db_writer import DatabaseWriter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class PDFProcessingPipeline:
    """
    Main pipeline class that orchestrates the entire PDF processing workflow
    """
    
    def __init__(self,
                 target_chunk_size: int = 8,
                 min_chunk_size: int = 2,
                 max_chunk_size: int = 16,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 16,
                 device: str = "cpu"):
        """
        Initialize the pipeline with all components
        
        Args:
            target_chunk_size: Target number of sentences per chunk
            min_chunk_size: Minimum sentences per chunk
            max_chunk_size: Maximum sentences per chunk
            embedding_model: Name of the sentence transformer model
            batch_size: Batch size for embedding generation
        """
        
        logger.info("Initializing PDF Processing Pipeline...")
        
        # Initialize S3 Reader
        logger.info("Connecting to S3...")
        self.s3_reader = S3Reader()
        
        # Initialize Chunker
        logger.info("Initializing semantic chunker...")
        self.chunker = HybridSemanticChunker(
            target_chunk_size=target_chunk_size,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            device=device
        )
        
        # Initialize Vector Embedder
        logger.info(f"Loading embedding model: {embedding_model} on {device}...")
        self.embedder = VectorEmbedder(model_name=embedding_model, device=device)
        self.batch_size = batch_size
        
        # Initialize Database Writer
        logger.info("Connecting to database...")
        self.db_writer = DatabaseWriter(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
            autocommit=False
        )
        
        logger.info("Pipeline initialized successfully!")
    
    def _process_pdf_with_data(self, s3_key: str, pdf_data: bytes) -> Optional[int]:
        """
        Internal method to process PDF when data is already downloaded
        
        Args:
            s3_key: S3 key of the PDF file
            pdf_data: Already downloaded PDF data
        
        Returns:
            Document ID if successful, None otherwise
        """
        
        logger.info(f"Processing PDF: {s3_key}")
        
        try:
            
            # Step 2: Extract metadata and text
            logger.info("Step 2: Extracting metadata and text...")
            metadata = self.s3_reader.get_pdf_metadata(s3_key, pdf_data)
            text = self.s3_reader.extract_text_from_pdf(pdf_data)
            
            # Step 2.5: Clean text (remove references, artifacts, etc.)
            logger.info("Step 2.5: Cleaning text (removing references and artifacts)...")
            text = self.s3_reader.clean_text(text, remove_references=True)
            
            if not text or len(text.strip()) < 100:
                logger.warning(f"Insufficient text extracted from {s3_key}")
                return None
            
            logger.info(f"Extracted and cleaned {len(text)} characters from PDF")
            
            # Step 3: Chunk the text
            logger.info("Step 3: Chunking text using hybrid semantic chunker...")
            chunks = self.chunker.chunk_text(text)
            logger.info(f"Created {len(chunks)} chunks")
            
            if not chunks:
                logger.warning(f"No chunks created for {s3_key}")
                return None
            
            # Step 4: Generate embeddings
            logger.info("Step 4: Generating embeddings...")
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedder.encode_texts(
                chunk_texts,
                batch_size=self.batch_size,
                show_progress=True
            )
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Step 5: Prepare document metadata
            logger.info("Step 5: Writing to database...")
            document_metadata = {
                's3_key': s3_key,
                'file_hash': metadata['file_hash'],
                'num_pages': metadata['num_pages'],
                'title': metadata.get('title'),
                'author': metadata.get('author'),
                'publication_year': None,  # Can be extracted if available
                'journal': None,  # Can be extracted if available
                'doi': None,  # Can be extracted if available
                'status': 'processing',
                'metadata': {
                    'subject': metadata.get('subject'),
                    'text_length': len(text),
                    'num_chunks': len(chunks)
                }
            }
            
            # Step 6: Write to database
            document_id = self.db_writer.insert_document_with_chunks(
                document_metadata=document_metadata,
                chunks=chunks,
                embeddings=embeddings
            )
            
            logger.info(f"✅ Successfully processed {s3_key} - Document ID: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"❌ Error processing {s3_key}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_single_pdf(self, s3_key: str) -> Optional[int]:
        """
        Process a single PDF through the entire pipeline
        
        Args:
            s3_key: S3 key of the PDF file
        
        Returns:
            Document ID if successful, None otherwise
        """
        
        # Download and process
        logger.info("Step 1: Downloading PDF from S3...")
        pdf_data = self.s3_reader.download_pdf(s3_key)
        return self._process_pdf_with_data(s3_key, pdf_data)
    
    def process_all_pdfs(self, skip_existing: bool = True) -> Dict[str, any]:
        """
        Process all PDFs in the S3 bucket
        
        Args:
            skip_existing: If True, skip PDFs that already exist in database (by file hash)
        
        Returns:
            Dictionary with processing statistics
        """
        
        logger.info("=" * 60)
        logger.info("Starting batch processing of all PDFs in S3 bucket")
        logger.info("=" * 60)
        
        # Get list of PDFs from S3
        pdf_keys = self.s3_reader.list_pdfs()
        logger.info(f"Found {len(pdf_keys)} PDFs in S3 bucket")
        
        if not pdf_keys:
            logger.warning("No PDFs found in S3 bucket")
            return {
                'total': 0,
                'processed': 0,
                'skipped': 0,
                'failed': 0
            }
        
        stats = {
            'total': len(pdf_keys),
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'document_ids': []
        }
        
        start_time = datetime.now()
        
        for i, pdf_key in enumerate(pdf_keys, 1):
            logger.info(f"\n[{i}/{len(pdf_keys)}] Processing: {pdf_key}")
            
            try:
                # Download PDF once
                pdf_data = self.s3_reader.download_pdf(pdf_key)
                
                # Check if document already exists (if skip_existing is True)
                if skip_existing:
                    file_hash = self.db_writer.calculate_file_hash(pdf_data)
                    
                    # Check if exists in database
                    existing_doc = self.db_writer.get_document_by_hash(file_hash)
                    if existing_doc:
                        logger.info(f"⏭️  Skipping (already exists): {pdf_key}")
                        stats['skipped'] += 1
                        continue
                
                # Process the PDF (pass pdf_data to avoid re-downloading)
                document_id = self._process_pdf_with_data(pdf_key, pdf_data)
                
                if document_id:
                    stats['processed'] += 1
                    stats['document_ids'].append(document_id)
                else:
                    stats['failed'] += 1
                    
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_key}: {e}")
                stats['failed'] += 1
                continue
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total PDFs: {stats['total']}")
        logger.info(f"✅ Processed: {stats['processed']}")
        logger.info(f"⏭️  Skipped: {stats['skipped']}")
        logger.info(f"❌ Failed: {stats['failed']}")
        logger.info(f"⏱️  Duration: {duration:.2f} seconds")
        logger.info(f"📊 Average: {duration/stats['total']:.2f} seconds per PDF")
        logger.info("=" * 60)
        
        return stats
    
    def reprocess_document(self, document_id: int) -> bool:
        """
        Reprocess an existing document (re-chunk and re-embed)
        
        Args:
            document_id: ID of the document to reprocess
        
        Returns:
            True if successful, False otherwise
        """
        
        logger.info(f"Reprocessing document ID: {document_id}")
        
        try:
            # Get document from database
            doc = self.db_writer.get_document(document_id)
            if not doc:
                logger.error(f"Document {document_id} not found")
                return False
            
            # Download PDF from S3
            pdf_data = self.s3_reader.download_pdf(doc['s3_key'])
            text = self.s3_reader.extract_text_from_pdf(pdf_data)
            
            # Delete existing chunks
            # (This would require adding a delete_chunks_by_document method to DatabaseWriter)
            logger.warning("Note: Old chunks are not deleted. Consider adding cascade delete or manual cleanup.")
            
            # Re-chunk
            chunks = self.chunker.chunk_text(text)
            
            # Re-embed
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedder.encode_texts(chunk_texts, batch_size=self.batch_size)
            
            # Insert new chunks
            inserted, failed = self.db_writer.insert_chunk_batch(
                document_id=document_id,
                chunks=chunks,
                embeddings=embeddings
            )
            
            # Update document status
            self.db_writer.update_document_status(
                document_id=document_id,
                status='completed',
                processed_date=datetime.now()
            )
            
            logger.info(f"✅ Reprocessed document {document_id}: {inserted} chunks inserted")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error reprocessing document {document_id}: {e}")
            return False


def main():
    """
    Main entry point for the pipeline
    """
    
    # Example usage
    pipeline = PDFProcessingPipeline()
    
    # Option 1: Process all PDFs in S3 bucket
    stats = pipeline.process_all_pdfs(skip_existing=True)
    
    # Option 2: Process a single PDF
    # document_id = pipeline.process_single_pdf('your-pdf-file.pdf')
    # if document_id:
    #     print(f"Successfully processed document ID: {document_id}")


if __name__ == "__main__":
    main()

