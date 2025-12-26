import os
import logging
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

from db_writer import DatabaseWriter
from evidence_extractor import EvidenceExtractor
from evidence_validator import EvidenceValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()


class EvidenceExtractPipeline:
    """
    Pipeline for extracting evidence from text chunks and validating it.
    """
    
    GOALS = [
        'ai_features',
        'performance_degradation',
        'causal_links'
    ]

    def __init__(self,
                 model: str = "deepseek-chat",
                 enable_validation: bool = True,
                 max_workers: int = 3):
        """
        Initialize the evidence extract pipeline.
        
        Args:
            model: The model to use for evidence extraction.
            enable_validation: Whether to enable validation of the evidence.
            max_workers: The maximum number of workers to use for evidence extraction.
        """
        
        logger.info("=" * 60)
        logger.info("INITIALIZING EVIDENCE EXTRACT PIPELINE...")
        logger.info("=" * 60)
        
        # Initialize database writer
        logger.info("Initializing database writer...")
        try:
            self.db_writer = DatabaseWriter(
                dbname=os.getenv('DB_NAME'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
                host=os.getenv('DB_HOST'),
                port=int(os.getenv('DB_PORT', 5432))
            )
        except Exception as e:
            logger.error(f"Failed to initialize database writer: {e}")
            raise
        
        logger.info("Database writer initialized successfully!")
        
        # Initialize evidence extractor
        logger.info("Initializing evidence extractor...")
        self.evidence_extractor = EvidenceExtractor(
            model=model
        )
        logger.info("Evidence extractor initialized successfully!")
        
        # Initialize evidence validator
        self.enable_validation = enable_validation
        if enable_validation:
            logger.info("Initializing evidence validator...")
            self.validator = EvidenceValidator(db_writer=self.db_writer)
            logger.info("Evidence validator initialized successfully!")
        else:
            logger.info("Validation is disabled. Skipping evidence validator initialization.")
        
        
        self.max_workers = max_workers
        
        logger.info("\n" + "=" * 60)
        logger.info("EVIDENCE EXTRACT PIPELINE INITIALIZED SUCCESSFULLY!")
        logger.info("=" * 60 + "\n")
        
    def extract_from_documents(self,
                               document_ids: List[int],
                               goals: List[str],
                               skip_extracted: bool = True) -> Dict:
        """
        Extract evidence from documents.
        
        Args:
            document_ids: The IDs of the documents to extract evidence from.
            goals: The goals to extract evidence for.
        Returns:
            Statistics dictionary containing the total number of documents, chunks, and evidence extracted.
        """
        
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTING EVIDENCE FROM DOCUMENTS...")
        logger.info("=" * 60 + "\n")
        
        logger.info(f"Goals: {', '.join(goals)}")
        logger.info(f"Max workers: {self.max_workers}")
        logger.info(f"Validation: {'Enabled' if self.enable_validation else 'Disabled'}")
        
        stats = {
            'documents_processed': 0,
            'total_chunks': 0,
            'chunks_extracted': 0,
            'evidence_found': 0,
            'evidence_validated': 0,
            'evidence_failed_validation': 0,
            'by_goal': dict.fromkeys(goals, 0),
            'api_requests': 0,
            'start_time': datetime.now(),
        }
        
        for doc_id in document_ids:
            logger.info("\n" + "=" * 60)
            logger.info(f"Processing document ID: {doc_id}")
            logger.info("=" * 60 + "\n")
            
            doc_stats = self.extract_from_document(
                document_id=doc_id,
                goals=goals,
                skip_extracted=skip_extracted
            )
            
            stats['documents_processed'] += 1
            stats['total_chunks'] += doc_stats['total_chunks']
            stats['chunks_extracted'] += doc_stats['chunks_extracted']
            stats['evidence_found'] += doc_stats['evidence_found']
            stats['evidence_validated'] += doc_stats['evidence_validated']
            stats['evidence_failed_validation'] += doc_stats['evidence_failed_validation']
            stats['api_requests'] += doc_stats['api_requests']
            
            for goal, count in doc_stats['by_goal'].items():
                stats['by_goal'][goal] += count
                
        stats['end_time'] = datetime.now()
        stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()
        
        self.print_statistics(stats)
        
        return stats
    
    def extract_from_document(self,
                              document_id: int,
                              goals: List[str],
                              skip_extracted: bool = True) -> Dict:
        """
        Extract evidence from a document.
        
        Args:
            document_id: The ID of the document to extract evidence from.
            goals: The goals to extract evidence for.
            skip_extracted: Skip chunks that have already been extracted.
        
        Returns:
            Statistics dictionary containing the total number of chunks, evidence found, and evidence validated.
        """
        

        stats = {
            'total_chunks': 0,
            'chunks_extracted': 0,
            'evidence_found': 0,
            'evidence_validated': 0,
            'evidence_failed_validation': 0,
            'by_goal': dict.fromkeys(goals, 0),
            'api_requests': 0,
            'start_time': datetime.now(),
        }
        
        doc = self.db_writer.get_document(document_id)
        
        if not doc:
            logger.error(f"Document with ID {document_id} not found")
            return stats
        
        logger.info(f"Document found: {doc['title']}")
        
        chunks = self.db_writer.get_chunks_by_document(document_id)
        stats['total_chunks'] = len(chunks)
        
        if not chunks:
            logger.error(f"No chunks found for document with ID {document_id}")
            return stats
        
        logger.info(f"Total chunks found: {stats['total_chunks']}")
        
        for chunk in chunks:
            chunk['document_id'] = document_id
        
        # Create a mapping from chunk_id to chunk for quick lookup during validation
        chunk_by_id = {chunk['chunk_id']: chunk for chunk in chunks}
        
        for goal in goals:
            logger.info(f"Processing goal: {goal}")
            
            chunks_to_process = chunks
            if skip_extracted:
                chunks_to_process = self.filter_unextracted_chunks(chunks_to_process, document_id, goal)
                
                skipped = len(chunks) - len(chunks_to_process)
                
                logger.info(f"Skipped {skipped} chunks that have already been extracted for goal: {goal}")
                
            if not chunks_to_process:
                logger.info(f"No chunks to process for goal: {goal}")
                continue
            
            logger.info(f"Processing {len(chunks_to_process)} chunks for goal: {goal}")
            
            extraction_results = self.evidence_extractor.extract_batch(
                chunks=chunks_to_process,
                prompt_type=goal,
                max_workers=self.max_workers
            )

            stats['api_requests'] += len(extraction_results)
            
            goal_evidence_count = 0
            goal_validated_count = 0
            goal_failed_validation_count = 0
            
            for result in extraction_results:
                if not result['success']:
                    logger.error(f"Error extracting evidence for chunk {result['chunk_id']}: {result['error']}")
                    continue
                
                chunk_id = result['chunk_id']
                evidence_response = result.get('evidence', {})
                
                # Extract evidence items based on goal type
                evidence_items = self.db_writer.extract_evidence_items_from_response(evidence_response, goal)
                
                if not evidence_items:
                    logger.debug(f"No evidence found for chunk {chunk_id} with goal {goal}")
                    continue
                
                # Validate evidence if validation is enabled
                if self.enable_validation:
                    validated_items = []
                    
                    for item in evidence_items:
                        # Use the quote field for validation (character-by-character exact)
                        quote = item.get('quote', '')
                        # Also check metadata for quote (if stored there by db_writer)
                        if not quote and 'metadata' in item:
                            quote = item['metadata'].get('quote', '')
                        # Fall back to excerpt if quote not available (backwards compatibility)
                        text_to_validate = quote if quote else item.get('excerpt', '')
                        chunk_text = chunk_by_id[chunk_id]['text']
                        
                        if text_to_validate:
                            validation_result = self.validator.validate_excerpt(
                                source_text=chunk_text,
                                excerpt=text_to_validate
                            )

                            item['validation'] = validation_result
                            
                            if validation_result['success']:
                                goal_validated_count += 1
                                validated_items.append(item)
                                # Use the chunk_id from validation result if available
                                validated_chunk_id = validation_result.get('chunk_id')
                                if validated_chunk_id:
                                    chunk_id = validated_chunk_id
                            else:
                                goal_failed_validation_count += 1
                                print(validation_result)
                                logger.warning(f"Validation failed for {'quote' if quote else 'excerpt'}: {text_to_validate[:100]}... in chunk {chunk_id}")
                    
                    evidence_items = validated_items
                    
                # Save evidence to database - need to pass the validated evidence items
                try:
                    saved_count = self.db_writer.save_extracted_evidence(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        evidence_items=evidence_items,  # Use validated evidence_items, not original result
                        prompt_type=goal,
                        model_used=result.get('model', 'unknown')
                    )
                    goal_evidence_count += saved_count
                    
                    if saved_count > 0:
                        logger.info(f"Saved {saved_count} evidence items for chunk {chunk_id} (goal: {goal})")
                    
                except Exception as e:
                    logger.error(f"Failed to save evidence for chunk {chunk_id}: {e}")
                    continue
            
            stats['by_goal'][goal] = goal_evidence_count
            stats['evidence_found'] += goal_evidence_count
            stats['evidence_validated'] += goal_validated_count
            stats['evidence_failed_validation'] += goal_failed_validation_count
            # Only count chunks_extracted once (not per goal) - move outside goal loop
            
            logger.info(f"Goal '{goal}' completed: {goal_evidence_count} evidence items saved")
        
        # Count unique chunks that were processed for any goal
        stats['chunks_extracted'] = len(chunks)
        
        stats['end_time'] = datetime.now()
        stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()
        
        return stats
    
    def filter_unextracted_chunks(self, chunks: List[Dict], document_id: int, goal: str) -> List[Dict]:
        """
        Filter chunks that have already been extracted for a specific goal.
        
        Args:
            chunks: List of chunks to filter
            document_id: Document ID
            goal: The goal/prompt_type to check for existing extractions
        
        Returns:
            List of chunks that haven't been extracted for this goal yet
        """
        
        unextracted_chunks = []
        
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            
            # Check if evidence already exists for this chunk and goal
            try:
                with self.db_writer.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            """
                            SELECT COUNT(*) FROM extracted_evidence 
                            WHERE chunk_id = %s AND document_id = %s AND excerpt_type = %s
                            """,
                            (chunk_id, document_id, goal)
                        )
                        count = cursor.fetchone()[0]
                        
                        if count == 0:
                            unextracted_chunks.append(chunk)
                        else:
                            logger.debug(f"Chunk {chunk_id} already has {count} evidence items for goal '{goal}'")
            
            except Exception as e:
                logger.error(f"Error checking existing evidence for chunk {chunk_id}: {e}")
                # If error checking, include the chunk to be safe
                unextracted_chunks.append(chunk)
        
        return unextracted_chunks
    
    def print_statistics(self, stats: Dict) -> None:
        """
        Print pipeline statistics.
        
        Args:
            stats: Statistics dictionary
        """
        
        logger.info("\n" + "=" * 60)
        logger.info("EVIDENCE EXTRACTION PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"📊 Documents processed: {stats['documents_processed']}")
        logger.info(f"📄 Total chunks: {stats['total_chunks']}")
        logger.info(f"⚙️  Chunks extracted: {stats['chunks_extracted']}")
        logger.info(f"🔍 Evidence found: {stats['evidence_found']}")
        logger.info(f"✅ Evidence validated: {stats['evidence_validated']}")
        logger.info(f"❌ Evidence failed validation: {stats['evidence_failed_validation']}")
        logger.info(f"🤖 API requests made: {stats['api_requests']}")
        logger.info(f"⏱️  Duration: {stats.get('duration', 0):.2f} seconds")
        
        if stats.get('evidence_found', 0) > 0:
            success_rate = (stats['evidence_validated'] / stats['evidence_found']) * 100
            logger.info(f"📈 Validation success rate: {success_rate:.1f}%")
        
        logger.info("\n📈 By Goal:")
        for goal, count in stats['by_goal'].items():
            logger.info(f"   {goal}: {count} evidence items")
        
        logger.info("=" * 60 + "\n")


def main():
    """
    Example usage of the evidence extraction pipeline
    """
    
    # Initialize pipeline
    pipeline = EvidenceExtractPipeline(
        model="deepseek-chat",
        enable_validation=True,
        max_workers=3
    )
    
    # Example: Extract evidence for specific document IDs and goals
    document_ids = [1,2,3,4,5,6,7,8,9,10,11,12]  # Replace with actual document IDs
    goals = ['ai_features', 'performance_degradation', 'causal_links']
    
    # Run the pipeline
    stats = pipeline.extract_from_documents(
        document_ids=document_ids,
        goals=goals,
        skip_extracted=True
    )
    
    return stats


if __name__ == "__main__":
    main()
                