import psycopg2
from psycopg2.extras import execute_values, Json
from psycopg2 import sql
from typing import List, Dict, Optional, Tuple, Any
import hashlib
import json
from datetime import datetime
import numpy as np
from contextlib import contextmanager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseWriter:
    def __init__(self,
                 dbname: str,
                 user: str,
                 password: str,
                 host: str = 'localhost',
                 port: int = 5432,
                 autocommit: bool = False):
        """
        Initialize the database writer
        
        Args:
            dbname: The name of the database
            user: The username to connect to the database
            password: The password to connect to the database
            host: The host of the database
            port: The port of the database
            autocommit: Whether to commit the transaction automatically
        """
        self.conn_params = {
            'dbname': dbname,
            'user': user,
            'password': password,
            'host': host,
            'port': port,
        }
        self.autocommit = autocommit
        self.conn = None
        self.cursor = None
        
        # Test connection
        self.test_connection()
    
    def test_connection(self):
        """
        Test the connection to the database
        
        Returns:
            True if successful, False otherwise
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Test basic connection
                    cursor.execute("SELECT version();")
                    version = cursor.fetchone()[0]
                    logger.info(f"Connected to PostgreSQL version: {version}")
                    
                    # Test pgvector extension
                    cursor.execute(
                        "SELECT extversion from pg_extension where extname = 'vector';"
                    )
                    
                    vector_version = cursor.fetchone()
                    if vector_version:
                        logger.info(f"pgvector extension version: {vector_version[0]}")
                    else:
                        logger.info("pgvector extension not found")
        except Exception as e:
            logger.error(f"Failed to test connection: {e}")
            raise e
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Returns:
            Connection to the database
        """
        
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            yield conn
            if self.autocommit:
                conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error in database operation: {e}")
            raise e
        finally:
            conn.close()
    
    def calculate_file_hash(self, file_content: bytes) -> str:
        """
        Calculate the SHA-256 hash of the file content
        
        Args:
            file_content: The content of the file to calculate the hash of
        
        Returns:
            The SHA-256 hash of the file content
        """
        
        return hashlib.sha256(file_content).hexdigest()
    
    def insert_document(self, 
                        s3_key: str,
                        file_hash: str,
                        num_pages: int,
                        title: Optional[str] = None,
                        author: Optional[str] = None,
                        publication_year: Optional[int] = None,
                        journal: Optional[str] = None,
                        doi: Optional[str] = None,
                        metadata: Optional[Dict] = None,
                        status: str = 'pending') -> Optional[int]:
        """
        Insert a document into the database
        
        Args:
            s3_key: The S3 key of the document
            file_hash: The hash of the document
            num_pages: The number of pages in the document
            title: Optional title of the document
            author: Optional author of the document
            publication_year: Optional publication year of the document
            journal: Optional journal of the document
            doi: Optional DOI of the document
            metadata: Optional metadata of the document
            status: The status of the document
        
        Returns:
            Document ID if successful, None otherwise
        """
        
        try: 
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Check for duplicate by file hash
                    cursor.execute(
                        "SELECT id FROM documents WHERE file_hash = %s",
                        (file_hash,)
                    )
                    
                    existing = cursor.fetchone()
                    
                    if existing:
                        logger.warning(f"Document with hash {file_hash} already exists")
                        return existing[0]
                    
                    # Insert new document
                    insert_query = """
                        INSERT INTO documents (
                            s3_key,
                            title,
                            author,
                            publication_year,
                            journal,
                            doi,
                            file_hash,
                            num_pages,
                            status,
                            metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """
                    
                    cursor.execute(
                        insert_query,
                        (
                            s3_key,
                            title,
                            author,
                            publication_year,
                            journal,
                            doi,
                            file_hash,
                            num_pages,
                            status,
                            Json(metadata or {})
                        )
                    )
                    
                    document_id = cursor.fetchone()[0]
                    conn.commit()
                    
                    logger.info(f"Inserted document with ID: {document_id}: {title or s3_key}")
                    return document_id
        except Exception as e:
            logger.error(f"Error inserting document: {e}")
            raise e
    
    def update_document_status(self,
                               document_id: int,
                               status: str,
                               error_message: Optional[str] = None,
                               processed_date: Optional[datetime] = None) -> bool:
        """
        Update the document processing status
        
        Args:
            document_id: The ID of the document to update
            status: The new status of the document
            error_message: Optional error message
            processed_date: Optional date the document was processed
        
        Returns:
            Boolean indicating success or failure
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    update_query = """
                        UPDATE documents
                        SET status = %s,
                            error_message = %s,
                            processed_date = COALESCE(%s, processed_date)
                        WHERE id = %s
                    """
                    
                    cursor.execute(
                        update_query,
                        (status, error_message, processed_date, document_id)
                    )
                    
                    conn.commit()
                    logger.info(f"Updated document {document_id} status to: {status}")
                    return True
        except Exception as e:
            logger.error(f"Error updating document status: {e}")
            raise e
        
    def get_document(self, document_id: int) -> Optional[Dict]:
        """
        Retrieve a document by its ID
        
        Args:
            document_id: The ID of the document to retrieve
        
        Returns: Document data as dict or None if not found
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT id, s3_key, title, author, publication_year,
                               journal, doi, file_hash, num_pages, upload_date,
                               processed_date, status, error_message, metadata
                        FROM documents
                        WHERE id = %s
                        """,
                        (document_id,)
                    )
                    
                    row = cursor.fetchone()
                    if not row:
                        logger.warning(f"Document with ID {document_id} not found")
                        return None

                    return {
                        'id': row[0],
                        's3_key': row[1],
                        'title': row[2],
                        'author': row[3],
                        'publication_year': row[4],
                        'journal': row[5],
                        'doi': row[6],
                        'file_hash': row[7],
                        'num_pages': row[8],
                        'upload_date': row[9],
                        'processed_date': row[10],
                        'status': row[11],
                        'error_message': row[12],
                        'metadata': row[13]
                    }
                    
        except Exception as e:
            logger.error(f"Error getting document: {e}")
            raise e
    
    def get_document_by_hash(self, file_hash: str) -> Optional[Dict]:
        """
        Retrieve a document by its file hash
        
        Args:
            file_hash: The hash of the document to retrieve
        
        Returns: Document data as dict or None if not found
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT id, s3_key, title, author, publication_year,
                               journal, doi, file_hash, num_pages, upload_date,
                               processed_date, status, error_message, metadata
                        FROM documents
                        WHERE file_hash = %s
                        """,
                        (file_hash,)
                    )
                    
                    row = cursor.fetchone()
                    if not row:
                        return None

                    return {
                        'id': row[0],
                        's3_key': row[1],
                        'title': row[2],
                        'author': row[3],
                        'publication_year': row[4],
                        'journal': row[5],
                        'doi': row[6],
                        'file_hash': row[7],
                        'num_pages': row[8],
                        'upload_date': row[9],
                        'processed_date': row[10],
                        'status': row[11],
                        'error_message': row[12],
                        'metadata': row[13]
                    }
                    
        except Exception as e:
            logger.error(f"Error getting document by hash: {e}")
            raise e
        
    def read_document_text(self, document_id: int) -> Optional[str]:
        """
        Read the text of a document by combining all chunks text
        
        Args:
            document_id: The ID of the document to read
        
        Returns: The text of the document
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT text
                        FROM chunks
                        WHERE document_id = %s
                        ORDER BY chunk_id ASC
                        """,
                        (document_id,)
                    )
                    
                    rows = cursor.fetchall()
                    document_text = '\n'.join([row[0] for row in rows])
                    return document_text
        
        except Exception as e:
            logger.error(f"Error reading document text: {e}")
            return None
        
    def delete_document(self, document_id: int) -> bool:
        """
        Delete document and all associated chunks (chunks are deleted automatically by cascade)
        
        Args:
            document_id: The ID of the document to delete
        
        Returns: True if successful, False otherwise
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM documents WHERE id = %s",
                        (document_id,)
                    )
                    conn.commit()
                    
                    logger.info(f"Deleted document {document_id}")
                    return True
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise e
    
    def insert_chunks(self,
                      document_id: int,
                      chunk_id: int,
                      text: str,
                      embedding: np.ndarray,
                      section_title: Optional[str] = None,
                      word_count: Optional[int] = None,
                      sentence_count: Optional[int] = None,
                      char_start: Optional[int] = None,
                      char_end: Optional[int] = None,
                      metadata: Optional[Dict] = None) -> Optional[int]:
        """
        Insert a chunk into the database
        
        Args:
            document_id: The ID of the document the chunk belongs to
            chunk_id: The ID of the chunk
            text: The text of the chunk
            embedding: The embedding of the chunk
            section_title: Optional section title
            word_count: Optional word count
            sentence_count: Optional sentence count
            char_start: Optional character start
            char_end: Optional character end
            metadata: Optional metadata
        
        Returns: chunk primary key ID or None if failed
        """
        
        try: 
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # convert embedding to list for pgvector
                    embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                    
                    # Calculate word_count 
                    if word_count is None:
                        word_count = len(text.split())
                    
                    insert_query = """
                        INSERT INTO chunks (
                            document_id, 
                            chunk_id,
                            text,
                            section_title,
                            word_count,
                            sentence_count,
                            char_start,
                            char_end,
                            metadata,
                            embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """
                    
                    cursor.execute(
                        insert_query,
                        (
                            document_id,
                            chunk_id,
                            text,
                            section_title,
                            word_count,
                            sentence_count,
                            char_start,
                            char_end,
                            Json(metadata or {}),
                            embedding_list
                        )
                    )
                    
                    chunk_pk_id = cursor.fetchone()[0]
                    conn.commit()
                    
                    logger.debug(f"Inserted chunk with ID: {chunk_pk_id} for document {document_id}")
                    return chunk_pk_id
        except Exception as e:
            logger.error(f"Error inserting chunk: {e}")
            raise e
    
    def insert_chunk_batch(self,
                           document_id: int,
                           chunks: List[Dict],
                           embeddings: np.ndarray) -> Tuple[int, int]:
        """
        Batch insert a list of chunks into the database
        
        Args:
            document_id: The ID of the document the chunks belong to
            chunks: List of chunks to insert
            embeddings: List of embeddings for the chunks
        
        Return: Tuple of (successful_inserts, failed_inserts)
        """
        
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch in number of chunks and embeddings: {len(chunks)} chunks vs {len(embeddings)} embeddings")
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    batch_data = []
                    
                    for i, chunk in enumerate(chunks):
                        embedding_list = embeddings[i].tolist() if isinstance(embeddings[i], np.ndarray) else embeddings[i]
                        
                        word_count = chunk.get('word_count') or len(chunk['text'].split())
                        
                        batch_data.append((
                            document_id,
                            chunk['chunk_id'],
                            chunk['text'],
                            chunk.get('section_title'),
                            word_count,
                            chunk.get('sentence_count'),
                            chunk.get('char_start'),
                            chunk.get('char_end'),
                            Json(chunk.get('metadata', {})),
                            embedding_list
                        ))
                    
                    # Execute batch insert AFTER loop
                    insert_query = """
                        INSERT INTO chunks (
                            document_id,
                            chunk_id,
                            text,
                            section_title,
                            word_count,
                            sentence_count,
                            char_start,
                            char_end,
                            metadata,
                            embedding
                        )
                        VALUES %s
                        ON CONFLICT (document_id, chunk_id) DO NOTHING
                    """
                    
                    execute_values(
                        cursor,
                        insert_query,
                        batch_data,
                        template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)"
                    )
                    
                    inserted_count = cursor.rowcount
                    conn.commit()
                    
                    logger.info(
                        f"Batch inserted {inserted_count} chunks out of {len(chunks)} for document {document_id}"
                    )
                    
                    failed_count = len(chunks) - inserted_count
                    return inserted_count, failed_count
        except Exception as e:
            logger.error(f"Error inserting chunk batch: {e}")
            raise e
        
    def get_chunks_by_document(self, document_id: int) -> List[Dict]:
        """
        Retrieve all chunks for a given document
        
        Args:
            document_id: The ID of the document to retrieve chunks for
        
        Returns: List of chunk dicts (without embeddings for efficiency)
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT id, chunk_id, text, section_title, word_count, sentence_count, created_at
                        FROM chunks
                        WHERE document_id = %s
                        ORDER BY chunk_id ASC
                        """,
                        (document_id,)
                    )
                    
                    rows = cursor.fetchall()
                    
                    return [
                        {
                            'id': row[0],
                            'chunk_id': row[1],
                            'text': row[2],
                            'section_title': row[3],
                            'word_count': row[4],
                            'sentence_count': row[5],
                            'created_at': row[6]
                        } for row in rows
                    ]
        except Exception as e:
            logger.error(f"Error getting chunks by document: {e}")
            raise e
        
    def get_chunk_with_embedding(self, chunk_id: int) -> Optional[Dict]:
        """
        Retrieve a single chunk including its embedding
        
        Args:
            chunk_id: The ID of the chunk to retrieve
        
        Returns: Dict with chunk data and embedding or None if not found
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT id, document_id, chunk_id, text, section_title, word_count, embedding           
                        FROM chunks
                        WHERE id = %s
                        """,
                        (chunk_id,)
                    )
                    
                    row = cursor.fetchone()
                    
                    if not row:
                        logger.warning(f"Chunk with ID {chunk_id} not found")
                        return None
                    
                    
                    return {
                        'id': row[0],
                        'document_id': row[1],
                        'chunk_id': row[2],
                        'text': row[3],
                        'section_title': row[4],
                        'word_count': row[5],
                        'embedding': row[6]
                    }    
                    
        except Exception as e:
            logger.error(f"Error getting chunk with embedding: {e}")
            raise e
    
    def search_similar_chunks(self,
                              query_embedding: np.ndarray,
                              top_k: int = 10,
                              document_id: Optional[int] = None,
                              similarity_threshold: Optional[float] = None) -> List[Dict]:
        """
        Search for similar chunks using vector similarity (cosine distance)
        
        Args:
            query_embedding: The embedding vector to search with
            top_k: Number of top results to return
            document_id: Optional filter to search within a specific document
            similarity_threshold: Optional minimum similarity score (0-1)
        
        Args:
            query_embedding: The embedding of the query
            top_k: Number of top results to return
            document_id: Optional filter to search within a specific document
            similarity_threshold: Optional minimum similarity score (0-1)
        
        Returns: List of dicts with chunk data and similarity scores
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Convert embedding to list for pgvector
                    embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
                    
                    # Build query with optional filters
                    query = """
                        SELECT 
                            c.id,
                            c.document_id,
                            c.chunk_id,
                            c.text,
                            c.section_title,
                            c.word_count,
                            c.sentence_count,
                            d.title as document_title,
                            d.author,
                            1 - (c.embedding <=> %s::vector) as similarity
                        FROM chunks c
                        JOIN documents d ON c.document_id = d.id
                    """
                    
                    conditions = []
                    params = [embedding_list]
                    
                    if document_id:
                        conditions.append("c.document_id = %s")
                        params.append(document_id)
                    
                    if similarity_threshold:
                        conditions.append("1 - (c.embedding <=> %s::vector) >= %s")
                        params.insert(1, embedding_list)
                        params.append(similarity_threshold)
                    
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                    
                    query += """
                        ORDER BY c.embedding <=> %s::vector
                        LIMIT %s
                    """
                    
                    params.append(embedding_list)
                    params.append(top_k)
                    
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    results = []
                    for row in rows:
                        results.append({
                            'id': row[0],
                            'document_id': row[1],
                            'chunk_id': row[2],
                            'text': row[3],
                            'section_title': row[4],
                            'word_count': row[5],
                            'sentence_count': row[6],
                            'document_title': row[7],
                            'author': row[8],
                            'similarity': float(row[9])
                        })
                    
                    logger.info(f"Found {len(results)} similar chunks")
                    return results
                    
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            raise e
    
    def hybrid_search(self,
                     query_embedding: np.ndarray,
                     keyword: Optional[str] = None,
                     top_k: int = 10,
                     section_title: Optional[str] = None) -> List[Dict]:
        """
        Hybrid search combining vector similarity and keyword filtering
        
        Args:
            query_embedding: The embedding vector to search with
            keyword: Optional keyword to search in text
            top_k: Number of top results to return
            section_title: Optional filter by section title
        
        Returns: List of dicts with chunk data and similarity scores
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
                    
                    query = """
                        SELECT 
                            c.id,
                            c.document_id,
                            c.chunk_id,
                            c.text,
                            c.section_title,
                            d.title as document_title,
                            1 - (c.embedding <=> %s::vector) as similarity
                        FROM chunks c
                        JOIN documents d ON c.document_id = d.id
                        WHERE 1=1
                    """
                    
                    params = [embedding_list]
                    
                    if keyword:
                        query += " AND c.text ILIKE %s"
                        params.append(f"%{keyword}%")
                    
                    if section_title:
                        query += " AND c.section_title ILIKE %s"
                        params.append(f"%{section_title}%")
                    
                    query += """
                        ORDER BY c.embedding <=> %s::vector
                        LIMIT %s
                    """
                    
                    params.append(embedding_list)
                    params.append(top_k)
                    
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    results = []
                    for row in rows:
                        results.append({
                            'id': row[0],
                            'document_id': row[1],
                            'chunk_id': row[2],
                            'text': row[3],
                            'section_title': row[4],
                            'document_title': row[5],
                            'similarity': float(row[6])
                        })
                    
                    logger.info(f"Hybrid search found {len(results)} chunks")
                    return results
                    
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            raise e
        
    def insert_document_with_chunks(self,
                                    document_metadata: Dict,
                                    chunks: List[Dict],
                                    embeddings: np.ndarray) -> int:
        """
        Insert a document and its chunks into the database
        
        Args:
            document_metadata: Dictionary containing document metadata
            chunks: List of chunks to insert
            embeddings: List of embeddings for the chunks
        
        Returns: 
            Document ID if successful
            Exception if failed
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Insert document
                    doc_query = """
                        INSERT INTO documents (
                            s3_key,
                            title,
                            author,
                            publication_year,
                            journal,
                            doi,
                            file_hash,
                            num_pages,
                            status,
                            metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """
                    
                    cursor.execute(
                        doc_query,
                        (
                            document_metadata['s3_key'],
                            document_metadata.get('title'),
                            document_metadata.get('author'),
                            document_metadata.get('publication_year'),
                            document_metadata.get('journal'),
                            document_metadata.get('doi'),
                            document_metadata['file_hash'],
                            document_metadata['num_pages'],
                            document_metadata.get('status', 'pending'),
                            Json(document_metadata.get('metadata', {}))
                        )
                    )
                    
                    document_id = cursor.fetchone()[0]
                    logger.info(f"Inserted document with ID: {document_id}: {document_metadata.get('title') or document_metadata['s3_key']}")
                    
                    batch_data = []
                    for i, chunk in enumerate(chunks):
                        embedding_list = embeddings[i].tolist() if isinstance(embeddings[i], np.ndarray) else embeddings[i]
                        word_count = chunk.get('word_count') or len(chunk['text'].split())
                        
                        batch_data.append((
                            document_id,
                            chunk['chunk_id'],
                            chunk['text'],
                            chunk.get('section_title'),
                            word_count,
                            chunk.get('sentence_count'),
                            chunk.get('char_start'),
                            chunk.get('char_end'),
                            Json(chunk.get('metadata', {})),
                            embedding_list
                        ))
                    
                    # Execute batch insert AFTER loop
                    chunk_query = """
                        INSERT INTO chunks (
                            document_id,
                            chunk_id,
                            text,
                            section_title,
                            word_count,
                            sentence_count,
                            char_start,
                            char_end,
                            metadata,
                            embedding
                        )
                        VALUES %s
                        ON CONFLICT (document_id, chunk_id) DO NOTHING
                    """
                    
                    execute_values(
                        cursor,
                        chunk_query,
                        batch_data,
                        template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)"
                    )
                    
                    inserted_count = cursor.rowcount
                    logger.info(f"Batch inserted {inserted_count} chunks out of {len(chunks)} for document {document_id}")
                    
                    # Update document status
                    cursor.execute(
                        """
                        UPDATE documents
                        SET status = 'completed'
                        WHERE id = %s
                        """,
                        (document_id,)
                    )
                    
                    conn.commit()
                    logger.info(f"Updated document {document_id} status to: completed")
                    return document_id
                    
        except Exception as e:
            logger.error(f"Error inserting document with chunks: {e}")
            raise e

    def insert_query(self, 
                     query_text: str,
                     query_hash: str,
                     llm_response: Dict,
                     retrieval_params: Dict = None,
                     retrieval_results: Dict = None,
                     query_embedding: np.ndarray = None) -> Optional[int]:
        """
        Insert a query record into the database
        
        Args:
            query_text: The text of the query
            query_hash: The hash of the query (SHA-256)
            llm_response: Dict containing:
                - output (str): The LLM's response
                - model (str, optional): Model name
            retrieval_params: Dict containing (optional):
                - top_k (int): Number of results requested
                - max_results (int): Maximum results limit
            retrieval_results: Dict containing (optional):
                - document_ids (List[int]): Retrieved document IDs
                - chunk_ids (List[int]): Retrieved chunk IDs
                - num_documents (int): Count of documents retrieved
                - num_chunks (int): Count of chunks retrieved
                - similarity_scores (List[float]): Similarity scores
                - avg_similarity (float): Average similarity
            query_embedding: The embedding vector of the query
        
        Returns:
            Query ID if successful, None otherwise
        """
        
        try:
            # Set defaults for optional parameters
            retrieval_params = retrieval_params or {}
            retrieval_results = retrieval_results or {}
            
            # Extract values with defaults
            top_k = retrieval_params.get('top_k', 10)
            max_results = retrieval_params.get('max_results', 10)
            
            document_ids = retrieval_results.get('document_ids')
            chunk_ids = retrieval_results.get('chunk_ids')
            num_documents_retrieved = retrieval_results.get('num_documents', 0)
            num_chunks_retrieved = retrieval_results.get('num_chunks', 0)
            similarity_scores = retrieval_results.get('similarity_scores')
            avg_similarity = retrieval_results.get('avg_similarity', 0.0)
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Convert embedding to list for pgvector (consistent with other methods)
                    embedding_list = None
                    if query_embedding is not None:
                        embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
                    
                    insert_query = """
                        INSERT INTO queries (
                            query_text,
                            query_hash,
                            llm_output,
                            llm_model,
                            top_k,
                            max_results,
                            query_embedding,
                            document_ids,
                            chunk_ids,
                            num_documents_retrieved,
                            num_chunks_retrieved,
                            similarity_scores,
                            avg_similarity
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """
                    
                    cursor.execute(insert_query, (
                        query_text,
                        query_hash,
                        llm_response['output'],
                        llm_response['model'],
                        top_k,
                        max_results,
                        embedding_list,
                        document_ids,
                        chunk_ids,
                        num_documents_retrieved,
                        num_chunks_retrieved,
                        similarity_scores,
                        avg_similarity
                    ))
                    
                    query_id = cursor.fetchone()[0]
                    conn.commit()
                    logger.info(f"Inserted query with ID: {query_id}")
                    return query_id
        except Exception as e:
            logger.error(f"Error inserting query: {e}")
            raise e
    
    def extract_evidence_items_from_response(self, evidence_response: Dict, prompt_type: str) -> List[Dict]:
        """
        Extract evidence items from the LLM response JSON structure
        
        Args:
            evidence_response: The evidence JSON response from the LLM (contains 'features', 'degradations', or 'causal_links')
            prompt_type: The type of prompt (ai_features, performance_degradation, causal_links)
        
        Returns:
            List of evidence items (only valid dictionaries)
        """
        
        # Get raw items based on prompt type
        if prompt_type == 'ai_features':
            raw_items = evidence_response.get('features', [])
        elif prompt_type == 'performance_degradation':
            raw_items = evidence_response.get('degradations', [])
        elif prompt_type == 'causal_links':
            raw_items = evidence_response.get('causal_links', [])
        else:
            logger.warning(f"Unknown prompt_type: {prompt_type}")
            return []
        
        # Filter out non-dictionary items and log the issues
        valid_items = []
        invalid_count = 0
        
        for item in raw_items:
            if isinstance(item, dict):
                valid_items.append(item)
            else:
                invalid_count += 1
                logger.debug(f"Skipping invalid item of type {type(item)} in {prompt_type}: {str(item)[:100]}")
        
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid items in {prompt_type} response (expected dictionaries, got other types)")
        
        return valid_items
    
    def normalize_evidence_item(self, item: Dict, prompt_type: str) -> Dict:
        """
        Normalize evidence items from different prompt types into a unified format
        
        Args:
            item: The evidence item from the extraction (varies by prompt_type)
            prompt_type: The type of prompt (ai_features, performance_degradation, causal_links)
        
        Returns:
            Dictionary with unified structure for database insertion
        """
        
        # Base unified structure matching what prompt templates actually return
        unified = {
            'excerpt_type': prompt_type,
            'excerpt': item.get('excerpt', ''),  # Word-for-word with cleaned spacing
            'summary': item.get('summary', ''),
            'relevance_score': item.get('relevance_score', 5),
            'justification_relevance': item.get('justification_relevance', ''),
            'metadata': {
                'quote': item.get('quote', ''),  # Character-by-character exact copy stored in metadata
            }
        }
        
        # Extract and process location_proof information from enhanced prompts
        location_proof = item.get('location_proof', {})
        
        # Handle location_proof as string (JSON) or dict
        if isinstance(location_proof, str):
            try:
                import json
                location_proof = json.loads(location_proof)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to parse location_proof JSON: {location_proof}")
                location_proof = {}
        
        # Extract location information
        location_data = {
            'starts_with': location_proof.get('starts_with', ''),
            'ends_with': location_proof.get('ends_with', ''),
            'position': location_proof.get('position', ''),
            'verified': location_proof.get('verified', False)
        }
        
        # Map prompt_type specific fields to unified format based on prompt templates
        if prompt_type == 'ai_features':
            unified['goal'] = f"AI Feature: {item.get('feature_name', 'Unknown')}"
            unified['metadata'] = {
                'category': item.get('category'),
                'feature_name': item.get('feature_name'),
                'prompt_type': prompt_type,
                'location_proof': location_data
            }
        
        elif prompt_type == 'performance_degradation':
            unified['goal'] = f"Performance Degradation: {item.get('category', 'Unknown')}"
            unified['metadata'] = {
                'category': item.get('category'),
                'severity': item.get('severity'),
                'justification_severity': item.get('justification_severity'),
                'prompt_type': prompt_type,
                'location_proof': location_data
            }
        
        elif prompt_type == 'causal_links':
            unified['goal'] = f"Causal Link: {item.get('ai_feature', 'Unknown')} → {item.get('performance_effect', 'Unknown')}"
            unified['metadata'] = {
                'ai_feature': item.get('ai_feature'),
                'performance_effect': item.get('performance_effect'),
                'causal_strength': item.get('causal_strength'),
                'justification_causal_strength': item.get('justification_causal_strength'),
                'evidence_type': item.get('evidence_type'),
                'prompt_type': prompt_type,
                'location_proof': location_data
            }
        
        else:
            # Fallback for unknown types
            unified['goal'] = f"Evidence extraction: {prompt_type}"
            unified['metadata'] = {
                'prompt_type': prompt_type,
                'location_proof': location_data,
                'original_item': item
            }
        
        return unified
    
    def save_extracted_evidence(self, 
                                chunk_id: int,
                                document_id: int,
                                evidence_items: List[Dict],
                                prompt_type: str,
                                model_used: str) -> int:
        """
        Save extracted evidence to database for a specific chunk
        
        Args:
            chunk_id: The ID of the chunk the evidence belongs to
            document_id: The ID of the document the evidence belongs to
            evidence_items: List of evidence items to save
            prompt_type: The type of prompt used to extract the evidence
            model_used: The model used to extract the evidence
        
        Returns:
            Number of evidence items saved
        """
        
        saved_count = 0
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    for item in evidence_items:
                        normalized_item = self.normalize_evidence_item(item, prompt_type)
                        
                        # Add model_used to metadata
                        normalized_item['metadata']['model_used'] = model_used
                        
                        insert_query = """
                            INSERT INTO extracted_evidence (
                                chunk_id,
                                document_id,
                                excerpt_type,
                                excerpt,
                                summary,
                                relevance_score,
                                justification_relevance,
                                validation_status,
                                validation_method,
                                validation_confidence,
                                metadata
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING id
                        """
                        
                        # Handle validation data from evidence_validator if present
                        validation_data = item.get('validation', {})
                        validation_status = 'valid' if validation_data.get('success', False) else 'pending'
                        validation_method = validation_data.get('method')
                        validation_confidence = validation_data.get('confidence')
                        
                        # Ensure confidence is a Python float, not numpy type
                        if validation_confidence is not None:
                            validation_confidence = float(validation_confidence)
                        
                        cursor.execute(insert_query, (
                            chunk_id,
                            document_id,
                            normalized_item['excerpt_type'],
                            normalized_item['excerpt'],
                            normalized_item['summary'],
                            normalized_item['relevance_score'],
                            normalized_item['justification_relevance'],
                            validation_status,
                            validation_method,
                            validation_confidence,
                            Json(normalized_item.get('metadata', {}))
                        ))
                        
                        saved_count += 1
                        
                    conn.commit()
                    logger.info(f"Saved {saved_count} evidence items for chunk {chunk_id}")
                    return saved_count
        except Exception as e:
            logger.error(f"Error saving extracted evidence: {e}")
            raise e
        
    def find_chunk_by_text(self, text: str) -> Optional[int]:
        """
        Find a chunk by a given substring of its text
        
        Args:
            text: The substring of the text to search for
        
        Returns:
            The chunk ID if found, None otherwise
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT chunk_id
                        FROM chunks
                        WHERE text ILIKE %s
                        """,
                        (f"%{text}%",)
                    )
                    
                    row = cursor.fetchone()
                    return row[0] if row else None
        except Exception as e:
            logger.error(f"Error finding chunk by text: {e}")
            raise e