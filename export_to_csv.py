import psycopg2
import csv
from typing import Optional
import os
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def export_evidence_to_csv(
    output_file: str = 'evidence_export.csv',
    dbname: str = None,
    user: str = None,
    password: str = None,
    host: str = 'localhost',
    port: int = 5432
):
    """
    Export extracted evidence from PostgreSQL database to CSV with specific columns.
    
    Args:
        output_file: Path to output CSV file
        dbname: Database name (defaults to env variable)
        user: Database user (defaults to env variable)
        password: Database password (defaults to env variable)
        host: Database host
        port: Database port
    """
    
    # Get database credentials from environment if not provided
    dbname = dbname or os.getenv('DB_NAME', 'postgres')
    user = user or os.getenv('DB_USER', 'postgres')
    password = password or os.getenv('DB_PASSWORD')
    
    if not password:
        logger.warning("No database password found. Using empty password.")
        password = ''
    
    try:
        # Connect to database
        logger.info(f"Connecting to database {dbname}@{host}:{port}")
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        
        cursor = conn.cursor()
        
        # Query to get all necessary data
        query = """
            SELECT 
                d.title AS source,
                CASE 
                    WHEN e.excerpt_type = 'ai_features' THEN 
                        CONCAT(
                            'Category: ', COALESCE(e.metadata->>'category', ''), 
                            ' | Feature: ', COALESCE(e.metadata->>'feature_name', '')
                        )
                    ELSE ''
                END AS ai_features,
                CASE 
                    WHEN e.excerpt_type = 'performance_degradation' THEN 
                        CONCAT(
                            'Category: ', COALESCE(e.metadata->>'category', ''), 
                            ' | Severity: ', COALESCE(e.metadata->>'severity', '')
                        )
                    ELSE ''
                END AS performance_degradation,
                CASE 
                    WHEN e.excerpt_type = 'causal_links' THEN 
                        CONCAT(
                            'AI Feature: ', COALESCE(e.metadata->>'ai_feature', ''), 
                            ' | Evidence Type: ', COALESCE(e.metadata->>'evidence_type', ''), 
                            ' | Causal Strength: ', COALESCE(e.metadata->>'causal_strength', ''), 
                            ' | Performance Effect: ', COALESCE(e.metadata->>'performance_effect', '')
                        )
                    ELSE ''
                END AS causal_links,
                e.excerpt,
                CASE 
                    WHEN e.excerpt_type = 'performance_degradation' THEN 
                        CONCAT(
                            'Severity Justification: ', COALESCE(e.metadata->>'justification_severity', ''), 
                            ' | Relevance Justification: ', COALESCE(e.justification_relevance, '')
                        )
                    WHEN e.excerpt_type = 'causal_links' THEN 
                        CONCAT(
                            'Causal Strength Justification: ', COALESCE(e.metadata->>'justification_causal_strength', ''), 
                            ' | Relevance Justification: ', COALESCE(e.justification_relevance, '')
                        )
                    ELSE e.justification_relevance
                END AS justification,
                CASE 
                    WHEN e.validation_status = 'valid' THEN 'y'
                    ELSE 'n'
                END AS validation
            FROM 
                extracted_evidence e
            JOIN 
                chunks c ON e.chunk_id = c.id
            JOIN 
                documents d ON e.document_id = d.id
            ORDER BY 
                d.title, e.created_at;
        """
        
        logger.info("Executing query to fetch evidence data...")
        cursor.execute(query)
        
        # Fetch all results
        results = cursor.fetchall()
        logger.info(f"Retrieved {len(results)} records from database")
        
        # Write to CSV
        logger.info(f"Writing to CSV file: {output_file}")
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            # Define column headers
            headers = [
                'Source',
                'Your finding about AI features',
                'Your finding about Human performance degradation types',
                'Your finding about Causal links between them',
                'Excerpt',
                'Justification',
                'Validation (y/n)'
            ]
            
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerow(headers)
            
            # Write data rows
            for row in results:
                writer.writerow(row)
        
        logger.info(f"Successfully exported {len(results)} records to {output_file}")
        
        # Close connections
        cursor.close()
        conn.close()
        
        return output_file
        
    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export evidence from PostgreSQL to CSV')
    parser.add_argument('--output', '-o', default='evidence_export.csv', 
                        help='Output CSV file path')
    parser.add_argument('--dbname', default=None, help='Database name')
    parser.add_argument('--user', default=None, help='Database user')
    parser.add_argument('--password', default=None, help='Database password')
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', type=int, default=5432, help='Database port')
    
    args = parser.parse_args()
    
    try:
        export_evidence_to_csv(
            output_file=args.output,
            dbname=args.dbname,
            user=args.user,
            password=args.password,
            host=args.host,
            port=args.port
        )
        
        print(f"✅ Export completed successfully! Output saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        exit(1)
