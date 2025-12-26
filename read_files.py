# read files from an S3 bucket
import boto3
import os
import io
import re
import hashlib
from dotenv import load_dotenv
from PyPDF2 import PdfReader
load_dotenv()

class S3Reader:
    def __init__(self):
        """
        Initialize the S3 reader
        
        Args:
            aws_access_key_id: The AWS access key ID
            aws_secret_access_key: The AWS secret access key
            aws_region: The AWS region
            s3_bucket_name: The name of the S3 bucket
        """
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_REGION')
        self.s3_bucket_name = os.getenv('S3_BUCKET_NAME')
        self.s3_client = boto3.client('s3', aws_access_key_id=self.aws_access_key_id, aws_secret_access_key=self.aws_secret_access_key, region_name=self.aws_region)

    def list_pdfs(self):
        """
        List all PDF files in the S3 bucket
        
        Returns:
            List of PDF file names
        """
        response = self.s3_client.list_objects_v2(
            Bucket=self.s3_bucket_name
        )
        
        pdfs = []
        
        if 'Contents' in response:
            pdfs = [obj['Key'] for obj in response['Contents']
                    if obj['Key'].endswith('.pdf')]
        
        return pdfs
    
    def download_pdf(self, s3_key: str):
        """
        Download a PDF file from the S3 bucket
        
        Args:
            s3_key: The S3 key of the PDF file
        
        Returns:
            PDF file data
            Exception if failed
        """
        response = self.s3_client.get_object(
            Bucket=self.s3_bucket_name,
            Key=s3_key
        )
        
        return response['Body'].read()
    
    def extract_text_from_pdf(self, pdf_data: bytes):
        """Extract text from a PDF file
        
        Args:
            pdf_data: The PDF file data
        
        Returns:
            Extracted text
        """
        pdf_file = io.BytesIO(pdf_data)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def clean_text(self, text: str, remove_references: bool = True) -> str:
        """
        Clean and preprocess extracted text
        
        Args:
            text: Raw text extracted from PDF
            remove_references: If True, remove references/bibliography section
        
        Returns:
            Cleaned text
            Exception if failed
        """
        if not text:
            return text
        
        # Remove references section if requested
        if remove_references:
            # Common reference section headers (case-insensitive)
            reference_patterns = [
                r'\n\s*(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography|WORKS CITED|Works Cited)\s*\n',
                r'\n\s*\d+\.\s*(?:REFERENCES|References)\s*\n'
            ]
            
            for pattern in reference_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Truncate text at the references section
                    text = text[:match.start()]
                    break
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double newline
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Standalone page numbers
        text = re.sub(r'\n\s*Page \d+\s*\n', '\n', text, flags=re.IGNORECASE)
        
        # Remove common PDF artifacts
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)  # Control characters
        
        return text.strip()
    
    def get_pdf_metadata(self, s3_key: str, pdf_data: bytes):
        """Get metadata from a PDF file
        
        Args:
            s3_key: The S3 key of the PDF file
            pdf_data: The PDF file data
        
        Returns:
            Metadata dictionary
        """
        pdf_file = io.BytesIO(pdf_data)
        pdf_file.seek(0) # reset the file pointer to the beginning of the file
        reader = PdfReader(pdf_file)
        
        metadata = {
            's3_key': s3_key,
            'num_pages': len(reader.pages),
            'file_hash': hashlib.sha256(pdf_data).hexdigest(),
        }
        
        if reader.metadata:
            metadata.update({
                'title': reader.metadata.title,
                'author': reader.metadata.author,
                'subject': reader.metadata.subject,
            })
            
        return metadata
