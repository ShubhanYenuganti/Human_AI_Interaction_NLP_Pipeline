# Human-AI Interaction NLP Pipeline

## 🚀 Overview

This is a comprehensive **Retrieval-Augmented Generation (RAG) system** designed for intelligent document processing and question-answering. The pipeline processes PDF documents from AWS S3, performs semantic chunking, generates vector embeddings, and provides an AI-powered query interface using Google Gemini and DeepSeek models.

### 🏗️ Architecture

The system consists of several interconnected components:

```
📁 S3 Bucket (PDFs) → 📄 PDF Processing → 🔄 Semantic Chunking → 🧠 Vector Embeddings → 💾 PostgreSQL (pgvector) → 🤖 RAG Query System
```

### 🎯 Key Features

- **PDF Processing Pipeline**: Automated extraction and processing of PDFs from S3
- **Hybrid Semantic Chunking**: Intelligent text segmentation using embedding-based similarity
- **Vector Search**: Fast similarity search using PostgreSQL with pgvector extension
- **Multi-Model AI Integration**: Support for Google Gemini and DeepSeek models
- **Evidence Extraction & Validation**: Advanced evidence extraction with validation mechanisms
- **Interactive Query Interface**: Terminal-based interactive mode for real-time Q&A
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## 🛠️ Components

### Core Modules

- **`pipeline.py`**: Main PDF processing pipeline
- **`gemini.py`**: RAG system with Google Gemini integration
- **`read_files.py`**: S3 PDF reader and text extraction
- **`chunk.py`**: Hybrid semantic text chunking
- **`vector.py`**: Vector embedding generation using sentence-transformers
- **`db_writer.py`**: PostgreSQL database operations with pgvector
- **`evidence_extractor.py`**: Evidence extraction using DeepSeek
- **`evidence_validator.py`**: Evidence validation and quality assessment
- **`evidence_extract_pipeline.py`**: End-to-end evidence extraction pipeline

## 📋 Prerequisites

- **Python 3.8+**
- **PostgreSQL 12+** with **pgvector extension**
- **AWS Account** with S3 access
- **Google Gemini API** access (for general querying)
- **DeepSeek API** access (for evidence extraction)

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Human_AI_Interaction_NLP_Pipeline
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up PostgreSQL Database

#### Install PostgreSQL and pgvector

**macOS (Homebrew):**
```bash
brew install postgresql
brew install pgvector
```

#### Create Database and Enable Extensions

```sql
-- Connect to PostgreSQL as superuser
sudo -u postgres psql

-- Create database and user
CREATE DATABASE nlp_pipeline;
CREATE USER nlp_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE nlp_pipeline TO nlp_user;

-- Connect to the new database
\c nlp_pipeline;

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant permissions
GRANT ALL ON SCHEMA public TO nlp_user;
```

#### Initialize Database Schema

```bash
# Run the provided SQL schema
psql -U nlp_user -d nlp_pipeline -f db1226.sql
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```env
# Database Configuration
DB_NAME=nlp_pipeline
DB_USER=nlp_user
DB_PASSWORD=your_secure_password
DB_HOST=localhost
DB_PORT=5432

# AWS S3 Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-pdf-bucket-name

# AI Model APIs
GEMINI_API_KEY=your_gemini_api_key # for general query
DEEPSEEK_API_KEY=your_deepseek_api_key  # for evidence extraction
```

### 5. API Keys Setup

#### Google Gemini API
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file as `GEMINI_API_KEY`

#### DeepSeek API (Optional)
1. Visit [DeepSeek Platform](https://platform.deepseek.com/)
2. Create an account and generate an API key
3. Add it to your `.env` file as `DEEPSEEK_API_KEY`

#### AWS S3 Setup
1. Create an AWS account and S3 bucket
2. Create IAM user with S3 read permissions
3. Add credentials to `.env` file

## 🚀 Usage

### 1. PDF Processing Pipeline

Process all PDFs in your S3 bucket:

```bash
python pipeline.py
```

### 2. RAG Query System

#### Interactive Mode

```bash
python gemini.py
```

This starts an interactive terminal where you can ask questions:

```
❓ Your question: What are the main findings about climate change?
```

### 3. Evidence Extraction Pipeline

```bash
python evidence_extract_pipeline.py
```

## 📊 Database Schema

The system uses four main tables:

- **`documents`**: Stores PDF metadata and processing status
- **`chunks`**: Contains text chunks with embeddings
- **`queries`**: Logs user queries and responses
- **`extracted_evidence`**: Stores extracted evidence with validation scores

## ⚙️ Configuration Options

### Chunking Parameters

- `target_chunk_size`: Target sentences per chunk (default: 8)
- `min_chunk_size`: Minimum sentences per chunk (default: 2)
- `max_chunk_size`: Maximum sentences per chunk (default: 16)

### Embedding Models

- Default: `sentence-transformers/all-MiniLM-L6-v2`
- Other options: `all-mpnet-base-v2`, `multi-qa-mpnet-base-dot-v1`

### Device Selection

- `cpu`: CPU processing (default)
- `cuda`: GPU acceleration with NVIDIA CUDA
- `mps`: Apple Silicon GPU acceleration

## 🔍 Troubleshooting

### Common Issues

1. **pgvector Extension Error**
   ```bash
   # Ensure pgvector is properly installed
   sudo apt-get install postgresql-16-pgvector
   ```

2. **CUDA Out of Memory**
   ```python
   # Reduce batch size or use CPU
   pipeline = PDFProcessingPipeline(batch_size=8, device="cpu")
   ```

3. **S3 Access Denied**
   - Verify AWS credentials in `.env`
   - Ensure S3 bucket permissions allow read access

4. **Database Connection Issues**
   ```bash
   # Test database connection
   psql -U nlp_user -d nlp_pipeline -h localhost
   ```

### Performance Optimization

- Use GPU acceleration for faster embedding generation
- Adjust `batch_size` based on available memory
- Consider using larger embedding models for better accuracy
- Monitor PostgreSQL performance with appropriate indexing

## 📈 Monitoring and Logs

The system provides comprehensive logging:
- Pipeline processing status
- Database operations
- API calls and responses
- Performance metrics
