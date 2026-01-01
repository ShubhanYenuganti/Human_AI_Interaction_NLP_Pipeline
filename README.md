# Human-AI Interaction NLP Pipeline for B. John Garrick Institute for the Risk Sciences.

**B. John Garrick Institute for the Risk Sciences**

## 🎯 Overview

An automated evidence extraction and validation system designed to identify and analyze research evidence from research documents about human-AI interaction. The pipeline leverages DeepSeek AI to extract evidence across three critical research domains: AI features, performance degradation, and causal links.

### 🏗️ Architecture

```
📁 S3 Bucket (PDFs) → 📄 PDF Processing → 🔄 Semantic Chunking → 🧠 Vector Embeddings → 💾 PostgreSQL (pgvector) → 🤖 Evidence Extraction & Validation (DeepSeek)
```

### 🎯 Primary Research Goals

1. **AI Features**: Identify and document AI system features and characteristics in human-AI interaction contexts
2. **Performance Degradation**: Extract evidence of performance issues, degradation patterns, and failure modes
3. **Causal Links**: Capture causal relationships between AI features and observed outcomes in human behavior

## 🛠️ Components

### Evidence Extraction Modules

- **`evidence_extract_pipeline.py`**: **[PRIMARY]** End-to-end evidence extraction pipeline
- **`evidence_extractor.py`**: Core evidence extraction logic using DeepSeek AI
- **`evidence_validator.py`**: Evidence validation and quality assessment
- **`prompt_templates.py`**: Structured prompts for evidence extraction tasks

### Supporting Infrastructure

- **`pipeline.py`**: PDF processing and embedding generation pipeline
- **`read_files.py`**: S3 PDF reader and text extraction
- **`chunk.py`**: Hybrid semantic text chunking
- **`vector.py`**: Vector embedding generation using sentence-transformers
- **`db_writer.py`**: PostgreSQL database operations with pgvector

### Optional Query Interface

- **`gemini.py`**: RAG-based query system with Google Gemini (for exploratory analysis)

## 📋 Prerequisites

- **Python 3.8+**
- **PostgreSQL 12+** with **pgvector extension**
- **AWS Account** with S3 access
- **DeepSeek API** access (**required** for evidence extraction)
- **Google Gemini API** access (optional - only for exploratory querying)

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
psql -U nlp_user -d nlp_pipeline -f db.sql
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

#### DeepSeek API (**Required**)
1. Visit [DeepSeek Platform](https://platform.deepseek.com/)
2. Create an account and generate an API key
3. Add it to your `.env` file as `DEEPSEEK_API_KEY`

#### Google Gemini API (Optional)
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file as `GEMINI_API_KEY`

#### AWS S3 Setup
1. Create an AWS account and S3 bucket
2. Upload your research PDF documents into the bucket
3. Create IAM user with S3 read permissions
4. Add credentials to `.env` file

## 🚀 Usage

### Quick Start: Evidence Extraction

#### 1. Process PDFs and Generate Embeddings

First, process your PDF documents from S3 to create vector embeddings:

```bash
python pipeline.py
```

This will:
- Download PDFs from your configured S3 bucket and store vector embeddings and metadata into the database

#### 2. Run Evidence Extraction Pipeline

Extract and validate evidence across all three research domains:

```bash
python evidence_extract_pipeline.py
```

The pipeline will:
- Analyze all processed document chunks
- Extract evidence for each research goal:
  - **AI Features**: System characteristics and capabilities
  - **Performance Degradation**: Issues and failure patterns
  - **Causal Links**: Relationships between features and outcomes
- Validate extracted evidence with confidence scoring
- Store results in the `extracted_evidence` table

#### 3. Review Extracted Evidence

Query the database to review extracted evidence:

```sql
-- View recent evidence extractions
SELECT excerpt_type, excerpt, relevance_score, validation_confidence
FROM extracted_evidence
ORDER BY created_at DESC
LIMIT 10;

-- View evidence by type
SELECT COUNT(*), excerpt_type
FROM extracted_evidence
GROUP BY excerpt_type;
```

### Optional: Exploratory Query Interface

For ad-hoc questions about your documents (requires Gemini API):

```bash
python gemini.py
```

Interactive terminal for real-time Q&A with your document corpus.omprehensive logging:
- Pipeline processing status
- Database operations
- API calls and responses
- Performance metrics
