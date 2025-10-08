# Configuration Guide

This guide explains all configuration options for the Agentic IT Support Chatbot.

## Table of Contents

- [Environment Variables](#environment-variables)
- [LLM Configuration](#llm-configuration)
- [Embedding Model Configuration](#embedding-model-configuration)
- [ChromaDB Configuration](#chromadb-configuration)
- [Document Ingestion](#document-ingestion)
- [API Configuration](#api-configuration)
- [Advanced Settings](#advanced-settings)

## Environment Variables

All configuration is managed through environment variables in the `.env` file.

### Creating Your Configuration

```bash
cp .env.sample .env
```

Edit `.env` with your preferred text editor.

## LLM Configuration

Configure the language model used for chat and decision-making.

### Groq (Default)

```bash
# API Key (required)
GROQ_API_KEY=gsk_your_api_key_here

# Model Selection
LLM_MODEL=llama3-70b-8192
# Options: llama3-70b-8192, llama3-8b-8192, mixtral-8x7b-32768

# Temperature (0.0 - 1.0)
LLM_TEMPERATURE=0.7

# Max Tokens
LLM_MAX_TOKENS=2048
```

### Getting a Groq API Key

1. Visit [console.groq.com](https://console.groq.com)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy and paste into your `.env` file

### Alternative LLM Providers

To use different LLM providers, modify `utils/call_llm_groq.py` to support:
- OpenAI
- Anthropic Claude
- Azure OpenAI
- Local models (Ollama, LM Studio)

## Embedding Model Configuration

Embeddings are generated locally for privacy.

```bash
# Model Name
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Embedding Dimension (must match model)
EMBEDDING_DIM=384
```

### Recommended Embedding Models

| Model | Dimension | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | Development, small KB |
| `all-mpnet-base-v2` | 768 | Medium | Better | Production, medium KB |
| `all-MiniLM-L12-v2` | 384 | Medium | Better | Balanced performance |

### Changing Embedding Models

⚠️ **Warning**: Changing the embedding model requires re-indexing all documents.

```bash
# 1. Change model in .env
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DIM=768

# 2. Clear existing database
rm -rf chroma_db/

# 3. Re-index documents
python -c "
from flows import create_indexing_flow
flow = create_indexing_flow()
flow.run({'source_dir': './data/docs'})
"
```

## ChromaDB Configuration

Configure the vector database for document storage.

```bash
# Persistence Directory
CHROMA_PERSIST_DIR=./chroma_db

# Collection Name
CHROMA_COLLECTION_NAME=it_support_docs

# Similarity Metric
# Options: cosine, l2, ip (inner product)
CHROMA_SIMILARITY_METRIC=cosine

# Number of results to retrieve
CHROMA_TOP_K=5

# Minimum similarity threshold (0.0 - 1.0)
CHROMA_MIN_SIMILARITY=0.3
```

### ChromaDB Storage

- **Local**: Stores in `CHROMA_PERSIST_DIR` directory
- **Client-Server**: Configure ChromaDB server URL if using separate instance

```bash
# For remote ChromaDB server
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

## Document Ingestion

Configure how documents are processed and indexed.

```bash
# Source Directory
INGESTION_SOURCE_DIR=./data/docs

# Supported File Extensions
INGESTION_FILE_TYPES=.txt,.md

# Chunk Size (characters)
CHUNK_SIZE=1000

# Chunk Overlap (characters)
CHUNK_OVERLAP=200

# Batch Size for Processing
INGESTION_BATCH_SIZE=10

# Recursive Directory Search
INGESTION_RECURSIVE=true
```

### Chunking Strategy

Adjust chunking parameters based on your document structure:

**Technical Docs** (code snippets, procedures):
```bash
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

**Long-form Guides** (tutorials, manuals):
```bash
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
```

**FAQs** (short Q&A):
```bash
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

## API Configuration

Configure the FastAPI server.

```bash
# Server Host
API_HOST=0.0.0.0

# Server Port
API_PORT=8000

# CORS Origins (comma-separated)
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com

# API Rate Limiting
RATE_LIMIT_PER_MINUTE=60

# Session Timeout (seconds)
SESSION_TIMEOUT=3600
```

### Production API Settings

```bash
# Enable/Disable Debug Mode
DEBUG=false

# Logging Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# Workers (for production)
WORKERS=4

# Request Timeout (seconds)
REQUEST_TIMEOUT=30
```

## Advanced Settings

### RAG Configuration

```bash
# Maximum context tokens for RAG
RAG_MAX_CONTEXT_TOKENS=2000

# Reranking (if implemented)
RAG_RERANK_ENABLED=false

# Hybrid search (keyword + semantic)
RAG_HYBRID_SEARCH=false
RAG_HYBRID_ALPHA=0.5  # 0=pure keyword, 1=pure semantic
```

### Intent Classification

```bash
# Intent classification threshold
INTENT_CONFIDENCE_THRESHOLD=0.6

# Number of previous turns to consider
INTENT_CONTEXT_TURNS=3
```

### Conversation Memory

```bash
# Memory backend (in_memory, redis, dynamodb)
MEMORY_BACKEND=in_memory

# Redis configuration (if using Redis)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Session expiry (seconds)
MEMORY_SESSION_EXPIRY=3600
```

### Troubleshooting Workflows

```bash
# Max troubleshooting steps
TROUBLESHOOT_MAX_STEPS=10

# Timeout per step (seconds)
TROUBLESHOOT_STEP_TIMEOUT=300
```

### Jira Integration (Not Implemented Yet)

```bash
# Jira URL
JIRA_URL=https://your-company.atlassian.net

# Jira Authentication
JIRA_EMAIL=your-email@company.com
JIRA_API_TOKEN=your_jira_api_token

# Project Key
JIRA_PROJECT_KEY=ITSUP

# Default Issue Type
JIRA_ISSUE_TYPE=Task
```

### Security Settings

```bash
# Redaction enabled
REDACTION_ENABLED=true

# Patterns to redact (regex)
REDACTION_PATTERNS=password,api_key,secret,token

# JWT Secret (for authentication)
JWT_SECRET=your-secret-key-change-this

# API Key Authentication (if enabled)
API_KEY_REQUIRED=false
API_KEY_HEADER=X-API-Key
```

## Configuration Validation

Validate your configuration:

```bash
# Check environment variables
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
print('GROQ_API_KEY:', 'Set' if os.getenv('GROQ_API_KEY') else 'Not set')
print('EMBEDDING_MODEL:', os.getenv('EMBEDDING_MODEL'))
print('CHROMA_PERSIST_DIR:', os.getenv('CHROMA_PERSIST_DIR'))
"
```

## Configuration Best Practices

1. **Never commit `.env`** - Use `.env.sample` as a template
2. **Use strong secrets** in production
3. **Set appropriate rate limits** to prevent abuse
4. **Enable logging** for troubleshooting
5. **Configure backups** for ChromaDB directory
6. **Use environment-specific configs** (dev, staging, prod)
7. **Document custom settings** for your team

## Environment-Specific Configurations

### Development

```bash
DEBUG=true
LOG_LEVEL=DEBUG
LLM_TEMPERATURE=0.7
CHROMA_PERSIST_DIR=./chroma_db_dev
```

### Production

```bash
DEBUG=false
LOG_LEVEL=INFO
LLM_TEMPERATURE=0.3
CHROMA_PERSIST_DIR=/var/lib/agentic_it/chroma_db
WORKERS=4
API_KEY_REQUIRED=true
```

## Troubleshooting Configuration

**Issue: API key not working**
- Verify key is correct and has no extra spaces
- Check if key has been revoked
- Ensure proper permissions/quota

**Issue: Embeddings failing**
- Verify model name is correct
- Check internet connection (first-time download)
- Ensure sufficient disk space

**Issue: ChromaDB errors**
- Check directory permissions
- Verify path exists
- Check disk space

For more help, see [Troubleshooting](Troubleshooting.md).
