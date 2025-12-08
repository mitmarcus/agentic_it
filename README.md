# IT Support Chatbot - Agentic RAG System

[![Testing CI](https://github.com/mitmarcus/agentic_it/actions/workflows/testing_ci.yaml/badge.svg)](https://github.com/mitmarcus/agentic_it/actions/workflows/testing_ci.yaml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-enabled-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

An intelligent IT support chatbot.
The system uses agentic decision-making and RAG (Retrieval-Augmented Generation) to answer IT questions, guide troubleshooting, and manage support tickets.

## 🚀 Features

- ✅ **Agentic Decision Making**: Intelligently routes queries to appropriate handlers
- ✅ **RAG Pipeline**: Retrieves relevant documentation before generating answers
- ✅ **Local Embeddings**: Privacy-first with sentence-transformers
- ✅ **Conversation Memory**: Maintains context across multi-turn conversations
- ✅ **Intent Classification**: Automatically categorizes user queries
- ✅ **Sensitive Data Redaction**: Automatically removes passwords, emails, API keys
- ✅ **Docker Compose**: Easy deployment with containerized services
- ✅ **Interactive Troubleshooting**: Guided step-by-step workflows
- ✅ **Observability & Tracing**: Full workflow observability with Langfuse integration
- ✅ **OS Detection**: Automatically detects user's OS and provides platform-specific instructions
- ✅ **Jira Integration**: Automated ticket creation with context and troubleshooting history
- ✅ **Network Status Monitoring**: Async status page scraping for incident detection (Playwright-based)

## 📋 Prerequisites

- Docker and Docker Compose
- API Keys:
  - Groq API key (free at https://console.groq.com)
  - (Optional) Langfuse API keys for tracing (free at https://cloud.langfuse.com)

## 🔧 Installation

### 1. Clone and Configure

```bash
git clone https://github.com/mitmarcus/agentic_it
cd agentic_it

# Copy and configure environment variables
cp .env.sample .env
# Edit .env and add your GROQ_API_KEY
# (Optional) Add LANGFUSE_* credentials for tracing
```

### 2. Start Services

```bash
# Build and start all services (ChromaDB + Chatbot + Frontend)
docker-compose up -d --build

# View logs
docker-compose logs -f chatbot

# Check health
curl http://localhost:8000/health
```

### 3. Index Documents

```bash
# Upload new documents (HTML, PDF, TXT, MD)
curl -X POST http://localhost:8000/upload \
  -F "files=@/path/to/document.html" \
  -F "files=@/path/to/guide.pdf"

# Check collection stats
curl http://localhost:8000/collection/info
```

## 📝 Usage

### API Endpoints

#### Query & Conversation

```bash
# Ask a question with context
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I set up VPN?",
    "session_id": "user123",
    "user_os": "Windows"
  }'

# Get conversation history
curl http://localhost:8000/session/user123/history

# Clear session
curl -X DELETE http://localhost:8000/session/user123

# Cleanup old sessions (>7 days)
curl -X POST http://localhost:8000/sessions/cleanup
```

#### Document Management

```bash
# Upload and auto-index documents
curl -X POST http://localhost:8000/upload \
  -F "files=@docs/vpn_guide.html" \
  -F "files=@docs/printer_setup.pdf"

# Manually trigger indexing
curl -X POST http://localhost:8000/index

# Get collection statistics
curl http://localhost:8000/collection/info

# Delete specific document
curl -X DELETE "http://localhost:8000/documents?doc_id=doc123"
```

### Local Development

```bash
# Create virtual environment
python3.13 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Install Playwright browsers (for status monitoring)
playwright install

# Run locally (without Docker)
uvicorn main:app --reload --port 8000
```

## 🎯 How It Works

### Query Flow

1. **Intent Classification**: Categorizes query (informative/troubleshooting)
2. **Query Embedding**: Generates vector representation using local model
3. **KB Search**: Retrieves top-k relevant documents from ChromaDB
4. **Decision Maker** (Agent): Decides next action:
   - `answer`: Generate response using retrieved docs
   - `clarify`: Ask for more details
   - `search_kb`: Search again with refined query
   - `troubleshoot`: Start guided workflow
   - `create_ticket`: Create Jira ticket with full context
   - `search_tickets`: Search for existing related tickets (not implemented)
5. **Generate Answer**: Uses Groq LLM with RAG context
6. **Format Response**: Prepares final response for user

### Indexing Flow

1. **Load Documents**: Read markdown/txt files from `data/docs/`
2. **Chunk Documents**: Split into ~500 char chunks with overlap
3. **Embed Chunks**: Generate embeddings locally (batch processing)
4. **Store in ChromaDB**: Index for fast semantic search

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_batch_node.py -v
pytest tests/test_async_flow.py -v

# Test individual utilities
python utils/call_llm.py          # Test LLM (OpenAI/Azure/Groq)
python utils/embedding_local.py
python utils/chromadb_client.py
python utils/intent_classifier.py
python utils/redactor.py
```

## 📊 Monitoring

View logs:

```bash
# Application logs
tail -f logs/app.log

# Docker logs
docker-compose logs -f chatbot
docker-compose logs -f chromadb
```

### 🔍 Workflow Tracing with Langfuse

The system includes comprehensive workflow observability using [Langfuse](https://langfuse.com/). This provides:

- **Flow-level tracing**: Start/end times, input/output data for entire workflows
- **Node-level observability**: Detailed tracking of each node's prep/exec/post phases
- **Error tracking**: Automatic capture of exceptions and stack traces
- **Performance metrics**: Execution times for each phase
- **Input/Output inspection**: View data flowing through each node

#### Setup Langfuse Tracing

1. **Get Langfuse credentials** (free tier available):

   - Sign up at https://cloud.langfuse.com
   - Or self-host: https://langfuse.com/docs/deployment/self-host

2. **Configure environment variables** in `.env`:

   ```env
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   TRACING_DEBUG=false
   ```

3. **Run your workflows** - tracing happens automatically!

4. **View traces** in your Langfuse dashboard to:
   - Debug flow execution
   - Optimize performance bottlenecks
   - Track errors and retries
   - Analyze data transformations

## CI/CD

This project uses GitHub Actions for continuous integration and deployment:

### Automated Checks

- **Test Suite**: All tests in `tests/` directory must pass
- **Docker Build**: Verifies Docker image builds successfully

## 🔒 Security

- ✅ **Local embeddings**: Document content never sent to external APIs
- ✅ **Sensitive data redaction**: Automatic removal of passwords, emails, API keys, tokens
- ✅ **Environment-based secrets**: API keys stored in `.env` (not in code)
- ✅ **CORS restrictions**: Configurable allowed origins
- ✅ **Input validation**: Pydantic models for all API requests
- ✅ **Dependency scanning**: Automated vulnerability checks in CI
- ✅ **Minimal attack surface**: CPU-only PyTorch, locked dependencies
