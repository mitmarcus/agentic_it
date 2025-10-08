# IT Support Chatbot - Agentic RAG System

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
- 🚧 **Interactive Troubleshooting**: Guided step-by-step workflows
- 🚧 **Jira Integration**: Ticket creation and search
- 🚧 **Major Incident Detection**: Alerts about known outages

## 📋 Prerequisites

- Docker and Docker Compose
- API Keys:
  - Groq API key (free at https://console.groq.com)

## 🔧 Installation

### 1. Clone and Configure

```bash
git clone https://github.com/mitmarcus/agentic_it
cd agentic_it

# Copy and configure environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 2. Start Services

```bash
# Start ChromaDB and Chatbot services
docker-compose up -d

# View logs
docker-compose logs -f chatbot
```

### 3. Index Documents

```bash
# Place your IT documentation in data/docs/ (in the docker container)
# Then index them:
curl -X POST http://localhost:8000/index
```

## 📝 Usage

### API Endpoints

**Health Check**

```bash
curl http://localhost:8000/health
```

**Ask a Question**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I set up VPN?",
    "session_id": "user123"
  }'
```

**Index Documents**

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"source_dir": "./data/docs"}'
```

**Get Conversation History**

```bash
curl http://localhost:8000/session/user123/history
```

**Clear Session**

```bash
curl -X DELETE http://localhost:8000/session/user123
```

### Python Test Script

```bash
# Install dependencies locally for testing
pip install -r requirements.txt

# Run test script
python test_chatbot.py
```

## 🎯 How It Works

### Query Flow

1. **Intent Classification**: Categorizes query (factual/troubleshooting/navigation)
2. **Query Embedding**: Generates vector representation using local model
3. **KB Search**: Retrieves top-k relevant documents from ChromaDB
4. **Decision Maker** (Agent): Decides next action:
   - `answer`: Generate response using retrieved docs
   - `clarify`: Ask for more details
   - `search_kb`: Search again with refined query
   - `troubleshoot`: Start guided workflow (not yet implemented)
   - `create_ticket`: Create Jira ticket (not yet implemented)
5. **Generate Answer**: Uses Groq LLM with RAG context
6. **Format Response**: Prepares final response for user

### Indexing Flow

1. **Load Documents**: Read markdown/txt files from `data/docs/`
2. **Chunk Documents**: Split into ~500 char chunks with overlap
3. **Embed Chunks**: Generate embeddings locally (batch processing)
4. **Store in ChromaDB**: Index for fast semantic search

## 🧪 Testing

```bash
# Test individual utilities
python utils/call_llm_groq.py
python utils/embedding_local.py
python utils/chromadb_client.py

# Test full pipeline
python test_chatbot.py

# Run unit tests (future)
pytest tests/
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

## 🔒 Security

- ✅ Local embeddings (no document content sent to external APIs)
- ✅ Sensitive data redaction (passwords, emails, API keys)
- ✅ Environment-based secrets management
- ✅ CORS restrictions
- ⚠️ Add authentication/authorization for production use
