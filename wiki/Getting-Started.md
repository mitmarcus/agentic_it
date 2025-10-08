# Getting Started

This guide will help you get the Agentic IT Support Chatbot up and running in minutes.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.9+**
- **Docker** (optional, for containerized deployment)
- **Git**

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/mitmarcus/agentic_it.git
cd agentic_it
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Copy the sample environment file and configure it:

```bash
cp .env.sample .env
```

Edit `.env` and set your configuration:

```bash
# LLM Configuration
GROQ_API_KEY=your_groq_api_key_here

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_db

# Document Ingestion
INGESTION_SOURCE_DIR=./data/docs
```

### 4. Prepare Knowledge Base Documents

Create a directory for your IT documentation:

```bash
mkdir -p data/docs
```

Add your IT documentation files (`.txt`, `.md`) to this directory. Examples:
- VPN setup guides
- Password reset procedures
- Printer troubleshooting steps
- Software installation guides

### 5. Index Documents (Offline)

Run the indexing flow to process and embed your documents:

```bash
python -c "from flows import create_indexing_flow; flow = create_indexing_flow(); flow.run({'source_dir': './data/docs'})"
```

This will:
- Load documents from the specified directory
- Chunk them into smaller pieces
- Generate embeddings using a local model
- Store everything in ChromaDB

### 6. Start the Chatbot

Run the FastAPI server:

```bash
python main.py
```

The server will start on `http://localhost:8000`.

### 7. Test the Chatbot

You can test the chatbot using:

**Option A: Interactive CLI Test**

```bash
python test_chatbot.py
```

**Option B: API Request**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I set up VPN?",
    "session_id": "test-session-123"
  }'
```

**Option C: Interactive Docs**

Visit `http://localhost:8000/docs` for the FastAPI interactive documentation.

## Docker Quick Start (Alternative)

If you prefer using Docker:

```bash
# Build and run with Docker Compose
docker-compose up --build
```

This will:
- Build the Docker image
- Start the chatbot service
- Start ChromaDB
- Expose the API on port 8000

## What's Next?

- **[Configuration](Configuration.md)** - Learn about advanced configuration options
- **[Architecture](Architecture.md)** - Understand how the system works
- **[API Documentation](API-Documentation.md)** - Explore the API endpoints
- **[Troubleshooting](Troubleshooting.md)** - Fix common issues

## Example Queries to Try

Once your chatbot is running, try these example queries:

1. **Factual Query**: "What is our VPN connection procedure?"
2. **Troubleshooting**: "My printer won't print"
3. **Navigation**: "Where can I find the password reset portal?"
4. **Clarification**: "I need help" (the bot will ask clarifying questions)

## Verifying Your Setup

To verify everything is working correctly:

```bash
# Test flow creation
python flows.py

# Run basic tests
python -m pytest tests/ -v
```

You should see success messages indicating all components are properly configured.

## Getting Help

If you run into issues:

- Check the [Troubleshooting](Troubleshooting.md) guide
- Review the logs in your terminal
- Ensure all environment variables are set correctly
- Verify your documents are in the correct format
