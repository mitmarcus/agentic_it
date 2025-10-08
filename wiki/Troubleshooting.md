# Troubleshooting Guide

Common issues and their solutions for the Agentic IT Support Chatbot.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Issues](#configuration-issues)
- [Runtime Issues](#runtime-issues)
- [Performance Issues](#performance-issues)
- [API Issues](#api-issues)
- [Database Issues](#database-issues)
- [Debugging Tips](#debugging-tips)

## Installation Issues

### Issue: Module not found errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'fastapi'
```

**Solutions:**

1. **Reinstall dependencies:**
```bash
pip install -r requirements.txt --force-reinstall
```

2. **Check virtual environment:**
```bash
which python  # Should point to venv
pip list      # Verify installed packages
```

3. **Upgrade pip:**
```bash
pip install --upgrade pip
```

### Issue: Python version incompatibility

**Symptoms:**
```
SyntaxError: invalid syntax
```

**Solutions:**

1. **Check Python version:**
```bash
python --version  # Should be 3.9+
```

2. **Use correct Python version:**
```bash
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Permission denied during installation

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**

1. **Don't use sudo with pip:**
```bash
# Bad
sudo pip install -r requirements.txt

# Good
pip install --user -r requirements.txt
# Or use virtual environment
```

2. **Fix permissions:**
```bash
chmod -R 755 ./agentic_it
```

## Configuration Issues

### Issue: API key not working

**Symptoms:**
```
Error: Invalid API key
```

**Solutions:**

1. **Verify .env file:**
```bash
cat .env | grep GROQ_API_KEY
# Should show: GROQ_API_KEY=gsk_...
```

2. **Check for extra spaces:**
```bash
# Bad (trailing space)
GROQ_API_KEY=gsk_abc123 

# Good
GROQ_API_KEY=gsk_abc123
```

3. **Regenerate API key:**
- Visit [console.groq.com](https://console.groq.com)
- Create new API key
- Update `.env` file

### Issue: Environment variables not loaded

**Symptoms:**
```
ValueError: GROQ_API_KEY not set
```

**Solutions:**

1. **Verify .env exists:**
```bash
ls -la .env
```

2. **Load explicitly:**
```python
from dotenv import load_dotenv
load_dotenv()  # Add this at the top of main.py
```

3. **Check file permissions:**
```bash
chmod 644 .env
```

### Issue: Wrong embedding dimension

**Symptoms:**
```
ValueError: Embedding dimension mismatch
```

**Solutions:**

1. **Match model and dimension:**
```bash
# For all-MiniLM-L6-v2
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384

# For all-mpnet-base-v2
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DIM=768
```

2. **Clear and re-index:**
```bash
rm -rf chroma_db/
# Re-run indexing flow
```

## Runtime Issues

### Issue: Server won't start

**Symptoms:**
```
Error: Address already in use
```

**Solutions:**

1. **Check port usage:**
```bash
lsof -ti:8000
```

2. **Kill existing process:**
```bash
kill -9 $(lsof -ti:8000)
```

3. **Use different port:**
```bash
# In .env
API_PORT=8001

# Or
uvicorn main:app --port 8001
```

### Issue: Out of memory errors

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Reduce batch size:**
```bash
# In .env
INGESTION_BATCH_SIZE=5  # Down from 10
```

2. **Process fewer documents:**
```bash
# Split documents into smaller batches
python -c "
from flows import create_indexing_flow
flow = create_indexing_flow()
flow.run({'source_dir': './data/docs/batch1'})
"
```

3. **Use lighter embedding model:**
```bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Issue: Slow response times

**Symptoms:**
- Queries taking > 10 seconds
- Timeouts

**Solutions:**

1. **Check ChromaDB size:**
```bash
du -sh chroma_db/
# If > 1GB, consider pruning
```

2. **Reduce retrieval count:**
```bash
# In .env
CHROMA_TOP_K=3  # Down from 5
```

3. **Enable caching:**
```python
# Add caching to embeddings
@lru_cache(maxsize=1000)
def embed_query(query: str):
    # ...
```

### Issue: LLM API errors

**Symptoms:**
```
Error: Rate limit exceeded
Error: Service unavailable
```

**Solutions:**

1. **Check API status:**
```bash
curl https://api.groq.com/health
```

2. **Implement retry logic:**
```python
# Already implemented in utils/call_llm_groq.py
# Increase retries if needed
```

3. **Reduce request frequency:**
```bash
# In .env
RATE_LIMIT_PER_MINUTE=30  # Down from 60
```

## Performance Issues

### Issue: Slow document indexing

**Symptoms:**
- Indexing taking > 1 minute for 10 documents

**Solutions:**

1. **Increase batch size:**
```bash
INGESTION_BATCH_SIZE=20  # Up from 10
```

2. **Use faster embedding model:**
```bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

3. **Optimize chunk size:**
```bash
CHUNK_SIZE=800  # Smaller chunks = more chunks but faster processing
```

### Issue: High memory usage

**Symptoms:**
- System running out of RAM

**Solutions:**

1. **Process documents in batches:**
```bash
# Split into smaller directories
./data/docs/batch1
./data/docs/batch2
```

2. **Clear conversation history:**
```bash
curl -X DELETE http://localhost:8000/sessions/{session_id}
```

3. **Restart service periodically:**
```bash
# Add to cron
0 */6 * * * systemctl restart agentic-it
```

## API Issues

### Issue: 500 Internal Server Error

**Symptoms:**
```json
{
  "error": "InternalServerError",
  "message": "An error occurred"
}
```

**Solutions:**

1. **Check logs:**
```bash
# Look for stack traces
tail -f logs/app.log
```

2. **Enable debug mode:**
```bash
# In .env
DEBUG=true
LOG_LEVEL=DEBUG
```

3. **Test components:**
```bash
python test_chatbot.py
```

### Issue: 404 Not Found

**Symptoms:**
```
404: Not Found
```

**Solutions:**

1. **Verify endpoint:**
```bash
# Correct endpoint
POST /chat

# Wrong endpoint
POST /api/chat  # No /api prefix
```

2. **Check API documentation:**
```
http://localhost:8000/docs
```

### Issue: CORS errors (Browser)

**Symptoms:**
```
Access to fetch blocked by CORS policy
```

**Solutions:**

1. **Add origin to .env:**
```bash
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

2. **Wildcard (development only):**
```bash
CORS_ORIGINS=*
```

## Database Issues

### Issue: ChromaDB connection errors

**Symptoms:**
```
ConnectionError: Could not connect to ChromaDB
```

**Solutions:**

1. **Check directory exists:**
```bash
ls -la chroma_db/
```

2. **Create directory:**
```bash
mkdir -p chroma_db
chmod 755 chroma_db
```

3. **Reset database:**
```bash
rm -rf chroma_db/
# Re-index documents
```

### Issue: No documents found

**Symptoms:**
```json
{
  "results": [],
  "total": 0
}
```

**Solutions:**

1. **Verify documents indexed:**
```python
from utils.chromadb_client import get_chromadb_client
client = get_chromadb_client()
collection = client.get_collection("it_support_docs")
print(f"Total documents: {collection.count()}")
```

2. **Re-index documents:**
```bash
python -c "
from flows import create_indexing_flow
flow = create_indexing_flow()
flow.run({'source_dir': './data/docs'})
"
```

3. **Check file formats:**
```bash
# Only .txt and .md are supported by default
ls data/docs/*.{txt,md}
```

### Issue: Embedding dimension mismatch

**Symptoms:**
```
ValueError: Embedding has wrong dimension
```

**Solutions:**

1. **Clear database:**
```bash
rm -rf chroma_db/
```

2. **Verify configuration:**
```bash
grep EMBEDDING .env
```

3. **Re-index with correct settings:**
```bash
# Ensure EMBEDDING_DIM matches model
python -c "from flows import create_indexing_flow; ..."
```

## Debugging Tips

### Enable Debug Logging

```bash
# In .env
DEBUG=true
LOG_LEVEL=DEBUG
```

### View Logs

```bash
# In Python
import logging
logging.basicConfig(level=logging.DEBUG)

# In Docker
docker-compose logs -f chatbot
```

### Test Individual Components

```bash
# Test LLM
python -c "
from utils.call_llm_groq import call_llm
response = call_llm('Hello', [])
print(response)
"

# Test embeddings
python -c "
from utils.embedding_local import get_embedding
embedding = get_embedding('test query')
print(f'Dimension: {len(embedding)}')
"

# Test ChromaDB
python -c "
from utils.chromadb_client import get_chromadb_client
client = get_chromadb_client()
print('ChromaDB connected')
"
```

### Validate Configuration

```bash
python -c "
from dotenv import load_dotenv
import os

load_dotenv()

required_vars = ['GROQ_API_KEY', 'EMBEDDING_MODEL', 'CHROMA_PERSIST_DIR']
for var in required_vars:
    value = os.getenv(var)
    status = '✓' if value else '✗'
    print(f'{status} {var}: {value or \"NOT SET\"}')"
```

### Check Flow Execution

```bash
# Test flow creation
python flows.py

# Test query flow
python -c "
from flows import create_query_flow
flow = create_query_flow()
result = flow.run({
    'user_query': 'test',
    'session_id': 'debug'
})
print(result)
"
```

### Profile Performance

```python
import time
import cProfile

def profile_query():
    from flows import create_query_flow
    flow = create_query_flow()
    
    start = time.time()
    result = flow.run({
        'user_query': 'How do I reset my password?',
        'session_id': 'profile-test'
    })
    duration = time.time() - start
    
    print(f"Query took {duration:.2f} seconds")
    return result

cProfile.run('profile_query()')
```

### Network Debugging

```bash
# Test Groq API connectivity
curl https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY"

# Test local API
curl -v http://localhost:8000/health

# Check DNS
nslookup api.groq.com
```

## Getting Additional Help

If issues persist:

1. **Check GitHub Issues:**
   - Search existing issues
   - Open new issue with details

2. **Provide Debug Information:**
   - Python version: `python --version`
   - Package versions: `pip list`
   - OS information: `uname -a`
   - Error logs and stack traces
   - Configuration (redact sensitive data)

3. **Enable Verbose Logging:**
   ```bash
   LOG_LEVEL=DEBUG python main.py > debug.log 2>&1
   ```

4. **Review Documentation:**
   - [Getting Started](Getting-Started.md)
   - [Configuration](Configuration.md)
   - [Architecture](Architecture.md)

## Common Error Messages Reference

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Missing dependency | `pip install -r requirements.txt` |
| `FileNotFoundError` | Missing file/directory | Check paths in `.env` |
| `ConnectionError` | Service unreachable | Check network/service status |
| `ValueError` | Invalid configuration | Verify `.env` settings |
| `MemoryError` | Insufficient RAM | Reduce batch sizes |
| `TimeoutError` | Request timeout | Increase timeout settings |
| `AuthenticationError` | Invalid API key | Check API key in `.env` |
| `RateLimitError` | Too many requests | Wait or increase limits |
