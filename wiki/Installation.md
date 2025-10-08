# Installation Guide

This guide provides detailed instructions for installing the Agentic IT Support Chatbot in various environments.

## Table of Contents

- [System Requirements](#system-requirements)
- [Local Development Installation](#local-development-installation)
- [Docker Installation](#docker-installation)
- [Production Deployment](#production-deployment)
- [Post-Installation Steps](#post-installation-steps)

## System Requirements

### Minimum Requirements

- **OS**: Linux, macOS, or Windows 10/11
- **Python**: 3.9 or higher
- **RAM**: 4 GB minimum (8 GB recommended)
- **Disk Space**: 5 GB minimum (for models and database)
- **CPU**: 2 cores minimum (4 cores recommended)

### Software Dependencies

- Python 3.9+
- pip (Python package manager)
- Git
- Docker and Docker Compose (for containerized deployment)

## Local Development Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/mitmarcus/agentic_it.git
cd agentic_it
```

### Step 2: Create Virtual Environment (Recommended)

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n agentic_it python=3.9
conda activate agentic_it
```

### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Dependencies Overview

The project requires:
- **FastAPI**: Web framework for the API
- **uvicorn**: ASGI server
- **chromadb**: Vector database for document storage
- **sentence-transformers**: Local embedding generation
- **groq**: LLM API client
- **pydantic**: Data validation
- **python-dotenv**: Environment variable management

### Step 4: Configure Environment

```bash
cp .env.sample .env
```

Edit `.env` with your settings (see [Configuration](Configuration.md) for details).

### Step 5: Verify Installation

```bash
# Test that all imports work
python -c "from flows import create_query_flow; print('✓ Installation successful')"

# Run flow tests
python flows.py
```

## Docker Installation

Docker provides an isolated and reproducible environment.

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+

### Step 1: Clone Repository

```bash
git clone https://github.com/mitmarcus/agentic_it.git
cd agentic_it
```

### Step 2: Configure Environment

```bash
cp .env.sample .env
# Edit .env with your configuration
```

### Step 3: Build and Run

```bash
# Build the Docker image
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

### Step 4: Verify Docker Installation

```bash
# Check service status
docker-compose ps

# Test the API
curl http://localhost:8000/health
```

### Docker Services

The `docker-compose.yml` includes:
- **chatbot**: Main application service
- **chromadb**: Vector database
- Shared volumes for persistence

## Production Deployment

### Cloud Deployment Options

#### AWS Deployment

```bash
# Use AWS ECS or EKS
# Store secrets in AWS Secrets Manager
# Use RDS or DynamoDB for session storage
```

#### Azure Deployment

```bash
# Use Azure Container Instances or AKS
# Store secrets in Azure Key Vault
# Use Azure Cosmos DB for session storage
```

#### Google Cloud Deployment

```bash
# Use Google Cloud Run or GKE
# Store secrets in Secret Manager
# Use Cloud Firestore for session storage
```

### Production Checklist

- [ ] Set strong environment variables
- [ ] Configure HTTPS/TLS
- [ ] Set up monitoring and logging
- [ ] Configure backup for ChromaDB
- [ ] Set up load balancing (if needed)
- [ ] Configure rate limiting
- [ ] Set up health checks
- [ ] Review security settings
- [ ] Configure auto-scaling (if needed)
- [ ] Set up CI/CD pipeline

### Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Environment Variables**: Use secret management services
3. **Network**: Restrict network access appropriately
4. **HTTPS**: Always use TLS in production
5. **Authentication**: Implement proper authentication
6. **Data**: Encrypt sensitive data at rest

## Post-Installation Steps

### 1. Index Initial Documents

```bash
# Create document directory
mkdir -p data/docs

# Add your IT documentation files
cp /path/to/your/docs/* data/docs/

# Run indexing
python -c "
from flows import create_indexing_flow
flow = create_indexing_flow()
flow.run({'source_dir': './data/docs'})
"
```

### 2. Test the Installation

```bash
# Run the test chatbot
python test_chatbot.py

# Run unit tests
python -m pytest tests/ -v
```

### 3. Start the Service

**Development:**
```bash
python main.py
```

**Production:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Verify Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## Updating the Installation

### Update Code

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### Update Docker

```bash
docker-compose pull
docker-compose up -d --build
```

## Uninstalling

### Local Installation

```bash
# Remove virtual environment
rm -rf venv/

# Remove database
rm -rf chroma_db/

# Remove cloned repository
cd .. && rm -rf agentic_it/
```

### Docker Installation

```bash
# Stop and remove containers
docker-compose down -v

# Remove images
docker rmi agentic_it_chatbot
```

## Troubleshooting Installation

### Common Issues

**Issue: Module not found errors**
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Issue: Permission denied**
```bash
# Solution: Check file permissions
chmod +x test_docker.sh
```

**Issue: Port already in use**
```bash
# Solution: Change port in .env or docker-compose.yml
# Or stop the service using the port
lsof -ti:8000 | xargs kill -9
```

For more troubleshooting help, see [Troubleshooting](Troubleshooting.md).
