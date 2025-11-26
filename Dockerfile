# Backend Dockerfile
# seems to work better than FROM python:slim
FROM python:3.13-slim 

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY utils/ ./utils/
COPY cremedelacreme/ ./cremedelacreme/
COPY langfuse_tracing/ ./langfuse_tracing/
COPY nodes.py .
COPY flows.py .
COPY models.py .
COPY main.py .

# Create directories
RUN mkdir -p logs data/docs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
