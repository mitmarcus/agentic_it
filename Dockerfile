# Stage 1: Builder
FROM python:3.13-slim AS builder

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies 
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary -r requirements.txt && \
    python -m spacy download en_core_web_sm && \
    playwright install --with-deps chromium

# Stage 2: Runtime - Minimal image
FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy installed packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache/ms-playwright /root/.cache/ms-playwright

RUN playwright install-deps chromium

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy application code
COPY utils/ ./utils/
COPY cremedelacreme/ ./cremedelacreme/
COPY langfuse_tracing/ ./langfuse_tracing/
COPY nodes.py .
COPY flows.py .
COPY models.py .
COPY main.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
