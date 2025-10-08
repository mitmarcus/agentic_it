"""
This is the chatbot FastAPI service.
"""
import os
import uuid
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import flows
from flows import get_flow
from utils.conversation_memory import conversation_memory

# Import models
from models import (
    QueryRequest,
    QueryResponse,
    IndexRequest,
    IndexResponse,
    HealthResponse,
    SessionClearResponse,
    SessionHistoryResponse,
    SessionCleanupResponse,
)

# Load environment variables 
# TODO: change this
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
log_file_enabled = os.getenv("LOG_FILE_ENABLED", "true").lower() == "true"
log_file_path = os.getenv("LOG_FILE_PATH", "./logs/app.log")

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path) if log_file_enabled else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)


# ============================================================================
# Lifespan Context Manager
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=" * 80)
    logger.info("IT Support Chatbot API Starting")
    logger.info("=" * 80)
    logger.info(f"Environment: {os.getenv('ENV', 'development')}")
    logger.info(f"ChromaDB Mode: {os.getenv('CHROMADB_MODE', 'client')}")
    logger.info(f"Embedding Model: {os.getenv('EMBED_MODEL', 'unknown')}")
    logger.info("=" * 80)
    
    # Test ChromaDB connection
    try:
        from utils.chromadb_client import initialize_client, get_collection_stats
        initialize_client()
        stats = get_collection_stats()
        logger.info(f"ChromaDB connected: {stats['count']} documents indexed")
    except Exception as e:
        logger.warning(f"ChromaDB connection check failed: {e}")
        logger.warning("Indexing may be required. Use POST /index to index documents.")
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("IT Support Chatbot API Shutting Down")


# Create FastAPI app
app = FastAPI(
    title="IT Support Chatbot API",
    description="Agentic RAG-based IT support chatbot",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
cors_origins = [
    os.getenv("CORS_ORIGIN_1", "http://localhost:3000"),
    os.getenv("CORS_ORIGIN_2", "http://localhost:8080"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query through the chatbot.
    
    Args:
        request: Query request with user question
    
    Returns:
        Chatbot response
        
    Raises:
        HTTPException: If query processing fails
    """
    try:
        # Generate or use provided session ID
        session_id = request.session_id or str(uuid.uuid4())
        user_id = request.user_id or "anonymous"
        
        logger.info(f"Processing query from session {session_id}: {request.query[:100]}")
        
        # Add user message to conversation memory
        conversation_memory.add_message(session_id, "user", request.query)
        
        # Get conversation turn count
        history = conversation_memory.get_conversation_history(session_id)
        turn_count = len([m for m in history if m["role"] == "user"])
        
        # Prepare shared store
        shared = {
            "session_id": session_id,
            "user_id": user_id,
            "user_query": request.query,
            "timestamp": datetime.now().isoformat(),
            "turn_count": turn_count,
            "response": {}
        }
        
        # Get and run the query flow
        flow = get_flow("query")
        flow.run(shared)
        
        # Extract response
        response_data = shared.get("response", {})
        response_text = response_data.get("text", "I'm sorry, I couldn't process your request.")
        action_taken = response_data.get("action_taken", "unknown")
        requires_followup = response_data.get("requires_followup", False)
        
        # Prepare metadata
        metadata = {
            "intent": shared.get("intent", {}),
            "retrieved_docs_count": len(shared.get("retrieved_docs", [])),
            "decision": shared.get("decision", {}),
            "turn_count": turn_count
        }
        
        conversation_memory.add_message(session_id, "assistant", response_text)
        
        logger.info(f"Query processed successfully. Action: {action_taken}")
        
        return {
            "response": response_text,
            "session_id": session_id,
            "action_taken": action_taken,
            "requires_followup": requires_followup,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest):
    """
    Index documents into the knowledge base.
    
    Args:
        request: Index request with optional source directory
    
    Returns:
        Indexing results
        
    Raises:
        HTTPException: If indexing fails
    """
    try:
        source_dir = request.source_dir or os.getenv("INGESTION_SOURCE_DIR", "./data/docs")
        
        logger.info(f"Starting document indexing from {source_dir}")
        
        # Prepare shared store
        shared = {
            "source_dir": source_dir
        }
        
        # Get and run the indexing flow
        flow = get_flow("indexing")
        flow.run(shared)
        
        # Extract results
        documents_loaded = len(shared.get("documents", []))
        chunks_created = len(shared.get("all_chunks", []))
        chunks_indexed = shared.get("indexed_count", 0)
        
        logger.info(f"Indexing completed: {documents_loaded} docs, {chunks_indexed} chunks indexed")
        
        return {
            "status": "success",
            "documents_loaded": documents_loaded,
            "chunks_created": chunks_created,
            "chunks_indexed": chunks_indexed
        }
        
    except Exception as e:
        logger.error(f"Error indexing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error indexing documents: {str(e)}")


@app.delete("/session/{session_id}", response_model=SessionClearResponse)
async def clear_session(session_id: str):
    """
    Clear a conversation session.
    
    Args:
        session_id: Session ID to clear
    
    Returns:
        Success message
    """
    try:
        conversation_memory.clear_session(session_id)
        logger.info(f"Session {session_id} cleared")
        
        return {"status": "success", "message": f"Session {session_id} cleared"}
        
    except Exception as e:
        logger.error(f"Error clearing session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")


@app.get("/session/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str, limit: int = 10):
    """
    Get conversation history for a session.
    
    Args:
        session_id: Session ID
        limit: Maximum number of messages to return
    
    Returns:
        Conversation history
    """
    try:
        history = conversation_memory.get_conversation_history(session_id, limit=limit)
        
        return {
            "session_id": session_id,
            "message_count": len(history),
            "messages": history
        }
        
    except Exception as e:
        logger.error(f"Error getting session history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting session history: {str(e)}")


@app.post("/maintenance/cleanup-sessions", response_model=SessionCleanupResponse)
async def cleanup_old_sessions(max_age_hours: int = 24):
    """
    Clean up old conversation sessions.
    
    Args:
        max_age_hours: Maximum age of sessions to keep
    
    Returns:
        Cleanup results
    """
    try:
        removed_count = conversation_memory.cleanup_old_sessions(max_age_hours)
        logger.info(f"Cleaned up {removed_count} old sessions")
        
        return {
            "status": "success",
            "sessions_removed": removed_count,
            "max_age_hours": max_age_hours
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error cleaning up sessions: {str(e)}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8080"))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload in development
        log_level=log_level.lower()
    )
