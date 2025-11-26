"""
This is the chatbot FastAPI service.
"""
import os
import uuid
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List

# Import flows
from flows import get_flow
from utils.conversation_memory import conversation_memory

# Import models
from models import (
    QueryRequest,
    QueryResponse,
    IndexRequest,
    IndexResponse,
    FileUploadResponse,
    HealthResponse,
    SessionClearResponse,
    SessionHistoryResponse,
    SessionCleanupResponse,
    CollectionInfoResponse,
)

# Load environment variables 
# TODO: change this
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
log_file_enabled = os.getenv("LOG_FILE_ENABLED", "true").lower() == "true"
log_file_path = os.getenv("LOG_FILE_PATH", "./logs/app.log")

# Create logs directory if it doesn't exist
if log_file_enabled:
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

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


@app.get("/collection/info", response_model=CollectionInfoResponse)
async def get_collection_info(
    limit: int = 100,
    offset: int = 0,
    collection_name: str = os.getenv("CHROMADB_COLLECTION", "it_support_docs")
):
    """
    Get information about documents in the ChromaDB collection.
    
    Args:
        limit: Maximum number of documents to return (default: 100)
        offset: Number of documents to skip (default: 0)
        collection_name: Name of the collection (default: from env)
    
    Returns:
        Collection information with document list
        
    Raises:
        HTTPException: If collection access fails
    """
    try:
        from utils.chromadb_client import get_collection
        
        collection = get_collection(collection_name=collection_name)
        total_count = collection.count()
        
        # Get all documents to group by source file
        result = collection.get(
            include=["documents", "metadatas"]
        )
        
        # Group documents by source file
        grouped_docs = {}
        ids = result.get("ids", []) or []
        docs = result.get("documents", []) or []
        metadatas = result.get("metadatas", []) or []
        
        for i in range(len(ids)):
            doc_text = docs[i] if i < len(docs) and docs[i] else ""
            metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            
            source_file = metadata.get("source_file", "Unknown")
            filename = metadata.get("filename", "Unknown")
            
            if source_file not in grouped_docs:
                grouped_docs[source_file] = {
                    "filename": filename,
                    "source_file": source_file,
                    "extension": metadata.get("extension", metadata.get("file_type", "unknown")),
                    "chunks": []
                }
            
            grouped_docs[source_file]["chunks"].append({
                "id": ids[i],
                "content": doc_text,
                "chunk_index": metadata.get("chunk_index", 0)
            })
        
        # Format grouped documents
        documents = []
        for source_file, doc_info in grouped_docs.items():
            # Sort chunks by index
            doc_info["chunks"].sort(key=lambda x: x.get("chunk_index", 0))
            
            # Get first chunk for preview
            first_chunk = doc_info["chunks"][0]["content"] if doc_info["chunks"] else ""
            doc_preview = first_chunk[:200] + "..." if len(first_chunk) > 200 else first_chunk
            
            documents.append({
                "id": f"{source_file}_grouped",
                "content": doc_preview,
                "metadata": {
                    "filename": doc_info["filename"],
                    "source_file": source_file,
                    "extension": doc_info["extension"],
                    "chunk_count": len(doc_info["chunks"])
                },
                "chunks": doc_info["chunks"]
            })
        
        # Apply pagination
        paginated_docs = documents[offset:offset + limit]
        
        logger.info(f"Retrieved {len(documents)} unique documents (grouped from {total_count} chunks)")
        
        return {
            "collection_name": collection_name,
            "total_documents": len(documents),
            "documents": paginated_docs
        }
        
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")


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
        user_os = request.user_os or "unknown"
        
        logger.info(f"Processing query from session {session_id} {user_os}: {request.query[:100]}")
        
        # Add user message to conversation memory
        conversation_memory.add_message(session_id, "user", request.query)
        
        # Get conversation turn count
        history = conversation_memory.get_conversation_history(session_id)
        turn_count = len([m for m in history if m["role"] == "user"])
        
        # Prepare shared store
        shared = {
            "session_id": session_id,
            "user_id": user_id,
            "user_os": user_os,
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


@app.post("/upload", response_model=FileUploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and index multiple documents into the knowledge base.
    
    Supports .txt, .md, .html, .pdf files up to 100MB each.
    
    Args:
        files: List of files to upload and index
    
    Returns:
        Upload and indexing results
        
    Raises:
        HTTPException: If upload or indexing fails
    """
    import tempfile
    import shutil
    from pathlib import Path
    
    # Configuration
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {".txt", ".md", ".html", ".pdf"}
    
    uploaded_files = []
    failed_files = []
    file_details = []
    
    # Create temporary directory for uploaded files
    temp_dir = tempfile.mkdtemp(prefix="upload_")
    
    try:
        logger.info(f"Processing {len(files)} uploaded files")
        
        # Save uploaded files to temporary directory
        for file in files:
            try:
                # Validate filename exists
                if not file.filename:
                    failed_files.append({
                        "filename": "unknown",
                        "error": "Missing filename"
                    })
                    continue
                
                # Validate file extension
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in ALLOWED_EXTENSIONS:
                    failed_files.append({
                        "filename": file.filename,
                        "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
                    })
                    logger.warning(f"Rejected file {file.filename}: unsupported extension {file_ext}")
                    continue
                
                # Read file content
                content = await file.read()
                file_size = len(content)
                
                # Validate file size
                if file_size > MAX_FILE_SIZE:
                    failed_files.append({
                        "filename": file.filename,
                        "error": f"File too large ({file_size / 1024 / 1024:.1f}MB). Max: 100MB"
                    })
                    logger.warning(f"Rejected file {file.filename}: too large ({file_size} bytes)")
                    continue
                
                # Validate file is not empty
                if file_size == 0:
                    failed_files.append({
                        "filename": file.filename,
                        "error": "File is empty"
                    })
                    logger.warning(f"Rejected file {file.filename}: empty file")
                    continue
                
                # Save to temporary directory
                temp_filepath = Path(temp_dir) / file.filename
                with open(temp_filepath, "wb") as f:
                    f.write(content)
                
                uploaded_files.append({
                    "filename": file.filename,
                    "filepath": str(temp_filepath),
                    "size_bytes": file_size
                })
                
                logger.info(f"Saved uploaded file: {file.filename} ({file_size} bytes)")
                
            except Exception as e:
                failed_files.append({
                    "filename": file.filename,
                    "error": str(e)
                })
                logger.error(f"Error processing file {file.filename}: {e}")
        
        # If no files were successfully uploaded, return early
        if not uploaded_files:
            return {
                "status": "failed",
                "files_uploaded": 0,
                "files_failed": len(failed_files),
                "chunks_indexed": 0,
                "file_details": failed_files
            }
        
        # Index the uploaded files
        logger.info(f"Indexing {len(uploaded_files)} uploaded files from {temp_dir}")
        
        shared = {
            "source_dir": temp_dir
        }
        
        # Get and run the indexing flow
        flow = get_flow("indexing")
        flow.run(shared)
        
        # Extract indexing results
        documents = shared.get("documents", [])
        all_chunks = shared.get("all_chunks", [])
        chunks_indexed = shared.get("indexed_count", 0)
        
        # Build detailed results for each file
        for uploaded_file in uploaded_files:
            # Find matching document
            matching_doc = None
            for doc in documents:
                if isinstance(doc, dict) and "metadata" in doc:
                    doc_metadata = doc.get("metadata", {})
                    if isinstance(doc_metadata, dict) and doc_metadata.get("filename") == uploaded_file["filename"]:
                        matching_doc = doc
                        break
            
            if matching_doc:
                # Count chunks for this file
                chunks_for_file = 0
                chunk_metadata_list = shared.get("chunk_metadata", [])
                if isinstance(chunk_metadata_list, list):
                    for chunk_meta in chunk_metadata_list:
                        if isinstance(chunk_meta, dict):
                            source_file = chunk_meta.get("source_file", "")
                            if isinstance(source_file, str) and source_file.endswith(uploaded_file["filename"]):
                                chunks_for_file += 1
                
                file_details.append({
                    "filename": uploaded_file["filename"],
                    "size_bytes": uploaded_file["size_bytes"],
                    "chunks_created": chunks_for_file,
                    "status": "indexed"
                })
            else:
                file_details.append({
                    "filename": uploaded_file["filename"],
                    "size_bytes": uploaded_file["size_bytes"],
                    "chunks_created": 0,
                    "status": "failed",
                    "error": "Not found in indexing results"
                })
        
        # Add failed files to details
        for failed_file in failed_files:
            file_details.append({
                "filename": failed_file["filename"],
                "size_bytes": 0,
                "chunks_created": 0,
                "status": "rejected",
                "error": failed_file["error"]
            })
        
        logger.info(f"Upload completed: {len(uploaded_files)} files uploaded, {chunks_indexed} chunks indexed")
        
        return {
            "status": "success" if not failed_files else "partial_success",
            "files_uploaded": len(uploaded_files),
            "files_failed": len(failed_files),
            "chunks_indexed": chunks_indexed,
            "file_details": file_details
        }
        
    except Exception as e:
        logger.error(f"Error during upload/indexing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")
    
    finally:
        # Cleanup temporary directory
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {e}")


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
