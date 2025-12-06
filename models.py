"""
Pydantic models for request/response validation.

This module contains all the data models used by the FastAPI endpoints.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Query Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for chat queries."""
    query: str = Field(..., description="User's question or request", min_length=1, max_length=10000)
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation continuity", max_length=100)
    user_id: Optional[str] = Field(None, description="Optional user identifier", max_length=100)
    user_os: Optional[str] = Field(None, description="User's operating system (Windows, macOS, Linux, etc.)", max_length=100)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "How do I connect to the VPN?",
                    "session_id": "abc123",
                    "user_id": "john.doe",
                    "user_os": "Windows 10"
                }
            ]
        }
    }


class QueryResponse(BaseModel):
    """Response model for chat queries."""
    response: str = Field(..., description="Chatbot's response to the user")
    session_id: str = Field(..., description="Session ID for this conversation")
    action_taken: str = Field(..., description="Action taken by the agent (answer, clarify, search_kb, etc.)")
    requires_followup: bool = Field(..., description="Whether the response requires user followup")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata about the response")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "response": "To connect to the VPN, follow these steps...",
                    "session_id": "abc123",
                    "action_taken": "answer",
                    "requires_followup": False,
                    "metadata": {
                        "intent": {"category": "technical_support"},
                        "retrieved_docs_count": 3,
                        "turn_count": 1
                    }
                }
            ]
        }
    }


# ============================================================================
# Indexing Models
# ============================================================================

class IndexRequest(BaseModel):
    """Request model for indexing documents."""
    source_dir: Optional[str] = Field(None, description="Source directory containing documents to index")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source_dir": "./data/docs"
                }
            ]
        }
    }


class IndexResponse(BaseModel):
    """Response model for indexing operation."""
    status: str = Field(..., description="Status of the indexing operation")
    documents_loaded: int = Field(..., description="Number of documents loaded")
    chunks_created: int = Field(..., description="Number of text chunks created")
    chunks_indexed: int = Field(..., description="Number of chunks successfully indexed")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "success",
                    "documents_loaded": 10,
                    "chunks_created": 150,
                    "chunks_indexed": 150
                }
            ]
        }
    }


# ============================================================================
# Session Models
# ============================================================================

class SessionClearResponse(BaseModel):
    """Response model for session clearing."""
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Confirmation message")


class SessionHistoryResponse(BaseModel):
    """Response model for session history."""
    session_id: str = Field(..., description="Session identifier")
    message_count: int = Field(..., description="Number of messages in history")
    messages: list = Field(..., description="List of conversation messages")


class SessionCleanupResponse(BaseModel):
    """Response model for session cleanup operation."""
    status: str = Field(..., description="Operation status")
    sessions_removed: int = Field(..., description="Number of sessions removed")
    max_age_hours: int = Field(..., description="Maximum age threshold used for cleanup")


# ============================================================================
# File Upload Models
# ============================================================================

class FileUploadResponse(BaseModel):
    """Response model for file upload and indexing."""
    status: str = Field(..., description="Upload operation status")
    files_uploaded: int = Field(..., description="Number of files successfully uploaded")
    files_failed: int = Field(..., description="Number of files that failed to upload")
    chunks_indexed: int = Field(..., description="Total chunks indexed from uploaded files")
    file_details: list = Field(default_factory=list, description="Details about each uploaded file")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "success",
                    "files_uploaded": 3,
                    "files_failed": 0,
                    "chunks_indexed": 45,
                    "file_details": [
                        {
                            "filename": "guide.txt",
                            "size_bytes": 1024,
                            "chunks_created": 15,
                            "status": "indexed"
                        }
                    ]
                }
            ]
        }
    }


# ============================================================================
# Collection Info Model
# ============================================================================

class CollectionInfoResponse(BaseModel):
    """Response model for collection information."""
    collection_name: str = Field(..., description="Name of the collection")
    total_documents: int = Field(..., description="Total number of documents in collection")
    documents: list = Field(default_factory=list, description="List of documents with metadata")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "collection_name": "it_support_docs",
                    "total_documents": 150,
                    "documents": [
                        {
                            "id": "doc123",
                            "content": "VPN setup guide...",
                            "metadata": {"source_file": "vpn.txt", "filename": "vpn.txt"}
                        }
                    ]
                }
            ]
        }
    }


# ============================================================================
# Document Deletion Model
# ============================================================================

class DeleteDocumentResponse(BaseModel):
    """Response model for document deletion."""
    status: str = Field(..., description="Deletion operation status")
    chunks_deleted: int = Field(..., description="Number of chunks deleted")
    source_file: str = Field(..., description="Source file that was deleted")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "success",
                    "chunks_deleted": 15,
                    "source_file": "./data/docs/guide.txt"
                }
            ]
        }
    }


# ============================================================================
# Health Check Model
# ============================================================================

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service health status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "timestamp": "2025-10-07T12:00:00",
                    "version": "1.0.0"
                }
            ]
        }
    }


# ============================================================================
# Feedback Models
# ============================================================================

class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""
    session_id: str = Field(..., description="Session ID from the original query")
    query: str = Field(..., description="Original user query")
    response: str = Field(..., description="Response that user is providing feedback on")
    feedback_type: str = Field(..., description="Type of feedback: positive, negative, or neutral")
    feedback_score: int = Field(0, description="Numeric score (-1 to 1, or 1-5 scale)", ge=-1, le=5)
    feedback_comment: Optional[str] = Field(None, description="Optional user comment")
    retrieved_doc_ids: Optional[List[str]] = Field(None, description="Document IDs used to generate this response")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "abc123",
                    "query": "How do I reset my VPN password?",
                    "response": "To reset your VPN password...",
                    "feedback_type": "positive",
                    "feedback_score": 1,
                    "feedback_comment": "This was very helpful!"
                }
            ]
        }
    }


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    feedback_id: str = Field(..., description="Unique ID for this feedback entry")
    message: str = Field(..., description="Confirmation message")
    timestamp: str = Field(..., description="Timestamp of feedback submission")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "feedback_id": "fb_abc123def",
                    "message": "Thank you for your feedback!",
                    "timestamp": "2025-10-07T12:00:00"
                }
            ]
        }
    }


class FeedbackStatsResponse(BaseModel):
    """Response model for feedback statistics."""
    total: int = Field(..., description="Total number of feedback entries")
    positive: int = Field(..., description="Number of positive feedback entries")
    negative: int = Field(..., description="Number of negative feedback entries")
    neutral: int = Field(..., description="Number of neutral feedback entries")
    positive_rate: float = Field(..., description="Percentage of positive feedback")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "total": 100,
                    "positive": 75,
                    "negative": 15,
                    "neutral": 10,
                    "positive_rate": 0.75
                }
            ]
        }
    }
