"""
Pydantic models for request/response validation.

This module contains all the data models used by the FastAPI endpoints.
"""
from typing import Dict, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Query Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for chat queries."""
    query: str = Field(..., description="User's question or request", min_length=1)
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation continuity")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    user_os: Optional[str] = Field(None, description="User's operating system (Windows, macOS, Linux, etc.)")

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
