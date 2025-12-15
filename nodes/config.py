import os
from typing import Dict

def ensure_response_dict(shared: Dict) -> None:
    """Ensure the response dictionary exists in shared store."""
    if "response" not in shared:
        shared["response"] = {}

def _get_float_env(name: str) -> float:
    """Read a required float from environment variables. Raises ValueError if missing or invalid."""
    raw_value = os.getenv(name)
    if raw_value is None:
        raise ValueError(f"Environment variable {name} must be set")
    try:
        return float(raw_value)
    except ValueError:
        raise ValueError(f"Environment variable {name}={raw_value} is not a valid float")


def _get_int_env(name: str) -> int:
    """Read a required int from environment variables. Raises ValueError if missing or invalid."""
    raw_value = os.getenv(name)
    if raw_value is None:
        raise ValueError(f"Environment variable {name} must be set")
    try:
        return int(raw_value)
    except ValueError:
        raise ValueError(f"Environment variable {name}={raw_value} is not a valid integer")

# Cache policy limits at module load
POLICY_LIMITS = {
    "clarify_confidence_threshold": _get_float_env("AGENT_CLARIFY_CONFIDENCE_THRESHOLD"),
    "doc_confidence_threshold": _get_float_env("AGENT_DOC_CONFIDENCE_THRESHOLD"),
    "rate_limit_answer_confidence": _get_float_env("AGENT_RATE_LIMIT_ANSWER_CONFIDENCE"),
    "system_error_confidence": _get_float_env("AGENT_SYSTEM_ERROR_CONFIDENCE"),
    "troubleshoot_escalate_failed_steps": _get_int_env("TROUBLESHOOT_ESCALATE_FAILED_STEPS"),
    "troubleshoot_fallback_failed_steps": _get_int_env("TROUBLESHOOT_FALLBACK_FAILED_STEPS"),
    "max_turns": _get_int_env("AGENT_MAX_TURNS"),
}

# Cache RAG configuration
_RAG_CONFIG = {
    "top_k": _get_int_env("RAG_TOP_K"),
    "min_score": _get_float_env("RAG_MIN_SCORE"),
    "max_context_tokens": int(os.getenv("RAG_MAX_CONTEXT_TOKENS", "2000")),
    "embedding_dim": int(os.getenv("EMBEDDING_DIM", "384")),
    "chunk_size": _get_int_env("INGESTION_CHUNK_SIZE"),
    "chunk_overlap": _get_int_env("INGESTION_CHUNK_OVERLAP"),
    "source_dir": os.getenv("INGESTION_SOURCE_DIR", "./data/docs"),
}

# Cache feature flags 
_FEATURE_FLAGS = {
    "rerank": os.getenv("RERANK_ENABLED", "true").lower() == "true",  # Keep: most impactful step
    "query_expansion": os.getenv("QUERY_EXPANSION_ENABLED", "false").lower() == "true",
    "hyde": os.getenv("HYDE_ENABLED", "false").lower() == "true",
}