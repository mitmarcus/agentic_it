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


def _get_bool_env(name: str, default: bool = False) -> bool:
    """Read a boolean from environment variables. Accepts: true/false, 1/0, yes/no, on/off (case-insensitive)."""
    raw_value = os.getenv(name)
    if raw_value is None:
        raise ValueError(f"Environment variable {name} must be set")
    
    raw_lower = raw_value.lower().strip()
    if raw_lower in ("true", "1", "yes", "on"):
        return True
    elif raw_lower in ("false", "0", "no", "off"):
        return False
    else:
        raise ValueError(f"Environment variable {name}={raw_value} is not a valid boolean (use: true/false, 1/0, yes/no, on/off)")


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
    "max_context_tokens": _get_int_env("RAG_MAX_CONTEXT_TOKENS"),
    "embedding_dim": _get_int_env("EMBEDDING_DIM"),
    "chunk_size": _get_int_env("INGESTION_CHUNK_SIZE"),
    "chunk_overlap": _get_int_env("INGESTION_CHUNK_OVERLAP"),
    "source_dir": os.getenv("INGESTION_SOURCE_DIR", "./data/docs"),
}

# Cache feature flags 
_FEATURE_FLAGS = {
    "rerank": _get_bool_env("RERANK_ENABLED"), 
    "query_expansion": _get_bool_env("QUERY_EXPANSION_ENABLED", default=False),
    "hyde": _get_bool_env("HYDE_ENABLED"),
}