"""
Centralized LLM configuration for consistent behavior across all nodes.

This module provides a single source of truth for all LLM-related parameters
including token limits and retry settings.
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """Centralized LLM configuration."""
    
    # Token limits (from env: LLM_MAX_TOKENS)
    max_tokens: int
    
    # Retry configuration (from env: LLM_MAX_RETRIES, LLM_RETRY_WAIT)
    max_retries: int
    retry_wait: int
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load configuration from environment variables."""
        max_tokens_str = os.getenv("LLM_MAX_TOKENS")
        max_retries_str = os.getenv("LLM_MAX_RETRIES")
        retry_wait_str = os.getenv("LLM_RETRY_WAIT")
        
        if not max_tokens_str:
            raise ValueError("LLM_MAX_TOKENS environment variable is required")
        if not max_retries_str:
            raise ValueError("LLM_MAX_RETRIES environment variable is required")
        if not retry_wait_str:
            raise ValueError("LLM_RETRY_WAIT environment variable is required")
        
        return cls(
            max_tokens=int(max_tokens_str),
            max_retries=int(max_retries_str),
            retry_wait=int(retry_wait_str),
        )


# Global singleton instance (lazy loaded)
_llm_config: Optional[LLMConfig] = None


def get_llm_config() -> LLMConfig:
    """
    Get global LLM configuration (lazy loaded, cached).
    
    Returns:
        LLMConfig: The global configuration instance
    """
    global _llm_config
    if _llm_config is None:
        _llm_config = LLMConfig.from_env()
    return _llm_config


def reset_llm_config():
    """Reset the global config (useful for testing)."""
    global _llm_config
    _llm_config = None
