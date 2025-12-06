"""
Centralized LLM configuration for consistent behavior across all nodes.

This module provides a single source of truth for all LLM-related parameters
including token limits, retry settings, and model configuration.
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """Centralized LLM configuration."""
    
    # Model settings (from env: GROQ_MODEL, LLM_TEMPERATURE)
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.3
    
    # Token limits (from env: LLM_MAX_TOKENS)
    max_tokens: int = 1024
    
    # Retry configuration (from env: LLM_MAX_RETRIES, LLM_RETRY_WAIT)
    max_retries: int = 3
    retry_wait: int = 2
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load configuration from environment variables with fallbacks."""
        return cls(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            retry_wait=int(os.getenv("LLM_RETRY_WAIT", "2")),
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
