"""
Groq LLM API wrapper.
Picked Groq because it has a free api and is actually pretty fast.
"""
from groq import Groq, RateLimitError
import os
import logging
from typing import Iterator

logger = logging.getLogger(__name__)


def _get_groq_client() -> tuple[Groq, str]:
    """
    Get configured Groq client and model name.
    
    Returns:
        Tuple of (Groq client, model name)
        
    Raises:
        ValueError: If API key or model not set
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    model = os.getenv("GROQ_MODEL", "")
    if not model:
        raise ValueError("GROQ_MODEL environment variable not set")
    
    return Groq(api_key=api_key), model


def call_llm(prompt: str, max_tokens: int = 1024) -> str:
    """
    Call Groq LLM API with the given prompt.
    
    Args:
        prompt: Text prompt for the LLM
        max_tokens: Maximum tokens in response (default: 1024)
    
    Returns:
        Generated text response
        
    Raises:
        ValueError: If API key not set or response is empty
        RateLimitError: If rate limit exceeded (for Node retry handling)
        Exception: If API call fails (propagates to Node for retry handling)
    """
    client, model = _get_groq_client()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        
        result = response.choices[0].message.content
        if not result:
            raise ValueError("Empty response from Groq API")
        
        return result
        
    except RateLimitError as e:
        # Log rate limit error with details
        logger.error(f"Rate limit exceeded: {e}")
        # Re-raise for Node fallback handling
        raise


def stream_llm(prompt: str, max_tokens: int = 1024) -> Iterator[str]:
    """
    Stream response from Groq LLM API.
    
    Args:
        prompt: Text prompt for the LLM
        max_tokens: Maximum tokens in response (default: 1024)
    
    Yields:
        Text chunks as they're generated
        
    Raises:
        ValueError: If API key not set
        RateLimitError: If rate limit exceeded
        Exception: If API call fails
    """
    client, model = _get_groq_client()
    
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
            stream=True,  # Enable streaming
        )
        
        for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
        
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded: {e}")
        raise

# Test standalone call_llm function
if __name__ == "__main__":
    # Load environment variables from .env file if present; I don't like load_dotenv but it works
    # if you know better ways please submit a PR
    from dotenv import load_dotenv
    load_dotenv()

    test_prompt = "What is the capital of France? Answer in one word."
    
    print("Testing Groq API...")
    try:
        response = call_llm(test_prompt)
        print(f"Response: {response}")
    except Exception as e:
        print(f"\nError during standalone test: {e}")
