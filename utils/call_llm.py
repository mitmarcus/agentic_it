"""
Unified LLM API wrapper supporting OpenAI, Azure OpenAI, and Groq.
Switch providers using LLM_PROVIDER environment variable.
"""
import os
from typing import Optional

# Lazy import logger to avoid circular imports in standalone mode
_logger = None

def _get_logger():
    global _logger
    if _logger is None:
        from utils.logger import get_logger
        _logger = get_logger(__name__)
    return _logger

# Cached client instances
_OPENAI_CLIENT = None
_AZURE_CLIENT = None
_GROQ_CLIENT = None


def _get_openai_client():
    """Get or create cached OpenAI client."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def _get_azure_client():
    """Get or create cached Azure OpenAI client."""
    global _AZURE_CLIENT
    if _AZURE_CLIENT is None:
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
        
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable not set")
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable not set")
        
        _AZURE_CLIENT = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
    return _AZURE_CLIENT


def _get_groq_client():
    """Get or create cached Groq client."""
    global _GROQ_CLIENT
    if _GROQ_CLIENT is None:
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("groq package not installed. Install with: pip install groq")
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        _GROQ_CLIENT = Groq(api_key=api_key)
    return _GROQ_CLIENT


def call_llm(
    prompt: str, 
    max_tokens: int = 1024,
    temperature: Optional[float] = None,
    provider: Optional[str] = None
) -> str:
    """
    Call LLM API with the given prompt. Supports OpenAI, Azure OpenAI, and Groq.
    
    Args:
        prompt: Text prompt for the LLM
        max_tokens: Maximum tokens in response (default: 1024)
        temperature: Temperature for sampling (default: from LLM_TEMPERATURE env var or 0.2)
        provider: Override provider ("openai", "azure", or "groq"). 
                 If None, uses LLM_PROVIDER env var (default: "groq")
    
    Returns:
        Generated text response
        
    Raises:
        ValueError: If API key/config not set or response is empty
        Exception: If API call fails (propagates to Node for retry handling)
        
    Environment Variables:
        LLM_PROVIDER: "openai", "azure", or "groq" (default: "groq")
        LLM_TEMPERATURE: Default temperature (default: 0.2)
        
        For OpenAI:
            OPENAI_API_KEY: Your OpenAI API key
            OPENAI_MODEL: Model name (default: "gpt-4o")
            
        For Azure OpenAI:
            AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
            AZURE_OPENAI_ENDPOINT: Azure endpoint URL
            AZURE_OPENAI_DEPLOYMENT: Deployment/model name
            AZURE_OPENAI_API_VERSION: API version (default: "2024-02-15-preview")
            
        For Groq:
            GROQ_API_KEY: Your Groq API key
            GROQ_MODEL: Model name (default: "llama-3.3-70b-versatile")
    """
    # Determine provider
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "groq").lower()
    else:
        provider = provider.lower()
    
    # Get temperature
    if temperature is None:
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    
    _get_logger().info(f"Calling LLM with provider: {provider}")
    
    try:
        if provider == "openai":
            return _call_openai(prompt, max_tokens, temperature)
        elif provider == "azure":
            return _call_azure(prompt, max_tokens, temperature)
        elif provider == "groq":
            return _call_groq(prompt, max_tokens, temperature)
        else:
            raise ValueError(
                f"Unknown LLM provider: {provider}. "
                f"Must be 'openai', 'azure', or 'groq'"
            )
    except Exception as e:
        _get_logger().error(f"LLM call failed with provider {provider}: {e}")
        raise


def _call_openai(prompt: str, max_tokens: int, temperature: float) -> str:
    """Call OpenAI API."""
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    client = _get_openai_client()
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    result = response.choices[0].message.content
    if not result:
        raise ValueError("Empty response from OpenAI API")
    
    return result


def _call_azure(prompt: str, max_tokens: int, temperature: float) -> str:
    """Call Azure OpenAI API."""
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT environment variable not set")
    
    client = _get_azure_client()
    
    response = client.chat.completions.create(
        model=deployment,  # Azure uses deployment name
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    result = response.choices[0].message.content
    if not result:
        raise ValueError("Empty response from Azure OpenAI API")
    
    return result


def _call_groq(prompt: str, max_tokens: int, temperature: float) -> str:
    """Call Groq API."""
    from groq import RateLimitError
    
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    client = _get_groq_client()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        result = response.choices[0].message.content
        if not result:
            raise ValueError("Empty response from Groq API")
        
        return result
        
    except RateLimitError as e:
        _get_logger().error(f"Groq rate limit exceeded: {e}")
        raise


# Test standalone call_llm function
if __name__ == "__main__":
    # Handle imports for standalone execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from dotenv import load_dotenv
    load_dotenv()
    
    # Disable file logging for test to avoid permission issues
    os.environ["LOG_FILE_ENABLED"] = "false"

    test_prompt = "What is the capital of France? Answer in one word."
    
    # Test the configured provider
    provider = os.getenv("LLM_PROVIDER", "groq")
    print(f"\nTesting {provider.upper()} API...")
    
    try:
        response = call_llm(test_prompt)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Optionally test all providers if their keys are set
    print("\n" + "="*50)
    print("Testing all available providers:")
    print("="*50)
    
    for test_provider in ["azure", "groq"]:
        # Check if provider is configured
        if test_provider == "azure" and not os.getenv("AZURE_OPENAI_API_KEY"):
            print(f"\n{test_provider.upper()}: Skipped (AZURE_OPENAI_API_KEY not set)")
            continue
        elif test_provider == "groq" and not os.getenv("GROQ_API_KEY"):
            print(f"\n{test_provider.upper()}: Skipped (GROQ_API_KEY not set)")
            continue
        
        print(f"\n{test_provider.upper()}:")
        try:
            response = call_llm(test_prompt, provider=test_provider)
            print(f"  ✓ Response: {response}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
