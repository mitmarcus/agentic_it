"""
Azure OpenAI API wrapper.
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

# Cached client instance
_AZURE_CLIENT = None


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
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
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


def call_llm(
    prompt: str, 
    max_tokens: int = 1024,
    temperature: Optional[float] = None
) -> str:
    """
    Call Azure OpenAI API with the given prompt.
    
    Args:
        prompt: Text prompt for the LLM
        max_tokens: Maximum tokens in response (default: 1024)
        temperature: Temperature for sampling (default: from LLM_TEMPERATURE env var or 0.2)
    
    Returns:
        Generated text response
        
    Raises:
        ValueError: If API key/config not set or response is empty
        Exception: If API call fails (propagates to Node for retry handling)
        
    Environment Variables:
        AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
        AZURE_OPENAI_ENDPOINT: Azure endpoint URL
        AZURE_OPENAI_DEPLOYMENT: Deployment/model name
        AZURE_OPENAI_API_VERSION: API version (default: "2024-12-01-preview")
        LLM_TEMPERATURE: Default temperature (default: 0.2)
    """
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT environment variable not set")
    
    # Get temperature
    if temperature is None:
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    
    _get_logger().info(f"Calling Azure OpenAI with deployment: {deployment}")
    
    try:
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
        
    except Exception as e:
        _get_logger().error(f"Azure OpenAI call failed: {e}")
        raise


# Test standalone call_llm function
if __name__ == "__main__":
    # Handle imports for standalone execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from dotenv import load_dotenv
    load_dotenv()

    test_prompt = "What is the capital of France? Answer in one word."
    
    print("\nTesting Azure OpenAI API...")
    
    try:
        response = call_llm(test_prompt)
        print(f"✓ Response: {response}")
    except Exception as e:
        print(f"✗ Error: {e}")
