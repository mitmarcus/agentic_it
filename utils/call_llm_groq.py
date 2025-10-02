"""
Groq LLM API wrapper.
Picked Groq because it has a free api and is actually pretty fast.
"""
from groq import Groq
import os


def call_llm(prompt: str) -> str:
    """
    Call Groq LLM API with the given prompt.
    
    Args:
        prompt: Text prompt for the LLM
    
    Returns:
        Generated text response
        
    Raises:
        ValueError: If API key not set or response is empty
        Exception: If API call fails (propagates to Node for retry handling)
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    model = os.getenv("GROQ_MODEL", "")
    if not model:
        raise ValueError("GROQ_MODEL environment variable not set")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    
    result = response.choices[0].message.content
    if not result:
        raise ValueError("Empty response from Groq API")
    
    return result

# Test standalone execution
if __name__ == "__main__":
    # Load environment variables from .env file if present; I don't like load_dotenv but it works
    # if you know better ways please submit a PR
    from dotenv import load_dotenv
    load_dotenv()

    test_prompt = "What is the capital of France? Answer in one word."
    
    print("Testing Groq API...")
    response = call_llm(test_prompt)
    print(f"Response: {response}")
