"""
Local embedding generation using sentence-transformers.
"""
import os
from typing import List, Optional

_MODEL = None # Cached model instance

def _load_model():
    """Load or return cached sentence-transformers model.
    
    Raises:
        Exception: If model loading fails (let Node retry mechanism handle it)
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    
    model_name = os.getenv("EMBED_MODEL")
    
    from sentence_transformers import SentenceTransformer
    
    print(f"Loading embedding model: {model_name}")
    _MODEL = SentenceTransformer(model_name)
    return _MODEL


def get_embedding(text: str) -> List[float]:
    """
    Generate embedding using local sentence-transformers.
    
    Args:
        text: Input text to embed
    
    Returns:
        Embedding vector (384-dim for all-MiniLM-L6-v2)
        
    Raises:
        Exception: If embedding generation fails (let Node retry mechanism handle it)
    """
    # Load model
    model_instance = _load_model()
    
    # Generate real embedding
    vec = model_instance.encode([text], normalize_embeddings=True)[0]
    return [float(x) for x in vec]

 # Test standalone embedding function
if __name__ == "__main__":
    # Load environment variables from .env file if present; I don't like load_dotenv but it works
    # if you know better ways please submit a PR
    from dotenv import load_dotenv
    load_dotenv()

    try:
        text = "Hello, world!"
        emb = get_embedding(text)
        print(f"Embedding dimension: {len(emb)}")
        print(f"First 5 values: {emb[:5]}")
    except Exception as e:
        print(f"\nError during standalone test: {e}")
