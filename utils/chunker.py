"""
Document chunking utilities for text processing.
"""
import os
from typing import Dict, Any


# Default chunking configuration
_DEFAULT_CHUNK_SIZE = 500
_DEFAULT_CHUNK_OVERLAP = 50
_DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")


def chunk_text(
    text: str,
    *,
    chunk_size: int = int(os.getenv("INGESTION_CHUNK_SIZE", _DEFAULT_CHUNK_SIZE)),
    chunk_overlap: int = int(os.getenv("INGESTION_CHUNK_OVERLAP", _DEFAULT_CHUNK_OVERLAP)),
    separators: tuple[str, ...] = _DEFAULT_SEPARATORS
) -> list[str]:
    """
    Split text into chunks with overlap for better context.
    
    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in characters (defaults to env var or 500)
        chunk_overlap: Overlap between chunks in characters (defaults to env var or 50)
        separators: Separators to try for splitting (paragraph, sentence, word)
    
    Returns:
        List of text chunks
        
    Example:
        >>> chunk_text("Long text here...", chunk_size=100, chunk_overlap=20)
        ['chunk1...', 'chunk2...']
    """
    
    if not text or len(text) == 0:
        return []
    
    # If text is smaller than chunk_size, return as single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position
        end = start + chunk_size
        
        # If this is the last chunk, take everything
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        
        # Try to find a good split point using separators
        split_point = end
        for separator in separators:
            # Look for separator near the end of the chunk
            search_start = max(start, end - len(separator) * 10)
            last_sep = text.rfind(separator, search_start, end)
            
            if last_sep > start:
                split_point = last_sep + len(separator)
                break
        
        # Add chunk
        chunk = text[start:split_point].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        next_start = split_point - chunk_overlap
        
        # Ensure we're making progress (avoid infinite loop)
        if next_start <= start:
            next_start = split_point
        
        start = next_start
    
    return chunks


def chunk_documents(
    documents: list[Dict[str, Any]],
    *,
    chunk_size: int = int(os.getenv("INGESTION_CHUNK_SIZE", _DEFAULT_CHUNK_SIZE)),
    chunk_overlap: int = int(os.getenv("INGESTION_CHUNK_OVERLAP", _DEFAULT_CHUNK_OVERLAP))
) -> list[Dict[str, Any]]:
    """
    Chunk multiple documents with metadata preservation.
    
    Args:
        documents: List of dicts with 'content' and 'metadata' keys
        chunk_size: Target chunk size (defaults to env var or 500)
        chunk_overlap: Overlap between chunks (defaults to env var or 50)
    
    Returns:
        List of chunk dicts with preserved and augmented metadata
        
    Example:
        >>> docs = [{"content": "...", "metadata": {"source": "file.txt"}}]
        >>> chunks = chunk_documents(docs)
        >>> # Returns: [{"content": "chunk1", "metadata": {..., "chunk_index": 0}}]
    """
    all_chunks = []
    
    for doc in documents:
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        
        # Chunk the content
        text_chunks = chunk_text(
            content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Create chunk dicts with metadata
        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "content": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks)
                }
            })
    
    return all_chunks


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of token count (assumes ~4 chars per token).
    
    Args:
        text: Input text
    
    Returns:
        Estimated token count
    """
    return len(text) // 4


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximately fit within token limit.
    
    Args:
        text: Input text
        max_tokens: Maximum token count
    
    Returns:
        Truncated text
    """
    max_chars = max_tokens * 4
    
    if len(text) <= max_chars:
        return text
    
    # Truncate and add indicator
    truncated = text[:max_chars]
    
    # Try to cut at last sentence
    last_period = truncated.rfind(". ")
    if last_period > max_chars * 0.8:  # At least 80% of target length
        truncated = truncated[:last_period + 1]
    
    return truncated + " [truncated]"


if __name__ == "__main__":
    # Test chunker
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Testing chunker...")
    
    # Test simple chunking
    test_text = """
    VPN Troubleshooting Guide
    
    If you're experiencing VPN connection issues, follow these steps:
    
    1. Check your internet connection. Make sure you have a stable connection.
    
    2. Restart the VPN client. Close the application completely and reopen it.
    
    3. Check firewall settings. Ensure the VPN is allowed through your firewall.
    
    4. Try a different server. Sometimes server load can cause connection issues.
    
    5. Update your VPN client to the latest version.
    
    If none of these steps work, please contact IT support.
    """
    
    chunks = chunk_text(test_text, chunk_size=100, chunk_overlap=20)
    
    print(f"\n✓ Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i+1} ({len(chunk)} chars): {chunk[:60]}...")
    
    # Test document chunking
    docs = [
        {
            "content": test_text,
            "metadata": {
                "source": "vpn_guide.md",
                "category": "networking"
            }
        }
    ]
    
    chunked_docs = chunk_documents(docs, chunk_size=150, chunk_overlap=30)
    
    print(f"\n✓ Chunked {len(chunked_docs)} document chunks:")
    for i, chunk_doc in enumerate(chunked_docs[:2]):  # Show first 2
        print(f"\n  Chunk {i+1}:")
        print(f"    Content: {chunk_doc['content'][:50]}...")
        print(f"    Index: {chunk_doc['metadata']['chunk_index']}/{chunk_doc['metadata']['total_chunks']}")
    
    # Test token estimation
    tokens = estimate_tokens(test_text)
    print(f"\n✓ Estimated {tokens} tokens")
    
    # Test truncation
    truncated = truncate_to_token_limit(test_text, max_tokens=50)
    print(f"\n✓ Truncated to 50 tokens ({len(truncated)} chars):")
    print(f"  {truncated[:100]}...")
