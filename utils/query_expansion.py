"""
Query Expansion Utility - Uses LLM to expand queries for better retrieval.

This implements Improvement #4 from the RAG optimization article:
- Expands user queries with synonyms, related terms, and rephrased versions
- Improves recall by matching documents that use different terminology
- Can generate multiple query variants for multi-query retrieval

Example:
    User: "VPN not working"
    Expanded: ["VPN not working", "VPN connection issues", "virtual private network problems", 
               "remote access failure", "VPN troubleshooting"]
"""

import os
import yaml
from typing import List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

# Cache for expanded queries to avoid redundant LLM calls
_query_cache: dict = {}


def expand_query(
    query: str,
    num_expansions: int = 3,
    call_llm_func = None
) -> List[str]:
    """
    Expand a query using LLM to generate related phrasings.
    
    Args:
        query: Original user query
        num_expansions: Number of alternative queries to generate
        call_llm_func: Function to call LLM (injected to avoid circular imports)
        
    Returns:
        List of query variants including the original
    """
    expansion_enabled = os.getenv("QUERY_EXPANSION_ENABLED", "true").lower() == "true"
    
    if not expansion_enabled:
        logger.debug("Query expansion disabled")
        return [query]
    
    # Check cache first
    cache_key = f"{query}:{num_expansions}"
    if cache_key in _query_cache:
        logger.debug(f"Query expansion cache hit for: {query[:50]}...")
        return _query_cache[cache_key]
    
    if call_llm_func is None:
        logger.warning("No LLM function provided for query expansion")
        return [query]
    
    try:
        prompt = f"""You are a query expansion assistant for an IT support knowledge base.

Given the user's query, generate {num_expansions} alternative phrasings that would help find relevant documentation.

Guidelines:
- Include synonyms and related technical terms
- Consider different ways users might phrase the same question
- Include both formal and informal variations
- Keep expansions relevant to IT support context

User Query: {query}

Output exactly {num_expansions} alternative queries, one per line. Do not include numbering or explanations.
Just the queries, nothing else."""

        response = call_llm_func(prompt)
        
        # Parse the response - each line is an alternative query
        expanded = [query]  # Always include original
        for line in response.strip().split('\n'):
            line = line.strip()
            # Skip empty lines and numbering like "1." or "- "
            if line and not line[0].isdigit() and not line.startswith('-'):
                expanded.append(line)
            elif line and line[0].isdigit():
                # Remove numbering like "1. " or "1) "
                parts = line.split('.', 1) if '.' in line[:3] else line.split(')', 1)
                if len(parts) > 1:
                    expanded.append(parts[1].strip())
        
        # Limit to requested number + original
        expanded = expanded[:num_expansions + 1]
        
        # Cache the result
        _query_cache[cache_key] = expanded
        
        logger.info(f"Query expansion: '{query[:30]}...' -> {len(expanded)} variants")
        logger.debug(f"Expanded queries: {expanded}")
        
        return expanded
        
    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        return [query]


def expand_query_with_keywords(
    query: str,
    keywords: List[str] = None,
    call_llm_func = None
) -> str:
    """
    Expand a query by incorporating extracted keywords for better search.
    
    This creates a single enhanced query rather than multiple variants.
    
    Args:
        query: Original user query
        keywords: Pre-extracted keywords from the query
        call_llm_func: Function to call LLM
        
    Returns:
        Enhanced query string
    """
    expansion_enabled = os.getenv("QUERY_EXPANSION_ENABLED", "true").lower() == "true"
    
    if not expansion_enabled:
        return query
    
    if not keywords:
        return query
    
    try:
        # Simple keyword-based expansion without LLM
        keyword_str = " ".join(keywords)
        enhanced = f"{query} {keyword_str}"
        
        logger.debug(f"Keyword-enhanced query: {enhanced[:100]}...")
        return enhanced
        
    except Exception as e:
        logger.error(f"Keyword expansion failed: {e}")
        return query


def generate_hypothetical_answer(
    query: str,
    call_llm_func = None
) -> str:
    """
    Generate a hypothetical answer (HyDE) for the query.
    
    HyDE (Hypothetical Document Embeddings) generates what a good answer
    might look like, then uses that for retrieval instead of the query.
    This can improve retrieval for complex questions.
    
    Args:
        query: User query
        call_llm_func: Function to call LLM
        
    Returns:
        Hypothetical answer text for embedding
    """
    hyde_enabled = os.getenv("HYDE_ENABLED", "false").lower() == "true"
    
    if not hyde_enabled:
        return query
    
    if call_llm_func is None:
        logger.warning("No LLM function provided for HyDE")
        return query
    
    try:
        prompt = f"""You are an IT support expert. Write a short, direct answer to this question as if you found it in documentation.
Keep it concise (2-3 sentences). Include specific technical terms.

Question: {query}

Answer:"""

        response = call_llm_func(prompt)
        
        # Combine query and hypothetical answer for embedding
        enhanced = f"{query}\n\n{response.strip()}"
        
        logger.info(f"Generated HyDE for query: {query[:50]}...")
        logger.debug(f"Hypothetical answer: {response[:200]}...")
        
        return enhanced
        
    except Exception as e:
        logger.error(f"HyDE generation failed: {e}")
        return query


def clear_expansion_cache():
    """Clear the query expansion cache."""
    global _query_cache
    _query_cache = {}
    logger.info("Query expansion cache cleared")


if __name__ == "__main__":
    # Test query expansion
    logging.basicConfig(level=logging.DEBUG)
    
    # Mock LLM function for testing
    def mock_llm(prompt):
        return """VPN connection problems and troubleshooting
Remote access not connecting
Virtual private network error resolution"""
    
    print("Testing query expansion:")
    query = "VPN not working"
    expanded = expand_query(query, num_expansions=3, call_llm_func=mock_llm)
    print(f"Original: {query}")
    print(f"Expanded: {expanded}")
    
    print("\nTesting keyword expansion:")
    enhanced = expand_query_with_keywords(
        query="How do I connect remotely?",
        keywords=["VPN", "remote", "connect", "access"]
    )
    print(f"Enhanced: {enhanced}")
    
    print("\nTesting HyDE:")
    os.environ["HYDE_ENABLED"] = "true"
    hyde = generate_hypothetical_answer(
        query="How do I reset my password?",
        call_llm_func=lambda p: "To reset your password, go to the IT portal at portal.company.com, click 'Forgot Password', enter your email, and follow the reset link."
    )
    print(f"HyDE result: {hyde}")
