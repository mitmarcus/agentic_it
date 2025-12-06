"""
Cross-encoder reranker for improving RAG retrieval quality.

Uses a cross-encoder model to re-score documents based on query-document pairs,
which is more accurate than bi-encoder similarity for final ranking.
"""
import os
from typing import Any
from utils.logger import get_logger

logger = get_logger(__name__)

# Lazy-loaded reranker instance
_RERANKER: Any = None


def get_reranker():
    """
    Get or initialize the cross-encoder reranker.
    
    Uses lazy loading to avoid import overhead until first use.
    Model is cached for subsequent calls.
    
    Returns:
        CrossEncoder instance
    """
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        
        model_name = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info(f"Loading reranker model: {model_name}")
        
        _RERANKER = CrossEncoder(model_name)
        logger.info("Reranker model loaded successfully")
    
    return _RERANKER


def rerank_results(
    query: str,
    results: list[dict],
    top_k: int | None = None
) -> list[dict]:
    """
    Rerank search results using a cross-encoder model.
    
    Cross-encoders process query-document pairs together, providing more
    accurate relevance scores than bi-encoder similarity at the cost of
    not being able to pre-compute document embeddings.
    
    Args:
        query: The user's search query
        results: List of result dicts with at least a 'document' key
        top_k: Number of top results to return (None = return all, reranked)
    
    Returns:
        Results sorted by rerank_score (descending), with rerank_score added
        
    Example:
        >>> results = [{"document": "VPN setup guide...", "score": 0.8}]
        >>> reranked = rerank_results("how to connect VPN", results, top_k=3)
        >>> print(reranked[0]["rerank_score"])
    """
    if not results:
        return results
    
    if not query.strip():
        logger.warning("Empty query provided to reranker, returning original results")
        return results
    
    try:
        reranker = get_reranker()
        
        # Create query-document pairs for cross-encoder
        pairs = [(query, r["document"]) for r in results]
        
        # Get cross-encoder scores (higher = more relevant)
        scores = reranker.predict(pairs)
        
        # Add rerank scores to results
        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)
            # Keep original vector score for comparison
            result["vector_score"] = result.get("score", 0.0)
        
        # Sort by rerank score (descending)
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        
        # Log score comparison for debugging
        if reranked:
            logger.debug(
                f"Reranking complete. "
                f"Top result: vector_score={reranked[0].get('vector_score', 0):.3f}, "
                f"rerank_score={reranked[0]['rerank_score']:.3f}"
            )
        
        # Return top_k if specified
        if top_k is not None:
            return reranked[:top_k]
        
        return reranked
        
    except Exception as e:
        logger.error(f"Reranking failed: {e}. Returning original results.")
        # Graceful fallback - return original results unchanged
        return results


if __name__ == "__main__":
    # Test the reranker
    test_query = "How do I connect to the VPN?"
    test_results = [
        {"document": "To connect to the VPN, open Cisco AnyConnect and enter vpn.company.com", "score": 0.85},
        {"document": "The printer is located on the 3rd floor near the elevator", "score": 0.82},
        {"document": "VPN troubleshooting: If connection fails, check your internet first", "score": 0.80},
        {"document": "Company holiday schedule for 2025", "score": 0.75},
    ]
    
    print(f"Query: {test_query}")
    print("\nOriginal order (by vector score):")
    for i, r in enumerate(test_results):
        print(f"  {i+1}. [{r['score']:.2f}] {r['document'][:50]}...")
    
    reranked = rerank_results(test_query, test_results.copy(), top_k=3)
    
    print("\nReranked order (by cross-encoder):")
    for i, r in enumerate(reranked):
        print(f"  {i+1}. [vec={r['vector_score']:.2f}, rerank={r['rerank_score']:.2f}] {r['document'][:50]}...")
