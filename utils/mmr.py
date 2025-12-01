"""
MMR (Maximal Marginal Relevance) Utility for diverse document retrieval.

This implements Improvement #3 from the RAG optimization article:
- MMR balances relevance and diversity
- Prevents returning multiple similar/redundant chunks
- Improves coverage of different aspects of a topic

MMR Score = λ * Relevance(doc, query) - (1-λ) * max(Similarity(doc, selected_docs))
"""

import os
import numpy as np
from typing import List, Dict, Optional, Callable
from utils.logger import get_logger

logger = get_logger(__name__)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (-1 to 1)
    """
    if not vec1 or not vec2:
        return 0.0
    
    arr1 = np.array(vec1, dtype=np.float32)
    arr2 = np.array(vec2, dtype=np.float32)
    
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(arr1, arr2) / (norm1 * norm2))


def calculate_mmr_scores(
    query_embedding: List[float],
    candidate_embeddings: List[List[float]],
    relevance_scores: List[float],
    selected_indices: List[int],
    lambda_param: float = 0.7
) -> List[float]:
    """
    Calculate MMR scores for candidate documents.
    
    MMR = λ * Relevance - (1-λ) * MaxSimilarity(to_selected)
    
    Args:
        query_embedding: Query embedding vector
        candidate_embeddings: List of candidate document embeddings
        relevance_scores: Pre-computed relevance scores for candidates
        selected_indices: Indices of already selected documents
        lambda_param: Balance between relevance and diversity (0-1)
                     Higher = more relevance, Lower = more diversity
                     
    Returns:
        List of MMR scores for each candidate
    """
    mmr_scores = []
    
    for i, (emb, rel_score) in enumerate(zip(candidate_embeddings, relevance_scores)):
        if i in selected_indices:
            # Already selected, give very low score
            mmr_scores.append(float('-inf'))
            continue
        
        # Relevance term
        relevance_term = lambda_param * rel_score
        
        # Diversity term: max similarity to any already selected document
        if selected_indices:
            max_sim = 0.0
            for sel_idx in selected_indices:
                if sel_idx < len(candidate_embeddings):
                    sim = cosine_similarity(emb, candidate_embeddings[sel_idx])
                    max_sim = max(max_sim, sim)
            diversity_penalty = (1 - lambda_param) * max_sim
        else:
            # No documents selected yet, no penalty
            diversity_penalty = 0.0
        
        mmr_score = relevance_term - diversity_penalty
        mmr_scores.append(mmr_score)
    
    return mmr_scores


def mmr_rerank(
    results: List[Dict],
    query_embedding: Optional[List[float]] = None,
    top_k: int = 5,
    lambda_param: float = 0.7,
    embedding_key: str = "embedding",
    score_key: str = "score"
) -> List[Dict]:
    """
    Apply MMR to rerank results for diversity.
    
    This is the main entry point for MMR reranking.
    
    Args:
        results: List of result dicts (must have embeddings if available)
        query_embedding: Query embedding (optional, will use score as proxy if not provided)
        top_k: Number of diverse results to return
        lambda_param: Balance between relevance (1) and diversity (0). Default 0.7
        embedding_key: Key in result dict for embedding vector
        score_key: Key in result dict for relevance score
        
    Returns:
        Reranked list of results with MMR diversity
    """
    mmr_enabled = os.getenv("MMR_ENABLED", "true").lower() == "true"
    
    if not mmr_enabled:
        logger.debug("MMR disabled, returning original results")
        return results[:top_k]
    
    if not results:
        return []
    
    if len(results) <= top_k:
        # Not enough results to need MMR
        return results
    
    # Get lambda from environment or use provided value
    lambda_param = float(os.getenv("MMR_LAMBDA", str(lambda_param)))
    
    try:
        # Extract embeddings if available
        has_embeddings = all(embedding_key in r and r[embedding_key] for r in results)
        
        if has_embeddings:
            candidate_embeddings = [r[embedding_key] for r in results]
        else:
            # No embeddings available, fall back to simple text-based diversity
            # Use document text hashes as a proxy
            logger.debug("No embeddings in results, using content-based MMR")
            candidate_embeddings = None
        
        # Get relevance scores
        relevance_scores = [r.get(score_key, 0) for r in results]
        
        # Normalize relevance scores to 0-1 range for MMR calculation
        max_score = max(relevance_scores) if relevance_scores else 1
        min_score = min(relevance_scores) if relevance_scores else 0
        score_range = max_score - min_score if max_score != min_score else 1
        normalized_scores = [(s - min_score) / score_range for s in relevance_scores]
        
        # Greedy MMR selection
        selected_indices = []
        selected_results = []
        
        for _ in range(min(top_k, len(results))):
            if candidate_embeddings and query_embedding:
                # Full MMR with embeddings
                mmr_scores = calculate_mmr_scores(
                    query_embedding=query_embedding,
                    candidate_embeddings=candidate_embeddings,
                    relevance_scores=normalized_scores,
                    selected_indices=selected_indices,
                    lambda_param=lambda_param
                )
            else:
                # Simplified MMR using text similarity
                mmr_scores = _simplified_mmr_scores(
                    results=results,
                    relevance_scores=normalized_scores,
                    selected_indices=selected_indices,
                    lambda_param=lambda_param
                )
            
            # Select document with highest MMR score
            best_idx = np.argmax(mmr_scores)
            selected_indices.append(int(best_idx))
            
            result_copy = results[best_idx].copy()
            result_copy['mmr_rank'] = len(selected_results) + 1
            result_copy['mmr_score'] = float(mmr_scores[best_idx])
            selected_results.append(result_copy)
        
        logger.debug(f"MMR selected {len(selected_results)} diverse results from {len(results)} candidates")
        return selected_results
        
    except Exception as e:
        logger.error(f"MMR reranking failed: {e}, returning original results")
        return results[:top_k]


def _simplified_mmr_scores(
    results: List[Dict],
    relevance_scores: List[float],
    selected_indices: List[int],
    lambda_param: float
) -> List[float]:
    """
    Calculate simplified MMR scores using text content overlap.
    Used when embeddings are not available.
    """
    mmr_scores = []
    
    for i, (result, rel_score) in enumerate(zip(results, relevance_scores)):
        if i in selected_indices:
            mmr_scores.append(float('-inf'))
            continue
        
        # Relevance term
        relevance_term = lambda_param * rel_score
        
        # Simple diversity using word overlap
        if selected_indices:
            doc_text = result.get('document', '').lower()
            doc_words = set(doc_text.split())
            
            max_overlap = 0.0
            for sel_idx in selected_indices:
                sel_text = results[sel_idx].get('document', '').lower()
                sel_words = set(sel_text.split())
                
                if doc_words and sel_words:
                    overlap = len(doc_words & sel_words) / len(doc_words | sel_words)
                    max_overlap = max(max_overlap, overlap)
            
            diversity_penalty = (1 - lambda_param) * max_overlap
        else:
            diversity_penalty = 0.0
        
        mmr_scores.append(relevance_term - diversity_penalty)
    
    return mmr_scores


if __name__ == "__main__":
    # Test MMR
    logging.basicConfig(level=logging.DEBUG)
    
    # Create test results with embeddings
    np.random.seed(42)
    
    # Simulate 3 clusters of similar documents
    base_emb1 = np.random.rand(384).tolist()
    base_emb2 = np.random.rand(384).tolist()
    base_emb3 = np.random.rand(384).tolist()
    
    def add_noise(emb, scale=0.1):
        return (np.array(emb) + np.random.randn(384) * scale).tolist()
    
    test_results = [
        {"id": "1a", "document": "VPN connection issues and troubleshooting", "score": 0.95, "embedding": add_noise(base_emb1, 0.05)},
        {"id": "1b", "document": "VPN setup and connection problems", "score": 0.93, "embedding": add_noise(base_emb1, 0.05)},
        {"id": "2a", "document": "Password reset process for employees", "score": 0.90, "embedding": add_noise(base_emb2, 0.05)},
        {"id": "2b", "document": "How to change your password", "score": 0.88, "embedding": add_noise(base_emb2, 0.05)},
        {"id": "3a", "document": "Email configuration for Outlook", "score": 0.85, "embedding": add_noise(base_emb3, 0.05)},
        {"id": "1c", "document": "VPN troubleshooting guide", "score": 0.82, "embedding": add_noise(base_emb1, 0.05)},
    ]
    
    query_emb = add_noise(base_emb1, 0.3)  # Query closer to VPN topic
    
    print("Original results (by score):")
    for r in test_results:
        print(f"  {r['id']}: {r['document'][:40]}... score={r['score']:.2f}")
    
    print("\nMMR reranked results (diverse):")
    mmr_results = mmr_rerank(test_results, query_embedding=query_emb, top_k=4, lambda_param=0.5)
    for r in mmr_results:
        print(f"  {r['id']}: {r['document'][:40]}... mmr={r.get('mmr_score', 0):.3f}")
