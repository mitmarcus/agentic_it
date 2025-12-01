"""
Hybrid Search Utility - Combines BM25 keyword search with vector similarity.

- BM25 excels at exact keyword matching
- Vector search excels at semantic similarity
- Combining both gives better recall and precision

Uses Reciprocal Rank Fusion (RRF) to combine scores from both methods.
"""

import os
from typing import Optional
from rank_bm25 import BM25Okapi
import re
from utils.logger import get_logger

logger = get_logger(__name__)

# Module-level BM25 index cache
_bm25_index: Optional[BM25Okapi] = None
_bm25_corpus: list = []
_bm25_doc_ids: list = []


def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25.
    Lowercases text and splits on non-alphanumeric characters.
    """
    # Lowercase and split on non-alphanumeric
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def build_bm25_index(documents: list[dict]) -> BM25Okapi:
    """
    Build a BM25 index from a list of documents.
    
    Args:
        documents: List of dicts with 'id' and 'text' keys
        
    Returns:
        BM25Okapi index object
    """
    global _bm25_index, _bm25_corpus, _bm25_doc_ids
    
    _bm25_corpus = []
    _bm25_doc_ids = []
    
    for doc in documents:
        doc_id = doc.get('id', '')
        text = doc.get('text', '')
        
        tokens = tokenize(text)
        _bm25_corpus.append(tokens)
        _bm25_doc_ids.append(doc_id)
    
    _bm25_index = BM25Okapi(_bm25_corpus)
    logger.info(f"Built BM25 index with {len(_bm25_corpus)} documents")
    
    return _bm25_index


def get_bm25_index() -> Optional[BM25Okapi]:
    """Get the current BM25 index (if built)."""
    return _bm25_index


def bm25_search(query: str, top_k: int = 10) -> list[tuple[str, float]]:
    """
    Search using BM25.
    
    Args:
        query: Search query
        top_k: Number of results to return
        
    Returns:
        List of (doc_id, bm25_score) tuples, sorted by score descending
    """
    if _bm25_index is None:
        logger.warning("BM25 index not built, returning empty results")
        return []
    
    query_tokens = tokenize(query)
    scores = _bm25_index.get_scores(query_tokens)
    
    # Create (doc_id, score) pairs and sort by score
    results = list(zip(_bm25_doc_ids, scores))
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:top_k]


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[tuple[str, float]],
    k: int = 60,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5
) -> list[dict]:
    """
    Combine vector and BM25 results using Reciprocal Rank Fusion.
    
    RRF Score = sum(1 / (k + rank)) for each ranking list
    
    Args:
        vector_results: Results from vector search, each with 'id' key
        bm25_results: Results from BM25 search as (doc_id, score) tuples
        k: RRF constant (default 60, standard in literature)
        vector_weight: Weight for vector search contribution
        bm25_weight: Weight for BM25 contribution
        
    Returns:
        Combined results sorted by RRF score, with 'rrf_score' added
    """
    rrf_scores = {}
    doc_data = {}  # Store full document data
    
    # Add vector search contributions
    for rank, result in enumerate(vector_results):
        doc_id = result.get('id', '')
        if doc_id:
            rrf_score = vector_weight * (1.0 / (k + rank + 1))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score
            doc_data[doc_id] = result  # Store the full result
    
    # Add BM25 contributions
    for rank, (doc_id, bm25_score) in enumerate(bm25_results):
        if doc_id:
            rrf_score = bm25_weight * (1.0 / (k + rank + 1))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score
            # If we haven't seen this doc from vector search, we won't have full data
            # In practice, we'll only return docs that have vector data
    
    # Build final results - only include docs we have full data for
    final_results = []
    for doc_id, rrf_score in rrf_scores.items():
        if doc_id in doc_data:
            result = doc_data[doc_id].copy()
            result['rrf_score'] = rrf_score
            # Mark if this was found by BM25 (for debugging)
            result['found_by_bm25'] = doc_id in [r[0] for r in bm25_results]
            final_results.append(result)
    
    # Sort by RRF score descending
    final_results.sort(key=lambda x: x.get('rrf_score', 0), reverse=True)
    
    return final_results


def hybrid_search(
    query: str,
    vector_results: list[dict],
    top_k: int = 5,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5
) -> list[dict]:
    """
    Perform hybrid search combining vector results with BM25.
    
    This is the main entry point for hybrid search.
    
    Args:
        query: The search query
        vector_results: Results from vector/embedding search
        top_k: Number of final results to return
        vector_weight: Weight for vector search (0-1)
        bm25_weight: Weight for BM25 search (0-1)
        
    Returns:
        Combined and reranked results
    """
    hybrid_enabled = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
    
    if not hybrid_enabled:
        logger.debug("Hybrid search disabled, returning vector results only")
        return vector_results[:top_k]
    
    if _bm25_index is None:
        logger.warning("BM25 index not initialized, falling back to vector-only search")
        return vector_results[:top_k]
    
    try:
        # Get weights from environment
        vector_weight = float(os.getenv("HYBRID_VECTOR_WEIGHT", str(vector_weight)))
        bm25_weight = float(os.getenv("HYBRID_BM25_WEIGHT", str(bm25_weight)))
        
        # Normalize weights
        total_weight = vector_weight + bm25_weight
        vector_weight /= total_weight
        bm25_weight /= total_weight
        
        # Get BM25 results (fetch more candidates for fusion)
        bm25_results = bm25_search(query, top_k=top_k * 3)
        
        # Combine using RRF
        combined = reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight
        )
        
        logger.debug(f"Hybrid search: {len(vector_results)} vector + {len(bm25_results)} BM25 -> {len(combined)} combined")
        
        return combined[:top_k]
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}, falling back to vector results")
        return vector_results[:top_k]


def rebuild_bm25_from_chromadb(collection) -> int:
    """
    Rebuild BM25 index from ChromaDB collection.
    
    Args:
        collection: ChromaDB collection object
        
    Returns:
        Number of documents indexed
    """
    try:
        # Get all documents from ChromaDB
        results = collection.get(include=["documents", "metadatas"])
        
        documents = []
        for i, doc_id in enumerate(results.get('ids', [])):
            text = results.get('documents', [])[i] if results.get('documents') else ""
            documents.append({
                'id': doc_id,
                'text': text
            })
        
        if documents:
            build_bm25_index(documents)
            logger.info(f"Rebuilt BM25 index from ChromaDB: {len(documents)} documents")
            return len(documents)
        else:
            logger.warning("No documents found in ChromaDB for BM25 indexing")
            return 0
            
    except Exception as e:
        logger.error(f"Failed to rebuild BM25 index from ChromaDB: {e}")
        return 0


if __name__ == "__main__":
    # Test the hybrid search
    logging.basicConfig(level=logging.DEBUG)
    
    # Build a test index
    test_docs = [
        {"id": "doc1", "text": "How to reset your password in the IT support portal"},
        {"id": "doc2", "text": "VPN connection issues and troubleshooting guide"},
        {"id": "doc3", "text": "Password policy and security requirements"},
        {"id": "doc4", "text": "Installing software on company laptops"},
    ]
    
    build_bm25_index(test_docs)
    
    # Test BM25 search
    query = "password reset"
    bm25_results = bm25_search(query, top_k=3)
    print(f"\nBM25 results for '{query}':")
    for doc_id, score in bm25_results:
        print(f"  {doc_id}: {score:.4f}")
    
    # Test hybrid search (simulating vector results)
    fake_vector_results = [
        {"id": "doc3", "text": "Password policy and security requirements", "vector_score": 0.85},
        {"id": "doc1", "text": "How to reset your password in the IT support portal", "vector_score": 0.80},
        {"id": "doc4", "text": "Installing software on company laptops", "vector_score": 0.65},
    ]
    
    hybrid_results = hybrid_search(query, fake_vector_results, top_k=3)
    print(f"\nHybrid results for '{query}':")
    for result in hybrid_results:
        print(f"  {result['id']}: rrf={result.get('rrf_score', 0):.4f}, vector={result.get('vector_score', 0):.2f}, bm25_found={result.get('found_by_bm25', False)}")
