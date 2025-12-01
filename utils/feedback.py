"""
Feedback Loop Utility - Track and learn from user feedback.

This implements Improvement #6 from the RAG optimization article:
- Collects thumbs up/down feedback on answers
- Logs retrieved documents with feedback for analysis
- Enables continuous improvement through feedback data
- Can be used to fine-tune reranking or improve document quality

The feedback is stored in a simple JSON file for analysis.
In production, this could be stored in a database or analytics system.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import hashlib
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeedbackEntry:
    """Represents a single feedback entry."""
    feedback_id: str
    timestamp: str
    session_id: str
    query: str
    response: str
    retrieved_doc_ids: List[str]
    retrieved_doc_scores: List[float]
    feedback_type: str  # "positive", "negative", "neutral"
    feedback_score: int  # 1-5 or -1/0/1
    feedback_comment: Optional[str] = None
    intent: Optional[str] = None
    latency_ms: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class FeedbackStore:
    """Simple file-based feedback storage."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize feedback store.
        
        Args:
            storage_path: Path to feedback JSON file (defaults to logs/feedback.jsonl)
        """
        self.storage_path = Path(storage_path or os.getenv(
            "FEEDBACK_STORAGE_PATH",
            "./logs/feedback.jsonl"
        ))
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: List[FeedbackEntry] = []
        
    def add_feedback(self, entry: FeedbackEntry) -> str:
        """
        Add a feedback entry.
        
        Args:
            entry: FeedbackEntry object
            
        Returns:
            Feedback ID
        """
        try:
            # Append to JSONL file
            with open(self.storage_path, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
            
            self._cache.append(entry)
            logger.info(f"Stored feedback {entry.feedback_id}: {entry.feedback_type}")
            return entry.feedback_id
            
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
            raise
    
    def get_feedback(self, limit: int = 100) -> List[Dict]:
        """
        Get recent feedback entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of feedback dicts
        """
        entries = []
        try:
            if self.storage_path.exists():
                with open(self.storage_path, "r") as f:
                    for line in f:
                        try:
                            entries.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Failed to read feedback: {e}")
        
        return entries[-limit:]
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get feedback statistics.
        
        Returns:
            Dict with feedback counts and averages
        """
        entries = self.get_feedback(limit=10000)
        
        if not entries:
            return {
                "total": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "positive_rate": 0.0
            }
        
        total = len(entries)
        positive = sum(1 for e in entries if e.get("feedback_type") == "positive")
        negative = sum(1 for e in entries if e.get("feedback_type") == "negative")
        neutral = total - positive - negative
        
        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "positive_rate": positive / total if total > 0 else 0.0
        }


# Global feedback store instance
_feedback_store: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    """Get or create the global feedback store."""
    global _feedback_store
    if _feedback_store is None:
        _feedback_store = FeedbackStore()
    return _feedback_store


def generate_feedback_id(session_id: str, query: str) -> str:
    """Generate a unique feedback ID."""
    content = f"{session_id}:{query}:{datetime.now().isoformat()}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def record_feedback(
    session_id: str,
    query: str,
    response: str,
    feedback_type: str,
    feedback_score: int = 0,
    feedback_comment: Optional[str] = None,
    retrieved_docs: Optional[List[Dict]] = None,
    intent: Optional[str] = None,
    latency_ms: Optional[float] = None
) -> str:
    """
    Record user feedback for a query-response pair.
    
    Args:
        session_id: User session identifier
        query: Original user query
        response: Generated response
        feedback_type: "positive", "negative", or "neutral"
        feedback_score: Numeric score (-1 to 1, or 1-5)
        feedback_comment: Optional user comment
        retrieved_docs: List of retrieved document dicts
        intent: Classified intent
        latency_ms: Response latency in milliseconds
        
    Returns:
        Feedback ID
    """
    feedback_enabled = os.getenv("FEEDBACK_ENABLED", "true").lower() == "true"
    
    if not feedback_enabled:
        logger.debug("Feedback collection disabled")
        return ""
    
    # Extract doc IDs and scores
    doc_ids = []
    doc_scores = []
    if retrieved_docs:
        for doc in retrieved_docs:
            doc_ids.append(doc.get("id", ""))
            # Try rerank score, then RRF score, then vector score
            score = doc.get("rerank_score", doc.get("rrf_score", doc.get("score", 0)))
            doc_scores.append(float(score))
    
    entry = FeedbackEntry(
        feedback_id=generate_feedback_id(session_id, query),
        timestamp=datetime.now().isoformat(),
        session_id=session_id,
        query=query,
        response=response[:500],  # Truncate long responses
        retrieved_doc_ids=doc_ids,
        retrieved_doc_scores=doc_scores,
        feedback_type=feedback_type,
        feedback_score=feedback_score,
        feedback_comment=feedback_comment,
        intent=intent,
        latency_ms=latency_ms
    )
    
    store = get_feedback_store()
    return store.add_feedback(entry)


def get_low_rated_queries(min_count: int = 5) -> List[Dict]:
    """
    Get queries with consistently negative feedback.
    
    Useful for identifying areas where the system needs improvement.
    
    Args:
        min_count: Minimum feedback count to consider
        
    Returns:
        List of problem queries with stats
    """
    store = get_feedback_store()
    entries = store.get_feedback(limit=10000)
    
    # Group by query (normalized)
    query_stats = {}
    for entry in entries:
        query = entry.get("query", "").lower().strip()
        if not query:
            continue
        
        if query not in query_stats:
            query_stats[query] = {"positive": 0, "negative": 0, "total": 0}
        
        query_stats[query]["total"] += 1
        if entry.get("feedback_type") == "positive":
            query_stats[query]["positive"] += 1
        elif entry.get("feedback_type") == "negative":
            query_stats[query]["negative"] += 1
    
    # Filter to queries with enough feedback and low positive rate
    problem_queries = []
    for query, stats in query_stats.items():
        if stats["total"] >= min_count:
            positive_rate = stats["positive"] / stats["total"]
            if positive_rate < 0.5:  # Less than 50% positive
                problem_queries.append({
                    "query": query,
                    "positive_rate": positive_rate,
                    "total_feedback": stats["total"],
                    **stats
                })
    
    # Sort by positive rate (worst first)
    problem_queries.sort(key=lambda x: x["positive_rate"])
    return problem_queries


def get_doc_effectiveness() -> List[Dict]:
    """
    Analyze which documents are most/least effective.
    
    Returns:
        List of documents with feedback effectiveness scores
    """
    store = get_feedback_store()
    entries = store.get_feedback(limit=10000)
    
    # Track effectiveness per document
    doc_stats = {}
    for entry in entries:
        feedback_type = entry.get("feedback_type", "neutral")
        doc_ids = entry.get("retrieved_doc_ids", [])
        
        for doc_id in doc_ids:
            if not doc_id:
                continue
            
            if doc_id not in doc_stats:
                doc_stats[doc_id] = {"positive": 0, "negative": 0, "total": 0}
            
            doc_stats[doc_id]["total"] += 1
            if feedback_type == "positive":
                doc_stats[doc_id]["positive"] += 1
            elif feedback_type == "negative":
                doc_stats[doc_id]["negative"] += 1
    
    # Calculate effectiveness score
    results = []
    for doc_id, stats in doc_stats.items():
        if stats["total"] > 0:
            effectiveness = (stats["positive"] - stats["negative"]) / stats["total"]
            results.append({
                "doc_id": doc_id,
                "effectiveness": effectiveness,
                **stats
            })
    
    # Sort by effectiveness (best first)
    results.sort(key=lambda x: x["effectiveness"], reverse=True)
    return results


# Cached feedback adjustments (refreshed periodically)
_feedback_adjustments_cache: Optional[Dict[str, float]] = None
_feedback_cache_timestamp: Optional[datetime] = None
_FEEDBACK_CACHE_TTL_SECONDS = 300  # 5 minutes


def get_feedback_adjustments(min_feedback_count: int = 3) -> Dict[str, float]:
    """
    Get feedback-based score adjustments for documents.
    
    Returns a dict mapping doc_id to adjustment factor:
    - Positive adjustment (0 to +0.15) for docs with good feedback
    - Negative adjustment (0 to -0.2) for docs with bad feedback
    
    Adjustments are cached for 5 minutes to avoid repeated file reads.
    
    Args:
        min_feedback_count: Minimum feedback entries needed to apply adjustment
        
    Returns:
        Dict mapping doc_id to score adjustment (-0.2 to +0.15)
    """
    global _feedback_adjustments_cache, _feedback_cache_timestamp
    
    # Return cached adjustments if still valid
    if _feedback_adjustments_cache is not None and _feedback_cache_timestamp is not None:
        age = (datetime.now() - _feedback_cache_timestamp).total_seconds()
        if age < _FEEDBACK_CACHE_TTL_SECONDS:
            return _feedback_adjustments_cache
    
    # Rebuild adjustments from feedback data
    adjustments: Dict[str, float] = {}
    
    try:
        store = get_feedback_store()
        entries = store.get_feedback(limit=10000)
        
        if not entries:
            _feedback_adjustments_cache = {}
            _feedback_cache_timestamp = datetime.now()
            return {}
        
        # Track per-document feedback
        doc_stats: Dict[str, Dict[str, int]] = {}
        for entry in entries:
            feedback_type = entry.get("feedback_type", "neutral")
            doc_ids = entry.get("retrieved_doc_ids", [])
            
            for doc_id in doc_ids:
                if not doc_id:
                    continue
                
                if doc_id not in doc_stats:
                    doc_stats[doc_id] = {"positive": 0, "negative": 0, "total": 0}
                
                doc_stats[doc_id]["total"] += 1
                if feedback_type == "positive":
                    doc_stats[doc_id]["positive"] += 1
                elif feedback_type == "negative":
                    doc_stats[doc_id]["negative"] += 1
        
        # Calculate adjustments for docs with enough feedback
        for doc_id, stats in doc_stats.items():
            if stats["total"] < min_feedback_count:
                continue
            
            positive_rate = stats["positive"] / stats["total"]
            negative_rate = stats["negative"] / stats["total"]
            
            # Positive feedback boosts score (max +0.15)
            # Negative feedback reduces score (max -0.2)
            # Neutral has no effect
            if positive_rate > 0.6:
                # Good doc: boost proportionally (0.6 rate = +0.05, 1.0 rate = +0.15)
                adjustments[doc_id] = (positive_rate - 0.6) * 0.375  # 0 to 0.15
            elif negative_rate > 0.5:
                # Bad doc: penalize proportionally (0.5 rate = -0.05, 1.0 rate = -0.2)
                adjustments[doc_id] = -(negative_rate - 0.4) * 0.333  # 0 to -0.2
        
        logger.info(f"Computed feedback adjustments for {len(adjustments)} documents")
        
    except Exception as e:
        logger.warning(f"Failed to compute feedback adjustments: {e}")
        adjustments = {}
    
    _feedback_adjustments_cache = adjustments
    _feedback_cache_timestamp = datetime.now()
    return adjustments


def apply_feedback_adjustments(results: List[Dict], score_key: str = "score") -> List[Dict]:
    """
    Apply feedback-based score adjustments to retrieval results.
    
    Documents with positive feedback history get boosted.
    Documents with negative feedback history get penalized.
    
    Args:
        results: List of retrieval result dicts (must have 'id' and score_key)
        score_key: Key for the score to adjust (default: "score")
        
    Returns:
        Results with adjusted scores and re-sorted by adjusted score
    """
    feedback_enabled = os.getenv("FEEDBACK_ADJUSTMENT_ENABLED", "true").lower() == "true"
    
    if not feedback_enabled or not results:
        return results
    
    adjustments = get_feedback_adjustments()
    
    if not adjustments:
        return results
    
    adjusted_results = []
    adjustments_applied = 0
    
    for result in results:
        doc_id = result.get("id", "")
        original_score = result.get(score_key, 0)
        
        # Apply adjustment if available
        adjustment = adjustments.get(doc_id, 0)
        adjusted_score = max(0, min(1, original_score + adjustment))  # Clamp to [0, 1]
        
        # Create new result with adjusted score
        adjusted_result = result.copy()
        adjusted_result[score_key] = adjusted_score
        
        # Track original score for transparency
        if adjustment != 0:
            adjusted_result["feedback_adjustment"] = adjustment
            adjusted_result["original_score"] = original_score
            adjustments_applied += 1
        
        adjusted_results.append(adjusted_result)
    
    # Re-sort by adjusted score
    adjusted_results.sort(key=lambda x: x.get(score_key, 0), reverse=True)
    
    if adjustments_applied > 0:
        logger.info(f"Applied feedback adjustments to {adjustments_applied}/{len(results)} results")
    
    return adjusted_results


if __name__ == "__main__":
    # Test feedback system
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing feedback recording:")
    
    # Record some test feedback
    feedback_id = record_feedback(
        session_id="test_session_1",
        query="How do I reset my VPN password?",
        response="To reset your VPN password, go to the IT portal...",
        feedback_type="positive",
        feedback_score=1,
        retrieved_docs=[
            {"id": "doc1", "score": 0.95},
            {"id": "doc2", "score": 0.87}
        ],
        intent="vpn_password_reset",
        latency_ms=1234.5
    )
    print(f"Recorded feedback: {feedback_id}")
    
    # Record negative feedback
    record_feedback(
        session_id="test_session_2",
        query="VPN connection keeps dropping",
        response="I'm not sure about that...",
        feedback_type="negative",
        feedback_score=-1,
        retrieved_docs=[{"id": "doc3", "score": 0.65}]
    )
    
    print("\nFeedback stats:")
    store = get_feedback_store()
    stats = store.get_feedback_stats()
    print(f"  Total: {stats['total']}")
    print(f"  Positive: {stats['positive']}")
    print(f"  Negative: {stats['negative']}")
    print(f"  Positive rate: {stats['positive_rate']:.1%}")
    
    print("\nRecent feedback:")
    for entry in store.get_feedback(limit=3):
        print(f"  - {entry['query'][:50]}... -> {entry['feedback_type']}")
