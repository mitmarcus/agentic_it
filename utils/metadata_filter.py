"""
Metadata Filtering Utility - Filter documents by structured metadata.

- Pre-filters documents by category, date, document type, etc.
- Reduces search space for faster and more relevant results
- Can be combined with intent classification for automatic filtering

Example metadata filters:
- category: "vpn", "password", "email", "hardware"
- doc_type: "how-to", "troubleshooting", "policy", "faq"
- date_range: filter by document age
- department: "IT", "HR", "Finance"
"""

import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from utils.logger import get_logger

logger = get_logger(__name__)


# Category mappings for automatic filtering based on intent/keywords
CATEGORY_MAPPINGS = {
    # VPN/Network related
    "vpn": ["vpn", "anyconnect", "remote access", "network", "connection", "firewall"],
    "network": ["wifi", "wireless", "ethernet", "internet", "connectivity", "dns"],
    
    # Account/Access related
    "password": ["password", "reset", "forgot", "credentials", "login", "authentication"],
    "access": ["access", "permission", "authorization", "role", "group", "admin"],
    
    # Software related
    "software": ["install", "update", "application", "program", "license", "version"],
    "email": ["email", "outlook", "mail", "calendar", "teams", "exchange"],
    
    # Hardware related
    "hardware": ["laptop", "computer", "monitor", "keyboard", "mouse", "printer", "device"],
    
    # General support
    "onboarding": ["new employee", "onboarding", "first day", "setup", "getting started"],
    "troubleshooting": ["error", "issue", "problem", "not working", "fix", "broken"],
}


def detect_category_from_query(query: str) -> Optional[str]:
    """
    Automatically detect document category from query text.
    
    Args:
        query: User query string
        
    Returns:
        Detected category or None if no clear match
    """
    query_lower = query.lower()
    
    # Score each category based on keyword matches
    category_scores = {}
    for category, keywords in CATEGORY_MAPPINGS.items():
        score = 0
        for keyword in keywords:
            if keyword in query_lower:
                # Longer keywords get higher weight
                score += len(keyword)
        if score > 0:
            category_scores[category] = score
    
    if category_scores:
        # Return category with highest score
        best_category = max(category_scores, key=category_scores.get)
        logger.debug(f"Detected category '{best_category}' from query (score: {category_scores[best_category]})")
        return best_category
    
    return None


def build_metadata_filter(
    category: Optional[str] = None,
    doc_type: Optional[str] = None,
    source_file_pattern: Optional[str] = None,
    max_age_days: Optional[int] = None,
    department: Optional[str] = None,
    custom_filters: Optional[Dict[str, Any]] = None
) -> Optional[Dict]:
    """
    Build a ChromaDB-compatible metadata filter.
    
    Args:
        category: Document category (vpn, password, email, etc.)
        doc_type: Type of document (how-to, troubleshooting, policy, faq)
        source_file_pattern: Regex pattern for source file names
        max_age_days: Maximum document age in days
        department: Department filter
        custom_filters: Additional custom filter conditions
        
    Returns:
        ChromaDB where clause dict or None if no filters
    """
    filters_enabled = os.getenv("METADATA_FILTER_ENABLED", "true").lower() == "true"
    
    if not filters_enabled:
        return None
    
    conditions = []
    
    # Category filter
    if category:
        conditions.append({"category": {"$eq": category}})
    
    # Document type filter
    if doc_type:
        conditions.append({"doc_type": {"$eq": doc_type}})
    
    # Department filter
    if department:
        conditions.append({"department": {"$eq": department}})
    
    # Source file pattern (uses $contains for partial match)
    if source_file_pattern:
        conditions.append({"source_file": {"$contains": source_file_pattern}})
    
    # Date filter (if documents have indexed_at timestamp)
    if max_age_days:
        cutoff = datetime.now() - timedelta(days=max_age_days)
        cutoff_str = cutoff.isoformat()
        conditions.append({"indexed_at": {"$gte": cutoff_str}})
    
    # Add any custom filters
    if custom_filters:
        for key, value in custom_filters.items():
            if isinstance(value, dict):
                conditions.append({key: value})
            else:
                conditions.append({key: {"$eq": value}})
    
    # Combine conditions with $and
    if not conditions:
        return None
    
    if len(conditions) == 1:
        return conditions[0]
    
    return {"$and": conditions}


def extract_metadata_hints(query: str) -> Dict[str, Any]:
    """
    Extract metadata filtering hints from query text.
    
    Looks for explicit mentions like:
    - "in the VPN guide"
    - "from last month"
    - "for new employees"
    
    Args:
        query: User query
        
    Returns:
        Dict of metadata hints that can be used for filtering
    """
    hints = {}
    query_lower = query.lower()
    
    # Detect category
    category = detect_category_from_query(query)
    if category:
        hints["category"] = category
    
    # Detect document type hints
    doc_type_patterns = {
        "how-to": r"how (to|do)|guide|tutorial|steps",
        "troubleshooting": r"fix|error|issue|problem|not working|broken",
        "policy": r"policy|rule|compliance|requirement|must|should",
        "faq": r"frequently|common question|faq",
    }
    
    for doc_type, pattern in doc_type_patterns.items():
        if re.search(pattern, query_lower):
            hints["doc_type"] = doc_type
            break
    
    # Detect recency hints
    recency_patterns = {
        7: r"recent|latest|this week|today",
        30: r"this month|last month|recent",
        90: r"this quarter|last quarter",
        365: r"this year|last year",
    }
    
    for days, pattern in recency_patterns.items():
        if re.search(pattern, query_lower):
            hints["max_age_days"] = days
            break
    
    # Detect audience hints
    if re.search(r"new employee|onboarding|first day|just started", query_lower):
        hints["audience"] = "new_employee"
    elif re.search(r"admin|administrator|it staff", query_lower):
        hints["audience"] = "admin"
    
    logger.debug(f"Extracted metadata hints: {hints}")
    return hints


def apply_filters_to_results(
    results: List[Dict],
    filters: Dict[str, Any]
) -> List[Dict]:
    """
    Post-filter results based on metadata (when pre-filtering isn't possible).
    
    Args:
        results: List of search results
        filters: Filter conditions
        
    Returns:
        Filtered results
    """
    if not filters:
        return results
    
    filtered = []
    for result in results:
        metadata = result.get("metadata", {})
        matches = True
        
        for key, condition in filters.items():
            if key.startswith("$"):
                continue  # Skip operators
            
            value = metadata.get(key)
            
            if isinstance(condition, dict):
                # Handle operators
                for op, op_value in condition.items():
                    if op == "$eq" and value != op_value:
                        matches = False
                    elif op == "$ne" and value == op_value:
                        matches = False
                    elif op == "$contains" and op_value not in str(value):
                        matches = False
                    elif op == "$gte" and value < op_value:
                        matches = False
                    elif op == "$lte" and value > op_value:
                        matches = False
            else:
                if value != condition:
                    matches = False
        
        if matches:
            filtered.append(result)
    
    logger.debug(f"Post-filtered {len(results)} results to {len(filtered)}")
    return filtered


def auto_filter_for_intent(intent: str, confidence: float = 0.0) -> Optional[Dict]:
    """
    Automatically generate metadata filter based on classified intent.
    
    Args:
        intent: Classified intent (e.g., "vpn_issue", "password_reset")
        confidence: Confidence score of intent classification
        
    Returns:
        Metadata filter dict or None
    """
    # Only apply auto-filtering with high confidence
    min_confidence = float(os.getenv("METADATA_FILTER_MIN_CONFIDENCE", "0.7"))
    
    if confidence < min_confidence:
        logger.debug(f"Intent confidence {confidence:.2f} below threshold {min_confidence}")
        return None
    
    # Intent to category mapping
    intent_categories = {
        "vpn_issue": "vpn",
        "vpn_setup": "vpn",
        "password_reset": "password",
        "password_policy": "password",
        "email_issue": "email",
        "software_install": "software",
        "hardware_issue": "hardware",
        "new_employee": "onboarding",
        "network_issue": "network",
    }
    
    category = intent_categories.get(intent)
    if category:
        return build_metadata_filter(category=category)
    
    return None


if __name__ == "__main__":
    # Test metadata filtering
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing category detection:")
    test_queries = [
        "How do I connect to VPN?",
        "My password is expired",
        "Outlook not syncing emails",
        "New laptop setup for new employee",
        "Fix printer connection error",
    ]
    
    for query in test_queries:
        category = detect_category_from_query(query)
        print(f"  '{query[:40]}...' -> {category}")
    
    print("\nTesting metadata hint extraction:")
    query = "How do I fix the recent VPN connection issue?"
    hints = extract_metadata_hints(query)
    print(f"  Query: {query}")
    print(f"  Hints: {hints}")
    
    print("\nTesting filter building:")
    filter_dict = build_metadata_filter(
        category="vpn",
        doc_type="troubleshooting",
        max_age_days=30
    )
    print(f"  Filter: {filter_dict}")
