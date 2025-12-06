"""
Intent classification utilities for categorizing user queries.

Binary classification:
- informative: User wants information, explanations, how-to guides, contact info
- troubleshooting: User has a problem/error they need help fixing
"""
from __future__ import annotations

import re
from typing import Literal


IntentType = Literal["informative", "troubleshooting"]


# Use frozensets for O(1) lookup
TROUBLESHOOTING_KEYWORDS = frozenset([
    "error", "issue", "problem", "not working", "broken", "failed",
    "can't", "cannot", "unable", "doesn't work", "won't", "crash",
    "bug", "fix", "troubleshoot", "debug", "solve", "help me",
    "stuck", "freeze", "freezing", "slow", "hang", "hanging",
    "fail", "failing", "stopped", "disconnect", "disconnected"
])

# Action words that suggest troubleshooting context
TROUBLESHOOTING_ACTIONS = frozenset([
    "reset", "restart", "reinstall", "update", "configure",
    "change", "modify", "adjust", "repair", "recover"
])

# Informative patterns - user seeking information, not fixing a problem
INFORMATIVE_KEYWORDS = frozenset([
    "what is", "what are", "who is", "when is", "why", "explain",
    "define", "describe", "tell me about", "information about",
    "where is", "where can i find", "how do i access", "show me",
    "how to", "how do i", "navigate to", "go to", "find", "locate",
    "who to contact", "who should i contact", "contact for"
])


def classify_intent(query: str) -> str:
    """
    Classify user query intent using rule-based heuristics.
    
    Args:
        query: User query text
    
    Returns:
        Intent type: 'informative' or 'troubleshooting'
    """
    query_lower = query.lower().strip()
    
    troubleshooting_score = 0.0
    informative_score = 0.0
    
    # Check troubleshooting signals
    for keyword in TROUBLESHOOTING_KEYWORDS:
        if keyword in query_lower:
            troubleshooting_score += 1.0
    
    for keyword in TROUBLESHOOTING_ACTIONS:
        if keyword in query_lower:
            troubleshooting_score += 0.5
    
    # Check informative signals
    for keyword in INFORMATIVE_KEYWORDS:
        if keyword in query_lower:
            informative_score += 1.0
    
    # Question mark without troubleshooting keywords suggests informative
    if "?" in query and troubleshooting_score == 0:
        informative_score += 0.3
    
    # Default to informative for ambiguous queries
    # (safer to provide info than start troubleshooting)
    if troubleshooting_score == 0 and informative_score == 0:
        return "informative"
    
    # Troubleshooting wins if it has higher score
    if troubleshooting_score > informative_score:
        return "troubleshooting"
    
    return "informative"


def is_greeting(query: str) -> bool:
    """Check if query is a greeting."""
    greetings = frozenset(["hello", "hi", "hey", "good morning", "good afternoon", "good evening"])
    query_lower = query.lower().strip()
    return any(greeting in query_lower for greeting in greetings)


def is_farewell(query: str) -> bool:
    """Check if query is a farewell."""
    farewells = frozenset(["bye", "goodbye", "see you", "thanks", "thank you"])
    query_lower = query.lower().strip()
    return any(farewell in query_lower for farewell in farewells)


def extract_keywords(query: str) -> list[str]:
    """
    Extract potential keywords from query.
    Removes stopwords and common words.
    """
    stopwords = frozenset([
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "can", "i", "you", "he", "she",
        "it", "we", "they", "this", "that", "these", "those", "to", "from",
        "in", "on", "at", "by", "for", "with", "about", "as", "of"
    ])
    
    # Tokenize and clean
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    
    return keywords


if __name__ == "__main__":
    # Test intent classification
    test_queries = [
        # Informative queries
        ("What is VPN and how does it work?", "informative"),
        ("Where can I find the IT support portal?", "informative"),
        ("How do I reset my password?", "informative"),
        ("Explain Active Directory", "informative"),
        ("Who should I contact about router issues?", "informative"),
        ("Show me how to configure Outlook", "informative"),
        # Troubleshooting queries
        ("My email is not working, I keep getting an error", "troubleshooting"),
        ("The printer is broken and won't print", "troubleshooting"),
        ("VPN keeps disconnecting", "troubleshooting"),
        ("Outlook is crashing when I open it", "troubleshooting"),
        ("Can't connect to wifi", "troubleshooting"),
    ]
    
    print("Intent Classification Tests:")
    print("-" * 60)
    for query, expected in test_queries:
        intent = classify_intent(query)
        status = "✓" if intent == expected else "✗"
        print(f"{status} Query: {query}")
        print(f"  Expected: {expected}, Got: {intent}")
