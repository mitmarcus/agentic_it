"""
Intent classification utilities for categorizing user queries.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Literal


IntentType = Literal["factual", "troubleshooting", "navigation"]


def classify_intent(query: str) -> Dict[str, Any]:
    """
    Classify user query intent using rule-based heuristics.
    
    Args:
        query: User query text
    
    Returns:
        Dict with intent type and confidence score
    """
    query_lower = query.lower().strip()
    
    # Troubleshooting patterns
    troubleshooting_keywords = [
        "error", "issue", "problem", "not working", "broken", "failed",
        "can't", "cannot", "unable", "doesn't work", "won't", "crash",
        "bug", "fix", "troubleshoot", "debug", "solve", "help me"
    ]
    
    # Navigation patterns
    navigation_keywords = [
        "where is", "where can i find", "how do i access", "show me",
        "navigate to", "go to", "find", "locate", "search for"
    ]
    
    # Question words that indicate factual queries
    factual_keywords = [
        "what is", "what are", "who is", "when is", "why", "explain",
        "define", "describe", "tell me about", "information about"
    ]
    
    # Action words that often indicate troubleshooting
    action_keywords = [
        "reset", "restart", "reinstall", "update", "configure",
        "change", "modify", "adjust"
    ]
    
    troubleshooting_score = 0.0
    navigation_score = 0.0
    factual_score = 0.0
    
    # Check for signals
    for keyword in troubleshooting_keywords:
        if keyword in query_lower:
            troubleshooting_score += 1.0
    
    for keyword in action_keywords:
        if keyword in query_lower:
            troubleshooting_score += 0.5
    
    for keyword in navigation_keywords:
        if keyword in query_lower:
            navigation_score += 1.0
    
    for keyword in factual_keywords:
        if keyword in query_lower:
            factual_score += 1.0
    
    # Question mark often indicates factual query
    if "?" in query:
        if troubleshooting_score == 0:
            factual_score += 0.3
    
    # Normalize scores
    total = troubleshooting_score + navigation_score + factual_score
    if total == 0:
        # Default to factual for ambiguous queries
        return {
            "intent": "factual",
            "confidence": 0.5
        }
    
    troubleshooting_score /= total
    navigation_score /= total
    factual_score /= total
    
    # Determine primary intent
    if troubleshooting_score > navigation_score and troubleshooting_score > factual_score:
        intent = "troubleshooting"
        confidence = troubleshooting_score
    elif navigation_score > factual_score:
        intent = "navigation"
        confidence = navigation_score
    else:
        intent = "factual"
        confidence = factual_score
    
    return {
        "intent": intent,
        "confidence": confidence
    }


def is_greeting(query: str) -> bool:
    """Check if query is a greeting."""
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    query_lower = query.lower().strip()
    return any(greeting in query_lower for greeting in greetings)


def is_farewell(query: str) -> bool:
    """Check if query is a farewell."""
    farewells = ["bye", "goodbye", "see you", "thanks", "thank you"]
    query_lower = query.lower().strip()
    return any(farewell in query_lower for farewell in farewells)


def extract_keywords(query: str) -> list[str]:
    """
    Extract potential keywords from query.
    Removes stopwords and common words.
    """
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "can", "i", "you", "he", "she",
        "it", "we", "they", "this", "that", "these", "those", "to", "from",
        "in", "on", "at", "by", "for", "with", "about", "as", "of"
    }
    
    # Tokenize and clean
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    
    return keywords


if __name__ == "__main__":
    # Test intent classification
    test_queries = [
        "What is VPN and how does it work?",
        "My email is not working, I keep getting an error",
        "Where can I find the IT support portal?",
        "How do I reset my password?",
        "Explain Active Directory",
        "The printer is broken and won't print",
        "Show me how to configure Outlook"
    ]
    
    for query in test_queries:
        result = classify_intent(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
