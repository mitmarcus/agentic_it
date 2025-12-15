from cremedelacreme import Node
from typing import Dict
from .config import _RAG_CONFIG, _FEATURE_FLAGS
from utils.logger import get_logger
from utils.query_expansion import expand_query, generate_hypothetical_answer
from utils.call_llm import call_llm
from core.llm_config import get_llm_config
from utils.conversation_memory import conversation_memory
from utils.embedding_local import get_embedding

# ============================================================================
# Node 2: EmbedQueryNode
# ============================================================================

logger = get_logger(__name__)

class EmbedQueryNode(Node):
    """Generate embedding for user query with follow-up detection and context enrichment.

    - Detects follow-up queries and enriches them with active topic context
    - Includes conversation context for better retrieval on follow-up questions
    - Can use LLM to generate alternative phrasings for better recall
    - Supports HyDE (Hypothetical Document Embeddings) for complex queries
    """
    
    # Follow-up indicators for local detection (no LLM needed)
    FOLLOW_UP_INDICATORS = frozenset([
        # OS mentions
        "windows", "linux", "mac", "macos", "ubuntu", "debian", "fedora", "ios", "android",
        # Confirmations  
        "yes", "yeah", "yep", "correct", "right", "exactly",
        # Negations
        "no", "nope", "didnt work", "doesnt work", "still not working",
        # Version/specifics
        "version", "365", "2019", "2021", "2023", "chrome", "firefox", "edge", "safari",
        # Continuation signals
        "what about", "how about", "also", "instead", "other",
    ])
    
    NEW_TOPIC_STARTERS = ("how to", "how do i", "what is", "where is", "who is", "can i", "why is", "when")
    GREETINGS = frozenset(["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye"])
    
    def __init__(self):
        # Retry 3 times if embedding fails
        config = get_llm_config()
        super().__init__(max_retries=config.max_retries, wait=config.retry_wait)
    
    def prep(self, shared: Dict) -> Dict:
        """Read user query, keywords, conversation context, and active topic."""
        session_id = shared.get("session_id", "")
        # Get recent conversation for context (last 2-3 exchanges)
        history = conversation_memory.get_formatted_history(session_id, limit=4, exclude_last=True)
        # Get active topic for follow-up detection
        active_topic = conversation_memory.get_active_topic(session_id)
        
        return {
            "query": shared.get("user_query", ""),
            "session_id": session_id,
            "keywords": shared.get("keywords", []),
            "conversation_context": history,
            "active_topic": active_topic,
            "turn_count": shared.get("turn_count", 1)
        }
    
    def exec(self, prep_data: Dict) -> Dict:
        """Generate embedding vector with follow-up detection and context enrichment."""
        query = prep_data["query"]
        conversation_context = prep_data.get("conversation_context", "")
        active_topic = prep_data.get("active_topic")
        turn_count = prep_data.get("turn_count", 1)
        
        if not query:
            raise ValueError("Empty query for embedding")
        
        # Detect if this is a follow-up and enrich query accordingly
        is_follow_up = False
        query_with_context = query
        
        if turn_count > 1:
            # Check if this is a follow-up to active topic
            if active_topic and self._is_follow_up(query):
                topic_text = active_topic.get("topic", "")
                if topic_text:
                    query_with_context = f"{topic_text} - {query}"
                    is_follow_up = True
                    logger.info(f"Follow-up detected: '{query}' -> '{query_with_context}'")
            
            # Fallback: extract terms from conversation if no active topic match
            elif conversation_context and not is_follow_up:
                context_terms = self._extract_context_terms(conversation_context)
                if context_terms:
                    query_with_context = f"{context_terms} {query}"
                    logger.debug(f"Enriched with context terms: {query_with_context[:100]}...")
        
        # Use cached feature flags (avoid per-request env reads)
        expansion_enabled = _FEATURE_FLAGS["query_expansion"]
        hyde_enabled = _FEATURE_FLAGS["hyde"]
        
        # Apply HyDE if enabled (generates hypothetical answer to embed)
        if hyde_enabled:
            logger.debug("Applying HyDE (Hypothetical Document Embeddings)...")
            enhanced_query = generate_hypothetical_answer(query_with_context, call_llm_func=call_llm)
        elif expansion_enabled:
            logger.debug("Applying query expansion...")
            expanded = expand_query(query_with_context, num_expansions=2, call_llm_func=call_llm)
            enhanced_query = " ".join(expanded)
        else:
            enhanced_query = query_with_context
        
        # Generate embedding for the (possibly enhanced) query
        embedding = get_embedding(enhanced_query)
        logger.debug(f"Generated embedding: {len(embedding)} dimensions")
        
        # Log what was actually embedded (for debugging)
        if enhanced_query != query:
            logger.info(f"Embedded enhanced query: '{enhanced_query[:150]}...'")
        if is_follow_up:
            logger.info(f"Follow-up query detected and processed")
        
        return {
            "embedding": embedding
        }
    
    def _is_follow_up(self, query: str) -> bool:
        """Detect if query is a follow-up (local, no LLM)."""
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        # New topic = not a follow-up
        if any(query_lower.startswith(starter) for starter in self.NEW_TOPIC_STARTERS):
            return False
        
        # Greetings = not a follow-up
        if query_lower in self.GREETINGS:
            return False
        
        # Short query (1-3 words) that's not a greeting = likely follow-up
        if len(words) <= 3:
            return True
        
        # Short query (≤5 words) with follow-up indicator = follow-up
        if len(words) <= 5:
            return any(indicator in query_lower for indicator in self.FOLLOW_UP_INDICATORS)
        
        return False
    
    def _extract_context_terms(self, conversation_context: str) -> str:
        """Extract key technical terms from conversation context for query enrichment."""
        import re
        
        # Find potential app/product names (capitalized words)
        caps_words = re.findall(r'\b[A-Z][a-z]+\b', conversation_context)
        # Common IT terms to preserve
        it_terms = ["outlook", "calendar", "email", "vpn", "wifi", "printer", "password", 
                    "connection", "connections", "error", "network", "teams", "office", "mac address"]
        
        found_terms = []
        context_lower = conversation_context.lower()
        for term in it_terms:
            if term in context_lower and term not in [t.lower() for t in found_terms]:
                found_terms.append(term)
        
        # Add capitalized words (likely app names)
        for word in caps_words[:3]:
            if word.lower() not in [t.lower() for t in found_terms]:
                found_terms.append(word)
        
        return " ".join(found_terms[:5])
    
    def exec_fallback(self, prep_res: Dict, exc: Exception) -> Dict:
        """Fallback: return zero vector if embedding fails."""
        logger.error(f"Embedding failed after retries: {exc}")
        embed_dim = _RAG_CONFIG["embedding_dim"]
        return {
            "embedding": [0.0] * embed_dim
        }
    
    def post(self, shared: Dict, prep_res: Dict, exec_res: Dict) -> str:
        """Write embedding to shared store."""
        shared["query_embedding"] = exec_res["embedding"]
        return "default"
