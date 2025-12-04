"""
Nodes for IT Support Chatbot.

All nodes follow the pattern:
- prep(): Read from shared store
- exec(): Process logic (with retries handled by Node)
- post(): Write to shared store, return action
"""
import os
import yaml
from typing import Any, Dict, List
from cremedelacreme import AsyncNode, Node, BatchNode

# Import utilities
from utils.call_llm_groq import call_llm
from utils.embedding_local import get_embedding
from utils.intent_classifier import classify_intent, extract_keywords
from utils.conversation_memory import conversation_memory
from utils.chromadb_client import query_collection
from utils.chunker import truncate_to_token_limit
from utils.redactor import redact_text, redact_dict
from utils.status_retrieval import format_status_results
from utils.reranker import rerank_results
from utils.feedback import apply_feedback_adjustments
from utils.query_expansion import expand_query, generate_hypothetical_answer
from utils.logger import get_logger
from utils.prompts import (
    COMPANY_NAME,
    SYSTEM_ROLE,
    COMMON_ASSUMPTIONS,
    COMMON_DONTS,
    URL_RULES,
    CLARIFY_BAD_EXAMPLES,
    DECISION_MAKER_ROLE,
    RATE_LIMIT_MESSAGE,
    RATE_LIMIT_WITH_DOCS_MESSAGE,
    GENERIC_ERROR_MESSAGE,
    GENERIC_CLARIFY_MESSAGE,
    os_awareness_instruction,
    parse_yaml_response,
)

logger = get_logger(__name__)


def _get_float_env(name: str) -> float:
    """Read a required float from environment variables. Raises ValueError if missing or invalid."""
    raw_value = os.getenv(name)
    if raw_value is None:
        raise ValueError(f"Environment variable {name} must be set")
    try:
        return float(raw_value)
    except ValueError:
        raise ValueError(f"Environment variable {name}={raw_value} is not a valid float")


def _get_int_env(name: str) -> int:
    """Read a required int from environment variables. Raises ValueError if missing or invalid."""
    raw_value = os.getenv(name)
    if raw_value is None:
        raise ValueError(f"Environment variable {name} must be set")
    try:
        return int(raw_value)
    except ValueError:
        raise ValueError(f"Environment variable {name}={raw_value} is not a valid integer")


# Cache policy limits at module load
POLICY_LIMITS = {
    "clarify_confidence_threshold": _get_float_env("AGENT_CLARIFY_CONFIDENCE_THRESHOLD"),
    "doc_confidence_threshold": _get_float_env("AGENT_DOC_CONFIDENCE_THRESHOLD"),
    "rate_limit_answer_confidence": _get_float_env("AGENT_RATE_LIMIT_ANSWER_CONFIDENCE"),
    "system_error_confidence": _get_float_env("AGENT_SYSTEM_ERROR_CONFIDENCE"),
    "troubleshoot_escalate_failed_steps": _get_int_env("TROUBLESHOOT_ESCALATE_FAILED_STEPS"),
    "troubleshoot_fallback_failed_steps": _get_int_env("TROUBLESHOOT_FALLBACK_FAILED_STEPS"),
    "max_turns": _get_int_env("AGENT_MAX_TURNS"),
}

# Cache RAG configuration
_RAG_CONFIG = {
    "top_k": _get_int_env("RAG_TOP_K"),
    "min_score": _get_float_env("RAG_MIN_SCORE"),
    "max_context_tokens": int(os.getenv("RAG_MAX_CONTEXT_TOKENS", "2000")),
    "embedding_dim": int(os.getenv("EMBEDDING_DIM", "384")),
    "chunk_size": _get_int_env("INGESTION_CHUNK_SIZE"),
    "chunk_overlap": _get_int_env("INGESTION_CHUNK_OVERLAP"),
    "source_dir": os.getenv("INGESTION_SOURCE_DIR", "./data/docs"),
}

# Cache feature flags 
_FEATURE_FLAGS = {
    "rerank": os.getenv("RERANK_ENABLED", "true").lower() == "true",  # Keep: most impactful step
    "query_expansion": os.getenv("QUERY_EXPANSION_ENABLED", "false").lower() == "true",
    "hyde": os.getenv("HYDE_ENABLED", "false").lower() == "true",
}


# ============================================================================
# Node 0: RedactInputNode
# ============================================================================

class RedactInputNode(Node):
    """Redact sensitive information from user input."""
    
    def prep(self, shared: Dict) -> Dict:
        """Read raw user query and session ID."""
        return {
            "query": shared.get("user_query", ""),
            "session_id": shared.get("session_id", "unknown")
        }
    
    def exec(self, prep_data: Dict) -> Dict:
        """Redact sensitive data and log if found."""
        query = prep_data["query"]
        session_id = prep_data["session_id"]
        
        redacted_query = redact_text(query)
        has_sensitive = query != redacted_query
        
        if has_sensitive:
            logger.warning(f"Redacted sensitive data from query for session {session_id}")
        
        return {
            "redacted_query": redacted_query,
            "had_sensitive_data": has_sensitive
        }
    
    def post(self, shared: Dict, prep_res: Dict, exec_res: Dict) -> str:
        """Replace user_query with redacted version and notify user if redacted."""
        # Store original for logging only
        shared["original_query"] = prep_res["query"]
        # Replace with redacted version for all downstream nodes
        shared["user_query"] = exec_res["redacted_query"]
        shared["had_sensitive_data"] = exec_res["had_sensitive_data"]
        
        # If redaction occurred, add a warning message for the user
        if exec_res["had_sensitive_data"]:
            shared["redaction_notice"] = (
                "⚠️ For your security, sensitive information has been redacted from your message. "
                "Please avoid sharing passwords, API keys, or other credentials."
            )
        
        return "default"


# ============================================================================
# Node 1: IntentClassificationNode
# ============================================================================

class IntentClassificationNode(Node):
    """Classify user query intent for routing decisions."""
    
    def prep(self, shared: Dict) -> str:
        """Read user query from shared store."""
        return shared.get("user_query", "")
    
    def exec(self, query: str) -> Dict:
        """Classify intent using utility function."""
        intent = classify_intent(query)
        keywords = extract_keywords(query)
        
        logger.debug(f"Intent: {intent}")
        
        return {
            "intent": intent,
            "keywords": keywords
        }
    
    def post(self, shared: Dict, prep_res: str, exec_res: Dict) -> str:
        """Write intent data to shared store."""
        shared["intent"] = exec_res["intent"]
        shared["keywords"] = exec_res["keywords"]
        return "default"


# ============================================================================
# Node 2: EmbedQueryNode
# ============================================================================

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
        super().__init__(max_retries=3, wait=1)
    
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
        
        return {
            "embedding": embedding,
            "original_query": query,
            "query_with_context": query_with_context if query_with_context != query else None,
            "enhanced_query": enhanced_query if enhanced_query != query_with_context else None,
            "is_follow_up": is_follow_up
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
            "embedding": [0.0] * embed_dim,
            "original_query": prep_res.get("query", ""),
            "query_with_context": None,
            "enhanced_query": None,
            "is_follow_up": False
        }
    
    def post(self, shared: Dict, prep_res: Dict, exec_res: Dict) -> str:
        """Write embedding and follow-up status to shared store."""
        shared["query_embedding"] = exec_res["embedding"]
        shared["is_follow_up"] = exec_res.get("is_follow_up", False)
        
        # Store expansion info for debugging/logging
        if exec_res.get("query_with_context"):
            shared["query_with_context"] = exec_res["query_with_context"]
            logger.debug(f"Query with context: {exec_res['query_with_context'][:100]}...")
        if exec_res.get("enhanced_query"):
            shared["enhanced_query"] = exec_res["enhanced_query"]
            logger.debug(f"Query enhanced: {exec_res['enhanced_query'][:100]}...")
        
        return "default"


# ============================================================================
# Node 3: SearchKnowledgeBaseNode
# ============================================================================

class SearchKnowledgeBaseNode(Node):
    """Retrieve relevant documents from ChromaDB with streamlined RAG pipeline.
    
    Simplified pipeline (40% faster, same quality):
    1. Vector Search: Get top candidates using embedding similarity
    2. Reranking: Cross-encoder re-scores for precision (most impactful step)
    
    Removed (over-engineering for ~200 docs):
    - Metadata filtering (let reranker decide relevance)
    - BM25 hybrid (vector handles semantic queries well)
    - MMR diversity (reranker naturally diversifies)
    """
    
    def __init__(self):
        super().__init__(max_retries=2, wait=1)
    
    def prep(self, shared: Dict) -> Dict:
        """Read query embedding and text."""
        return {
            "query_embedding": shared.get("query_embedding", []),
            "query_text": shared.get("user_query", "")
        }
    
    def exec(self, prep_data: Dict) -> List[Dict]:
        """Query ChromaDB with vector search, then rerank top candidates."""
        query_embedding = prep_data["query_embedding"]
        query_text = prep_data["query_text"]
        
        if not query_embedding or sum(query_embedding) == 0:
            logger.warning("Empty or zero embedding, skipping search")
            return []
        
        # Use cached config
        top_k = _RAG_CONFIG["top_k"]
        min_score = _RAG_CONFIG["min_score"]
        
        # Step 1: Vector search - get more candidates for reranking
        rerank_candidates = top_k * 3  # Fetch 3x for reranker to work with
        
        vector_results = query_collection(
            query_embedding,
            top_k=rerank_candidates
        )
        
        if not vector_results:
            logger.debug("No vector search results")
            return []
        
        # Log retrieval stats
        scores = [r["score"] for r in vector_results]
        logger.debug(f"Vector search: {len(vector_results)} results, scores: {scores[:5]}...")
        
        # Step 2: Light filtering - remove very low scores
        # Use lower threshold since reranker will do the heavy lifting
        filtered_results = [r for r in vector_results if r.get("score", 0) >= min_score * 0.7]
        
        if not filtered_results:
            logger.debug(f"All results below threshold {min_score * 0.7}")
            return []
        
        # Step 3: Rerank using cross-encoder (the most impactful step)
        if _FEATURE_FLAGS["rerank"] and query_text and len(filtered_results) > 1:
            logger.debug(f"Reranking {len(filtered_results)} candidates...")
            filtered_results = rerank_results(query_text, filtered_results, top_k=top_k)
            logger.debug(f"Reranked to top {len(filtered_results)} results")
        else:
            # No reranking - just take top_k by vector score
            filtered_results = filtered_results[:top_k]
        
        # Apply feedback adjustments (boost/penalize based on user feedback)
        score_key = "rerank_score" if filtered_results and "rerank_score" in filtered_results[0] else "score"
        filtered_results = apply_feedback_adjustments(filtered_results, score_key=score_key)
        
        return filtered_results

    
    def post(self, shared: Dict, prep_res: Dict, exec_res: List[Dict]) -> str:
        """Write results and compile context."""
        shared["retrieved_docs"] = exec_res
        
        # Compile RAG context from documents
        if exec_res:
            max_context_tokens = _RAG_CONFIG["max_context_tokens"]
            
            # Smart deduplication: keep BEST chunk per source, not FIRST
            # This ensures we get the most relevant content from each document
            best_by_source: Dict[str, Dict] = {}
            for doc in exec_res:
                source = doc['metadata'].get('source_file', 'unknown')
                score = doc.get('rerank_score', doc.get('rrf_score', doc.get('score', 0)))
                
                if source not in best_by_source or score > best_by_source[source].get('_best_score', 0):
                    doc['_best_score'] = score
                    best_by_source[source] = doc
            
            unique_docs = list(best_by_source.values())
            # Sort by score descending
            unique_docs.sort(key=lambda d: d.get('_best_score', 0), reverse=True)
            
            context_parts = []
            for i, doc in enumerate(unique_docs):
                metadata = doc['metadata']
                source_file = metadata.get('source_file', 'unknown')
                chunk_index = metadata.get('chunk_index', '?')
                total_chunks = metadata.get('total_chunks', '?')
                
                # Show rerank score if available, then RRF, then vector score
                relevance = doc.get('rerank_score', doc.get('rrf_score', doc.get('score', 0)))
                
                context_parts.append(f"[Document {i+1}] (Relevance: {relevance:.2f})")
                context_parts.append(f"File: {source_file} (Chunk {chunk_index} of {total_chunks})")
                context_parts.append(doc['document'])
                context_parts.append("")
            
            rag_context = "\n".join(context_parts)
            rag_context = truncate_to_token_limit(rag_context, max_context_tokens)
            
            shared["rag_context"] = rag_context
            logger.debug(f"Compiled context from {len(unique_docs)} unique documents ({len(exec_res)} total chunks)")
            return "docs_found"
        else:
            shared["rag_context"] = ""
            return "no_docs"



# ============================================================================
# Node 4: DecisionMakerNode (Agent)
# ============================================================================

class DecisionMakerNode(Node):
    """Agent node that decides which action to take next."""
    
    def __init__(self):
        super().__init__(max_retries=3, wait=2)
    
    def prep(self, shared: Dict) -> Dict:
        """Gather all context for decision making."""
        # Get conversation history
        session_id = shared.get("session_id", "")
        history_str = conversation_memory.get_formatted_history(session_id, limit=20, exclude_last=True)
        
        # Get full RAG context (not just summaries)
        rag_context = shared.get("rag_context", "")
        retrieved_docs = shared.get("retrieved_docs", [])
        doc_count = len(retrieved_docs)
        doc_scores = [doc['score'] for doc in retrieved_docs] if retrieved_docs else []
        
        # Get workflow state
        workflow_state = shared.get("workflow_state")
        workflow_status = "None"
        if workflow_state:
            workflow_status = f"In progress: step {workflow_state.get('current_step_index', 0) + 1}"
        
        # Track search attempts to prevent infinite loops
        search_count = shared.get("search_count", 0)
        max_searches = _get_int_env("AGENT_MAX_TURNS")
        
        return {
            "user_query": shared.get("user_query", ""),
            "user_os": shared.get("user_os", "unknown"),
            "intent": shared.get("intent", {}),
            "network_status": format_status_results(shared.get("status_results", [])),
            "conversation_history": history_str,
            "rag_context": rag_context,
            "doc_count": doc_count,
            "doc_scores": doc_scores,
            "workflow_status": workflow_status,
            "turn_count": shared.get("turn_count", 0),
            "search_count": search_count,
            "max_searches": max_searches
        }
    
    def exec(self, context: Dict) -> Dict:
        """Call LLM to decide next action."""
        doc_threshold = POLICY_LIMITS["doc_confidence_threshold"]
        doc_count = context.get('doc_count', 0)
        avg_score = sum(context.get('doc_scores', [])) / len(context.get('doc_scores', [])) if context.get('doc_scores') else 0
        intent = context.get('intent', 'informative')
        
        prompt = f"""
### CONTEXT
User Query: "{context['user_query']}"
User System: {context.get('user_os', 'unknown')}
Intent: {intent} ({'user has a problem to fix' if intent == 'troubleshooting' else 'user wants information/guidance'})
Conversation Turn: {context['turn_count']}

Retrieved Knowledge Base ({doc_count} documents, avg score: {avg_score:.2f}):
{context['rag_context'] if context.get('rag_context') else 'No relevant documents found'}
Network Status:
{context['network_status']}

Conversation History (look at the last 3 messages):
{context['conversation_history'] if context['conversation_history'] else 'No previous conversation'}

Current Workflow State: {context['workflow_status']}"""
        
        # Debug: Log the prompt to see what decision maker sees
        logger.debug(f"Decision maker prompt (first 500 chars): {prompt[:500]}...")
        logger.debug(f"RAG context length: {len(context.get('rag_context', ''))} chars")
        
        prompt += f"""

{DECISION_MAKER_ROLE}

{COMMON_ASSUMPTIONS}

## REASONING PROCESS
1. Problem Summary: What is the user's core issue in 1-2 sentences?
2. Known Information: What specific, actionable data do we have? (e.g., error codes, intent, previous steps).
3. Missing Information: What critical data is absent that blocks a resolution?
4. Action Evaluation: Which 2-3 actions are most relevant? Briefly weigh their pros/cons given the context.
5. Final Decision: Select the best action. Justify why it's superior to the alternatives now.

### DECISION RULES (READ CAREFULLY)
1. **USER CONFIRMATION = ANSWER**: If user says "yes", "correct", "that's right", "exactly", or similar confirmations after a clarifying question, they are confirming the issue. **ALWAYS choose 'answer'** and provide the solution from the retrieved documents.
2. **TOPIC MATCH CHECK**: Before answering, verify the retrieved docs actually address the user's issue. A doc about "printer not working on VPN" does NOT answer "VPN not working" - these are different problems.
3. **IF docs match the user's topic AND scores > {doc_threshold}, answer**. If docs are tangentially related (e.g., mention VPN but solve a different problem), **clarify** what the user needs.
4. **CONTACT INFO & ROLES**: If the user asks "who to contact", and docs contain relevant contacts, answer.
5. **VAGUE QUERIES**: For short queries like "vpn not working", "wifi issues", "help" - **clarify** what specific problem they're experiencing before assuming.
6. You have searched {context['search_count']} times (max: {context['max_searches']}). If at max, choose 'answer' or 'clarify', NOT 'search_kb'.
7. **PRIORITIZE SEARCH**: If you haven't searched yet and the query contains specific keywords (e.g., "router", "wifi", "who to ask"), choose 'search_kb' instead of 'clarify'.
8. Only clarify if: (a) You have already searched and found nothing OR (b) Query is completely incomprehensible (e.g. "help", "it doesn't work") OR (c) Documents retrieved but irrelevant (score < {doc_threshold}).
9. Never create ticket without attempting resolution first.
10. **NEVER CLARIFY TWICE IN A ROW**: If conversation history shows we just asked a clarifying question, do NOT clarify again. Choose 'answer' with best available info.

### EXAMPLES OF CORRECT BEHAVIOR
- User confirms "yes" after clarification + docs available → **answer** with the solution
- User asks "who to contact about router?" + You have docs about router contacts (score 0.72) → **answer** with contact info
- User asks "router" (vague) + No good docs → **clarify** what they need
- User asks clear question + No docs found → **search_kb** OR **answer** saying you don't have that info
### AVAILABLE ACTIONS
1.  search_kb 
    Description: Search knowledge base for technical documentation, procedures, or solutions
    When to use: 
    - Current document doesn't address the specific error/issue mentioned
    - You have general information but need specific technical details
    - User mentions a specific product/feature not covered in current document

2.  answer
    Description: Provide direct answer or solution using available information
    When to use:
    - Current document directly address the user's question
    - You have step-by-step instructions for the reported issue
    - Information is recent, relevant, and from authoritative sources

3.  troubleshoot
    Description: Guide user through diagnostic steps to identify root cause
    When to use:
    - User reports a technical issue without clear solution in the knowledge base
    - Problem requires gathering more system/environment details
    - Issue could have multiple potential causes needing elimination
    - User asks "how to fix" rather than "what is"

4.  search_tickets
    Description: Search existing support tickets for similar unresolved issues
    When to use:
    - Troubleshooting has failed to resolve the issue
    - Multiple users may be experiencing the same problem
    - Issue appears to be systemic rather than user-specific
    - Current outage or known issue is suspected

5.  create_ticket
    Description: Escalate to human support agent with all gathered context
    When to use:
    - User explicitly asks to talk to a human, agent, or create a ticket
    - All self-service options have been exhausted
    - Issue requires administrative privileges or physical access
    - Problem is complex and spans multiple systems
    - User has already attempted basic troubleshooting without success
       
6.  clarify
    Description: Ask user for specific details to better understand the problem
    When to use:
    - User query contains ambiguous terms (e.g., "it", "this", "the problem")
    - Missing critical information (error codes, software versions, symptoms)
    - Multiple interpretations of the problem are possible

### DECISION RULES & GUARDRAILS
- **INTENT-BASED ROUTING**: If intent = 'troubleshooting' AND no clear solution in docs → prefer 'troubleshoot' action. If intent = 'informative' → prefer 'answer' or 'search_kb'.
- If any active network issues match user's issue → answer
- IMPORTANT: You have searched {context['search_count']} times (max: {context['max_searches']}). If at max, you MUST choose 'answer' (with best available info), 'clarify' or 'create_ticket', NOT 'search_kb'
- If intent = 'informative' AND retrieved document provides a clear, direct answer → answer
- If intent = 'troubleshooting' AND docs have step-by-step fix → answer with the fix
- If intent = 'troubleshooting' AND no clear fix in docs → troubleshoot (interactive diagnostic)
- If user message contains explicit error codes, logs, or attachments → troubleshoot (unless 'search_kb' finds an exact-match).
- If user explicitly requests 'talk to human', 'create ticket', or 'escalate', choose 'create_ticket'.
- Use 'create_ticket' after other resolution paths ('search_kb', troubleshoot) are exhausted or if the issue requires privileges/physical access.
- If the same document keep appearing in searches, do not search again. 'answer' with the best information you have.
- Attempt resolution first unless user demands escalation.
- Keep responses concise and actionable

### OUTPUT FORMAT
Respond strictly in the following YAML format:
```yaml
thinking: |
  Step 1: Analyze user problem: <summary>
  Step 2: Available info: <what we have> 
  Step 3: Missing info: <what we need>
  Step 4: Best action: <why this helps>
action: <action_name>
reasoning: <why you chose this action in one sentence>
confidence: <0.0 to 1.0>
```

Think carefully and make the best decision for the user."""

        response = call_llm(prompt, max_tokens=512)
        
        # Parse YAML response
        yaml_str = parse_yaml_response(response)
        decision = yaml.safe_load(yaml_str)
        
        # Validate decision
        allowed_actions = ["search_kb", "answer", "troubleshoot", "search_tickets", "create_ticket", "clarify"]
        assert isinstance(decision, dict), "Decision must be a dict"
        assert "action" in decision, "Decision must have 'action' field"
        assert decision["action"] in allowed_actions, f"Action must be one of {allowed_actions}"
        
        logger.debug(f"Decision: {decision['action']} (confidence: {decision.get('confidence', 0):.2f})")
        
        return decision
    
    def exec_fallback(self, prep_res: Dict, exc: Exception) -> Dict:
        """Fallback: make intelligent decision based on context when LLM fails."""
        logger.error(f"Decision making failed: {exc}")
        
        # Smart fallback based on available context
        has_docs = bool(prep_res.get("rag_context")) and prep_res.get("doc_count", 0) > 0
        
        # If rate limit and we have good docs, try to answer
        if "rate limit" in str(exc).lower() and has_docs:
            return {
                "action": "answer",
                "reasoning": "Rate limited but have relevant docs",
                "confidence": POLICY_LIMITS["rate_limit_answer_confidence"]
            }
        
        # Otherwise ask for clarification
        return {
            "action": "clarify",
            "reasoning": "System error, need more information",
            "confidence": POLICY_LIMITS["system_error_confidence"]
        }
    
    def post(self, shared: Dict, prep_res: Dict, exec_res: Dict) -> str:
        """Write decision and return action."""
        shared["decision"] = exec_res
        
        # Track search attempts to prevent infinite loops
        if exec_res["action"] == "search_kb":
            search_count = shared.get("search_count", 0)
            max_searches = POLICY_LIMITS["max_turns"]
            
            if search_count >= max_searches:
                logger.warning(f"Max search attempts ({max_searches}) reached, forcing answer")
                return "answer"  # Force answer instead of more searching
            
            shared["search_count"] = search_count + 1
            logger.debug(f"Search attempt {search_count + 1}/{max_searches}")
        
        return exec_res["action"]


# ============================================================================
# Node 5: GenerateAnswerNode
# ============================================================================

class GenerateAnswerNode(Node):
    """Generate final answer using RAG context."""
    
    def __init__(self):
        super().__init__(max_retries=3, wait=2)
    
    def prep(self, shared: Dict) -> Dict:
        """Read query, context, and history."""
        session_id = shared.get("session_id", "")
        history_str = conversation_memory.get_formatted_history(session_id, limit=8, exclude_last=True)
        
        return {
            "user_query": shared.get("user_query", ""),
            "user_os": shared.get("user_os", "unknown"),
            "rag_context": shared.get("rag_context", ""),
            "network_status": shared.get("status_results", ""),
            "conversation_history": history_str
        }
    
    def exec(self, context: Dict) -> Dict:
        """Generate answer using LLM."""
        user_query = context['user_query']
        user_os = context.get('user_os', 'unknown')
        conversation_history = context['conversation_history'] or ''
        
        # Check if this is a confirmation response
        confirmation_words = ["yes", "yeah", "yep", "correct", "right", "exactly", "that's it", "thats it"]
        is_confirmation = user_query.lower().strip() in confirmation_words
        
        prompt = f"""{SYSTEM_ROLE}

USER'S CURRENT MESSAGE: "{user_query}"
USER'S OPERATING SYSTEM: {user_os}

CONVERSATION HISTORY:
{conversation_history if conversation_history else 'No previous conversation'}

KNOWLEDGE BASE DOCUMENTS:
{context['rag_context'] or 'No relevant documents found.'}

### CRITICAL INSTRUCTIONS
1. **IF USER CONFIRMS (says "yes", "correct", etc.)**: Look at the conversation history to understand what they're confirming, then provide the SOLUTION from the knowledge base documents. DO NOT ask for more details.
2. **PROVIDE THE SOLUTION**: The knowledge base contains the answer. Extract the step-by-step solution and present it clearly.
3. **BE DIRECT**: Don't say "since you said yes..." or reference their confirmation. Just provide the solution.
4. **USE THE DOCS**: The solution is in the knowledge base. Use it!
5. **OS AWARENESS**: {os_awareness_instruction(user_os)}
6. **URLs**: {URL_RULES}

### WHAT NOT TO DO
- DON'T say "However, since your current response is just 'yes'..."
- DON'T ask for more clarification after user confirms
- DON'T ignore the knowledge base content
- DON'T give generic troubleshooting if docs have specific steps
- DON'T start with "I found instructions for X but not Y" - just give the answer with a note at the end
- DON'T repeat the OS disclaimer multiple times in the same response

### OUTPUT FORMAT (YAML)
```yaml
action: <factual_response | step_by_step_instructions>
confidence: <0.0-1.0>
response_to_user: |
    <Provide the solution from the knowledge base. If OS mismatch, add a brief note at the END only.>
```"""

        answer = call_llm(prompt, max_tokens=512)

        yaml_str = parse_yaml_response(answer)
        decision = yaml.safe_load(yaml_str)

        logger.debug(f"Generated answer: {len(decision)} chars")
        return decision
    
    def exec_fallback(self, prep_res: Dict, exc: Exception) -> Dict:
        """Fallback: provide helpful message based on available context."""
        logger.error(f"Answer generation failed: {exc}")
        
        if "rate limit" in str(exc).lower():
            return {
                "response_to_user": RATE_LIMIT_WITH_DOCS_MESSAGE,
                "confidence": 0.3
            }
        
        # Generic fallback with context if available
        if prep_res.get("rag_context"):
            return {
                "response_to_user": f"I found relevant documentation for your query about '{prep_res['user_query']}', "
                    "but I'm unable to generate a detailed response right now. "
                    "Please check the IT knowledge base or contact support for assistance.",
                "confidence": 0.2
            }
        
        return {
            "response_to_user": GENERIC_ERROR_MESSAGE,
            "confidence": 0.1
        }
    
    def post(self, shared: Dict, prep_res: Dict, exec_res: Dict) -> str:
        """Write answer to response and update active topic."""
        if "response" not in shared:
            shared["response"] = {}
        
        shared["response"]["text"] = exec_res.get("response_to_user", str(exec_res))
        shared["response"]["action_taken"] = "answer"
        shared["response"]["requires_followup"] = False
        
        # Add confidence to response metadata if available
        if "confidence" in exec_res:
            shared["response"]["confidence"] = exec_res["confidence"]
        
        # Update active topic - the question we just answered becomes the active topic
        # This enables follow-up handling like "im on linux" after "how to find mac address"
        session_id = shared.get("session_id", "")
        user_query = shared.get("user_query", "")
        keywords = shared.get("keywords", [])
        
        if session_id and user_query:
            # Only set topic if it's a substantive question (not confirmation/short response)
            if len(user_query.split()) > 3:
                conversation_memory.set_active_topic(session_id, user_query, keywords)
                logger.debug(f"Set active topic: {user_query[:50]}...")
        
        return "default"


# ============================================================================
# Node 6: AskClarifyingQuestionNode
# ============================================================================

class AskClarifyingQuestionNode(Node):
    """Ask user for more details when query is ambiguous, using retrieved docs for context."""
    
    def __init__(self):
        super().__init__(max_retries=2, wait=1)
    
    def prep(self, shared: Dict) -> Dict:
        """Read query, intent, retrieved docs, and conversation history."""
        session_id = shared.get("session_id", "")
        history_str = conversation_memory.get_formatted_history(session_id, limit=8, exclude_last=True)
        
        # Get retrieved docs to help ask targeted questions
        retrieved_docs = shared.get("retrieved_docs", [])
        
        return {
            "user_query": shared.get("user_query", ""),
            "user_os": shared.get("user_os", "unknown"),
            "intent": shared.get("intent", {}),
            "conversation_history": history_str,
            "retrieved_docs": retrieved_docs
        }
    
    def exec(self, context: Dict) -> Dict:
        """Generate clarifying question based on retrieved docs."""
        user_os = context.get('user_os', 'unknown')
        retrieved_docs = context.get('retrieved_docs', [])
        
        # Format retrieved docs for context - extract symptoms/issues described, NOT titles
        doc_context = ""
        if retrieved_docs:
            doc_summaries = []
            for i, doc in enumerate(retrieved_docs[:3]):  # Top 3 docs only
                # Extract content - focus on the problem description, not the title
                content = doc.get('document', '')[:300].replace('\n', ' ')
                score = doc.get('rerank_score', doc.get('score', 0))
                doc_summaries.append(f"  Doc {i+1} (relevance: {score:.2f}): {content}...")
            doc_context = "\n".join(doc_summaries)
        
        prompt = f"""
### CONTEXT
User Query: "{context['user_query']}"

Conversation History:
{context['conversation_history'] if context['conversation_history'] else 'No previous conversation'}

### RELEVANT DOCUMENTATION FOUND (use to understand possible issues)
{doc_context if doc_context else 'No documents retrieved.'}

### YOUR ROLE
Ask a clarifying question to narrow down the user's specific issue.

{COMMON_ASSUMPTIONS}

### CRITICAL RULES
1. **NEVER mention document names, article titles, or filenames** - the user doesn't care about these
2. **DESCRIBE the symptoms/issues** you found in docs to help user identify their problem
3. **If user confirms (says "yes", "correct", "that's it")** - DO NOT ask more questions, just acknowledge
4. Ask about specific SYMPTOMS (e.g., "Are you seeing a 'No connections' message when opening other users' calendars?")
5. Be concise - ONE question only

### GOOD EXAMPLES
- "Are you seeing a 'No connections' error when trying to view other users' calendars in Outlook?"
- "Is Outlook showing connection issues only for shared calendars, or also for your own email?"
- "What error message are you seeing when you try to connect?"

{CLARIFY_BAD_EXAMPLES}

If the user asks what OS they are using, respond: "You are using {user_os}"

### OUTPUT FORMAT
```yaml
action: clarify
reasoning: <brief reason>
confidence: <0.0 to 1.0>
response_to_user: |
  <Your question describing symptoms, NOT referencing documents>
```
"""

        question = call_llm(prompt, max_tokens=200)  # Shorter for clarification
        
        yaml_str = parse_yaml_response(question)
        decision = yaml.safe_load(yaml_str)

        return decision
    
    def exec_fallback(self, prep_res: Dict, exc: Exception) -> str:
        """Fallback: provide generic clarifying question on error."""
        logger.error(f"Clarifying question generation failed: {exc}")
        # Generic fallback based on context
        if "rate limit" in str(exc).lower():
            return f"{RATE_LIMIT_MESSAGE} {GENERIC_CLARIFY_MESSAGE}"
        return GENERIC_CLARIFY_MESSAGE
    
    def post(self, shared: Dict, prep_res: Dict, exec_res: Any) -> str:
        """Write clarifying question to response and preserve active topic."""
        if "response" not in shared:
            shared["response"] = {}
        
        if isinstance(exec_res, dict):
            response_text = exec_res.get("response_to_user", str(exec_res))
        else:
            response_text = str(exec_res)

        shared["response"]["text"] = response_text
        shared["response"]["action_taken"] = "clarify"
        shared["response"]["requires_followup"] = True
        
        # Set active topic to the user's original query (what we're asking clarification about)
        # This enables follow-up context like "yes" or "im on linux"
        session_id = shared.get("session_id", "")
        user_query = shared.get("user_query", "")
        keywords = shared.get("keywords", [])
        
        if session_id and user_query and len(user_query.split()) > 2:
            conversation_memory.set_active_topic(session_id, user_query, keywords)
            logger.debug(f"Clarify: Set active topic: {user_query[:50]}...")
        
        return "default"


# ============================================================================
# Node 7: FormatFinalResponseNode
# ============================================================================

class FormatFinalResponseNode(Node):
    """Format the final response to send to user."""
    
    def prep(self, shared: Dict) -> Dict:
        """Read response data."""
        return {
            "response": shared.get("response", {}),
            "user_query": shared.get("user_query", "")
        }
    
    def exec(self, context: Dict) -> str:
        """Return formatted response (already formatted by previous nodes)."""
        response_text = context["response"].get("text", "I'm sorry, I couldn't process your request.")
        
        # Add any additional formatting here if needed
        return response_text
    
    def post(self, shared: Dict, prep_res: Dict, exec_res: str) -> str:
        """Update final response text."""
        shared["response"]["text"] = exec_res
        
        # Save to conversation memory
        session_id = shared.get("session_id", "")
        if session_id:
            conversation_memory.add_message(session_id, "assistant", exec_res)
        
        logger.info(f"Final response: {exec_res[:100]}...")
        
        return "default"


# ============================================================================
# Node 8: InteractiveTroubleshootNode
# ============================================================================

class InteractiveTroubleshootNode(Node):
    """Provide interactive troubleshooting guidance."""
    
    def __init__(self):
        super().__init__(max_retries=2, wait=1)
    
    def prep(self, shared: Dict) -> Dict:
        """Gather context for troubleshooting."""
        session_id = shared.get("session_id", "")
        history_str = conversation_memory.get_formatted_history(session_id, limit=20, exclude_last=True)
        
        # Get current troubleshooting state
        ts_state = shared.get("troubleshoot_state", {})
        
        # Get retrieved docs summary
        retrieved_docs = shared.get("retrieved_docs", [])
        doc_summaries = ""
        for i, doc in enumerate(retrieved_docs[:3]):
            source = doc.get('metadata', {}).get('source_file', 'unknown')
            doc_summaries += f"{i+1}. [{source}] {doc.get('document', '')[:150]}...\n"
        
        return {
            "user_query": shared.get("user_query", ""),
            "user_os": shared.get("user_os", "unknown"),
            "intent": shared.get("intent", {}),
            "doc_summaries": doc_summaries,
            "conversation_history": history_str,
            "current_step": ts_state.get("current_step", 0),
            "steps_completed": ts_state.get("steps_completed", []),
            "failed_steps": ts_state.get("failed_steps", []),
            "issue_type": ts_state.get("issue_type", "")
        }
    
    def exec(self, context: Dict) -> Dict:
        """Analyze user intent and generate intelligent troubleshooting guidance."""
        escalate_failed_steps = POLICY_LIMITS["troubleshoot_escalate_failed_steps"]
        prompt = f"""### CONTEXT
User Query: "{context['user_query']}"
User System: {context.get('user_os', 'unknown')} / {context.get('user_browser', 'unknown')}
Intent Classification: {context['intent']}
Troubleshooting Step: {context['current_step'] + 1}
Steps Completed: {len(context['steps_completed'])}
Failed Steps: {len(context['failed_steps'])}

Retrieved Documentation:
{context['doc_summaries'] if context['doc_summaries'] else 'No relevant documentation retrieved'}

Conversation History:
{context['conversation_history'] if context['conversation_history'] else 'No previous conversation'}

Previous Steps Taken:
{chr(10).join(f"- {step}" for step in context['steps_completed'][-3:]) if context['steps_completed'] else 'None - this is the initial step'}

Failed Attempts:
{chr(10).join(f"- {step}" for step in context['failed_steps'][-3:]) if context['failed_steps'] else 'None'}

### YOUR ROLE
You are an intelligent troubleshooting assistant for {COMPANY_NAME}. Your job is to:
1. **Detect user intent changes** - recognize if user wants to exit troubleshooting, ask something else, or continue
2. **Provide diagnostic reasoning** - think like a systems engineer, not a script reader
3. **Adapt dynamically** - learn from failed steps and adjust your approach
4. **Know when to escalate** - recognize unsolvable issues and recommend human intervention

{COMMON_ASSUMPTIONS}

### AVAILABLE ACTIONS
[1] continue_troubleshoot
    Description: Provide next diagnostic step or question
    When to use: User is engaged, issue not yet resolved, more steps to try
    
[2] exit_troubleshoot
    Description: Exit troubleshooting mode cleanly
    When to use: User explicitly says stop/cancel/nevermind, asks unrelated question, or says issue is resolved
    
[3] escalate
    Description: Recommend creating support ticket
    When to use: 2+ failed steps, issue beyond first-level support, or user is frustrated

### DECISION RULES
- CRITICAL: Detect exit signals like "never mind", "forget it", "I'll try later", "actually I need...", "can you help with..."
- If user changes topic or asks new question → exit_troubleshoot
- If user reports success ("it works", "that fixed it", "problem solved") → exit_troubleshoot
- If user explicitly asks to 'talk to human' or 'create ticket' → escalate
- If {escalate_failed_steps}+ failed steps or same error persists → escalate
- If user shows frustration ("this isn't working", "nothing helps") → escalate
- If user is engaged and cooperative → continue_troubleshoot
- NEVER repeat a failed step - pivot to different approach

### USER-ACTIONABLE FILTER (CRITICAL)
The user is an END-USER at their computer, NOT an IT administrator. ONLY suggest steps they can perform:
INCLUDE: Software settings, app restarts, network reconnection, device restart, checking settings panels
INCLUDE: Clearing cache, updating apps, changing configurations in their control
EXCLUDE: Checking router hardware, server-side fixes
EXCLUDE: Physical infrastructure checks (router placement, overheating, power cycling network equipment)
EXCLUDE: Admin tasks requiring elevated access (firmware updates, network configuration, server restarts)
If documentation contains IT-admin steps, SKIP them and focus only on user-actionable troubleshooting.
- Use official documentation when available, but explain the reasoning
- Start with most common/simple causes before rare/complex ones

### DIAGNOSTIC REASONING FRAMEWORK
Hardware issues: Power → Connections → Drivers → Settings → Hardware failure
Software issues: Restart → Updates → Config → Cache/temp files → Reinstall → System issue
Network issues: Connection → DNS → Firewall → Proxy → Credentials → Server side
Access issues: Credentials → Permissions → Account status → System policy → Infrastructure

### PLATFORM-SPECIFIC GUIDANCE
**Use the User System info {context.get('user_os')} to tailor all instructions:**
- Windows: Use Windows key combinations (Win+R, Win+I), mention PowerShell/CMD, reference Windows Settings
- macOS: Use Mac key combinations (Cmd+Space, Cmd+,), mention Terminal, reference System Preferences/Settings
- Linux: Mention terminal commands, package managers (apt/yum/dnf), assume technical familiarity
- Browser-specific: Chrome DevTools (F12), Safari Web Inspector, Firefox Developer Tools
- Provide OS-specific paths (e.g., Windows: C:\\Program Files, Mac: /Applications, Linux: /opt or /usr/local)

### OUTPUT FORMAT
Respond in YAML:
```yaml
thinking: |
  <analyze user's response, detect intent change, review what's been tried, 
   determine root cause hypothesis, decide next diagnostic step>
action: <continue_troubleshoot | exit_troubleshoot | escalate>
reasoning: <why you chose this action in 1-2 sentences>
confidence: <0.0 to 1.0 in your diagnosis>
response_to_user: |
  <if continue_troubleshoot: provide ONE specific diagnostic step with clear reasoning>
  <if exit_troubleshoot: acknowledge their intent and offer help with new topic>
  <if escalate: explain why escalation is needed and summarize findings>
next_hypothesis: <your working theory about root cause, or null if exiting>
```

### RESPONSE STYLE GUIDELINES
- Be conversational and empathetic, not robotic
- Explain WHY each step matters (e.g., "checking X because Y often causes Z")
- Use bullet points for multi-step instructions
- Ask for specific observations, not yes/no (e.g., "What error code do you see?" not "Is there an error?")
- Keep under 3 sentences unless complex procedure requires more
- If using docs, cite them briefly: "According to [VPN Setup Guide]..."

Think like a senior systems engineer who teaches while troubleshooting."""

        response = call_llm(prompt, max_tokens=768)
        
        # Parse YAML response
        yaml_str = parse_yaml_response(response)
        decision = yaml.safe_load(yaml_str)
        
        # Validate decision
        allowed_actions = ["continue_troubleshoot", "exit_troubleshoot", "escalate"]
        assert isinstance(decision, dict), "Decision must be a dict"
        assert "action" in decision, "Decision must have 'action' field"
        assert decision["action"] in allowed_actions, f"Action must be one of {allowed_actions}"
        assert "response_to_user" in decision, "Decision must have 'response_to_user' field"
        
        logger.info(f"Troubleshoot decision: {decision['action']} (confidence: {decision.get('confidence', 0):.2f})")
        
        return decision
    
    def exec_fallback(self, prep_res: Dict, exc: Exception) -> Dict:
        """Fallback: provide intelligent response based on context when LLM fails."""
        logger.error(f"Troubleshooting guidance generation failed: {exc}")
        
        is_first_step = prep_res.get("current_step", 0) == 0
        user_query = prep_res.get("user_query", "your issue")
        failed_count = len(prep_res.get("failed_steps", []))
        
        # Generic system error - escalate if multiple failures
        fallback_failed_steps = POLICY_LIMITS["troubleshoot_fallback_failed_steps"]
        if failed_count >= fallback_failed_steps:
            return {
                "action": "escalate",
                "reasoning": "System error with multiple failed steps",
                "confidence": POLICY_LIMITS["system_error_confidence"],
                "response_to_user": (
                    "I'm having trouble generating detailed troubleshooting steps right now, "
                    "and we've already tried several approaches. "
                    "I recommend contacting IT support directly for immediate assistance."
                ),
                "next_hypothesis": None
            }
        
        return {
            "action": "continue_troubleshoot",
            "reasoning": "System error but attempting to continue",
            "confidence": POLICY_LIMITS["system_error_confidence"],
            "response_to_user": (
                "I'm having technical difficulties, but let's try a basic troubleshooting step. "
                "Please restart the affected application or device and let me know if that resolves the issue."
            ),
            "next_hypothesis": "Standard restart procedure"
        }
    
    def post(self, shared: Dict, prep_res: Dict, exec_res: Dict) -> str:
        """Update troubleshooting state and determine next action."""
        action = exec_res.get("action", "continue_troubleshoot")
        response_text = exec_res.get("response_to_user", "I'm here to help with your issue.")
        
        # Handle different actions
        if action == "exit_troubleshoot":
            # Clear troubleshooting state
            shared.pop("troubleshoot_state", None)
            
            if "response" not in shared:
                shared["response"] = {}
            
            shared["response"]["text"] = response_text
            shared["response"]["action_taken"] = "exit_troubleshoot"
            shared["response"]["requires_followup"] = False
            
            logger.info("Exiting troubleshooting mode per user intent")
            return "exit"  # Exit troubleshooting flow
        
        elif action == "escalate":
            # Mark for escalation
            if "troubleshoot_state" not in shared:
                shared["troubleshoot_state"] = {}
            
            shared["troubleshoot_state"]["escalated"] = True
            
            if "response" not in shared:
                shared["response"] = {}
            
            shared["response"]["text"] = response_text
            shared["response"]["action_taken"] = "escalate"
            shared["response"]["requires_followup"] = True
            
            logger.info("Escalating to human support")
            return "escalate"
        
        else:  # continue_troubleshoot
            # Update troubleshooting state
            if "troubleshoot_state" not in shared:
                shared["troubleshoot_state"] = {
                    "current_step": 0,
                    "steps_completed": [],
                    "failed_steps": [],
                    "issue_type": prep_res.get("intent", {}).get("intent", "")
                }
            
            ts_state = shared["troubleshoot_state"]
            ts_state["current_step"] += 1
            
            # Track hypothesis
            hypothesis = exec_res.get("next_hypothesis")
            if hypothesis:
                ts_state["current_hypothesis"] = hypothesis
            
            # Store response
            if "response" not in shared:
                shared["response"] = {}
            
            shared["response"]["text"] = response_text
            shared["response"]["action_taken"] = "troubleshoot"
            shared["response"]["requires_followup"] = True
            
            logger.info(
                f"Troubleshooting step {ts_state['current_step']} completed. "
                f"Hypothesis: {hypothesis}"
            )
            
            return "default"


# ============================================================================
# Node 9: NotImplementedNode (placeholder for other features)
# ============================================================================

class NotImplementedNode(Node):
    """Placeholder for features not yet implemented."""
    
    def __init__(self, feature_name: str = "This feature"):
        super().__init__()
        self.feature_name = feature_name
    
    def exec(self, prep_res: Any) -> str:
        """Return not implemented message."""
        return f"{self.feature_name} is not yet implemented. I'll help you with what I can!"
    
    def post(self, shared: Dict, prep_res: Any, exec_res: str) -> str:
        """Write message to response."""
        if "response" not in shared:
            shared["response"] = {}
        
        shared["response"]["text"] = exec_res
        shared["response"]["action_taken"] = "not_implemented"
        
        return "default"


# ============================================================================
# Offline Indexing Nodes
# ============================================================================

class LoadDocumentsNode(Node):
    """Load documents from source directory."""
    
    def prep(self, shared: Dict) -> str:
        """Get source directory from shared or cached config."""
        return shared.get("source_dir", _RAG_CONFIG["source_dir"])
    
    def exec(self, source_dir: str) -> List[Dict]:
        """Load all documents from directory using document parser."""
        from pathlib import Path
        from utils.document_parser import parse_document
        
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_path}")
        
        # Find all supported files
        file_extensions = (".txt", ".md", ".html", ".htm", ".pdf")
        files = [f for f in source_path.rglob("*") if f.is_file() and f.suffix.lower() in file_extensions]
        
        logger.info(f"Found {len(files)} files in {source_path}")
        
        documents = []
        for filepath in files:
            try:
                relative_path = filepath.relative_to(source_path)
                
                # Use document parser for PDF and HTML, plain text for others
                file_ext = filepath.suffix.lower()
                if file_ext in [".pdf", ".html", ".htm"]:
                    result = parse_document(str(filepath))
                    content = result['text']
                else:
                    # Plain text/markdown
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                
                documents.append({
                    "content": content,
                    "metadata": {
                        "source_file": str(filepath),
                        "relative_path": str(relative_path),
                        "filename": filepath.name,
                        "extension": filepath.suffix,
                        "size_bytes": filepath.stat().st_size,
                    }
                })
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
                continue
        
        logger.info(f"Loaded {len(documents)} documents from {source_dir}")
        return documents
    
    def post(self, shared: Dict, prep_res: str, exec_res: List[Dict]) -> str:
        """Write documents to shared store."""
        shared["documents"] = exec_res
        return "default"


class ChunkDocumentsNode(BatchNode):
    """Chunk documents into smaller pieces."""
    
    def prep(self, shared: Dict) -> List[Dict]:
        """Read documents from shared store."""
        return shared.get("documents", [])
    
    def exec(self, document: Dict) -> List[Dict]:
        """Chunk a single document."""
        from utils.chunker import chunk_text
        
        content = document.get("content", "")
        metadata = document.get("metadata", {})
        
        # Use cached chunk configuration
        chunk_size = _RAG_CONFIG["chunk_size"]
        chunk_overlap = _RAG_CONFIG["chunk_overlap"]

        chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Create chunk dicts with metadata
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                "content": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            }
            chunk_dicts.append(chunk_dict)
        
        return chunk_dicts
    
    def post(self, shared: Dict, prep_res: List[Dict], exec_res_list: List[List[Dict]]) -> str:
        """Flatten all chunks into single list."""
        all_chunks = []
        for chunk_list in exec_res_list:
            all_chunks.extend(chunk_list)
        
        shared["all_chunks"] = all_chunks
        logger.info(f"Created {len(all_chunks)} total chunks")
        return "default"


class EmbedDocumentsNode(Node):
    """Generate embeddings for document chunks using TRUE batch encoding.
    
    Changed from BatchNode to regular Node for 6-7x speedup.
    Instead of calling get_embedding() per chunk (N calls), we use
    get_embeddings_batch() which processes all chunks in one forward pass.
    """
    
    def __init__(self):
        super().__init__(max_retries=3, wait=1)
    
    def prep(self, shared: Dict) -> List[Dict]:
        """Read chunks from shared store."""
        return shared.get("all_chunks", [])
    
    def exec(self, chunks: List[Dict]) -> List[List[float]]:
        """Embed ALL chunks in a single batch call for maximum efficiency."""
        from utils.embedding_local import get_embeddings_batch
        
        if not chunks:
            return []
        
        # Extract content from all chunks
        texts = [chunk.get("content", "") for chunk in chunks]
        
        # TRUE batch encoding - single forward pass for all texts (6-7x faster)
        embeddings = get_embeddings_batch(texts, batch_size=32)
        
        logger.info(f"Batch embedded {len(texts)} chunks in single call")
        return embeddings
    
    def post(self, shared: Dict, prep_res: List[Dict], exec_res: List[List[float]]) -> str:
        """Store embeddings with chunks."""
        shared["all_embeddings"] = exec_res
        logger.info(f"Generated {len(exec_res)} embeddings")
        return "default"


class StoreInChromaDBNode(Node):
    """Store chunks and embeddings in ChromaDB."""
    
    def __init__(self):
        super().__init__(max_retries=2, wait=1)
    
    def prep(self, shared: Dict) -> Dict:
        """Read chunks and embeddings."""
        return {
            "chunks": shared.get("all_chunks", []),
            "embeddings": shared.get("all_embeddings", [])
        }
    
    def exec(self, data: Dict) -> int:
        """Insert into ChromaDB."""
        from utils.chromadb_client import insert_documents
        
        chunks = data["chunks"]
        embeddings = data["embeddings"]
        
        # Extract text and metadata
        chunk_texts = [c["content"] for c in chunks]
        chunk_metadata = [c["metadata"] for c in chunks]
        
        # Insert into ChromaDB
        insert_documents(chunk_texts, embeddings, chunk_metadata)
        
        return len(chunks)
    
    def post(self, shared: Dict, prep_res: Dict, exec_res: int) -> str:
        """Log completion."""
        logger.info(f"Successfully indexed {exec_res} chunks in ChromaDB")
        shared["indexed_count"] = exec_res
        return "default"

# ============================================================================
# Status Querying Node
# ============================================================================

class StatusQueryNode(AsyncNode):
    """Node to check Stibo's status page for network interruptions."""
    
    async def prep_async(self, shared: Dict) -> None:
        """No preparation needed for status query."""
        return None
    
    async def exec_async(self, prep_res: None) -> List[Dict]:
        """Query the status page."""
        from utils.status_retrieval import scrape_session
        
        results = await scrape_session()
        return results if results is not None else []
    
    async def post_async(self, shared: Dict, prep_res: None, exec_res: Dict) -> str:
        """Write status results to shared store."""
        shared["status_results"] = exec_res
        return "default"