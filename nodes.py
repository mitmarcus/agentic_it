"""
Nodes for IT Support Chatbot.

All nodes follow the pattern:
- prep(): Read from shared store
- exec(): Process logic (with retries handled by Node)
- post(): Write to shared store, return action
"""
import os
import yaml
import logging
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

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


POLICY_LIMITS = {
    "clarify_confidence_threshold": _get_float_env("AGENT_CLARIFY_CONFIDENCE_THRESHOLD"),
    "doc_confidence_threshold": _get_float_env("AGENT_DOC_CONFIDENCE_THRESHOLD"),
    "rate_limit_answer_confidence": _get_float_env("AGENT_RATE_LIMIT_ANSWER_CONFIDENCE"),
    "system_error_confidence": _get_float_env("AGENT_SYSTEM_ERROR_CONFIDENCE"),
    "troubleshoot_escalate_failed_steps": _get_int_env("TROUBLESHOOT_ESCALATE_FAILED_STEPS"),
    "troubleshoot_fallback_failed_steps": _get_int_env("TROUBLESHOOT_FALLBACK_FAILED_STEPS"),
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
        intent_result = classify_intent(query)
        keywords = extract_keywords(query)
        
        logger.info(f"Intent: {intent_result['intent']} (confidence: {intent_result['confidence']:.2f})")
        
        return {
            "intent_data": intent_result,
            "keywords": keywords
        }
    
    def post(self, shared: Dict, prep_res: str, exec_res: Dict) -> str:
        """Write intent data to shared store."""
        shared["intent"] = exec_res["intent_data"]
        shared["keywords"] = exec_res["keywords"]
        return "default"


# ============================================================================
# Node 2: EmbedQueryNode
# ============================================================================

class EmbedQueryNode(Node):
    """Generate embedding for user query."""
    
    def __init__(self):
        # Retry 3 times if embedding fails
        super().__init__(max_retries=3, wait=1)
    
    def prep(self, shared: Dict) -> str:
        """Read user query."""
        return shared.get("user_query", "")
    
    def exec(self, query: str) -> List[float]:
        """Generate embedding vector."""
        if not query:
            raise ValueError("Empty query for embedding")
        
        embedding = get_embedding(query)
        logger.info(f"Generated embedding: {len(embedding)} dimensions")
        return embedding
    
    def exec_fallback(self, prep_res: str, exc: Exception) -> List[float]:
        """Fallback: return zero vector if embedding fails."""
        logger.error(f"Embedding failed after retries: {exc}")
        # Return zero vector as fallback
        embed_dim = int(os.getenv("EMBEDDING_DIM", "384"))
        return [0.0] * embed_dim
    
    def post(self, shared: Dict, prep_res: str, exec_res: List[float]) -> str:
        """Write embedding to shared store."""
        shared["query_embedding"] = exec_res
        return "default"


# ============================================================================
# Node 3: SearchKnowledgeBaseNode
# ============================================================================

class SearchKnowledgeBaseNode(Node):
    """Retrieve relevant documents from ChromaDB."""
    
    def __init__(self):
        super().__init__(max_retries=2, wait=1)
    
    def prep(self, shared: Dict) -> List[float]:
        """Read query embedding."""
        return shared.get("query_embedding", [])
    
    def exec(self, query_embedding: List[float]) -> List[Dict]:
        """Query ChromaDB for similar documents."""
        if not query_embedding or sum(query_embedding) == 0:
            logger.warning("Empty or zero embedding, skipping search")
            return []
        
        # Fetch candidates for coverage
        top_k = _get_int_env("RAG_TOP_K")
        # Threshold for quality filtering
        min_score = _get_float_env("RAG_MIN_SCORE")
        
        # Query collection
        results = query_collection(query_embedding, top_k=top_k)
        
        # Log scores for debugging
        if results:
            scores = [r["score"] for r in results]
            logger.info(f"Retrieved scores: {scores}, min_score threshold: {min_score}")
        
        # Filter by minimum score
        filtered_results = [r for r in results if r["score"] >= min_score]
        
        logger.info(f"Found {len(results)} results, {len(filtered_results)} above threshold")
        
        return filtered_results
    
    def post(self, shared: Dict, prep_res: List[float], exec_res: List[Dict]) -> str:
        """Write results and compile context."""
        shared["retrieved_docs"] = exec_res
        
        # Compile RAG context from documents
        if exec_res:
            max_context_tokens = int(os.getenv("RAG_MAX_CONTEXT_TOKENS", "2000"))
            
            # Improved: Deduplicate by source_file to avoid redundant chunks from same doc
            seen_sources = set()
            unique_docs = []
            for doc in exec_res:
                source = doc['metadata'].get('source_file', 'unknown')
                if source not in seen_sources:
                    seen_sources.add(source)
                    unique_docs.append(doc)
            
            context_parts = []
            for i, doc in enumerate(unique_docs):
                metadata = doc['metadata']
                source_file = metadata.get('source_file', 'unknown')
                chunk_index = metadata.get('chunk_index', '?')
                total_chunks = metadata.get('total_chunks', '?')
                
                # Improved: More context about document structure
                context_parts.append(f"[Document {i+1}] (Relevance: {doc['score']:.2f})")
                context_parts.append(f"File: {source_file} (Chunk {chunk_index} of {total_chunks})")
                context_parts.append(doc['document'])
                context_parts.append("")
            
            rag_context = "\n".join(context_parts)
            rag_context = truncate_to_token_limit(rag_context, max_context_tokens)
            
            shared["rag_context"] = rag_context
            logger.info(f"Compiled context from {len(unique_docs)} unique documents ({len(exec_res)} total chunks)")
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
        clarify_threshold = POLICY_LIMITS["clarify_confidence_threshold"]
        doc_threshold = POLICY_LIMITS["doc_confidence_threshold"]
        doc_count = context.get('doc_count', 0)
        avg_score = sum(context.get('doc_scores', [])) / len(context.get('doc_scores', [])) if context.get('doc_scores') else 0
        
        prompt = f"""
### CONTEXT
User Query: "{context['user_query']}"
User System: {context.get('user_os', 'unknown')}
Intent Classification: {context['intent'].get('intent', 'unknown')} (confidence: {context['intent'].get('confidence', 0):.2f})
Conversation Turn: {context['turn_count']}

Retrieved Knowledge Base ({doc_count} documents, avg score: {avg_score:.2f}):
{context['rag_context'] if context.get('rag_context') else 'No relevant documents found'}
Network Status:
{context['network_status']}

Conversation History (look at the last 3 messages):
{context['conversation_history'] if context['conversation_history'] else 'No previous conversation'}

Current Workflow State: {context['workflow_status']}"""
        
        # Debug: Log the prompt to see what decision maker sees
        logger.info(f"Decision maker prompt (first 500 chars): {prompt[:500]}...")
        logger.info(f"RAG context length: {len(context.get('rag_context', ''))} chars")
        
        prompt += """

### YOUR ROLE
You are the decision-making component of an IT support chatbot for Stibo Systems. Your job is to 
analyze the context and decide the next action to help the employee in our office environment efficiently.

### ASSUMPTIONS
- Assume the user is an employee working in the Stibo Systems office unless they explicitly state otherwise (e.g., "I am at home", "remote", "public wifi").
- Prioritize office-related solutions first.

## REASONING PROCESS
1. Problem Summary: What is the user's core issue in 1-2 sentences?
2. Known Information: What specific, actionable data do we have? (e.g., error codes, intent, previous steps).
3. Missing Information: What critical data is absent that blocks a resolution?
4. Action Evaluation: Which 2-3 actions are most relevant? Briefly weigh their pros/cons given the context.
5. Final Decision: Select the best action. Justify why it's superior to the alternatives now.

### DECISION RULES (READ CAREFULLY)
1. **IF you have retrieved documents with scores > {doc_threshold}, YOU MUST ANSWER** - The knowledge base likely has relevant information.
2. **LOOK DEEPER**: Check the document text for specific names, phone numbers, or steps even if the relevance score is moderate.
3. **CONTACT INFO & ROLES**: If the user asks "who to contact" or "what does [Name] do", and the documents contain names linked to roles or topics, YOU MUST ANSWER. Look for "Contact [Name] for [Topic]" or "[Name] is responsible for [Topic]".
4. **PHRASE MATCH**: If the user mentions a specific phrase (e.g. "lights are green") and a document contains that phrase, YOU MUST ANSWER based on that document.
5. **DO NOT ask clarifying questions if you have good documents** - Use the retrieved context to answer directly.
6. You have searched {context['search_count']} times (max: {context['max_searches']}). If at max, choose 'answer' or 'clarify', NOT 'search_kb'.
7. **PRIORITIZE SEARCH**: If you haven't searched yet and the query contains specific keywords (e.g., "router", "wifi", "who to ask"), choose 'search_kb' instead of 'clarify'.
8. Only clarify if: (a) You have already searched and found nothing OR (b) Query is completely incomprehensible (e.g. "help", "it doesn't work") OR (c) Documents retrieved but irrelevant (score < {doc_threshold}).
9. Never create ticket without attempting resolution first.

### EXAMPLES OF CORRECT BEHAVIOR
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
- If any active network issues match user's issue → answer
- IMPORTANT: You have searched {context['search_count']} times (max: {context['max_searches']}). If at max, you MUST choose 'answer' (with best available info), 'clarify' or 'create_ticket', NOT 'search_kb'
- If intent confidence < {clarify_threshold:.2f} AND query lacks technical terms → clarify. If query has technical terms (e.g. "router", "vpn", "wifi"), prefer 'search_kb'.
- If intent is factual AND retrieved document provides a clear, direct solution → answer
- If user message contains explicit error codes, logs, or attachments → troubleshoot (unless 'search_kb' finds an exact-match).
- If intent = troubleshooting + no workflow started → troubleshoot (or answer if we have the document)
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
        yaml_str = response.split("```yaml")[1].split("```")[0].strip() if "```yaml" in response else response
        decision = yaml.safe_load(yaml_str)
        
        # Validate decision
        allowed_actions = ["search_kb", "answer", "troubleshoot", "search_tickets", "create_ticket", "clarify"]
        assert isinstance(decision, dict), "Decision must be a dict"
        assert "action" in decision, "Decision must have 'action' field"
        assert decision["action"] in allowed_actions, f"Action must be one of {allowed_actions}"
        
        logger.info(f"Decision: {decision['action']} (confidence: {decision.get('confidence', 0):.2f})")
        
        return decision
    
    def exec_fallback(self, prep_res: Dict, exc: Exception) -> Dict:
        """Fallback: make intelligent decision based on context when LLM fails."""
        logger.error(f"Decision making failed: {exc}")
        
        # Smart fallback based on available context
        has_docs = bool(prep_res.get("rag_context")) and prep_res.get("doc_count", 0) > 0
        intent = prep_res.get("intent", {}).get("intent", "unknown")
        confidence = prep_res.get("intent", {}).get("confidence", 0)
        
        # If rate limit and we have good docs, try to answer
        doc_conf_threshold = POLICY_LIMITS["doc_confidence_threshold"]
        if "rate limit" in str(exc).lower() and has_docs and confidence > doc_conf_threshold:
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
            max_searches = int(os.getenv("AGENT_MAX_TURNS", "5"))
            
            if search_count >= max_searches:
                logger.warning(f"Max search attempts ({max_searches}) reached, forcing answer")
                return "answer"  # Force answer instead of more searching
            
            shared["search_count"] = search_count + 1
            logger.info(f"Search attempt {search_count + 1}/{max_searches}")
        
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
        user_os = context.get('user_os', 'unknown')
        
        prompt = f"""You are a helpful IT support assistant for Stibo Systems.

    USER INFO
    - OS: {user_os}

    COMPANY NETWORK STATUS
    {context['network_status'] if context['network_status'] else 'No company network status information available.'}

    KNOWLEDGE BASE CONTEXT
    {context['rag_context'] if context['rag_context'] else 'No relevant documents found in knowledge base.'}

    Conversation History:
    {context['conversation_history'] if context['conversation_history'] else 'No previous conversation.'}

    INSTRUCTIONS (condensed)
    - Answer using ONLY the provided context.
    - Be concise (3–5 sentences). Use bullet points for procedures.
    - Tailor instructions to the user's OS ({user_os}); avoid mentioning OS unless necessary.
    - If docs match network issues, mention known service issues.
    - If context is insufficient, say so and offer to create a ticket.
    - Never include sensitive info (passwords, keys, PII).
    - Format contact info as: Name - Role/Topic - Contact Details.
    - Do not use markdown bolding in your response.

    OUTPUT (YAML)
1. Use OS-specific language, paths, and commands.
2. Assume user is a standard employee; do not suggest admin tasks (e.g. firmware updates, router reboots).
3. Do NOT mention the user's operating system unless it is critical for the specific instruction or the user asks.
4. If the user's problem matches issues under company network status, say that there are known issues related to the affected service
5. If context insufficient, say so, and offer to create a ticket
6. NEVER include sensitive information (passwords, keys, personal data)
7. Be friendly and professional
8. CONTACT INFO: When providing contact information, format it clearly: Name - Role/Topic - Contact Details.

    

### AVAILABLE ANSWER FORMATS
1.  factual_response
    Description: Provide direct factual information or definitions
    When to use: For "what is" questions or factual queries

2.  step_by_step_instructions
    Description: Explain a solution or fix for a reported issue
    When to use: When documents provide specific solutions to problems

3.  reference_summary
    Description: Summarize key information from documentation
    When to use: When user needs comprehensive but concise information

### DECISION RULES & GUARDRAILS
- STRICT SOURCE-BASED ANSWERS: Only provide information that is directly supported by the "Retrieved Knowledge Documents".
- NO HALLUCINATION: Never invent steps, commands, error codes, or solutions not present in the source documents.
- BE DIRECT AND CONCISE: Get straight to the answer without unnecessary preamble or fluff.
- USE CLEAR FORMATTING: Apply bullet points for lists and numbered steps for procedures to enhance readability.
- CITE SOURCES NATURALLY: Reference documents implicitly (e.g., "According to our documentation...", "The knowledge base indicates...").
- MAINTAIN SECURITY: Never include or infer passwords, API keys, or sensitive information.
- ACKNOWLEDGE LIMITATIONS: If the available documents don't fully answer the question, state what information you can provide and what's missing.

### OUTPUT FORMAT
Respond in YAML:
```yaml
action: <factual_response | step_by_step_instructions | reference_summary>
confidence: <0.0 to 1.0>
# Confidence scoring guide:
# 1.0 = Direct answer found in "Retrieved Knowledge Documents" (e.g. exact steps, specific facts)
# 0.8-0.9 = Answer inferred from documents or general knowledge with high certainty
# 0.5-0.7 = Partial answer or general advice without specific documentation
# < 0.5 = Guessing or unable to answer
response_to_user: |
    <if factual_response: provide 1-2 sentences that would express the anwser to user's request>
    <if step_by_step_instructions: provide a bulletpoint step-by-step list from the document>
    <if reference_summary: anlyze the conversation history and summarize the key point on one message in 3-5 sentences>
```
### RESPONSE STYLE GUIDELINES
- Be conversational and empathetic, not robotic
- Use bullet points for step_by_step_instructions
- If using docs, cite them briefly: "According to [VPN Setup Guide]..."

Provide the most direct and helpful answer possible using only the verified information from available sources."""

        answer = call_llm(prompt, max_tokens=512)

        yaml_str = answer.split("```yaml")[1].split("```")[0].strip() if "```yaml" in answer else answer
        decision = yaml.safe_load(yaml_str)

        logger.info(f"Generated answer: {len(decision)} chars")
        return decision
    
    def exec_fallback(self, prep_res: Dict, exc: Exception) -> str:
        """Fallback: provide helpful message based on available context."""
        logger.error(f"Answer generation failed: {exc}")
        
        if "rate limit" in str(exc).lower():
            return ("I'm currently experiencing high API usage. However, based on the available documentation, "
                    "I can see information related to your query. Please try again in a few minutes, "
                    "or contact IT support directly for immediate assistance.")
        
        # Generic fallback with context if available
        if prep_res.get("rag_context"):
            return (f"I found relevant documentation for your query about '{prep_res['user_query']}', "
                    "but I'm unable to generate a detailed response right now. "
                    "Please check the IT knowledge base or contact support for assistance.")
        
        return ("I'm having trouble generating a response right now. "
                "Please contact IT support for direct assistance with your query.")
    
    def post(self, shared: Dict, prep_res: Dict, exec_res: Dict) -> str:
        """Write answer to response."""
        if "response" not in shared:
            shared["response"] = {}
        
        shared["response"]["text"] = exec_res.get("response_to_user", str(exec_res))
        shared["response"]["action_taken"] = "answer"
        shared["response"]["requires_followup"] = False
        
        # Add confidence to response metadata if available
        if "confidence" in exec_res:
            shared["response"]["confidence"] = exec_res["confidence"]
        
        return "default"


# ============================================================================
# Node 6: AskClarifyingQuestionNode
# ============================================================================

class AskClarifyingQuestionNode(Node):
    """Ask user for more details when query is ambiguous."""
    
    def __init__(self):
        super().__init__(max_retries=2, wait=1)
    
    def prep(self, shared: Dict) -> Dict:
        """Read query, intent, and conversation history."""
        session_id = shared.get("session_id", "")
        history_str = conversation_memory.get_formatted_history(session_id, limit=8, exclude_last=True)
        
        return {
            "user_query": shared.get("user_query", ""),
            "user_os": shared.get("user_os", "unknown"),
            "intent": shared.get("intent", {}),
            "conversation_history": history_str
        }
    
    def exec(self, context: Dict) -> Dict:
        """Generate clarifying question."""
        user_os = context.get('user_os', 'unknown')
        prompt = f"""
### CONTEXT
User Query: "{context['user_query']}"
Intent Classification: {context['intent'].get('intent', 'unclear')} (confidence: {context['intent'].get('confidence', 0):.2f})

Conversation History:
{context['conversation_history'] if context['conversation_history'] else 'No previous conversation'}

### YOUR ROLE
You are the Clarification Specialist component of an IT support AI agent for Stibo Systems. Your job is to generate precise, non-redundant clarifying questions that efficiently gather missing information relevant to our office environment.

### ASSUMPTIONS
- Assume the user is an employee working in the Stibo Systems office unless they explicitly state otherwise.
- Do NOT ask "where are you?" or "are you in the office?".

## REASONING PROCESS
1.  Identify the Gap: Determine what specific information is missing based on the decision engine's analysis and conversation context.
2.  Avoid Redundancy: Check conversation history to ensure you're not asking for information already provided.
3.  Choose Question Type: Select the most efficient question format (open-ended, multiple choice, or specific request).
4.  Optimize Wording: Phrase the question to be clear, specific, and easy for the user to answer.
5.  Validate Helpfulness: Ensure the question will actually move the conversation toward resolution.

If the user asks what OS they are using, respond: "You are using {user_os}"

Be concise (1-2 sentences) and friendly.
### AVAILABLE QUESTION STRATEGIES
1.  specific_detail
    Description: Ask for a precise piece of information (error code, version number, etc.)
    When to use: When you need one specific data point to proceed

2.  scenario_clarification  
    Description: Clarify the context or environment where the issue occurs
    When to use: When the problem context is ambiguous or unclear

3.  symptom_elaboration
    Description: Ask for more details about symptoms or error messages
    When to use: When the problem description is too vague

4.  multiple_choice
    Description: Offer limited choices to quickly narrow down possibilities
    When to use: When there are common scenarios that need differentiation

### DECISION RULES & GUARDRAILS
- BE SPECIFIC: Never ask vague questions like "Can you tell me more?" or "What seems to be the problem?"
- AVOID REPETITION: Do not ask for information that's already in the conversation history
- ONE QUESTION AT A TIME: Ask only one clear question per interaction to avoid confusion
- PRESERVE CONTEXT: Reference the current problem to keep the conversation focused
- MAINTAIN PROFESSIONAL TONE: Be polite and technical without being overly formal
- CONSIDER INTENT CONFIDENCE: If intent confidence is low, focus on understanding the core issue first

### OUTPUT FORMAT
Respond with a YAML object containing your clarifying question and metadata:

```yaml
action: <specific_detail|scenario_clarification|symptom_elaboration|multiple_choice>
reasoning: <why you chose this action in 1-2 sentences>
confidence: <0.0 to 1.0 in your diagnosis>
response_to_user: |
  <if specific_detail: Ask a direct, single question to get one precise piece of information (e.g., "What is the exact error code?").>
  <if scenario_clarification: Ask about the context or environment where the issue occurs (e.g., "Are you running this locally or in production?").>
  <if symptom_elaboration: Ask for more detailed descriptions of the symptoms or error messages (e.g., "What exactly does the error message say?").>
  <if multiple_choice: Offer 2-4 clear, distinct choices to quickly narrow down the problem (e.g., "Is it A) X, B) Y, or C) Z?").>
```
### RESPONSE STYLE GUIDELINES
- Be conversational and empathetic, not robotic
- Use bullet points for step_by_step_instructions

Generate the most efficient clarifying question that will provide the missing information needed to resolve the user's issue.
"""

        question = call_llm(prompt, max_tokens=256)  # Limit tokens for clarification
        
        yaml_str = question.split("```yaml")[1].split("```")[0].strip() if "```yaml" in question else question
        decision = yaml.safe_load(yaml_str)

        return decision
    
    def exec_fallback(self, prep_res: Dict, exc: Exception) -> str:
        """Fallback: provide generic clarifying question on error."""
        logger.error(f"Clarifying question generation failed: {exc}")
        # Generic fallback based on context
        if "rate limit" in str(exc).lower():
            return "I'm experiencing high load right now. Could you please provide more details about your issue so I can help you better?"
        return "Could you please provide more details about your issue?"
    
    def post(self, shared: Dict, prep_res: Dict, exec_res: Any) -> str:
        """Write clarifying question to response."""
        if "response" not in shared:
            shared["response"] = {}
        
        if isinstance(exec_res, dict):
            response_text = exec_res.get("response_to_user", str(exec_res))
        else:
            response_text = str(exec_res)

        shared["response"]["text"] = response_text
        shared["response"]["action_taken"] = "clarify"
        shared["response"]["requires_followup"] = True
        
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
            "user_browser": shared.get("user_browser", "unknown"),
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
Intent Classification: {context['intent'].get('intent', 'unknown')} (confidence: {context['intent'].get('confidence', 0):.2f})
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
You are an intelligent troubleshooting assistant for Stibo Systems. Your job is to:
1. **Detect user intent changes** - recognize if user wants to exit troubleshooting, ask something else, or continue
2. **Provide diagnostic reasoning** - think like a systems engineer, not a script reader
3. **Adapt dynamically** - learn from failed steps and adjust your approach
4. **Know when to escalate** - recognize unsolvable issues and recommend human intervention

### ASSUMPTIONS
- Assume the user is in the Stibo Systems office environment.
- Prioritize office network troubleshooting steps (e.g., "Check if you are connected to 'Stibo-Corp' wifi") over home networking steps.

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
- Use official documentation when available, but explain the reasoning
- Start with most common/simple causes before rare/complex ones

### DIAGNOSTIC REASONING FRAMEWORK
For hardware issues: Power → Connections → Drivers → Settings → Hardware failure
For software issues: Restart → Updates → Config → Cache/temp files → Reinstall → System issue
For network issues: Connection → DNS → Firewall → Proxy → Credentials → Server side
For access issues: Credentials → Permissions → Account status → System policy → Infrastructure

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
        yaml_str = response.split("```yaml")[1].split("```")[0].strip() if "```yaml" in response else response
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
            
            return "continue"


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
        """Get source directory from shared or env."""
        return shared.get("source_dir", os.getenv("INGESTION_SOURCE_DIR", "./data/docs"))
    
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
        
        # Chunk configuration from environment
        chunk_size = _get_int_env("INGESTION_CHUNK_SIZE")
        chunk_overlap = _get_int_env("INGESTION_CHUNK_OVERLAP")

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


class EmbedDocumentsNode(BatchNode):
    """Generate embeddings for document chunks."""
    
    def __init__(self):
        super().__init__(max_retries=3, wait=1)
    
    def prep(self, shared: Dict) -> List[Dict]:
        """Read chunks from shared store."""
        return shared.get("all_chunks", [])
    
    def exec(self, chunk_dict: Dict) -> List[float]:
        """Embed a single chunk."""
        content = chunk_dict.get("content", "")
        embedding = get_embedding(content)
        return embedding
    
    def post(self, shared: Dict, prep_res: List[Dict], exec_res_list: List[List[float]]) -> str:
        """Store embeddings with chunks."""
        shared["all_embeddings"] = exec_res_list
        logger.info(f"Generated {len(exec_res_list)} embeddings")
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