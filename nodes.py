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


def _get_float_env(name: str, default: float) -> float:
    """Safely read a float from environment variables."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return float(default)
    try:
        return float(raw_value)
    except ValueError:
        logger.warning("Invalid float for %s=%s, using default %.2f", name, raw_value, default)
        return float(default)


def _get_int_env(name: str, default: int) -> int:
    """Safely read an int from environment variables."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return int(default)
    try:
        return int(raw_value)
    except ValueError:
        logger.warning("Invalid int for %s=%s, using default %d", name, raw_value, default)
        return int(default)


POLICY_LIMITS = {
    "clarify_confidence_threshold": _get_float_env("AGENT_CLARIFY_CONFIDENCE_THRESHOLD", 0.7),
    "doc_confidence_threshold": _get_float_env("AGENT_DOC_CONFIDENCE_THRESHOLD", 0.6),
    "rate_limit_answer_confidence": _get_float_env("AGENT_RATE_LIMIT_ANSWER_CONFIDENCE", 0.5),
    "system_error_confidence": _get_float_env("AGENT_SYSTEM_ERROR_CONFIDENCE", 0.3),
    "troubleshoot_escalate_failed_steps": _get_int_env("TROUBLESHOOT_ESCALATE_FAILED_STEPS", 3),
    "troubleshoot_fallback_failed_steps": _get_int_env("TROUBLESHOOT_FALLBACK_FAILED_STEPS", 2),
}


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
        
        top_k = int(os.getenv("RAG_TOP_K", "3"))
        min_score = float(os.getenv("RAG_MIN_SCORE", "0.6"))
        
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
            
            context_parts = []
            for i, doc in enumerate(exec_res):
                context_parts.append(f"[Document {i+1}] (score: {doc['score']:.2f})")
                context_parts.append(f"Source: {doc['metadata'].get('source_file', 'unknown')}")
                context_parts.append(doc['document'])
                context_parts.append("")
            
            rag_context = "\n".join(context_parts)
            rag_context = truncate_to_token_limit(rag_context, max_context_tokens)
            
            shared["rag_context"] = rag_context
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
        history = conversation_memory.get_conversation_history(session_id, limit=10)
        
        # Format history - exclude the very last message (current query)
        history_str = ""
        for msg in history[:-1][-20:]:  # Last 10 exchanges, excluding current query
            history_str += f"{msg['role']}: {msg['content']}\n"
        
        # Get retrieved docs summary
        retrieved_docs = shared.get("retrieved_docs", [])
        doc_summaries = ""
        for i, doc in enumerate(retrieved_docs[:3]):
            doc_summaries += f"{i+1}. {doc['document'][:100]}... (score: {doc['score']:.2f})\n"
        
        # Get workflow state
        workflow_state = shared.get("workflow_state")
        workflow_status = "None"
        if workflow_state:
            workflow_status = f"In progress: step {workflow_state.get('current_step_index', 0) + 1}"
        
        # Track search attempts to prevent infinite loops
        search_count = shared.get("search_count", 0)
        max_searches = int(os.getenv("AGENT_MAX_TURNS", "5"))
        
        return {
            "user_query": shared.get("user_query", ""),
            "user_os": shared.get("user_os", "unknown"),
            "intent": shared.get("intent", {}),
            "network_status": format_status_results(shared.get("status_results", [])),
            "conversation_history": history_str,
            "doc_summaries": doc_summaries,
            "workflow_status": workflow_status,
            "turn_count": shared.get("turn_count", 0),
            "search_count": search_count,
            "max_searches": max_searches
        }
    
    def exec(self, context: Dict) -> Dict:
        """Call LLM to decide next action."""
        clarify_threshold = POLICY_LIMITS["clarify_confidence_threshold"]
        prompt = f"""
### CONTEXT
User Query: "{context['user_query']}"
User System: {context.get('user_os', 'unknown')}
Intent Classification: {context['intent'].get('intent', 'unknown')} (confidence: {context['intent'].get('confidence', 0):.2f})
Conversation Turn: {context['turn_count']}

Network Status:
{context['network_status']}

Retrieved Documents from knowledge base:
{context['doc_summaries'] if context['doc_summaries'] else 'No documents retrieved'}

Conversation History (look at the last 3 messages):
{context['conversation_history'] if context['conversation_history'] else 'No previous conversation'}

Current Workflow State: {context['workflow_status']}

### YOUR ROLE
You are the decision-making component of an IT support chatbot. Your job is to 
analyze the context and decide the next action to help the employee efficiently.

## REASONING PROCESS
1. Problem Summary: What is the user's core issue in 1-2 sentences?
2. Known Information: What specific, actionable data do we have? (e.g., error codes, intent, previous steps).
3. Missing Information: What critical data is absent that blocks a resolution?
4. Action Evaluation: Which 2-3 actions are most relevant? Briefly weigh their pros/cons given the context.
5. Final Decision: Select the best action. Justify why it's superior to the alternatives now.

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
- If intent confidence < {clarify_threshold:.2f} OR query contains ambiguous terms (e.g., "it", "that", "the problem") OR critical info is missing → clarify
- If intent is factual AND retrieved document provides a clear, direct solution → answer
- If user message contains explicit error codes, logs, or attachments → troubleshoot (unless 'search_kb' finds an exact-match).
- If intent = troubleshooting + no workflow started → troubleshoot (or answer if we have the document)
- Use 'create_ticket' after other resolution paths ('search_kb', troubleshoot) are exhausted or if the issue requires privileges/physical access.
- If the same document keep appearing in searches, do not search again. 'answer' with the best information you have.
- Never create ticket without attempting resolution first
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
        has_docs = bool(prep_res.get("doc_summaries"))
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
        history = conversation_memory.get_conversation_history(session_id, limit=10)
        
        # Format last 2 exchanges (4 messages), excluding current query
        history_str = ""
        for msg in history[:-1][-4:]:
            history_str += f"{msg['role']}: {msg['content']}\n"
        
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
        
        prompt = f"""### YOUR ROLE
You are a helpful IT support assistant. Provide accurate, concise answers based on official documentation.

### USER INFORMATION
Operating System: {user_os}

### COMPANY NETWORK STATUS
{context['network_status'] if context['network_status'] else 'No company network status information available.'}

### CONTEXT FROM KNOWLEDGE BASE
{context['rag_context'] if context['rag_context'] else 'No relevant documents found in knowledge base.'}

Conversation History:
{context['conversation_history'] if context['conversation_history'] else 'No previous conversation.'}

### YOUR ROLE
You are the Answer Generator component of an IT support AI agent. Your specific function is to provide direct answers and solutions using the available information, ONLY when the decision engine has chosen the 'answer' action.

### INSTRUCTIONS
1. Answer using ONLY information from the context above
2. Be concise but complete - aim for 3-5 sentences
3. Use bullet points for step-by-step instructions
4. **IMPORTANT: Tailor all instructions to the user's operating system ({user_os})**
   - For Windows: Use Windows-specific paths, commands, and UI elements
   - For macOS: Use Mac-specific paths, commands, and UI elements
   - For Linux: Use Linux-specific commands and paths
5. If the user asks about their OS, tell them: "You are using {user_os}"
6. If the user's problem matches issues under company network status, say that there are known issues related to the affected service
7. If context insufficient, say so and offer to create a ticket
8. NEVER include sensitive information (passwords, keys, personal data)
9. Be friendly and professional

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
confidence: <0.0 to 1.0 in your diagnosis>
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
    
    def post(self, shared: Dict, prep_res: Dict, exec_res: str) -> str:
        """Write answer to response."""
        if "response" not in shared:
            shared["response"] = {}
        
        shared["response"]["text"] = exec_res.get("response_to_user", str(exec_res))
        shared["response"]["action_taken"] = "answer"
        shared["response"]["requires_followup"] = False
        
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
        history = conversation_memory.get_conversation_history(session_id, limit=10)
        
        # Format last 2 exchanges, excluding current query
        history_str = ""
        for msg in history[:-1][-4:]:
            history_str += f"{msg['role']}: {msg['content']}\n"
        
        return {
            "user_query": shared.get("user_query", ""),
            "user_os": shared.get("user_os", "unknown"),
            "intent": shared.get("intent", {}),
            "conversation_history": history_str
        }
    
    def exec(self, context: Dict) -> str:
        """Generate clarifying question."""
        user_os = context.get('user_os', 'unknown')
        prompt = f"""
### CONTEXT
User Query: "{context['user_query']}"
Intent Classification: {context['intent'].get('intent', 'unclear')} (confidence: {context['intent'].get('confidence', 0):.2f})

Conversation History:
{context['conversation_history'] if context['conversation_history'] else 'No previous conversation'}

### YOUR ROLE
You are the Clarification Specialist component of an IT support AI agent. Your job is to generate precise, non-redundant clarifying questions that efficiently gather missing information.

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


Generate the most efficient clarifying question that will provide the missing information needed to resolve the user's issue.
"""

        question = call_llm(prompt, max_tokens=256)  # Limit tokens for clarification
        return question.strip()
    
    def exec_fallback(self, prep_res: Dict, exc: Exception) -> str:
        """Fallback: provide generic clarifying question on error."""
        logger.error(f"Clarifying question generation failed: {exc}")
        # Generic fallback based on context
        if "rate limit" in str(exc).lower():
            return "I'm experiencing high load right now. Could you please provide more details about your issue so I can help you better?"
        return "Could you please provide more details about your issue?"
    
    def post(self, shared: Dict, prep_res: Dict, exec_res: str) -> str:
        """Write clarifying question to response."""
        if "response" not in shared:
            shared["response"] = {}
        
        shared["response"]["text"] = exec_res.get("response_to_user", str(exec_res))
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
        history = conversation_memory.get_conversation_history(session_id, limit=10)
        
        # Get current troubleshooting state
        ts_state = shared.get("troubleshoot_state", {})
        
        # Format conversation history
        history_str = ""
        for msg in history[:-1][-20:]:  # Last 10 exchanges, excluding current query
            history_str += f"{msg['role']}: {msg['content']}\n"
        
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
You are an intelligent troubleshooting assistant. Your job is to:
1. **Detect user intent changes** - recognize if user wants to exit troubleshooting, ask something else, or continue
2. **Provide diagnostic reasoning** - think like a systems engineer, not a script reader
3. **Adapt dynamically** - learn from failed steps and adjust your approach
4. **Know when to escalate** - recognize unsolvable issues and recommend human intervention

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
        """Load all documents from directory."""
        from utils.document_loader import load_documents_from_directory
        
        documents = load_documents_from_directory(source_dir)
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
        
        chunk_size = int(os.getenv("INGESTION_CHUNK_SIZE", "500"))
        chunk_overlap = int(os.getenv("INGESTION_CHUNK_OVERLAP", "50"))

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
    
    async def exec_async(self, prep_res: None) -> Dict:
        """Query the status page."""
        from utils.status_retrieval import scrape_session
        
        results = await scrape_session()
        return results
    
    async def post_async(self, shared: Dict, prep_res: None, exec_res: Dict) -> str:
        """Write status results to shared store."""
        shared["status_results"] = exec_res
        return "default"