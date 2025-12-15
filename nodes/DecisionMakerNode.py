from typing import Dict
from cremedelacreme import Node
from utils.logger import get_logger
from .config import _get_int_env, POLICY_LIMITS
from utils.call_llm import call_llm
from utils.conversation_memory import conversation_memory
from core.llm_config import get_llm_config
from utils.status_retrieval import format_status_results
import yaml

from utils.prompts import (
    COMMON_ASSUMPTIONS,
    DECISION_MAKER_ROLE,
    get_decision_rules,
    parse_yaml_response,
)
# ============================================================================
# Node 4: DecisionMakerNode (Agent)
# ============================================================================
logger = get_logger(__name__)

class DecisionMakerNode(Node):
    """Agent node that decides which action to take next."""
    
    def __init__(self):
        config = get_llm_config()
        super().__init__(max_retries=config.max_retries, wait=config.retry_wait)
    
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
        
        # Check if we're in an active troubleshooting session
        troubleshoot_state = shared.get("troubleshoot_state", {})
        in_troubleshoot = bool(troubleshoot_state) and not troubleshoot_state.get("escalated", False)
        
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
            "max_searches": max_searches,
            "in_troubleshoot": in_troubleshoot,
            "troubleshoot_step": troubleshoot_state.get("current_step", 0)
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

Conversation History (look at the last few messages):
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

{get_decision_rules(doc_threshold, context)}
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
    - User indicates their problem is fixed (e.g. "it works", "that fixed it", "I figured it out")

6.  clarify
    Description: Ask user for specific details to better understand the problem
    When to use:
    - User query is extremely vague ("it", "this", "the problem" with no context)
    - Query is just one word like "help" or "broken"
    - ONLY use if intent = 'informative' AND docs don't match query

### DECISION RULES & GUARDRAILS
- **CRITICAL - TROUBLESHOOTING MODE**: If intent = 'troubleshooting', you MUST choose either 'troubleshoot' or 'answer'. NEVER choose 'clarify' for troubleshooting intent - users with problems need solutions, not clarification questions.
- **INTENT-BASED ROUTING**: 
  * If intent = 'troubleshooting' AND docs have step-by-step fix → 'answer' with the fix
  * If intent = 'troubleshooting' AND no clear fix → 'troubleshoot' (interactive diagnostic)
  * If intent = 'informative' → prefer 'answer' or 'search_kb'
- If any active network issues match user's issue → answer
- IMPORTANT: You have searched {context['search_count']} times (max: {context['max_searches']}). If at max, you MUST choose 'answer' (with best available info), 'clarify' or 'create_ticket', NOT 'search_kb'
- If user message contains explicit error codes, logs, or attachments → troubleshoot (unless 'search_kb' finds an exact-match).
- If user explicitly requests 'talk to human', 'create ticket', or 'escalate', choose 'create_ticket'.
- Use 'create_ticket' after other resolution paths ('search_kb', troubleshoot) are exhausted, if the issue requires privileges/physical access, or if the issue is resolved.
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
issue_fixed: <true or false - true if the user confirmed problem is fixed>
```

Think carefully and make the best decision for the user."""

        config = get_llm_config()
        response = call_llm(prompt, max_tokens=config.max_tokens)
        
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
        
        # Track search attempts to prevent infinite loops (check FIRST, before troubleshoot routing)
        if exec_res["action"] == "search_kb":
            search_count = shared.get("search_count", 0)
            max_searches = POLICY_LIMITS["max_turns"]
            
            if search_count >= max_searches:
                logger.warning(f"Max search attempts ({max_searches}) reached, forcing answer (even during troubleshooting)")
                return "answer"  # Force answer instead of more searching
            
            shared["search_count"] = search_count + 1
            logger.debug(f"Search attempt {search_count + 1}/{max_searches}")
        
        # If we're in an active troubleshooting session, continue with troubleshoot node
        # to let it handle user responses (confirmations, "it's fixed", etc.)
        # NOTE: This check comes AFTER loop protection to ensure max_searches is enforced
        if prep_res.get("in_troubleshoot", False):
            logger.info(f"In active troubleshooting session (step {prep_res.get('troubleshoot_step', 0)}), routing to troubleshoot node")
            return "troubleshoot"
        
        return exec_res["action"]

