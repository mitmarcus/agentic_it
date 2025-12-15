from typing import Dict, Any
from cremedelacreme import Node
from utils.logger import get_logger
from utils.call_llm import call_llm
from utils.conversation_memory import conversation_memory
from core.llm_config import get_llm_config
import yaml

from utils.prompts import (
    COMMON_ASSUMPTIONS,
    CLARIFY_BAD_EXAMPLES,
    GENERIC_CLARIFY_MESSAGE,
    RATE_LIMIT_MESSAGE,
    parse_yaml_response,
)

# ============================================================================
# Node 6: AskClarifyingQuestionNode
# ============================================================================

logger = get_logger(__name__)

class AskClarifyingQuestionNode(Node):
    """Ask user for more details when query is ambiguous, using retrieved docs for context."""
    
    def __init__(self):
        config = get_llm_config()
        super().__init__(max_retries=config.max_retries, wait=config.retry_wait)
    
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

        config = get_llm_config()
        question = call_llm(prompt, max_tokens=config.max_tokens)  # Shorter for clarification
        
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
