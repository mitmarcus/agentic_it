from typing import Dict
from cremedelacreme import Node
from utils.logger import get_logger
from core.llm_config import get_llm_config
from utils.conversation_memory import conversation_memory
from utils.call_llm import call_llm
from utils.prompts import (
    SYSTEM_ROLE,
    USER_ACTIONABLE_FILTER,
    URL_RULES,
    RATE_LIMIT_WITH_DOCS_MESSAGE,
    GENERIC_ERROR_MESSAGE,
    os_awareness_instruction,
    parse_yaml_response
)
import yaml
from .config import ensure_response_dict


# ============================================================================
# Node 5: GenerateAnswerNode
# ============================================================================
logger = get_logger(__name__)

class GenerateAnswerNode(Node):
    """Generate final answer using RAG context."""
    
    def __init__(self):
        config = get_llm_config()
        super().__init__(max_retries=config.max_retries, wait=config.retry_wait)
    
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
        
        # Detect if user is explicitly mentioning an OS in their query
        target_os, os_instruction = os_awareness_instruction(user_query, user_os)
        
        prompt = f"""{SYSTEM_ROLE}

USER'S CURRENT MESSAGE: "{user_query}"
USER'S OPERATING SYSTEM: {target_os}

CONVERSATION HISTORY:
{conversation_history if conversation_history else 'No previous conversation'}

KNOWLEDGE BASE DOCUMENTS:
{context['rag_context'] or 'No relevant documents found.'}

{USER_ACTIONABLE_FILTER}

### CRITICAL INSTRUCTIONS
1. **OS PRIORITY**: If user explicitly mentions an OS (Windows/Linux/Mac), provide instructions for THAT OS ONLY, even if docs are for a different OS. If no docs match their OS, say "I don't have specific {target_os} instructions. Please contact IT support."
2. **IF USER CONFIRMS (says "yes", "correct", etc.)**: Look at the conversation history to understand what they're confirming, then provide the SOLUTION from the knowledge base documents. DO NOT ask for more details.
3. **PROVIDE THE SOLUTION**: The knowledge base contains the answer. Extract the step-by-step solution and present it clearly.
4. **BE DIRECT**: Don't say "since you said yes..." or reference their confirmation. Just provide the solution.
5. **USE THE DOCS**: The solution is in the knowledge base. Use it!
6. **OS AWARENESS**: {os_instruction}
7. **URLs**: {URL_RULES}
8. **FILTER ADMIN TASKS**: When extracting steps from documentation, AUTOMATICALLY SKIP any physical/administrative tasks (router checks, hardware placement, infrastructure tasks). Only include steps the user can perform from their workstation.

### WHAT NOT TO DO
- DON'T say "However, since your current response is just 'yes'..."
- DON'T ask for more clarification after user confirms
- DON'T ignore the knowledge base content
- DON'T give generic troubleshooting if docs have specific steps
- DON'T start with "I found instructions for X but not Y" - just give the answer with a note at the end
- DON'T repeat the OS disclaimer multiple times in the same response
- DON'T include router resets, physical checks, or any infrastructure tasks
- DON'T suggest tasks requiring IT administrator access

### OUTPUT FORMAT (YAML)
```yaml
action: <factual_response | step_by_step_instructions>
confidence: <0.0-1.0>
response_to_user: |
    <Provide the solution from the knowledge base. If OS mismatch, add a brief note at the END only.>
```"""

        config = get_llm_config()
        answer = call_llm(prompt, max_tokens=config.max_tokens)

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
        ensure_response_dict(shared)
        
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

