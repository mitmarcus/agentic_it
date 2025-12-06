"""
Common system prompts and prompt components for the IT support chatbot.
Centralizes repeated prompt text to ensure consistency and easier maintenance.
"""

# Company name - single source of truth
COMPANY_NAME = "Stibo Systems"

# Base system role
SYSTEM_ROLE = f"You are a helpful IT support assistant for {COMPANY_NAME}."

# Common assumptions that apply to all nodes
COMMON_ASSUMPTIONS = f"""### ASSUMPTIONS
- This is a COMPANY IT support chatbot. ALWAYS assume the user is asking about {COMPANY_NAME} company resources.
- When user asks about "VPN", "wifi", "email", "password", etc. - they mean the COMPANY VPN, COMPANY wifi, COMPANY email, etc.
- Do NOT ask "are you asking about the company X?" - of course they are, this is the company IT chatbot.
- Assume the user is an employee working in the {COMPANY_NAME} office unless they explicitly state otherwise (e.g., "I am at home", "remote", "public wifi").
- Prioritize office-related solutions first."""

# URL handling rules
URL_RULES = """- When including URLs, do NOT add punctuation (periods, commas) immediately after them. Put URLs on their own line or add a space before any punctuation.
- DON'T put periods or punctuation directly after URLs (breaks the link)"""

# OS awareness instruction template
def detect_explicit_os(query: str) -> str | None:
    """Detect if user explicitly mentions an OS in their query.
    
    Returns:
        OS name ("Windows", "Linux", "macOS") or None
    """
    query_lower = query.lower()
    if any(word in query_lower for word in ["windows", "win", "win10", "win11"]):
        return "Windows"
    elif any(word in query_lower for word in ["linux", "ubuntu", "debian", "fedora"]):
        return "Linux"
    elif any(word in query_lower for word in ["mac", "macos", "osx"]):
        return "macOS"
    return None

def os_awareness_instruction(user_query: str, user_os: str) -> tuple[str, str]:
    """Generate OS awareness instruction based on explicit mentions.
    
    Args:
        user_query: The user's query text
        user_os: Detected OS from browser/system
        
    Returns:
        Tuple of (target_os, instruction)
    """
    explicit_os_mention = detect_explicit_os(user_query)
    target_os = explicit_os_mention or user_os
    
    if explicit_os_mention:
        # User explicitly asked for this OS - no disclaimer needed
        instruction = f"The user specifically asked for {explicit_os_mention} instructions. Provide ONLY {explicit_os_mention} steps. DO NOT add any OS disclaimer notes."
    else:
        # Using detected OS - add disclaimer only if docs are for different OS
        instruction = f"The user is on {target_os}. If the docs match {target_os}, provide them WITHOUT any note. ONLY if docs are for a DIFFERENT OS, add a brief note at the END: 'Note: These steps are for [Other OS]. For {target_os}-specific steps, contact IT support.'"
    
    return target_os, instruction

# Clarification rules
CLARIFICATION_RULES = """### CLARIFICATION RULES
1. **CHECK HISTORY FIRST**: Never ask the same clarifying question twice. If last message was clarifying question and user responded, move forward.
2. **USER CONFIRMED**: If user says "yes", "correct", "exactly" after your question, they confirmed - provide the solution.
3. Only clarify if:
   - Query is completely vague ("help", "it doesn't work")
   - Retrieved docs don't match the actual question
   - This is the FIRST time asking
4. **NEVER** clarify after user has already responded to your clarification"""

# Common bad examples for clarifying questions
CLARIFY_BAD_EXAMPLES = """### BAD EXAMPLES (NEVER DO THIS)
- "...similar to the problem in the Outlook Calendar article?" ❌
- "...as described in document XYZ?" ❌
- "...like in the No-connections guide?" ❌
- "Are you asking about the company VPN?" ❌
- "Do you mean Stibo's email system?" ❌
- Asking same question twice after user responds "yes" ❌"""

# Common what not to do rules
COMMON_DONTS = """### WHAT NOT TO DO
- DON'T mention document names, article titles, or filenames to the user
- DON'T ask if they mean the company resource - always assume company context
- DON'T repeat the same disclaimer multiple times in one response
- DON'T say "However, since your current response is just 'yes'..."
- DON'T give generic troubleshooting if docs have specific steps"""

# YAML output format helper
def yaml_output_format(fields: dict[str, str]) -> str:
    """
    Generate YAML output format instruction.
    
    Args:
        fields: Dict of field_name -> description
        
    Example:
        yaml_output_format({
            "action": "<action_name>",
            "confidence": "<0.0-1.0>",
            "response": "<your response>"
        })
    """
    field_lines = "\n".join(f"{k}: {v}" for k, v in fields.items())
    return f"""### OUTPUT FORMAT (YAML)
```yaml
{field_lines}
```"""


# Decision maker specific
DECISION_MAKER_ROLE = f"""### YOUR ROLE
You are the decision-making component of an IT support chatbot for {COMPANY_NAME}. Your job is to 
analyze the context and decide the next action to help the employee in our office environment efficiently."""

def get_decision_rules(doc_threshold: float, context: dict) -> str:
    """Generate decision rules for DecisionMakerNode.
    
    Args:
        doc_threshold: Minimum relevance score for documents
        context: Context dict containing search_count and max_searches
    """
    return f"""### DECISION RULES (READ CAREFULLY)
1. **CRITICAL: CHECK CONVERSATION HISTORY FIRST**: Look at the last assistant message in conversation history. If it was a clarifying question AND user responded, you MUST move forward with 'answer' or 'troubleshoot'. NEVER ask the same clarifying question twice. NEVER clarify after user responds "yes" to your question.
2. **USER CONFIRMATION = ANSWER**: If user says "yes", "correct", "that's right", "exactly", or similar confirmations after a clarifying question, they are confirming the issue. **ALWAYS choose 'answer'** and provide the solution from the retrieved documents.
3. **TOPIC MATCH CHECK**: Before answering, verify the retrieved docs actually address the user's issue. A doc about "printer not working on VPN" does NOT answer "VPN not working" - these are different problems.
4. **IF docs match the user's topic AND scores > {doc_threshold}, answer**. If docs are tangentially related (e.g., mention VPN but solve a different problem), **clarify** what the user needs.
5. **CONTACT INFO & ROLES**: If the user asks "who to contact", and docs contain relevant contacts, answer.
6. **VAGUE QUERIES**: For short queries like "vpn not working", "wifi issues", "help" - **clarify** what specific problem they're experiencing before assuming.
7. You have searched {context['search_count']} times (max: {context['max_searches']}). If at max, choose 'answer' or 'clarify', NOT 'search_kb'.
8. **PRIORITIZE SEARCH**: If you haven't searched yet and the query contains specific keywords (e.g., "router", "wifi", "who to ask"), choose 'search_kb' instead of 'clarify'.
9. Only clarify if: (a) You have already searched and found nothing OR (b) Query is completely incomprehensible (e.g. "help", "it doesn't work") OR (c) Documents retrieved but irrelevant (score < {doc_threshold}) OR (d) This is the FIRST time asking.
10. Never create ticket without attempting resolution first.

### EXAMPLES OF CORRECT BEHAVIOR
- User confirms "yes" after clarification + docs available → **answer** with the solution
- User asks "who to contact about router?" + You have docs about router contacts (score 0.72) → **answer** with contact info
- User asks "router" (vague) + No good docs → **clarify** what they need
- User asks clear question + No docs found → **search_kb** OR **answer** saying you don't have that info"""

# Troubleshoot specific  
TROUBLESHOOT_ROLE = f"""### YOUR ROLE
You are an intelligent troubleshooting assistant for {COMPANY_NAME}. Your job is to:
1. Analyze the user's problem based on available documentation
2. Guide them through diagnostic steps
3. Provide clear, actionable solutions"""


# =============================================================================
# Fallback message templates
# =============================================================================

RATE_LIMIT_MESSAGE = (
    "I'm currently experiencing high API usage. Please try again in a few minutes, "
    "or contact IT support directly for immediate assistance."
)

RATE_LIMIT_WITH_DOCS_MESSAGE = (
    "I'm currently experiencing high API usage. However, based on the available documentation, "
    "I can see information related to your query. Please try again in a few minutes, "
    "or contact IT support directly for immediate assistance."
)

GENERIC_ERROR_MESSAGE = (
    "I'm having trouble generating a response right now. "
    "Please contact IT support for direct assistance with your query."
)

GENERIC_CLARIFY_MESSAGE = "Could you please provide more details about your issue?"


# =============================================================================
# Helper functions
# =============================================================================

def parse_yaml_response(response: str) -> str:
    """Extract YAML content from LLM response.
    
    Handles responses wrapped in ```yaml ... ``` code blocks.
    Returns the raw response if no YAML block found.
    """
    if "```yaml" in response:
        return response.split("```yaml")[1].split("```")[0].strip()
    return response


def context_header(
    user_query: str,
    user_os: str = "unknown",
    intent: str = "",
    turn_count: int = 0,
    conversation_history: str = "",
    rag_context: str = "",
    doc_count: int = 0,
    avg_score: float = 0.0,
) -> str:
    """Generate a standard context header for LLM prompts.
    
    Args:
        user_query: The user's current query
        user_os: User's operating system
        intent: Classified intent
        turn_count: Current conversation turn
        conversation_history: Formatted conversation history
        rag_context: Retrieved document context
        doc_count: Number of retrieved documents
        avg_score: Average document relevance score
    """
    header = f"""### CONTEXT
User Query: "{user_query}"
User System: {user_os}"""
    
    if intent:
        header += f"\nIntent Classification: {intent}"
    
    if turn_count:
        header += f"\nConversation Turn: {turn_count}"
    
    if rag_context:
        header += f"""

Retrieved Knowledge Base ({doc_count} documents, avg score: {avg_score:.2f}):
{rag_context}"""
    
    if conversation_history:
        header += f"""

Conversation History:
{conversation_history}"""
    
    return header