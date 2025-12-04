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
def os_awareness_instruction(user_os: str) -> str:
    """Generate OS awareness instruction for prompts."""
    return f"""- The user is on {user_os}. If the docs are for a different OS, just provide the answer normally, then add a SHORT note at the END like "Note: These steps are for [Windows/Mac]. For {user_os}-specific steps, contact IT support.\""""

# Common bad examples for clarifying questions
CLARIFY_BAD_EXAMPLES = """### BAD EXAMPLES (NEVER DO THIS)
- "...similar to the problem in the Outlook Calendar article?" ❌
- "...as described in document XYZ?" ❌
- "...like in the No-connections guide?" ❌
- "Are you asking about the company VPN?" ❌
- "Do you mean Stibo's email system?" ❌"""

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