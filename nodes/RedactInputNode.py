from cremedelacreme import Node
from typing import Dict
from utils.redactor import redact_text
from utils.logger import get_logger

# ============================================================================
# Node 0: RedactInputNode
# ============================================================================
logger = get_logger(__name__)

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
        # Replace with redacted version for all downstream nodes
        shared["user_query"] = exec_res["redacted_query"]
        
        # If redaction occurred, add a warning message for the user
        if exec_res["had_sensitive_data"]:
            shared["redaction_notice"] = (
                "⚠️ For your security, sensitive information has been redacted from your message. "
                "Please avoid sharing passwords, API keys, or other credentials."
            )
        
        return "default"