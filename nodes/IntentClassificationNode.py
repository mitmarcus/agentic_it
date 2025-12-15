from cremedelacreme import Node
from typing import Dict
from utils.logger import get_logger
from utils.intent_classifier import classify_intent, extract_keywords
# ============================================================================
# Node 1: IntentClassificationNode
# ============================================================================
logger = get_logger(__name__)

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