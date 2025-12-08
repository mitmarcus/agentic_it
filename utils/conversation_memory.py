"""
Session-based conversation memory for interactive workflows.
Stores workflow state, conversation history, and step tracking.
"""
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import OrderedDict


class ConversationMemory:
    """
    In-memory store for conversation sessions.
    Tracks workflow state, conversation history, and active topic.
    
    Uses OrderedDict with LRU eviction to bound memory usage.
    """
    
    # Maximum sessions to keep (LRU eviction)
    MAX_SESSIONS = 1000
    
    def __init__(self):
        self._sessions: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
    
    def _evict_if_needed(self):
        """Evict oldest sessions if we exceed MAX_SESSIONS (call while holding lock)."""
        while len(self._sessions) > self.MAX_SESSIONS:
            # Pop oldest (first) item
            self._sessions.popitem(last=False)
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Get or create session data.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session data dictionary
        """
        with self._lock:
            if session_id not in self._sessions:
                self._evict_if_needed()
                self._sessions[session_id] = {
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(),
                    "last_activity": datetime.now().isoformat(),
                    "conversation_history": [],
                    "workflow_state": None,
                    "active_topic": None,  # Track the current topic being discussed
                }
            else:
                # Move to end (most recently used) for LRU
                self._sessions.move_to_end(session_id)
            
            # Update last activity
            self._sessions[session_id]["last_activity"] = datetime.now().isoformat()
            
            return self._sessions[session_id]
    
    def set_active_topic(self, session_id: str, topic: str, keywords: Optional[List[str]] = None):
        """
        Set the active topic being discussed (e.g., "how to find MAC address").
        
        Args:
            session_id: Session identifier
            topic: The main topic/question being discussed
            keywords: Key terms related to this topic
        """
        session = self.get_session(session_id)
        session["active_topic"] = {
            "topic": topic,
            "keywords": keywords or [],
            "set_at": datetime.now().isoformat()
        }
    
    def get_active_topic(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the active topic being discussed.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Active topic dict with 'topic' and 'keywords', or None
        """
        session = self.get_session(session_id)
        return session.get("active_topic")
    
    def clear_active_topic(self, session_id: str):
        """Clear the active topic (user switched topics)."""
        session = self.get_session(session_id)
        session["active_topic"] = None
    
    def add_message(self, session_id: str, role: str, content: str):
        """
        Add message to conversation history.
        
        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
        """
        session = self.get_session(session_id)
        session["conversation_history"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent conversation history.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
        
        Returns:
            List of message dictionaries
        """
        session = self.get_session(session_id)
        return session["conversation_history"][-limit:]
    
    def set_workflow_state(self, session_id: str, workflow_state: Dict[str, Any]):
        """
        Store workflow state for interactive troubleshooting.
        
        Args:
            session_id: Session identifier
            workflow_state: Workflow state dictionary with:
                - issue: Original issue description
                - all_steps: List of all troubleshooting steps
                - current_step_index: Current step number
                - completed_steps: List of completed step indices
                - status: 'in_progress', 'resolved', 'escalated'
        """
        session = self.get_session(session_id)
        session["workflow_state"] = workflow_state
    
    def get_workflow_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current workflow state.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Workflow state dictionary or None
        """
        session = self.get_session(session_id)
        return session.get("workflow_state")
    
    def clear_workflow_state(self, session_id: str):
        """
        Clear workflow state (issue resolved or abandoned).
        
        Args:
            session_id: Session identifier
        """
        session = self.get_session(session_id)
        session["workflow_state"] = None
    
    def is_in_workflow(self, session_id: str) -> bool:
        """
        Check if session is currently in an interactive workflow.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if in workflow, False otherwise
        """
        workflow_state = self.get_workflow_state(session_id)
        return workflow_state is not None and workflow_state.get("status") == "in_progress"
    
    def clear_session(self, session_id: str):
        """
        Clear all session data.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """
        Remove sessions older than max_age_hours.
        
        Args:
            max_age_hours: Maximum session age in hours
        """
        from datetime import datetime, timedelta
        
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=max_age_hours)
            to_remove = []
            
            for session_id, session_data in self._sessions.items():
                last_activity = datetime.fromisoformat(session_data["last_activity"])
                if last_activity < cutoff:
                    to_remove.append(session_id)
            
            for session_id in to_remove:
                del self._sessions[session_id]
            
            return len(to_remove)
    
    def get_formatted_history(self, session_id: str, limit: int = 10, exclude_last: bool = False) -> str:
        """
        Get conversation history formatted as a string.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to include in the output string
            exclude_last: Whether to exclude the most recent message (useful if current query is handled separately)
            
        Returns:
            Formatted string "Role: Content\n"
        """
        fetch_limit = limit + 1 if exclude_last else limit
        history = self.get_conversation_history(session_id, fetch_limit)
        
        if exclude_last and history:
            history = history[:-1]
            
        formatted = ""
        for msg in history:
            formatted += f"{msg['role']}: {msg['content']}\n"
        return formatted


# Global conversation memory instance
conversation_memory = ConversationMemory()


if __name__ == "__main__":
    # Test conversation memory
    memory = ConversationMemory()
    
    # Create session
    session_id = "test-123"
    
    # Add messages
    memory.add_message(session_id, "user", "My computer won't turn on")
    memory.add_message(session_id, "assistant", "Let me help you troubleshoot that.")
    
    # Get history
    history = memory.get_conversation_history(session_id)
    print(f"Conversation history: {len(history)} messages")
    
    # Set workflow state
    workflow = {
        "issue": "Computer won't turn on",
        "all_steps": [
            {"step": 1, "action": "Check power cable"},
            {"step": 2, "action": "Try different outlet"},
        ],
        "current_step_index": 0,
        "completed_steps": [],
        "status": "in_progress"
    }
    memory.set_workflow_state(session_id, workflow)
    
    # Check workflow
    print(f"In workflow: {memory.is_in_workflow(session_id)}")
    
    # Get workflow
    state = memory.get_workflow_state(session_id)
    print(f"Workflow state: {state}")
