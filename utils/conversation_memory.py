"""
Session-based conversation memory for interactive workflows.
Stores workflow state, conversation history, and step tracking.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime


class ConversationMemory:
    """
    In-memory store for conversation sessions.
    Tracks workflow state and conversation history.
    """
    
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Get or create session data.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session data dictionary
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat(),
                "conversation_history": [],
                "workflow_state": None,
            }
        
        # Update last activity
        self._sessions[session_id]["last_activity"] = datetime.now().isoformat()
        
        return self._sessions[session_id]
    
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
        
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = []
        
        for session_id, session_data in self._sessions.items():
            last_activity = datetime.fromisoformat(session_data["last_activity"])
            if last_activity < cutoff:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self._sessions[session_id]
        
        return len(to_remove)


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
