from typing import Dict
from cremedelacreme import Node
from utils.call_llm import call_llm
from utils.conversation_memory import conversation_memory
from utils.logger import get_logger
import re
from utils.ticket_manager import find_existing_ticket, write_ticket
import json

# ============================================================================
# Ticket Creation Node
# ============================================================================
logger = get_logger(__name__)

class TicketCreationNode(Node):
    """Create a ticket."""
    def __init__(self):
        super().__init__(max_retries=3, wait=2)

    def prep(self, shared: Dict) -> Dict:
        """Gather context for ticket creation."""
        session_id = shared.get("session_id", "")
        history = conversation_memory.get_conversation_history(session_id, limit=50)
        
        # Check if ticket already exists
        existing_ticket = find_existing_ticket(shared, history)

        # check if issue was fixed during troubleshooting
        fixed = shared.get("decision", {}).get("issue_fixed")
        
        # get current troubleshooting state
        ts_state = shared.get("troubleshoot_state", {})

        history_str = ""
        for msg in history[:-1][-20:]:
            history_str += f"{msg['role']}: {msg['content']}\n"

        # get retrieved docs summary
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
            "log_history": history,
            "steps_completed": ts_state.get("steps_completed", []),
            "failed_steps": ts_state.get("failed_steps", []),
            "issue_type": ts_state.get("issue_type", ""),
            "existing_ticket": existing_ticket,
            "issue_fixed": fixed
        }

    def exec(self, context: Dict) -> Dict:
        """Analyze user intent and generate a ticket."""
        try:
            if context.get("existing_ticket"):
                logger.info("Existing ticket found.")
                return context["existing_ticket"]

            prompt = f"""### CONTEXT
    User Query: "{context['user_query']}"
    User System: {context.get('user_os', 'unknown')} / {context.get('user_browser', 'unknown')}
    Retrieved Documentation:
    {context['doc_summaries'] if context['doc_summaries'] else 'No relevant documentation retrieved'}

    Conversation History:
    {context['conversation_history'] if context['conversation_history'] else 'No previous conversation'}

    Previous Steps Taken:
    {chr(10).join(f"- {step}" for step in context['steps_completed'][-3:]) if context['steps_completed'] else 'None - this is the initial step'}

    Failed Attempts:
    {chr(10).join(f"- {step}" for step in context['failed_steps'][-3:]) if context['failed_steps'] else 'None'}

    Issue fixed during troubleshooting: {context['issue_fixed']}

    ### YOUR ROLE
    You are an intelligent tech support assistant. Your job is to create a detailed support ticket for the user's issue, the troubleshooting steps taken so far in the provided conversation, and relevant context.

    ### TICKET FIELDS
    - Summary: A short title summarizing the issue
    - Description: A detailed description of the issue, including relevant context and troubleshooting steps taken


    ### OUTPUT FORMAT
    Respond ONLY with a valid JSON ticket object. IMPORTANT: All field values must be on a single line with no literal newlines. Use \\n for line breaks within strings.

    ```json
    {{
        "fixed": {str(context['issue_fixed']).lower()},
        "project": "AI Service Desk",
        "issue_type": "IT Help",
        "request_type": "IT Help - Detailed",
        "summary": "<short title summarizing the issue>",
        "description": "<detailed description including context and troubleshooting steps taken>",
        "priority": "L3-Medium",
        "assignee": "Automatic",
        "status": "Waiting for support"
    }}
    ```

    ### DESCRIPTION GUIDELINES
    - Use numbered, comma-separated lists for multi-step troubleshooting processes (ex. 1. step one, 2. step two, etc.)
    - Include all relevant context from conversation
    - If using docs, cite them briefly: "According to [VPN Setup Guide]..."
    """

            logger.info("Calling LLM for ticket creation...")
            llm_response = call_llm(prompt, max_tokens=768)
            
            # Handle both dict and string responses from call_llm
            if isinstance(llm_response, dict):
                response_text = llm_response.get('content', '') or llm_response.get('text', '') or str(llm_response)
                logger.info(f"LLM returned dict, extracted text: {response_text[:200]}...")
            else:
                response_text = str(llm_response)
                logger.info(f"LLM response received: {response_text[:200]}...")
            
            ticket = None
            
            try:
                m = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', response_text)
                if m:
                    json_str = m.group(1)
                else:
                    # try to find any JSON object (non-greedy to avoid capturing extra content)
                    m = re.search(r'(\{(?:[^{}]|\{[^{}]*\})*\})', response_text)
                    if m:
                        logger.info("Found JSON in response (no code block)")
                        json_str = m.group(1)
                    else:
                        json_str = None
                
                if json_str:
                    # clean
                    ticket = json.loads(json_str)
                            
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse ticket JSON: {e}")
                logger.error(f"Attempted to parse: {m.group(1) if m else 'No match found'}")
                raise ValueError(f"Failed to parse ticket from LLM response: {e}")
            
            if not ticket:
                logger.error("No ticket JSON found in LLM response")
                logger.error(f"Full response: {response_text}")
                raise ValueError("No ticket JSON found in LLM response")
            
            logger.info(f"Ticket created successfully: {ticket.get('summary', 'N/A')}")

            result = write_ticket(
                ticket, 
                filename=ticket.get("summary", "ticket").replace(" ", "_") + ".json", 
                conversation_history=context['log_history']
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Exception in exec: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise  # Re-raise to trigger exec_fallback

    def exec_fallback(self, prep_res: Dict, exc: Exception) -> Dict:
        """Fallback: provide intelligent response based on context when LLM fails."""
        logger.error(f"Ticket creation failed, using fallback: {exc}")

        ticket = {
            "project": "AI Service Desk",
            "issue_type": "IT Help",
            "request_type": "IT Help - Detailed",
            "summary": f"Support needed: {prep_res['user_query'][:50]}",
            "description": f"User query: {prep_res['user_query']}\nSystem error occurred during ticket creation.",
            "priority": "L3-Medium",
            "assignee": "Automatic",
            "status": "Waiting for support",
        }

        logger.info(f"Fallback ticket created: {ticket['summary']}")
        result = write_ticket(ticket, conversation_history=prep_res['log_history'])
        
        return result

    def post(self, shared: Dict, prep_res: Dict, exec_res: Dict) -> str:
        """Store ticket and update shared state."""
        if not exec_res or not isinstance(exec_res, dict):
            logger.error(f"Invalid exec_res in post: {exec_res}")
            if "response" not in shared:
                shared["response"] = {}
            shared["response"]["text"] = "I encountered an error creating your ticket. Please try again."
            return "default"

        status = exec_res.get("status")
        ticket_link = exec_res.get("ticket_link", "")
        message = exec_res.get("message", "")
        issue_fixed = prep_res.get("issue_fixed", False)

        # store link in case user asks again
        if ticket_link:
            shared["ticket_link"] = ticket_link
            logger.info(f"Ticket link stored: {ticket_link}")

        if "response" not in shared:
            shared["response"] = {}
        
        # handle different statuses
        if status == "already_exists":
            if issue_fixed:
                shared["response"]["text"] = (
                    f"I found that I already created a documentation ticket for this resolved issue.\n\n"
                    f"{ticket_link}\n\n"
                    f"The ticket has been closed since your issue was resolved. "
                    f"If you have any other questions, feel free to ask!"
                )
                shared["response"]["action_taken"] = "ticket_exists_resolved"
                shared["response"]["requires_followup"] = False
            else:
                shared["response"]["text"] = (
                    f"I found that a ticket was already created for this issue.\n\n"
                    f"{ticket_link}\n\n"
                    f"Our support team will review this and get back to you shortly. "
                    f"If you have additional information, please let me know."
                )
                shared["response"]["action_taken"] = "ticket_exists"
                shared["response"]["requires_followup"] = True

        elif status == "jira_unavailable":
            if issue_fixed:
                shared["response"]["text"] = (
                    f"I'm glad I could help resolve your issue! "
                    f"I've saved a documentation ticket locally for future reference.\n\n"
                    f"{ticket_link}\n\n"
                    f"If you have any other questions, feel free to ask!"
                )
                shared["response"]["action_taken"] = "create_resolved_ticket_local"
                shared["response"]["requires_followup"] = False
            else:
                shared["response"]["text"] = (
                    f"I was unable to connect to our ticketing system at this time. "
                    f"However, I've saved your issue details locally.\n\n"
                    f"{ticket_link}\n\n"
                    f"Please contact IT support directly, or try creating a ticket again later. "
                    f"I can help you with other questions in the meantime."
                )
                shared["response"]["action_taken"] = "ticket_saved_locally"
                shared["response"]["requires_followup"] = True
            
        elif status == "success":
            if issue_fixed:
                shared["response"]["text"] = (
                    f"I'm glad I could help resolve your issue! I've created a documentation ticket that has been automatically closed.\n\n"
                    f"Ticket Link: {ticket_link}\n"
                    f"This ticket documents the solution for future reference. "
                    f"If you have any other questions, feel free to ask!"
                )
                shared["response"]["action_taken"] = "create_resolved_ticket"
                shared["response"]["requires_followup"] = False
            else:
                shared["response"]["text"] = (
                    f"I've created a support ticket for your issue.\n\n"
                    f"Ticket Link: {ticket_link}\n\n"
                    f"Our support team will review this and get back to you shortly."
                )
                shared["response"]["action_taken"] = "create_ticket"
                shared["response"]["requires_followup"] = True
            
        else:
            logger.warning(f"Unknown ticket status: {status}")
            shared["response"]["text"] = (
                f"I've processed your ticket request.\n\n"
                f"{message}\n\n"
                f"Please contact IT support if you need immediate assistance."
            )
            shared["response"]["action_taken"] = "ticket_created"
            shared["response"]["requires_followup"] = True
        
        return "default"        