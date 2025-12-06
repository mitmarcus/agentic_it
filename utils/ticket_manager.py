# ticket_manager.py
from __future__ import annotations
import json
import uuid
import datetime
import re
from pathlib import Path
from typing import Optional, Dict, List
import os
import requests
from requests.auth import HTTPBasicAuth

def create_jira_issue(ticket: Dict) -> str:
    """
    Create a Jira issue using the provided ticket dictionary and Jira client.
    Returns the issue key of the created ticket.
    """

    url = f"{os.getenv("JIRA_URL")}/rest/api/2/issue"
    project_key = os.getenv("PROJECT_KEY")


    print(f"url: {url}")
    print(f"project_key: {project_key}")

    issue_data = {
        "fields": {
            "project": {
                "key": project_key
            },
            "summary": ticket["summary"],
            "description": ticket["description"],
            "issuetype": {
                "name": ticket.get("issue_type", "IT Help")
            },
            "priority": {
                "name": ticket.get("priority", "L3-Medium")
            }
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('AIS_API_TOKEN')}"
    }
     
    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(issue_data)
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        issue_info = response.json()
        print(f"✓ Issue created successfully!")
        print(f"  Issue Key: {issue_info['key']}")
        print(f"  Issue URL: {os.getenv("JIRA_URL")}/browse/{issue_info['key']}")

        if ticket.get("fixed"):
            close_issue(issue_info['key'])
        
        return issue_info
        
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP Error: {e}")
        print(f"  Response: {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"✗ Request Error: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected Error: {e}")
        return None

def add_attachment(log_file: Path, issue_key: str) -> bool:
    """
    Add an attachment log file to the specified Jira issue.
    Returns True if successful, False otherwise.
    """
    url = f"{os.getenv("JIRA_URL")}/rest/api/2/issue/{issue_key}/attachments"
    
    headers = {
        "X-Atlassian-Token": "no-check",
        "Authorization": f"Bearer {os.getenv('AIS_API_TOKEN')}"
    }
    
    try:
        with log_file.open("rb") as f:
            files = {
                'file': (log_file.name, f, 'text/plain')
            }
            response = requests.post(
                url,
                headers=headers,
                files=files
            )
        
        response.raise_for_status()
        
        print(f"✓ Attachment '{log_file.name}' added to issue {issue_key} successfully!")
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP Error: {e}")
        print(f"  Response: {response.text}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Request Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected Error: {e}")
        return False

def write_ticket(
    ticket: Dict,
    out_dir: Optional[str] = "out",
    filename: Optional[str] = None,
    conversation_history: Optional[List[Dict]] = None,
) -> Path:
    """
    Write the provided ticket dictionary as a JSON file, and a log file with the dialogue if provided.
    Handles directory creation and filename generation.
    """
    base = Path(__file__).resolve().parent
    out_path = (base / out_dir) if out_dir else base
    out_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    file_path = out_path / filename if filename else out_path / f"ticket_{ts}.json"
    log_filename = file_path.stem + "_log.txt"
    log_path = out_path / log_filename

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(ticket, f, indent=2)

    if conversation_history:
        with log_path.open("w", encoding="utf-8") as log_f:
            log_f.write(f"Ticket created at {ts} UTC\n\n")
            for i, msg in enumerate(conversation_history, 1):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "") or ""
                log_f.write(f"{role}:\n{content}\n\n")

    jira_result = create_jira_issue(ticket)
    
    if jira_result and jira_result.get("key"):
        add_attachment(log_path, jira_result["key"])
        jira_link = f"{os.getenv('JIRA_URL')}/browse/{jira_result['key']}"
        return {
            "status": "success",
            "ticket_link": jira_link,
            "message": f"Ticket created successfully: {jira_result['key']}"
        }
    else:
        print(f"Jira creation failed, ticket saved locally: {file_path}")
        return {
            "status": "jira_unavailable",
            "ticket_link": f"Local ticket saved: {file_path.name}",
            "message": "Unable to connect to Jira. Ticket saved locally for manual creation."
        }

def find_existing_ticket(shared: Dict, history: List[Dict]) -> str:
    """Return existing ticket link if found in shared store or conversation history, else None."""
    if shared.get("ticket_link"):
        return {
            "status": "already_exists",
            "ticket_link": shared["ticket_link"],
            "message": "A ticket was already created for this conversation"
        }

    # scans bot msgs
    for msg in reversed(history):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "") or ""
        
        # checks for jira ticket
        jira_url_pattern = r'(https?://[^\s]+/browse/[A-Z]+-\d+|Ticket Link: https?://[^\s]+/browse/[A-Z]+-\d+)'
        ticket_link_match = re.search(jira_url_pattern, content)
        
        if ticket_link_match:
            ticket_url = ticket_link_match.group(1)
            return {
                "status": "already_exists",
                "ticket_link": ticket_url,
                "message": "A ticket was already created for this issue"
            }
        
        # checks for local ticket
        elif "Local ticket saved:" in content:
            local_match = re.search(r'Local ticket saved: ([^\s\(]+)', content)
            if local_match:
                return {
                    "status": "already_exists",
                    "ticket_link": f"Local ticket: {local_match.group(1)}",
                    "message": "A ticket was already created locally for this issue"
                }

    return None

def close_issue(ticket_id: str) -> str:
    """
    Close the Jira issue with the specified ticket ID.
    Returns True if successful, False otherwise.
    """
    url = f"{os.getenv("JIRA_URL")}/rest/api/2/issue/{ticket_id}/transitions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('AIS_API_TOKEN')}"
    }
    
    transition_data = {
        "transition": {
            "id": "801"
        },
        "fields": {
            "resolution": {
                "name": "Auto-closed"
            }
        }
    }
    
    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(transition_data)
        )
        response.raise_for_status()
        
        print(f"✓ Issue {ticket_id} closed successfully!")
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP Error: {e}")
        print(f"  Response: {response.text}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Request Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected Error: {e}")
        return False

def main():
    """
    Example usage of the create_jira_issue function
    """

    ticket = {
                "project": "AI Service Desk",
                "summary": "rest test ticket",
                "description": "this is a ticket",
                "issue_type": "IT Help", # if this isn't set explicitly, it fails
                "request_type": "IT Help - Detailed",
                "priority": "L3-Medium",
                "assignee": "Automatic",
                "status": "Waiting for support"
            }
    
    # Example: Create a task
    result = create_jira_issue(ticket)
    add_attachment(Path("requirements.txt"), result["key"]) if result else None
    
    if result:
        print(f"\nFull response: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    main()