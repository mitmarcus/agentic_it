# ticket_manager.py
from __future__ import annotations
import json
import uuid
import datetime
import re
from pathlib import Path
from typing import Optional, Dict, List


def write_ticket_file(
    ticket: Dict,
    out_dir: Optional[str] = "out",
    filename: Optional[str] = None,
) -> Path:
    """
    Write the provided ticket dictionary as a JSON file.
    Handles directory creation and filename generation.
    """
    base = Path(__file__).resolve().parent
    out_path = (base / out_dir) if out_dir else base
    out_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if not filename:
        filename = f"ticket_{ts}_{uuid.uuid4().hex}.json"

    else:
        filename = f"{filename}_{ts}.json"

    file_path = out_path / filename

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(ticket, f, ensure_ascii=False, indent=2)

    return file_path

def find_existing_ticket(shared: Dict, history: List[Dict]) -> Dict:
    """Return existing ticket dict if found in shared store or conversation history, else None."""
    # Check shared store first
    if shared.get("ticket"):
        return shared["ticket"]

    # Scan assistant messages for the exact phrase "I've created a support ticket for your issue."
    for msg in reversed(history):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "") or ""
        
        # Check for the verbatim phrase
        if "I've created a support ticket for your issue." in content:
            # Extract the ticket JSON from this message
            try:
                # look for JSON block with ticket structure
                m = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', content)
                if not m:
                    m = re.search(r'(\{[\s\S]*\})', content)
                
                if m:
                    obj = json.loads(m.group(1))
                    # Must have ALL the key ticket fields to be considered a real ticket
                    if (isinstance(obj, dict) and 
                        "project" in obj and 
                        "summary" in obj and 
                        "description" in obj and
                        obj.get("project") == "AI Service Desk"):
                        return obj
            except Exception:
                pass
            
            # If phrase found but no valid ticket JSON, return a placeholder
            # to indicate a ticket was already created
            return {
                "project": "AI Service Desk",
                "summary": "Previously created ticket",
                "description": "A ticket was already created in this conversation",
                "issue_type": "Unknown",
                "request_type": "IT Help - Detailed",
                "priority": "L3-Medium",
                "assignee": "Automatic",
                "status": "Waiting for support"
            }

    return None