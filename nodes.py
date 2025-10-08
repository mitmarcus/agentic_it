"""
Nodes for IT Support Chatbot.

All nodes follow the pattern:
- prep(): Read from shared store
- exec(): Process logic (with retries handled by Node)
- post(): Write to shared store, return action
"""
import os
import yaml
import logging
from typing import Any, Dict, List
from cremedelacreme import Node, BatchNode, AsyncNode

# Import utilities
from utils.call_llm_groq import call_llm
from utils.embedding_local import get_embedding
from utils.intent_classifier import classify_intent, extract_keywords
from utils.conversation_memory import conversation_memory
from utils.chromadb_client import query_collection
from utils.chunker import truncate_to_token_limit
from utils.redactor import redact_text, redact_dict

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Node 1: IntentClassificationNode
# ============================================================================

class IntentClassificationNode(Node):
    """Classify user query intent for routing decisions."""
    
    def prep(self, shared: Dict) -> str:
        """Read user query from shared store."""
        return shared.get("user_query", "")
    
    def exec(self, query: str) -> Dict:
        """Classify intent using utility function."""
        intent_result = classify_intent(query)
        keywords = extract_keywords(query)
        
        logger.info(f"Intent: {intent_result['intent']} (confidence: {intent_result['confidence']:.2f})")
        
        return {
            "intent_data": intent_result,
            "keywords": keywords
        }
    
    def post(self, shared: Dict, prep_res: str, exec_res: Dict) -> str:
        """Write intent data to shared store."""
        shared["intent"] = exec_res["intent_data"]
        shared["keywords"] = exec_res["keywords"]
        return "default"


# ============================================================================
# Node 2: EmbedQueryNode
# ============================================================================

class EmbedQueryNode(Node):
    """Generate embedding for user query."""
    
    def __init__(self):
        # Retry 3 times if embedding fails
        super().__init__(max_retries=3, wait=1)
    
    def prep(self, shared: Dict) -> str:
        """Read user query."""
        return shared.get("user_query", "")
    
    def exec(self, query: str) -> List[float]:
        """Generate embedding vector."""
        if not query:
            raise ValueError("Empty query for embedding")
        
        embedding = get_embedding(query)
        logger.info(f"Generated embedding: {len(embedding)} dimensions")
        return embedding
    
    def exec_fallback(self, prep_res: str, exc: Exception) -> List[float]:
        """Fallback: return zero vector if embedding fails."""
        logger.error(f"Embedding failed after retries: {exc}")
        # Return zero vector as fallback
        embed_dim = int(os.getenv("EMBEDDING_DIM", "384"))
        return [0.0] * embed_dim
    
    def post(self, shared: Dict, prep_res: str, exec_res: List[float]) -> str:
        """Write embedding to shared store."""
        shared["query_embedding"] = exec_res
        return "default"


# ============================================================================
# Node 3: SearchKnowledgeBaseNode
# ============================================================================

class SearchKnowledgeBaseNode(Node):
    """Retrieve relevant documents from ChromaDB."""
    
    def __init__(self):
        super().__init__(max_retries=2, wait=1)
    
    def prep(self, shared: Dict) -> List[float]:
        """Read query embedding."""
        return shared.get("query_embedding", [])
    
    def exec(self, query_embedding: List[float]) -> List[Dict]:
        """Query ChromaDB for similar documents."""
        if not query_embedding or sum(query_embedding) == 0:
            logger.warning("Empty or zero embedding, skipping search")
            return []
        
        top_k = int(os.getenv("RAG_TOP_K", "3"))
        min_score = float(os.getenv("RAG_MIN_SCORE", "0.7"))
        
        # Query collection
        results = query_collection(query_embedding, top_k=top_k)
        
        # Log scores for debugging
        if results:
            scores = [r["score"] for r in results]
            logger.info(f"Retrieved scores: {scores}, min_score threshold: {min_score}")
        
        # Filter by minimum score
        filtered_results = [r for r in results if r["score"] >= min_score]
        
        logger.info(f"Found {len(results)} results, {len(filtered_results)} above threshold")
        
        return filtered_results
    
    def post(self, shared: Dict, prep_res: List[float], exec_res: List[Dict]) -> str:
        """Write results and compile context."""
        shared["retrieved_docs"] = exec_res
        
        # Compile RAG context from documents
        if exec_res:
            max_context_tokens = int(os.getenv("RAG_MAX_CONTEXT_TOKENS", "2000"))
            
            context_parts = []
            for i, doc in enumerate(exec_res):
                context_parts.append(f"[Document {i+1}] (score: {doc['score']:.2f})")
                context_parts.append(f"Source: {doc['metadata'].get('source_file', 'unknown')}")
                context_parts.append(doc['document'])
                context_parts.append("")
            
            rag_context = "\n".join(context_parts)
            rag_context = truncate_to_token_limit(rag_context, max_context_tokens)
            
            shared["rag_context"] = rag_context
            return "docs_found"
        else:
            shared["rag_context"] = ""
            return "no_docs"


# ============================================================================
# Node 4: DecisionMakerNode (Agent)
# ============================================================================

class DecisionMakerNode(Node):
    """Agent node that decides which action to take next."""
    
    def __init__(self):
        super().__init__(max_retries=3, wait=2)
    
    def prep(self, shared: Dict) -> Dict:
        """Gather all context for decision making."""
        # Get conversation history
        session_id = shared.get("session_id", "")
        history = conversation_memory.get_conversation_history(session_id, limit=10)
        
        # Format history - exclude the very last message (current query)
        history_str = ""
        for msg in history[:-1][-6:]:  # Last 3 exchanges, excluding current query
            history_str += f"{msg['role']}: {msg['content']}\n"
        
        # Get retrieved docs summary
        retrieved_docs = shared.get("retrieved_docs", [])
        doc_summaries = ""
        for i, doc in enumerate(retrieved_docs[:3]):
            doc_summaries += f"{i+1}. {doc['document'][:100]}... (score: {doc['score']:.2f})\n"
        
        # Get workflow state
        workflow_state = shared.get("workflow_state")
        workflow_status = "None"
        if workflow_state:
            workflow_status = f"In progress: step {workflow_state.get('current_step_index', 0) + 1}"
        
        # Track search attempts to prevent infinite loops
        search_count = shared.get("search_count", 0)
        max_searches = int(os.getenv("AGENT_MAX_TURNS", "5"))
        
        return {
            "user_query": shared.get("user_query", ""),
            "intent": shared.get("intent", {}),
            "conversation_history": history_str,
            "doc_summaries": doc_summaries,
            "workflow_status": workflow_status,
            "turn_count": shared.get("turn_count", 0),
            "search_count": search_count,
            "max_searches": max_searches
        }
    
    def exec(self, context: Dict) -> Dict:
        """Call LLM to decide next action."""
        prompt = f"""### CONTEXT
User Query: "{context['user_query']}"
Intent Classification: {context['intent'].get('intent', 'unknown')} (confidence: {context['intent'].get('confidence', 0):.2f})
Conversation Turn: {context['turn_count']}

Retrieved Documents:
{context['doc_summaries'] if context['doc_summaries'] else 'No documents retrieved'}

Conversation History (last 3 messages):
{context['conversation_history'] if context['conversation_history'] else 'No previous conversation'}

Current Workflow State: {context['workflow_status']}

### YOUR ROLE
You are the decision-making component of an IT support chatbot. Your job is to 
analyze the context and decide the next action to help the employee efficiently.

### AVAILABLE ACTIONS
[1] search_kb
    Description: Search knowledge base for more information
    When to use: Current retrieved docs insufficient or off-topic
    
[2] answer
    Description: Provide direct answer using current context
    When to use: Have sufficient context to answer confidently
    
[3] troubleshoot
    Description: Start interactive troubleshooting workflow
    When to use: User has technical problem requiring step-by-step guidance
    
[4] search_tickets
    Description: Search for similar existing tickets (not implemented yet)
    When to use: Problem seems unresolvable, check if others have same issue
    
[5] create_ticket
    Description: Create new ticket for IT support (not implemented yet)
    When to use: Cannot resolve issue, needs human intervention
    
[6] clarify
    Description: Ask user for more specific details
    When to use: Query is ambiguous or missing critical information

### DECISION RULES
- IMPORTANT: You have searched {context['search_count']} times (max: {context['max_searches']}). If at max, you MUST choose 'answer' or 'clarify', NOT 'search_kb'
- If confidence < 0.7 in understanding query → clarify
- If intent = factual + good docs found → answer
- If intent = troubleshooting + no workflow started → troubleshoot (or answer if we have docs)
- If in workflow + stuck → search_tickets
- If same docs keep appearing across searches → answer with what you have
- Never create ticket without attempting resolution first
- Keep responses concise and actionable

### OUTPUT FORMAT
Respond in YAML:
```yaml
thinking: |
  <your step-by-step reasoning process>
action: <action_name>
reasoning: <why you chose this action in one sentence>
confidence: <0.0 to 1.0>
```

Think carefully and make the best decision for the user."""

        response = call_llm(prompt, max_tokens=512)
        
        # Parse YAML response
        yaml_str = response.split("```yaml")[1].split("```")[0].strip() if "```yaml" in response else response
        decision = yaml.safe_load(yaml_str)
        
        # Validate decision
        allowed_actions = ["search_kb", "answer", "troubleshoot", "search_tickets", "create_ticket", "clarify"]
        assert isinstance(decision, dict), "Decision must be a dict"
        assert "action" in decision, "Decision must have 'action' field"
        assert decision["action"] in allowed_actions, f"Action must be one of {allowed_actions}"
        
        logger.info(f"Decision: {decision['action']} (confidence: {decision.get('confidence', 0):.2f})")
        
        return decision
    
    def exec_fallback(self, prep_res: Dict, exc: Exception) -> Dict:
        """Fallback: make intelligent decision based on context when LLM fails."""
        logger.error(f"Decision making failed: {exc}")
        
        # Smart fallback based on available context
        has_docs = bool(prep_res.get("doc_summaries"))
        intent = prep_res.get("intent", {}).get("intent", "unknown")
        confidence = prep_res.get("intent", {}).get("confidence", 0)
        
        # If rate limit and we have good docs, try to answer
        if "rate limit" in str(exc).lower() and has_docs and confidence > 0.6:
            return {
                "action": "answer",
                "reasoning": "Rate limited but have relevant docs",
                "confidence": 0.5
            }
        
        # Otherwise ask for clarification
        return {
            "action": "clarify",
            "reasoning": "System error, need more information",
            "confidence": 0.3
        }
    
    def post(self, shared: Dict, prep_res: Dict, exec_res: Dict) -> str:
        """Write decision and return action."""
        shared["decision"] = exec_res
        
        # Track search attempts to prevent infinite loops
        if exec_res["action"] == "search_kb":
            search_count = shared.get("search_count", 0)
            max_searches = int(os.getenv("AGENT_MAX_TURNS", "5"))
            
            if search_count >= max_searches:
                logger.warning(f"Max search attempts ({max_searches}) reached, forcing answer")
                return "answer"  # Force answer instead of more searching
            
            shared["search_count"] = search_count + 1
            logger.info(f"Search attempt {search_count + 1}/{max_searches}")
        
        return exec_res["action"]


# ============================================================================
# Node 5: GenerateAnswerNode
# ============================================================================

class GenerateAnswerNode(Node):
    """Generate final answer using RAG context."""
    
    def __init__(self):
        super().__init__(max_retries=3, wait=2)
    
    def prep(self, shared: Dict) -> Dict:
        """Read query, context, and history."""
        session_id = shared.get("session_id", "")
        history = conversation_memory.get_conversation_history(session_id, limit=10)
        
        # Format last 2 exchanges (4 messages), excluding current query
        history_str = ""
        for msg in history[:-1][-4:]:
            history_str += f"{msg['role']}: {msg['content']}\n"
        
        return {
            "user_query": shared.get("user_query", ""),
            "rag_context": shared.get("rag_context", ""),
            "conversation_history": history_str
        }
    
    def exec(self, context: Dict) -> str:
        """Generate answer using LLM."""
        prompt = f"""### YOUR ROLE
You are a helpful IT support assistant. Provide accurate, concise answers based on official documentation.

### CONTEXT FROM KNOWLEDGE BASE
{context['rag_context'] if context['rag_context'] else 'No relevant documents found in knowledge base.'}

### CONVERSATION HISTORY
{context['conversation_history'] if context['conversation_history'] else 'No previous conversation.'}

### USER QUESTION
{context['user_query']}

### INSTRUCTIONS
1. Answer using ONLY information from the context above
2. Be concise but complete - aim for 3-5 sentences
3. Use bullet points for step-by-step instructions
4. If context insufficient, say so and offer to create a ticket
5. NEVER include sensitive information (passwords, keys, personal data)
6. Be friendly and professional

### YOUR ANSWER"""

        answer = call_llm(prompt, max_tokens=512)
        logger.info(f"Generated answer: {len(answer)} chars")
        return answer
    
    def exec_fallback(self, prep_res: Dict, exc: Exception) -> str:
        """Fallback: provide helpful message based on available context."""
        logger.error(f"Answer generation failed: {exc}")
        
        if "rate limit" in str(exc).lower():
            return ("I'm currently experiencing high API usage. However, based on the available documentation, "
                    "I can see information related to your query. Please try again in a few minutes, "
                    "or contact IT support directly for immediate assistance.")
        
        # Generic fallback with context if available
        if prep_res.get("rag_context"):
            return (f"I found relevant documentation for your query about '{prep_res['user_query']}', "
                    "but I'm unable to generate a detailed response right now. "
                    "Please check the IT knowledge base or contact support for assistance.")
        
        return ("I'm having trouble generating a response right now. "
                "Please contact IT support for direct assistance with your query.")
    
    def post(self, shared: Dict, prep_res: Dict, exec_res: str) -> str:
        """Write answer to response."""
        if "response" not in shared:
            shared["response"] = {}
        
        shared["response"]["text"] = exec_res
        shared["response"]["action_taken"] = "answer"
        shared["response"]["requires_followup"] = False
        
        return "default"


# ============================================================================
# Node 6: AskClarifyingQuestionNode
# ============================================================================

class AskClarifyingQuestionNode(Node):
    """Ask user for more details when query is ambiguous."""
    
    def __init__(self):
        super().__init__(max_retries=2, wait=1)
    
    def prep(self, shared: Dict) -> Dict:
        """Read query, intent, and conversation history."""
        session_id = shared.get("session_id", "")
        history = conversation_memory.get_conversation_history(session_id, limit=10)
        
        # Format last 2 exchanges, excluding current query
        history_str = ""
        for msg in history[:-1][-4:]:
            history_str += f"{msg['role']}: {msg['content']}\n"
        
        return {
            "user_query": shared.get("user_query", ""),
            "intent": shared.get("intent", {}),
            "conversation_history": history_str
        }
    
    def exec(self, context: Dict) -> str:
        """Generate clarifying question."""
        prompt = f"""### CONVERSATION HISTORY
{context['conversation_history'] if context['conversation_history'] else 'No previous conversation.'}

### CURRENT USER MESSAGE
"{context['user_query']}"

### INTENT CLASSIFICATION
Intent: {context['intent'].get('intent', 'unclear')} (confidence: {context['intent'].get('confidence', 0):.2f})

### YOUR TASK
Based on the conversation history and the user's current message, generate a specific, 
helpful clarifying question to better understand their issue. Consider what you already 
know from the conversation.

Be concise (1-2 sentences) and friendly.

Clarifying question:"""

        question = call_llm(prompt, max_tokens=256)  # Limit tokens for clarification
        return question.strip()
    
    def exec_fallback(self, prep_res: Dict, exc: Exception) -> str:
        """Fallback: provide generic clarifying question on error."""
        logger.error(f"Clarifying question generation failed: {exc}")
        # Generic fallback based on context
        if "rate limit" in str(exc).lower():
            return "I'm experiencing high load right now. Could you please provide more details about your issue so I can help you better?"
        return "Could you please provide more details about your issue?"
    
    def post(self, shared: Dict, prep_res: Dict, exec_res: str) -> str:
        """Write clarifying question to response."""
        if "response" not in shared:
            shared["response"] = {}
        
        shared["response"]["text"] = exec_res
        shared["response"]["action_taken"] = "clarify"
        shared["response"]["requires_followup"] = True
        
        return "default"


# ============================================================================
# Node 7: FormatFinalResponseNode
# ============================================================================

class FormatFinalResponseNode(Node):
    """Format the final response to send to user."""
    
    def prep(self, shared: Dict) -> Dict:
        """Read response data."""
        return {
            "response": shared.get("response", {}),
            "user_query": shared.get("user_query", "")
        }
    
    def exec(self, context: Dict) -> str:
        """Return formatted response (already formatted by previous nodes)."""
        response_text = context["response"].get("text", "I'm sorry, I couldn't process your request.")
        
        # Add any additional formatting here if needed
        return response_text
    
    def post(self, shared: Dict, prep_res: Dict, exec_res: str) -> str:
        """Update final response text."""
        shared["response"]["text"] = exec_res
        
        # Save to conversation memory
        session_id = shared.get("session_id", "")
        if session_id:
            conversation_memory.add_message(session_id, "assistant", exec_res)
        
        logger.info(f"Final response: {exec_res[:100]}...")
        
        return "default"


# ============================================================================
# Node 8: InteractiveTroubleshootNode
# ============================================================================

class InteractiveTroubleshootNode(Node):
    """Provide interactive troubleshooting guidance."""
    
    def __init__(self):
        super().__init__(max_retries=2, wait=1)
    
    def prep(self, shared: Dict) -> Dict:
        """Gather context for troubleshooting."""
        session_id = shared.get("session_id", "")
        history = conversation_memory.get_conversation_history(session_id, limit=5)
        
        # Get current troubleshooting state
        ts_state = shared.get("troubleshoot_state", {})
        
        return {
            "user_query": shared.get("user_query", ""),
            "intent": shared.get("intent", {}),
            "retrieved_docs": shared.get("retrieved_docs", []),
            "conversation_history": history,
            "current_step": ts_state.get("current_step", 0),
            "steps_completed": ts_state.get("steps_completed", []),
            "issue_type": ts_state.get("issue_type", "")
        }
    
    def exec(self, context: Dict) -> str:
        """Generate troubleshooting steps."""
        # Build context for LLM
        doc_context = ""
        if context["retrieved_docs"]:
            doc_context = "\n".join([
                f"- {doc['metadata'].get('title', 'Untitled')}: {doc['text'][:200]}..."
                for doc in context["retrieved_docs"][:3]
            ])
        
        history_str = ""
        if context["conversation_history"]:
            history_str = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in context["conversation_history"][-3:]
            ])
        
        # Analyze conversation to determine troubleshooting phase
        is_first_step = context['current_step'] == 0
        has_documentation = bool(doc_context)
        has_history = bool(history_str)
        
        prompt = f"""You are an IT support troubleshooting assistant. Guide the user step-by-step through their technical issue.

**CURRENT SITUATION:**
• User says: "{context['user_query']}"
• Issue type: {context['intent'].get('intent', 'unknown')}
• Troubleshooting step: {context['current_step'] + 1}
{f"• Steps tried: {', '.join(context['steps_completed'])}" if context['steps_completed'] else ""}

**AVAILABLE CONTEXT:**
{f"Documentation:\n{doc_context[:300]}..." if has_documentation else "• No official documentation found"}

{f"Recent conversation:\n{history_str}" if has_history else ""}

**YOUR TASK:**
{"**Initial Assessment** - Understand the problem:" if is_first_step else "**Next Step** - Based on what they've reported:"}
{"1. Ask 1-2 diagnostic questions (e.g., error messages, when it started, what they've tried)" if is_first_step else "1. Analyze their last response to determine what worked/didn't work"}
{"2. Explain what you're checking for and why" if is_first_step else "2. Provide ONE clear, actionable step (not multiple options)"}
{"3. Give first simple troubleshooting step to try" if is_first_step else "3. Explain what this step checks/fixes"}
4. Ask them to report specific results or observations

**SMART TROUBLESHOOTING PRINCIPLES:**
• Follow the diagnostic tree: Check simple/common issues before complex ones
• Don't repeat steps they've already mentioned trying
• If a step failed, don't suggest the same approach - pivot to alternatives
• Look for patterns: error messages, timing, specific scenarios
• Know when to escalate: If 3+ steps fail, suggest creating a ticket
{f"• Use official docs when relevant: {doc_context[:150]}..." if has_documentation else "• Without docs, rely on general IT best practices"}

**RESPONSE STYLE:**
Be conversational, not robotic. Format with bullet points for clarity. Keep it concise (3-5 sentences max).

Your response:"""

        response = call_llm(prompt, max_tokens=512)
        return response
    
    def exec_fallback(self, prep_res: Dict, exc: Exception) -> str:
        """Fallback: provide basic troubleshooting guidance on error."""
        logger.error(f"Troubleshooting guidance generation failed: {exc}")
        
        is_first_step = prep_res.get("current_step", 0) == 0
        user_query = prep_res.get("user_query", "your issue")
        
        if "rate limit" in str(exc).lower():
            if is_first_step:
                return (f"I understand you're experiencing an issue with {user_query}. "
                        "I'm currently experiencing high load, but I can still help. "
                        "Could you tell me: When did this start happening? "
                        "Are you seeing any error messages? "
                        "What have you already tried?")
            else:
                return ("I'm experiencing high load right now. "
                        "Based on what you've told me, please try restarting the device/application "
                        "and let me know if that resolves the issue.")
        
        return ("I'm having trouble generating detailed troubleshooting steps right now. "
                "Please contact IT support directly for immediate assistance, "
                "or try again in a few minutes.")
    
    def post(self, shared: Dict, prep_res: Dict, exec_res: str) -> str:
        """Update troubleshooting state and response."""
        # Update troubleshooting state
        if "troubleshoot_state" not in shared:
            shared["troubleshoot_state"] = {
                "current_step": 0,
                "steps_completed": [],
                "issue_type": prep_res["intent"].get("intent", "")
            }
        
        shared["troubleshoot_state"]["current_step"] += 1
        
        # Store response
        if "response" not in shared:
            shared["response"] = {}
        
        shared["response"]["text"] = exec_res
        shared["response"]["action_taken"] = "troubleshoot"
        shared["response"]["requires_followup"] = True
        
        logger.info(f"Troubleshooting step {shared['troubleshoot_state']['current_step']} completed")
        
        return "default"


# ============================================================================
# Node 9: NotImplementedNode (placeholder for other features)
# ============================================================================

class NotImplementedNode(Node):
    """Placeholder for features not yet implemented."""
    
    def __init__(self, feature_name: str = "This feature"):
        super().__init__()
        self.feature_name = feature_name
    
    def exec(self, prep_res: Any) -> str:
        """Return not implemented message."""
        return f"{self.feature_name} is not yet implemented. I'll help you with what I can!"
    
    def post(self, shared: Dict, prep_res: Any, exec_res: str) -> str:
        """Write message to response."""
        if "response" not in shared:
            shared["response"] = {}
        
        shared["response"]["text"] = exec_res
        shared["response"]["action_taken"] = "not_implemented"
        
        return "default"


# ============================================================================
# Offline Indexing Nodes
# ============================================================================

class LoadDocumentsNode(Node):
    """Load documents from source directory."""
    
    def prep(self, shared: Dict) -> str:
        """Get source directory from shared or env."""
        return shared.get("source_dir", os.getenv("INGESTION_SOURCE_DIR", "./data/docs"))
    
    def exec(self, source_dir: str) -> List[Dict]:
        """Load all documents from directory."""
        from utils.document_loader import load_documents_from_directory
        
        documents = load_documents_from_directory(source_dir)
        logger.info(f"Loaded {len(documents)} documents from {source_dir}")
        return documents
    
    def post(self, shared: Dict, prep_res: str, exec_res: List[Dict]) -> str:
        """Write documents to shared store."""
        shared["documents"] = exec_res
        return "default"


class ChunkDocumentsNode(BatchNode):
    """Chunk documents into smaller pieces."""
    
    def prep(self, shared: Dict) -> List[Dict]:
        """Read documents from shared store."""
        return shared.get("documents", [])
    
    def exec(self, document: Dict) -> List[Dict]:
        """Chunk a single document."""
        from utils.chunker import chunk_text
        
        content = document.get("content", "")
        metadata = document.get("metadata", {})
        
        chunk_size = int(os.getenv("INGESTION_CHUNK_SIZE", "500"))
        chunk_overlap = int(os.getenv("INGESTION_CHUNK_OVERLAP", "50"))

        chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Create chunk dicts with metadata
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                "content": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            }
            chunk_dicts.append(chunk_dict)
        
        return chunk_dicts
    
    def post(self, shared: Dict, prep_res: List[Dict], exec_res_list: List[List[Dict]]) -> str:
        """Flatten all chunks into single list."""
        all_chunks = []
        for chunk_list in exec_res_list:
            all_chunks.extend(chunk_list)
        
        shared["all_chunks"] = all_chunks
        logger.info(f"Created {len(all_chunks)} total chunks")
        return "default"


class EmbedDocumentsNode(BatchNode):
    """Generate embeddings for document chunks."""
    
    def __init__(self):
        super().__init__(max_retries=3, wait=1)
    
    def prep(self, shared: Dict) -> List[Dict]:
        """Read chunks from shared store."""
        return shared.get("all_chunks", [])
    
    def exec(self, chunk_dict: Dict) -> List[float]:
        """Embed a single chunk."""
        content = chunk_dict.get("content", "")
        embedding = get_embedding(content)
        return embedding
    
    def post(self, shared: Dict, prep_res: List[Dict], exec_res_list: List[List[float]]) -> str:
        """Store embeddings with chunks."""
        shared["all_embeddings"] = exec_res_list
        logger.info(f"Generated {len(exec_res_list)} embeddings")
        return "default"


class StoreInChromaDBNode(Node):
    """Store chunks and embeddings in ChromaDB."""
    
    def __init__(self):
        super().__init__(max_retries=2, wait=1)
    
    def prep(self, shared: Dict) -> Dict:
        """Read chunks and embeddings."""
        return {
            "chunks": shared.get("all_chunks", []),
            "embeddings": shared.get("all_embeddings", [])
        }
    
    def exec(self, data: Dict) -> int:
        """Insert into ChromaDB."""
        from utils.chromadb_client import insert_documents
        
        chunks = data["chunks"]
        embeddings = data["embeddings"]
        
        # Extract text and metadata
        chunk_texts = [c["content"] for c in chunks]
        chunk_metadata = [c["metadata"] for c in chunks]
        
        # Insert into ChromaDB
        insert_documents(chunk_texts, embeddings, chunk_metadata)
        
        return len(chunks)
    
    def post(self, shared: Dict, prep_res: Dict, exec_res: int) -> str:
        """Log completion."""
        logger.info(f"Successfully indexed {exec_res} chunks in ChromaDB")
        shared["indexed_count"] = exec_res
        return "default"
