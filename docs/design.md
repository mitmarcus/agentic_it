# Design Doc: IT Support Chatbot

## 1. Requirements

### Problem Statement

IT support teams face repetitive inquiries about common issues (password resets, VPN setup, printer troubleshooting) that consume significant time. Employees often struggle to find relevant documentation or articulate their problems effectively, leading to unnecessary ticket creation and delayed resolution.

### Solution

An **Agentic IT Support Chatbot** that:

- Answers factual IT questions using a local knowledge base (RAG with ChromaDB)
- Guides employees through interactive troubleshooting workflows
- Monitors company network status page for ongoing incidents
- Redacts sensitive information from user inputs before processing
- Maintains conversation context across multi-turn interactions
- Operates with privacy-first principles (local embeddings, no sensitive data to external APIs)
- Provides OS-specific guidance based on user's operating system
- Supports document upload and indexing via API
- Includes observability via Langfuse tracing integration

### User Stories

1. **Quick Answer**: "As an employee, I want to quickly get answers to common IT questions without creating a ticket."

   - Example: "What is our VPN connection procedure?"
   - Expected: Bot retrieves relevant documentation and provides step-by-step instructions.

2. **Interactive Troubleshooting**: "As an employee with a printer issue, I want guided troubleshooting steps."

   - Example: "My printer won't print."
   - Expected: Bot asks diagnostic questions, provides solutions, tracks progress.

3. **Network Status Check**: "As an employee, I want to know if my issue is part of a widespread problem."

   - Example: "Is there a network outage?"
   - Expected: Bot checks company status page and informs about known service disruptions.

4. **Ticket Creation**: "As an employee with an unresolved issue, I want the bot to create a properly formatted ticket."

   - Example: After failed troubleshooting, bot offers to create ticket with collected context.
   - **Status**: Not yet implemented - placeholder node exists.

5. **Privacy Protection**: "As an IT admin, I want sensitive information redacted from logs and tickets."

   - Expected: Bot automatically redacts passwords, API keys, personal data before logging.

6. **Document Upload**: "As an IT admin, I want to upload new documentation to the knowledge base."
   - Example: Upload PDF or HTML files via API endpoint.
   - Expected: Documents are parsed, chunked, embedded, and indexed into ChromaDB.

### Success Criteria

- **Deflection Rate**: 40%+ of queries resolved without ticket creation
- **Response Time**: < 3 seconds for factual queries, < 10 seconds for RAG retrieval
- **User Satisfaction**: 80%+ positive feedback on bot interactions
- **Privacy Compliance**: Zero sensitive data leaks to external LLM APIs

---

## 2. Flow Design

> Notes for AI:
>
> 1. Consider the design patterns of agent, map-reduce, rag, and workflow. Apply them if they fit.
> 2. Present a concise, high-level description of the workflow.

### Applicable Design Patterns

1. **Agent Pattern**: The DecisionMakerNode uses agentic decision-making to route between different actions (answer, clarify, troubleshoot, search_kb, search_tickets, create_ticket).

2. **RAG Pattern**:

   - **Offline**: Ingest IT documentation → Chunk (semantic) → Embed (local model) → Store in ChromaDB
   - **Online**: User query → Embed → Retrieve top-k documents → Generate answer

3. **Workflow Pattern**: Interactive troubleshooting follows a sequential workflow with user feedback loops.

### Flow High-Level Design

The system has four main flows:

1. **QueryFlow**: Main query answering flow with agent routing
2. **IndexingFlow**: Offline document indexing for RAG
3. **SimpleQueryFlow**: Simplified flow for testing (direct search → answer)
4. **NetworkStatusFlow**: Async flow for checking company status page

#### Follow-Up Context Strategy

To handle multi-turn conversations correctly (e.g., "how to find mac address" → "im on linux"), the **`EmbedQueryNode`** handles follow-up detection locally (no LLM needed):

1. **Active Topic Tracking**: `conversation_memory` stores the last answered topic
2. **Follow-up Detection**: Short queries (≤5 words) with OS/version/confirmation words are detected as follow-ups
3. **Query Enrichment**: Follow-ups are enriched with topic context (e.g., "im on linux" → "how to find mac address - im on linux")
4. **Fallback**: If no active topic, extract IT terms from conversation history

This prevents the system from misinterpreting context switches and ensures continuity.

```mermaid
flowchart TD
    subgraph QueryFlow["Main Query Flow"]
        A[RedactInputNode] --> B[IntentClassificationNode]
        B --> C[EmbedQueryNode<br/>+ follow-up detection]
        C --> D[SearchKnowledgeBaseNode]
        D -->|docs_found| E[DecisionMakerNode]
        D -->|no_docs| E

        E -->|answer| F[GenerateAnswerNode]
        E -->|clarify| G[AskClarifyingQuestionNode]
        E -->|search_kb| C
        E -->|troubleshoot| H[InteractiveTroubleshootNode]
        E -->|search_tickets| I[NotImplementedNode]
        E -->|create_ticket| J[NotImplementedNode]

        F --> K[FormatFinalResponseNode]
        G --> K
        H --> K
        I --> K
        J --> K
    end
```

```mermaid
flowchart LR
    subgraph IndexingFlow["Offline Indexing Flow"]
        L[LoadDocumentsNode] --> M[ChunkDocumentsNode]
        M --> N[EmbedDocumentsNode]
        N --> O[StoreInChromaDBNode]
    end
```

```mermaid
flowchart LR
    subgraph NetworkStatusFlow["Network Status Flow"]
        P[StatusQueryNode]
    end
```

### Detailed Node Flow

#### Main Query Flow Nodes:

1. **RedactInputNode**

   - Redacts sensitive information (passwords, API keys, tokens) from user input
   - Stores redacted query for all downstream processing

2. **IntentClassificationNode**

   - Classifies query intent: **informative** (seeking information) or **troubleshooting** (has a problem to fix)
   - Extracts keywords for search optimization
   - Intent influences DecisionMaker routing (troubleshooting intent biases toward troubleshoot action)

3. **EmbedQueryNode**

   - Generates embedding for user query using local sentence-transformers
   - Retries up to 3 times on failure

4. **SearchKnowledgeBaseNode**

   - Queries ChromaDB with query embedding
   - Filters results by minimum score threshold
   - Returns "docs_found" or "no_docs" action

5. **DecisionMakerNode** (Agent)

   - Context: user_query, user_os, intent, conversation_history, retrieved_docs, network_status
   - Actions: `search_kb`, `troubleshoot`, `search_tickets`, `create_ticket`, `answer`, `clarify`
   - Returns action name to route flow

6. **GenerateAnswerNode**

   - Formats answer using LLM (Groq API)
   - Uses retrieved docs as context
   - Provides OS-specific instructions

7. **AskClarifyingQuestionNode**

   - Generates specific clarifying questions when query is ambiguous
   - Avoids redundant questions based on conversation history

8. **InteractiveTroubleshootNode**

   - Detects user intent changes (exit, continue, escalate)
   - Provides diagnostic reasoning and OS-specific steps
   - Tracks troubleshooting state and failed steps

9. **FormatFinalResponseNode**

   - Formats final response and saves to conversation memory

10. **NotImplementedNode**
    - Placeholder for ticket search and creation (not yet implemented)

#### Offline Indexing Flow:

```
LoadDocumentsNode → ChunkDocumentsNode → EmbedDocumentsNode → StoreInChromaDBNode
```

- **LoadDocumentsNode**: Loads .txt, .md, .html, .pdf files from source directory
- **ChunkDocumentsNode** (BatchNode): Semantic chunking with embedding-based coherence
- **EmbedDocumentsNode** (BatchNode): Generates embeddings for each chunk
- **StoreInChromaDBNode**: Inserts chunks and embeddings into ChromaDB

---

## 3. Utility Functions

> Notes for AI:
>
> 1. Understand the utility function definition thoroughly by reviewing the doc.
> 2. Include only the necessary utility functions, based on nodes in the flow.

### Implemented Utilities

1. **Call LLM** (`utils/call_llm_groq.py`)

   - **Input**: `prompt: str`, `max_tokens: int = 1024`
   - **Output**: `response: str`
   - **Necessity**: Used by most nodes for LLM-based tasks (answer generation, decision making)
   - **Implementation**: Uses Groq API with configurable model
   - **Environment Variables**: `GROQ_API_KEY`, `GROQ_MODEL`

2. **Get Embedding** (`utils/embedding_local.py`)

   - **Input**: `text: str`
   - **Output**: `vector: List[float]` (384-dim for all-MiniLM-L6-v2)
   - **Necessity**: Used for query embedding and document indexing
   - **Implementation**: Uses sentence-transformers with model caching
   - **Environment Variables**: `EMBED_MODEL`

3. **Intent Classifier** (`utils/intent_classifier.py`)

   - **Input**: `query: str`
   - **Output**: `str` - Returns intent type: "informative" or "troubleshooting"
   - **Necessity**: Helps DecisionMaker route queries appropriately
   - **Implementation**: Rule-based binary classification using frozenset keyword matching
   - **Intent Definitions**:
     - `informative`: User wants information, explanations, how-to guides, contact info
     - `troubleshooting`: User has a problem/error they need help fixing
   - **Additional Functions**: `extract_keywords()`, `is_greeting()`, `is_farewell()`

4. **Conversation Memory** (`utils/conversation_memory.py`)

   - **Input**: `session_id: str`, operations (add_message, get_history, set_workflow_state)
   - **Output**: Session data, conversation history, workflow state
   - **Necessity**: Maintains context across multi-turn conversations
   - **Implementation**: In-memory session store with workflow tracking and cleanup

5. **ChromaDB Client** (`utils/chromadb_client.py`)

   - **Input**: Various (query_vector, document_chunks, metadata)
   - **Output**: Search results, insertion confirmations
   - **Necessity**: Interface for vector database operations
   - **Functions**:
     - `initialize_client()` - Creates ChromaDB client (server or persistent mode)
     - `get_collection()` - Gets or creates collection with cosine similarity
     - `insert_documents()` - Inserts chunks with embeddings and metadata
     - `query_collection()` - Queries collection with embedding vector
     - `delete_documents_by_source()` - Deletes documents by source file
     - `get_collection_stats()` - Returns collection statistics
   - **Environment Variables**: `CHROMADB_MODE`, `CHROMADB_HOST`, `CHROMADB_PORT`, `CHROMADB_COLLECTION`, `CHROMADB_PERSIST_DIR`

6. **Document Chunker** (`utils/chunker.py`)

   - **Input**: `text: str, chunk_size: int, chunk_overlap: int`
   - **Output**: `List[str]` (text chunks)
   - **Necessity**: Breaks large documents into embeddable chunks
   - **Implementation**: Semantic chunking using spaCy sentence tokenization and embedding similarity
   - **Functions**:
     - `chunk_text()` - Main chunking function
     - `semantic_chunk_sentences()` - Groups sentences by semantic coherence
     - `truncate_to_token_limit()` - Truncates text to token limit
   - **Environment Variables**: `INGESTION_CHUNK_SIZE`, `INGESTION_CHUNK_OVERLAP`, `INGESTION_SIMILARITY_THRESHOLD`, `SPACY_MODEL`

7. **Redactor** (`utils/redactor.py`)

   - **Input**: `text: str, patterns: tuple[str, ...]`
   - **Output**: `redacted_text: str`
   - **Necessity**: Removes sensitive information before logging/processing
   - **Patterns**: API keys, passwords, tokens, AWS keys, private keys
   - **Functions**:
     - `redact_text()` - Redacts patterns from text
     - `redact_dict()` - Recursively redacts dictionary values
     - `is_sensitive()` - Checks if text contains sensitive data
     - `get_redaction_summary()` - Reports what was redacted

8. **Document Parser** (`utils/document_parser.py`)

   - **Input**: `filepath: str`
   - **Output**: `{"text": str, "metadata": dict}`
   - **Necessity**: Parses various document formats for indexing
   - **Supported Formats**: .pdf (with pdfplumber), .html/.htm (with BeautifulSoup)
   - **Classes**:
     - `PDFParser` - Extracts text with formatting from PDFs
     - `HTMLParser` - Extracts text from HTML with style preservation

9. **Status Retrieval** (`utils/status_retrieval.py`)
   - **Input**: None (reads from company status page)
   - **Output**: `List[Dict]` with event titles and messages
   - **Necessity**: Checks company network status for ongoing incidents
   - **Implementation**: Uses Playwright for async web scraping
   - **Functions**:
     - `scrape_session()` - Main async scraping function
     - `format_status_results()` - Formats results for display

### Not Yet Implemented

10. **Jira Client** (`utils/jira_client.py`) ⚠️ **PLANNED**
    - **Purpose**: Interface with Jira for ticket operations
    - **Planned Functions**:
      - `search_tickets(jql: str) -> List[dict]`
      - `create_ticket(project: str, summary: str, description: str, **kwargs) -> dict`
      - `update_ticket(ticket_id: str, fields: dict) -> dict`
      - `add_comment(ticket_id: str, comment: str)`
    - **Environment Variables**: `JIRA_URL`, `JIRA_USER`, `JIRA_API_TOKEN`, `JIRA_PROJECT`

---

## 4. Data Design

> Notes for AI: Try to minimize data redundancy

### Shared Store Schema

The shared store uses an in-memory dictionary structure for real-time operations, with ChromaDB for persistent knowledge storage.

```python
shared = {
    # Session Context
    "session_id": "uuid-string",
    "user_os": "windows",  # User's operating system for tailored responses

    # Input Processing
    "original_query": "My VPN keeps disconnecting, password is abc123",
    "user_query": "My VPN keeps disconnecting, password is [REDACTED]",  # After redaction
    "had_sensitive_data": True,  # Flag if redaction occurred
    "redaction_notice": "Your message contained sensitive data that was removed for security.",

    # Intent Classification (binary)
    "intent": "troubleshooting",  # String: "informative" or "troubleshooting"

    # Conversation State (from ConversationMemory)
    "conversation_history": [
        {"role": "user", "content": "...", "timestamp": "..."},
        {"role": "assistant", "content": "...", "timestamp": "..."}
    ],

    # Network Status Results (from StatusQueryNode)
    "status_results": [
        {"title": "Network Outage - Building A", "message": "Investigation in progress"}
    ],

    # RAG Results
    "query_embedding": [0.123, 0.456, ...],  # 384-dim vector
    "retrieved_docs": [
        {
            "id": "doc-123",
            "content": "VPN troubleshooting steps...",
            "metadata": {
                "source_file": "vpn_guide.md",
                "category": "networking"
            },
            "score": 0.92
        }
    ],

    # Decision Making (Agent Loop)
    "decision": "answer",  # search_kb | answer | clarify | troubleshoot | not_implemented
    "search_count": 0,  # Tracks number of KB searches (max 3)

    # Troubleshooting State (from ConversationMemory workflow_state)
    "troubleshoot_state": {
        "issue": "VPN disconnection",
        "steps": [
            "Check internet connection",
            "Restart VPN client",
            "Check firewall settings"
        ],
        "current_step_index": 1,
        "status": "in_progress"  # in_progress | resolved | failed
    },

    # Final Response
    "response": "Based on the documentation, here are the steps...",
    "needs_input": False,  # True if waiting for user response
    "is_clarifying": False,  # True if asking clarifying question
}
```

### ChromaDB Schema

**Collection Name**: `it_support_docs` (from env: `CHROMADB_COLLECTION`)

**Document Structure**:

```python
{
    "id": "source_file-chunk_index",  # e.g., "vpn_guide.md-0"
    "embedding": [0.123, 0.456, ...],  # 384-dim vector
    "document": "Text content of the chunk",
    "metadata": {
        "source_file": "vpn_setup_guide.md",
        "chunk_index": 0,
        "title": "VPN Setup Guide",  # Extracted from document
        "indexed_at": "2025-01-15T10:30:00Z"
    }
}
```

### Conversation Memory Schema

Managed by `ConversationMemory` class in `utils/conversation_memory.py`:

```python
session_data = {
    "session_id": "uuid",
    "created_at": "ISO timestamp",
    "last_activity": "ISO timestamp",
    "conversation_history": [
        {"role": "user", "content": "...", "timestamp": "..."},
        {"role": "assistant", "content": "...", "timestamp": "..."}
    ],
    "workflow_state": {
        "issue": "VPN problem",
        "steps": ["step 1", "step 2"],
        "current_step_index": 0,
        "status": "in_progress"
    }
}
```

---

## 5. Node Design

> Notes for AI: Carefully decide whether to use Batch/Async Node/Flow.

### Shared Store Access Patterns

All nodes follow the pattern: **Read in `prep()` → Process in `exec()` → Write in `post()`**

### Node Specifications

---

#### 1. RedactInputNode (NEW)

- **Type**: Regular Node
- **Purpose**: Remove sensitive data (passwords, tokens, API keys) from user input
- **Steps**:
  - **prep**: Read `shared["user_query"]`
  - **exec**: Call `redact_text(query)` from `utils/redactor.py`
  - **post**:
    - Store original in `shared["original_query"]`
    - Write redacted to `shared["user_query"]`
    - Set `shared["had_sensitive_data"]` flag
    - Return `"default"`

---

#### 2. IntentClassificationNode

- **Type**: Regular Node
- **Purpose**: Classify user query intent for routing decisions
- **Steps**:
  - **prep**: Read `shared["user_query"]`
  - **exec**: Call `classify_intent(query)` utility
  - **post**: Write `shared["intent"]`, return `"default"`

---

#### 3. EmbedQueryNode

- **Type**: Regular Node
- **Purpose**: Generate embedding for user query
- **Max Retries**: 3
- **Steps**:
  - **prep**: Read `shared["user_query"]`
  - **exec**: Call `get_embedding(query)` utility
  - **post**: Write `shared["query_embedding"]`, return `"default"`

---

#### 4. SearchKnowledgeBaseNode

- **Type**: Regular Node
- **Purpose**: Retrieve relevant documents from ChromaDB
- **Max Retries**: 2
- **Steps**:
  - **prep**: Read `shared["query_embedding"]`, `shared.get("search_count", 0)`
  - **exec**: Query ChromaDB with embedding, top_k=5
  - **post**:
    - Write `shared["retrieved_docs"]`
    - Increment `shared["search_count"]`
    - Return `"default"`

---

#### 5. DecisionMakerNode (Agent Node)

- **Type**: Regular Node with Agent pattern
- **Purpose**: Decide next action based on context
- **Max Retries**: 2, Wait: 1 second
- **Steps**:
  - **prep**: Read query, intent, docs, search_count, conversation_history
  - **exec**:
    - Call LLM with decision prompt
    - Available actions: `search_kb`, `answer`, `clarify`, `troubleshoot`, `not_implemented`
    - Parse YAML response
  - **post**:
    - Write `shared["decision"]`
    - Return decision as action (routes to appropriate node)

---

#### 6. GenerateAnswerNode

- **Type**: Regular Node
- **Purpose**: Generate final answer using retrieved context
- **Max Retries**: 2, Wait: 1 second
- **Steps**:
  - **prep**: Read query, retrieved_docs, conversation_history, user_os, redaction_notice
  - **exec**: Call LLM with answer generation prompt
  - **post**: Write `shared["response"]`, return `"default"`

---

#### 7. AskClarifyingQuestionNode

- **Type**: Regular Node
- **Purpose**: Ask user for more details when query is ambiguous
- **Max Retries**: 2, Wait: 1 second
- **Steps**:
  - **prep**: Read query, intent, retrieved_docs, conversation_history
  - **exec**: Call LLM to generate clarifying question
  - **post**:
    - Write `shared["response"]`
    - Set `shared["is_clarifying"] = True`
    - Return `"default"`

---

#### 8. InteractiveTroubleshootNode

- **Type**: Regular Node
- **Purpose**: Guide user through troubleshooting steps
- **Max Retries**: 2, Wait: 1 second
- **Steps**:
  - **prep**: Read query, docs, conversation_history, workflow_state from memory
  - **exec**:
    - Generate troubleshooting steps if new
    - Advance steps based on user feedback
    - Call LLM to format response
  - **post**:
    - Write `shared["response"]`
    - Update `shared["troubleshoot_state"]`
    - Set `shared["needs_input"] = True`
    - Return `"default"`

---

#### 9. FormatFinalResponseNode

- **Type**: Regular Node
- **Purpose**: Format response with metadata for API output
- **Steps**:
  - **prep**: Read response, is_clarifying, needs_input, had_sensitive_data
  - **exec**: Compile response object with all flags
  - **post**: Write final `shared["response"]`, return `"default"`

---

#### 10. NotImplementedNode

- **Type**: Regular Node
- **Purpose**: Placeholder for unimplemented features (ticket creation, Jira integration)
- **Steps**:
  - **prep**: Read query
  - **exec**: Return message indicating feature is not available
  - **post**: Write `shared["response"]`, return `"default"`

---

#### 11. StatusQueryNode (NEW)

- **Type**: AsyncNode
- **Purpose**: Check company network status page for ongoing incidents
- **Steps**:
  - **prep_async**: Read `shared["user_query"]`
  - **exec_async**: Call `scrape_session()` with Playwright
  - **post_async**: Write `shared["status_results"]`, return `"default"`

---

### Indexing Flow Nodes

#### 12. LoadDocumentsNode

- **Type**: Regular Node
- **Purpose**: Load documents from source directory
- **Steps**:
  - **prep**: Read `shared["source_dir"]`, `shared.get("source_file")`
  - **exec**:
    - Parse files using `document_parser.py`
    - Handle PDF and HTML formats
  - **post**: Write `shared["documents"]`, return `"default"`

---

#### 13. ChunkDocumentsNode

- **Type**: BatchNode
- **Purpose**: Split documents into semantic chunks
- **Steps**:
  - **prep**: Return list from `shared["documents"]`
  - **exec(doc)**: Call `chunk_text(doc["text"])` for each document
  - **post**: Write `shared["chunks"]` (flattened list), return `"default"`

---

#### 14. EmbedDocumentsNode

- **Type**: BatchNode
- **Purpose**: Generate embeddings for all chunks
- **Steps**:
  - **prep**: Return `shared["chunks"]`
  - **exec(chunk)**: Call `get_embedding(chunk)` for each chunk
  - **post**: Write `shared["embeddings"]`, return `"default"`

---

#### 15. StoreInChromaDBNode

- **Type**: Regular Node
- **Purpose**: Store embedded chunks in ChromaDB
- **Steps**:
  - **prep**: Read chunks, embeddings, metadata from shared
  - **exec**: Call `insert_documents()` from `chromadb_client.py`
  - **post**: Write `shared["index_result"]`, return `"default"`

---

## 6. Implementation Plan

### Phase 1: Foundation ✅ COMPLETE

- ✅ Set up Docker Compose environment (FastAPI, ChromaDB, Langfuse)
- ✅ Implement core utilities: LLM wrapper (Groq), embeddings (sentence-transformers), intent classifier
- ✅ Implement conversation memory with session management
- ✅ Implement ChromaDB client utility (server and persistent modes)
- ✅ Implement semantic document chunker (spaCy-based)
- ✅ Implement document parser (PDF and HTML)
- ✅ Build offline indexing flow: `LoadDocuments → ChunkDocuments → EmbedChunks → StoreInChromaDB`
- ✅ Test with IT documentation from Confluence exports

### Phase 2: Basic RAG ✅ COMPLETE

- ✅ Implement `IntentClassificationNode` (rule-based binary: informative/troubleshooting)
- ✅ Implement `EmbedQueryNode` (with retry logic)
- ✅ Implement `SearchKnowledgeBaseNode` (ChromaDB cosine similarity)
- ✅ Implement `GenerateAnswerNode` (with context from RAG)
- ✅ Build simple Q&A flow: `SimpleQueryFlow` for testing
- ✅ Test with factual IT queries

### Phase 3: Agentic Routing ✅ COMPLETE

- ✅ Implement `DecisionMakerNode` (agent pattern with action routing)
- ✅ Implement `AskClarifyingQuestionNode`
- ✅ Implement `InteractiveTroubleshootNode` (workflow state management)
- ✅ Implement `FormatFinalResponseNode`
- ✅ Build main `QueryFlow` with decision routing (agent loop)
- ✅ Test multi-turn conversations with session management

### Phase 4: Security & Observability ✅ COMPLETE

- ✅ Implement redactor utility (pattern-based sensitive data removal)
- ✅ Implement `RedactInputNode` at flow entry
- ✅ Add Langfuse tracing with `@trace_flow` decorator
- ✅ Add logging throughout nodes
- ✅ Test redaction and tracing end-to-end

### Phase 5: Status & Upload Features ✅ COMPLETE

- ✅ Implement `StatusQueryNode` (async with Playwright)
- ✅ Implement `NetworkStatusFlow` for status page checking
- ✅ Implement file upload endpoint with PDF/HTML parsing
- ✅ Implement document re-indexing and deletion
- ✅ Test file upload and status checking

### Phase 6: Jira Integration ⚠️ PLANNED

- ⚠️ Implement Jira client utility (`utils/jira_client.py`)
- ⚠️ Implement ticket search node
- ⚠️ Implement ticket creation node
- ⚠️ Implement major incident detection
- ⚠️ Replace `NotImplementedNode` with actual ticket functionality
- ⚠️ Test ticket operations end-to-end

---

## 7. Optimization Strategy

### Initial Evaluation

- **Human Intuition**: Test with real employee queries
- **Metrics to Track**:
  - Query routing accuracy (intent classification)
  - RAG retrieval quality (relevance of top-k docs)
  - Answer quality (factual accuracy, helpfulness)
  - Response latency per node

### RAG Improvements

Based on the article "Practical Improvements for RAG Systems", the following optimizations have been evaluated:

#### 1. Reranking (Cross-Encoder) ✅ ACTIVE

- **File**: `utils/reranker.py`
- **Purpose**: Re-scores top candidates using a cross-encoder for higher precision
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (configurable)
- **Config**: `RERANKER_ENABLED=true`, `RERANKER_MODEL`

#### 2. Hybrid Search (BM25 + Vector) ❌ REMOVED

- **Status**: Removed in commit 76c5719 due to excessive latency
- **Reason**: BM25 index building and RRF fusion added too much overhead for real-time queries

#### 3. MMR Diversity (Maximal Marginal Relevance) ❌ REMOVED

- **Status**: Removed in commit 76c5719 due to excessive latency
- **Reason**: Diversity calculation on embeddings added latency without significant retrieval quality improvement

#### 4. Query Expansion

- **File**: `utils/query_expansion.py`
- **Purpose**: Uses LLM to generate alternative phrasings for better recall
- **Features**: Standard expansion, HyDE (Hypothetical Document Embeddings)
- **Config**: `QUERY_EXPANSION_ENABLED=false`, `HYDE_ENABLED=false`
- **Warning**: Adds LLM latency per query (disabled by default)

#### 5. Metadata Filtering ❌ REMOVED

- **Status**: Removed in commit 76c5719 due to excessive latency
- **Reason**: Query analysis and pre-filtering overhead outweighed filtering benefits

#### 6. Feedback Loop ✅ ACTIVE

- **File**: `utils/feedback.py`
- **Purpose**: Collects user feedback AND adjusts retrieval scores based on feedback history
- **Endpoints**: `POST /feedback`, `GET /feedback/stats`
- **Storage**: JSONL file at `logs/feedback.jsonl`
- **Config**: `FEEDBACK_ENABLED=true`, `FEEDBACK_ADJUSTMENT_ENABLED=true`, `FEEDBACK_STORAGE_PATH`

**Feedback-Aware Retrieval**:

- Documents with >60% positive feedback are boosted up to +0.15
- Documents with >50% negative feedback are penalized up to -0.2
- Minimum 3 feedback entries required before adjustments apply
- Adjustments are cached for 5 minutes to avoid repeated file reads
- Applied after reranking, before final result selection

### Search Pipeline Order

The search pipeline applies improvements in this order (simplified after latency optimization):

1. **Vector Search** → Initial retrieval from ChromaDB
2. **Reranking** → Re-score with cross-encoder for precision (if enabled)
3. **Feedback Adjustment** → Boost/penalize based on user feedback history

```
Query → Embed → Vector Search → [Reranking] → [Feedback Adjustment] → Top-K Results
```

**Note**: BM25 hybrid search, MMR diversity, and metadata filtering were removed in commit 76c5719 due to excessive latency. The simplified pipeline provides faster response times while maintaining acceptable retrieval quality.

### Flow-Level Optimizations

1. **Context Management**:

   - Limit conversation history to last 3-5 exchanges
   - Use RAG to retrieve only most relevant context
   - Implement sliding window for long conversations

2. **Retrieval Quality**:

   - Tune `RAG_TOP_K` and `RAG_MIN_SCORE` parameters
   - Implement re-ranking after initial retrieval (cross-encoder)

3. **Agent Decision Making**:
   - Provide clear, non-overlapping action definitions
   - Include few-shot examples in decision maker prompt
   - Implement confidence thresholds for automatic escalation

### Micro-Optimizations

1. **Prompt Engineering**:

   - Use structured output format (YAML) for reliability
   - Include explicit instructions to avoid sensitive data
   - Add examples for complex tasks (ticket creation)

2. **Caching**:

   - Cache embeddings for frequently asked questions
   - Cache LLM responses for identical queries (with session awareness)
   - Cache Jira search results (with TTL)

3. **Model Selection**:
   - Use faster Groq models for simple tasks (classification)
   - Reserve larger models for complex reasoning (decision making)
   - Fine-tune local embedding model on IT domain

---

## 8. Reliability & Testing

### Node-Level Reliability

1. **Retry Configuration**:

   - LLM calls: `max_retries=3, wait=2` (handle rate limits)
   - Database operations: `max_retries=2, wait=1`
   - External APIs (Jira): `max_retries=3, wait=5`

2. **Validation in `exec()`**:

   ```python
   def exec(self, prep_res):
       response = call_llm(prompt)
       # Validate structure
       parsed = yaml.safe_load(response)
       assert "action" in parsed
       assert parsed["action"] in ALLOWED_ACTIONS
       return parsed
   ```

3. **Graceful Fallbacks**:
   ```python
   def exec_fallback(self, prep_res, exc):
       logging.error(f"Node failed after retries: {exc}")
       return "I'm having trouble processing that. Could you rephrase your question?"
   ```

### Testing Strategy

Implemented tests in `tests/` directory:

1. **Node Tests** (pytest):

   - `test_batch_node.py` - BatchNode functionality
   - `test_async_batch_node.py` - AsyncBatchNode
   - `test_async_parallel_batch_node.py` - Parallel batch processing

2. **Flow Tests**:

   - `test_flow_basic.py` - Basic flow execution
   - `test_flow_composition.py` - Nested flows
   - `test_async_flow.py` - AsyncFlow functionality

3. **Framework Tests**:
   - `test_fall_back.py` - Fallback mechanism
   - `test_tracing.py` - Langfuse integration

### Langfuse Observability

All flows use `@trace_flow` decorator for tracing:

```python
from langfuse_tracing import trace_flow

@trace_flow
def create_query_flow():
    # Flow is automatically traced
    ...
```

Provides:

- End-to-end request tracing
- Node execution timing
- LLM call monitoring
- Error tracking

---

## 9. Deployment Architecture

### Container Structure

```yaml
# docker-compose.yml structure
services:
  chromadb:
    - Persistent vector database
    - Stores document embeddings
    - Port: 8001 (mapped from internal 8000)

  langfuse-server:
    - Observability platform
    - Trace storage and visualization
    - Port: 3000

  chatbot:
    - FastAPI + Uvicorn server
    - Node orchestration engine
    - Groq API client for LLM calls
    - Local embedding generation
    - Port: 8000
    - Depends on: chromadb, langfuse-server
```

### Environment Configuration

**Key Settings** (from `.env`):

- **LLM**: `GROQ_API_KEY`, `GROQ_MODEL=llama-3.3-70b-versatile`
- **Embeddings**: `EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2` (local)
- **ChromaDB**: `CHROMADB_MODE=server`, `CHROMADB_HOST=chromadb:8000`
- **RAG**: `RAG_TOP_K=5`, `RAG_MIN_SCORE=0.0`
- **Agent**: `MAX_SEARCH_KB_ITERATIONS=3`
- **Langfuse**: `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`

### Security Considerations

1. **API Keys**: Stored in `.env`, never logged
2. **Sensitive Data**: Redacted by `RedactInputNode` before processing
3. **Local Embeddings**: No document content sent to external services
4. **CORS**: Configured for frontend origins
5. **Session Cleanup**: Automatic cleanup of stale sessions

---

## 10. Future Enhancements

### Immediate Priorities

1. **Jira Integration**: Implement ticket search and creation
2. **Major Incident Detection**: Check for known issues before troubleshooting
3. **Better Chunking**: Improve semantic chunking for complex documents

### Phase 2 Features

1. **Multi-Language Support**: Detect language and route appropriately
2. **Proactive Notifications**: Alert users about known issues
3. **Analytics Dashboard**: Track common issues and resolution rates
4. **Fine-Tuned Embeddings**: Train on IT documentation corpus
5. **Human-in-the-Loop**: Route uncertain responses for review
6. **Feedback Collection**: Improve responses based on user ratings

### Advanced Capabilities

1. **Multi-Modal Support**: Process screenshots and error images
2. **Automated Resolution**: Execute simple fixes with consent
3. **Integration Hub**: Connect to AD, ServiceNow, Teams
4. **Code Execution**: Run diagnostic scripts (sandboxed)

---

## Appendix A: File Structure

```
agentic_it/
├── main.py                          # FastAPI entry point (endpoints, lifespan)
├── nodes.py                         # All node definitions (14 nodes)
├── flows.py                         # Flow orchestration (4 flows)
├── models.py                        # Pydantic request/response models
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container image definition
├── docker-compose.yml               # Multi-container orchestration
├── cremedelacreme/                  # Node/Flow framework
│   ├── __init__.py                 # Core abstractions
│   └── __init__.pyi                # Type stubs
├── langfuse_tracing/               # Tracing utilities
│   ├── __init__.py
│   ├── core.py                     # Core tracing logic
│   ├── decorator.py                # @trace_flow decorator
│   └── config.py                   # Langfuse configuration
├── docs/
│   └── design.md                   # This document
├── utils/
│   ├── call_llm_groq.py           # LLM wrapper (Groq)
│   ├── embedding_local.py         # Local embedding generation
│   ├── intent_classifier.py       # Rule-based intent classification
│   ├── conversation_memory.py     # Session management
│   ├── chromadb_client.py         # Vector DB interface
│   ├── chunker.py                 # Semantic document chunking
│   ├── redactor.py                # Sensitive data removal
│   ├── document_parser.py         # PDF/HTML parsing
│   └── status_retrieval.py        # Status page scraping
├── data/
│   ├── docs/                       # IT documentation (HTML from Confluence)
│   └── uploads/                    # Uploaded documents
├── logs/
│   └── chatbot_*.log              # Application logs
├── tests/
│   ├── test_batch_node.py         # BatchNode tests
│   ├── test_async_flow.py         # AsyncFlow tests
│   ├── test_flow_basic.py         # Basic flow tests
│   └── test_tracing.py            # Langfuse tracing tests
├── frontend/                       # Next.js chat UI
│   ├── app/
│   └── package.json
└── chroma_data/                    # ChromaDB persistent storage
```

---

## Appendix B: Example Interactions

### Example 1: Factual Query

```
User: What is our VPN connection procedure?
Flow: [RedactInput] → [IntentClassify] → [EmbedQuery] → [SearchKB] → [DecisionMaker: answer] → [GenerateAnswer]
Bot: To connect to the VPN:
     1. Open Cisco AnyConnect
     2. Enter vpn.company.com
     3. Use your AD credentials
     4. Approve MFA request

     Source: AnyConnect How-to-setup documentation
```

### Example 2: Troubleshooting

```
User: My printer won't print
Flow: [DecisionMaker: troubleshoot] → [InteractiveTroubleshoot]
Bot: Let's troubleshoot this together. First, is the printer showing any error lights?
User: Yes, orange light blinking
Bot: That indicates a paper jam. Can you open the paper tray and check?
User: Found it, removed the paper
Bot: Great! Try printing a test page now.
User: It works!
Bot: Excellent! Issue resolved.
```

### Example 3: Network Status Check

```
User: Is there an outage?
Flow: [NetworkStatusFlow] → [StatusQueryNode (async)]
Bot: Current network status:
     ⚠️ SharePoint: Degraded performance (investigating)
     ✅ VPN: Operational
     ✅ Email: Operational
```

### Example 4: Clarification Needed

```
User: It's not working
Flow: [DecisionMaker: clarify] → [AskClarifyingQuestion]
Bot: I'd like to help! Could you tell me:
     - What application or service isn't working?
     - What error message do you see?
     - When did this start happening?
```

### Example 5: Sensitive Data Redaction

```
User: My password abc123 doesn't work for VPN
Flow: [RedactInput] → ...
Bot: ⚠️ Note: Sensitive information was removed from your message for security.

     Let's help with your VPN login issue. Are you using your Active Directory
     credentials? If so, try resetting your password at password.company.com
```

---

## Appendix C: Prompt Templates

### Decision Maker Prompt

````yaml
### CONTEXT
User Query: "{user_query}"
Intent Classification: {intent}
Search Count: {search_count}/3

Retrieved Documents:
{doc_summaries}

Conversation History:
{conversation_history}

### AVAILABLE ACTIONS
[1] search_kb
    When: Need more information from knowledge base (max 3 searches)

[2] answer
    When: Have sufficient context to answer directly

[3] troubleshoot
    When: User has technical problem requiring step-by-step guidance

[4] clarify
    When: Query is ambiguous or needs more details

[5] not_implemented
    When: User requests ticket creation or Jira-related features

### DECISION RULES
- If search_count >= 3 and still no good docs → answer with what you have
- If intent = informative + good docs found → answer
- If intent = troubleshooting → troubleshoot
- If query is too vague → clarify

### OUTPUT FORMAT
Respond in YAML:
```yaml
thinking: |
  <your step-by-step reasoning>
action: <action_name>
````

```

### Answer Generation Prompt

```

### YOUR ROLE

You are a helpful IT support assistant. Provide accurate, concise answers
based on official documentation.

### CONTEXT FROM KNOWLEDGE BASE

{rag_context}

### CONVERSATION HISTORY

{conversation_history}

### USER QUESTION

{user_query}

### USER OPERATING SYSTEM

{user_os or "Unknown"}

### INSTRUCTIONS

1. Answer using ONLY information from the context above
2. Be concise but complete - aim for 3-5 sentences
3. Use bullet points for step-by-step instructions
4. If context insufficient, acknowledge and provide general guidance
5. Tailor instructions to user's OS if known
6. Be friendly and professional

### YOUR ANSWER

```

### Troubleshooting Prompt

```

### YOUR ROLE

Generate troubleshooting steps for an IT issue.

### CONTEXT

{retrieved_docs}

### USER ISSUE

{user_query}

### INSTRUCTIONS

1. Generate 3-5 specific troubleshooting steps
2. Order from simple to complex
3. Make each step actionable and verifiable
4. Include what to check after each step

### OUTPUT FORMAT (YAML)

```yaml
steps:
  - "Step 1: Check X and verify Y"
  - "Step 2: Try doing Z"
  - "Step 3: If still failing, check A"
```

```

---

**End of Design Document**

---

**Version**: 1.6
**Last Updated**: Dec 1, 2025
**Author**: Marcus Mitelea
```
