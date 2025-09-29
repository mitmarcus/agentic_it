# Design Doc: Agentic IT Support Chatbot

## 1) Requirements

### Problem Statement

Employees and site visitors need instant IT support answers (how-tos, access requests, troubleshooting). The chatbot should answer using an internal knowledge base (RAG). When knowledge is missing, it should actively crawl allowed websites to gather the needed information. If the user is dissatisfied, angry, or explicitly asks for a human, the bot must file a Jira ticket with a concise summary and what was tried.

### Key Capabilities

- RAG over a ChromaDB index with semantic chunking and embeddings.
- Intelligent, agentic crawling when context is insufficient:
  - Start from a configured seed URL.
  - Use Playwright to render JS-heavy pages and collect content.
  - Decide which links to follow iteratively until enough context is found.
- Sentiment and intent analysis of user messages to detect anger/dissatisfaction or a request for a human handoff.
- Automatic Jira ticket creation with context, attempted steps, and citations.
- Served over HTTP (FastAPI). Entire app runs in a container (Docker).

### In-Scope Dependencies (if needed)

- Core: PyYAML, fastapi, uvicorn, python-multipart
- RAG: chromadb
- Crawling: playwright (for browser automation), beautifulsoup4, requests, crawl4ai (optional helper)
- Other: pydantic (FastAPI models), logging

### Assumptions & Constraints

- Allowed domains are configured; the crawler must respect that scope and robots.txt where applicable.
- ChromaDB runs as a separate server (container/service) that this app connects to over the network; persistence configured via env.
- LLM provider is abstracted behind utility functions (model choice can change without code changes here).
- Time and page limits apply to crawling to maintain latency and cost budgets.
- Jira credentials and project key are provided via environment variables.

### User Stories

- As an employee, I ask “How do I reset my VPN password?” and receive a step-by-step answer with citations.
- As a user, if the bot doesn’t find relevant KB entries, it proactively crawls our IT help site and then answers.
- As a frustrated user, I say “This still doesn’t work. I need a human.” and a Jira ticket is filed with the chat transcript, attempted steps, and links used.
- As an admin, I can control seed URLs, allowed domains, top_k retrieval, crawl depth, and rate limits via config.

### Non-Functional

- Latency: first-token < 3s typical when answer found in RAG; crawl path can extend to tens of seconds but bounded.
- Reliability: node-level retries; graceful fallbacks; clear stop conditions.
- Observability: structured logs; trace IDs across nodes.
- Security: restrict crawler to allowed domains; redact secrets; adhere to robots.txt if configured.

---

## 2) Applicable Design Patterns

- Agent: Decide next action (answer / crawl / escalate) based on context.
- RAG: Retrieve relevant chunks from ChromaDB to ground answers.
- Workflow: Deterministic stages (ingest -> retrieve -> decide -> answer or crawl -> loop -> answer -> sentiment/escalate).
- Batch + Async + Parallel: Crawl multiple links concurrently using AsyncParallelBatchNode/Flow.

---

## 3) Flow High-Level Design

At a high level, the chatbot follows this loop:

1. Ingest request and normalize context
2. Embed query and retrieve from ChromaDB
3. Agent decides: answer / crawl / escalate
4. If crawl: plan links -> fetch pages (parallel) -> extract & semantic-chunk -> embed & upsert -> loop back to retrieve
5. Generate final answer with citations
6. Sentiment/intent check; optionally create Jira ticket

```mermaid
flowchart TD
    A[IngestRequest] --> B[EmbedQuery]
    B --> C[RetrieveFromChroma]
    C --> D{DecideAction}

    D -->|answer| E[GenerateAnswer]
    D -->|crawl| F[PlanLinks]
    D -->|escalate| M[CreateJiraTicket]

    F --> G[AsyncParallel CrawlPages]
    G --> H[ExtractAndSemanticChunk]
    H --> I[EmbedAndUpsertToChroma]
    I --> C  %% loop: re-retrieve with updated KB

    E --> J[SentimentAndIntent]
    J -->|escalate| M
    J -->|ok| K[Finish]
    M --> K[Finish]
```

Stop conditions for the crawl loop: max depth, max pages, time budget, or “enough context” signal from DecideAction.

---

## 4) Utility Functions

> Note: Utilities are external helpers (I/O, APIs). No try/except inside utilities—let Node retries handle errors.

1. call_llm (utils/call_llm.py)

   - Input: messages or prompt (str|list)
   - Output: response text (str)
   - Necessity: All LLM steps (decisions, generation, classification).

2. embed_texts (utils/embedding.py)

   - Input: list[str]
   - Output: list[list[float]]
   - Necessity: Semantic embeddings for queries and chunks.

3. chroma_client (utils/chroma.py)

   - Input: config (host, collection, auth)
   - Output: client handle; methods: upsert, query(top_k), delete
   - Necessity: RAG storage and retrieval.

4. semantic_chunk (utils/chunking.py)

   - Input: page text (str)
   - Output: list[str] semantic chunks (use text semantics, heading structure, similarity-based boundaries)
   - Necessity: Higher-quality chunking vs naive fixed-size; improves retrieval precision.

5. fetch_page_playwright (utils/crawl.py)

   - Input: url (str), timeout, user-agent, cookies (optional)
   - Output: rendered HTML (str), final_url
   - Necessity: JS-heavy sites; Playwright renders reliably.

6. extract_text_and_links (utils/html.py)

   - Input: html (str), base_url
   - Output: text (str), links (list[str])
   - Necessity: Convert to clean text and collect candidate links (BeautifulSoup4).

7. choose_links (utils/link_planner.py)

   - Input: question (str), current hits (snippets), candidate links with context
   - Output: prioritized link plan [{url, reason}] (size N, deduped, filtered by allowed domains)
   - Necessity: Agentic selection of next crawl targets.

8. jira_create_issue (utils/jira.py)

   - Input: project_key, title, description, labels, reporter, attachments(optional)
   - Output: {key, url}
   - Necessity: Human handoff flow.

9. html_to_markdown / markdown_to_text (utils/textio.py)

   - Input: html or markdown
   - Output: markdown or plain text
   - Necessity: Normalization for chunking and context window.

10. rate_limit / semaphore (utils/limits.py)
    - Input: concurrency limit, qps
    - Output: limiter handle
    - Necessity: Prevent overload and rate-limit breaches (crawler & LLM).

Optional: crawl4ai helpers for sitemap, frontier management, dedupe.

---

## 5) Data Design (Shared Store)

> Use a well-designed shared dict to avoid duplication and keep nodes decoupled.

```python
shared = {
  "config": {
    "seed_url": "https://helpdesk.example.com",
    "allowed_domains": ["example.com"],
    "top_k": 5,
    "max_crawl_depth": 2,
    "max_pages": 20,
    "crawl_timeout_s": 15,
    "concurrency": 5,
    "rag_collection": "it_support",
    "answer_min_relevance": 0.75,
  },

  "user": {
    "id": None,
    "locale": "en",
    "sentiment": None,           # {positive|neutral|negative}
    "wants_human": False,
  },

  "chat": {
    "query": None,
    "history": [],                # previous turns (optional)
    "answer": None,
    "citations": [],              # list of {url, snippet}
  },

  "retrieval": {
    "q_embedding": None,
    "hits": [],                   # [{chunk, url, score}]
    "enough_context": False,
  },

  "crawl": {
    "frontier": [],               # list of URLs to visit next
    "visited": set(),             # URLs seen
    "plan": [],                   # [{url, reason}]
    "pages": {},                  # url -> {html, text}
    "new_chunks": [],             # semantic chunks extracted this round
    "embedded": [],               # embeddings parallel to new_chunks
    "depth_map": {},              # url -> depth
  },

  "jira": {
    "should_escalate": False,
    "issue": None,               # {key, url}
  },

  "telemetry": {
    "trace_id": None,
    "events": [],                # list of logs/events for debugging
  }
}
```

---

## 6) Node Design

> Types: Regular, BatchNode, AsyncNode, AsyncParallelBatchNode. All steps optional; avoid shared access in exec() per guidelines.

1. IngestRequest (Regular)

   - prep: Read HTTP payload; normalize {user.id, query, locale} into shared["user"], shared["chat"].
   - exec: N/A (or light validation)
   - post: Initialize telemetry trace; return "default".

2. EmbedQuery (Regular)

   - prep: Read shared["chat"]["query"].
   - exec: Compute query embedding.
   - post: Store in shared["retrieval"]["q_embedding"].

3. RetrieveFromChroma (Regular)

   - prep: q_embedding, top_k, collection handle.
   - exec: Query ChromaDB; return hits with scores and metadata (url, title, chunk_id).
   - post: shared["retrieval"]["hits"] = hits.

4. DecideAction (Regular; Agent)

   - prep: Gather question, hits summaries (top_k), thresholds, crawl counters (pages, depth).
   - exec: LLM decides next step; outputs YAML with fields:
     - action: {answer|crawl|escalate}
     - reason: str
     - enough_context: bool
     - desired_links: optional list of candidate URLs when action=crawl
   - post: Set shared["retrieval"]["enough_context"]. If crawl, seed plan/frontier; return action string.

5. GenerateAnswer (Regular)

   - prep: question, top hits (content + metadata), chat history.
   - exec: LLM grounded answer with citations; constrained to use provided context; if insufficient, say so and hint at escalation.
   - post: Store answer and citations in shared["chat"].

6. PlanLinks (Regular)

   - prep: question, current hits, last crawled pages' links (from crawl.pages), allowed_domains, visited, depth_map.
   - exec: choose_links utility ranks links using relevance to the question and novelty; cap to N.
   - post: shared["crawl"]["plan"] and extend frontier with new URLs and depths.

7. CrawlPages (AsyncParallelBatchNode)

   - prep_async: Return iterable of next URLs from frontier (respect depth, limits, allowed domains, not visited).
   - exec_async(item=url): Fetch with Playwright; return {url, html}.
   - post_async: Store html per url in shared["crawl"]["pages"].

8. ExtractAndSemanticChunk (BatchNode)

   - prep: Iterate over newly crawled pages.
   - exec(page): extract_text_and_links -> semantic_chunk(text) -> return {url, chunks, links}.
   - post: Update crawl.pages[url].text; aggregate new_chunks; update frontier with filtered links.

9. EmbedAndUpsertToChroma (BatchNode)

   - prep: shared["crawl"]["new_chunks"].
   - exec(chunk): embed; return (chunk_id, vector, metadata{url})
   - post: Upsert all to Chroma; clear new_chunks; mark index updated.

10. SentimentAndIntent (Regular)

- prep: last user message and conversation snippets.
- exec: LLM classification (angry/dissatisfied? asks for human?).
- post: Set shared["user"]["sentiment"] and ["wants_human"]; set jira.should_escalate accordingly.

11. CreateJiraTicket (Regular)

- prep: question, answer (if any), citations, crawl attempts/plan, relevant errors.
- exec: jira_create_issue; title derived from question; description includes what was tried and links.
- post: Store issue metadata in shared["jira"]["issue"].

12. Finish (Regular)

- prep/exec: N/A
- post: Finalize response payload for API; add telemetry.

### Transitions

- IngestRequest >> EmbedQuery >> RetrieveFromChroma >> DecideAction
- DecideAction - "answer" >> GenerateAnswer >> SentimentAndIntent
- DecideAction - "crawl" >> PlanLinks >> CrawlPages >> ExtractAndSemanticChunk >> EmbedAndUpsertToChroma >> RetrieveFromChroma >> DecideAction
- DecideAction - "escalate" >> CreateJiraTicket >> Finish
- SentimentAndIntent - "escalate" >> CreateJiraTicket >> Finish
- SentimentAndIntent - "ok" >> Finish

### Edge Cases & Guards

- Crawl limits: max_depth, max_pages, time budget; short-circuit when enough_context=True.
- Robots.txt honored if enabled; disallow external domains.
- Duplicate URLs and content dedupe.
- Non-HTML content (PDF): skip or convert later (out-of-scope initial version).
- Jira failures: fall back to instruct user to contact support email; keep transcript in logs.

---

## 7) API Surface (FastAPI)

- POST /chat: {user_id, query, locale?, history?} -> {answer, citations, escalated?, jira_issue?}
- GET /health: readiness/liveness

Use python-multipart for potential file uploads in future (e.g., screenshots). Responses include citations and trace_id for debugging.

---

## 8) Containerization & Runtime

- Packaging: Docker (single-service is sufficient and simpler than Kubernetes initially).
- Base image: python:3.11-slim; install Playwright and deps (chromium).
- Run server via uvicorn (FastAPI) in the container.
- Environment variables:
  - CHROMA_HOST/PORT/COLLECTION
  - JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY
  - ALLOWED_DOMAINS, SEED_URL, MAX_DEPTH, MAX_PAGES, TOP_K
  - LLM_PROVIDER/MODEL and API credentials as needed
- Ports: expose 8000 by default.
- ChromaDB: separate server/container. Configure via env: CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION.

---

## 9) Reliability, Observability, and Limits

- Retries: Enable Node max_retries where relevant (LLM, network I/O) with backoff wait.
- Fallbacks: exec_fallback to return clear messages rather than raising.
- Concurrency: semaphore for Playwright sessions; configurable.
- Logging: structured JSON logs; include trace_id and node name.
- Tests: unit tests for utilities (HTML extraction, link selection), flow tests for happy path and crawl loop, sentiment classification edge cases.

---

## 10) Initial Milestones

1. Skeleton FastAPI service wiring for basic RAG-only Q&A.
2. Add Playwright crawler subflow with semantic chunking and Chroma upsert; loop back to answer.
3. Add sentiment/intent checkpoint and Jira integration.
4. Containerize with Dockerfile; verify headless browser works in container; document env vars.
5. Hardening: limits, logging, tests, and basic dashboards.

---

## 11) Out of Scope (v1)

- Authenticated areas requiring SSO for crawling.
- Parsing binary formats (PDF/Office docs) — can be added later.
- Multi-language summarization beyond simple locale hints.
