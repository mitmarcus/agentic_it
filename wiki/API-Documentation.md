# API Documentation

Complete API reference for the Agentic IT Support Chatbot.

## Table of Contents

- [Base URL](#base-url)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
- [Request/Response Models](#requestresponse-models)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

## Base URL

**Development:**
```
http://localhost:8000
```

**Production:**
```
https://your-domain.com
```

## Authentication

### API Key Authentication (Optional)

If enabled, include the API key in the request header:

```http
X-API-Key: your-api-key-here
```

Configure in `.env`:
```bash
API_KEY_REQUIRED=true
API_KEY_HEADER=X-API-Key
```

## Endpoints

### 1. Health Check

Check if the service is running.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-08T10:30:00Z"
}
```

**Status Codes:**
- `200 OK` - Service is healthy
- `503 Service Unavailable` - Service is down

### 2. Chat

Send a message to the chatbot.

**Endpoint:** `POST /chat`

**Request Body:**
```json
{
  "message": "How do I set up VPN?",
  "session_id": "user-session-123",
  "user_id": "employee@company.com"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | User's query or message |
| `session_id` | string | Yes | Unique session identifier |
| `user_id` | string | No | User identifier (email or ID) |

**Response:**
```json
{
  "response": "To set up VPN, follow these steps:\n1. Download Cisco AnyConnect...",
  "session_id": "user-session-123",
  "timestamp": "2025-01-08T10:30:00Z",
  "metadata": {
    "intent": "factual",
    "action_taken": "answer",
    "confidence": 0.92,
    "sources": [
      {
        "title": "VPN Setup Guide",
        "score": 0.89
      }
    ]
  }
}
```

**Status Codes:**
- `200 OK` - Successful response
- `400 Bad Request` - Invalid request format
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

### 3. Index Documents

Trigger document indexing (admin operation).

**Endpoint:** `POST /index`

**Request Body:**
```json
{
  "source_dir": "./data/docs",
  "recursive": true,
  "file_types": [".txt", ".md"]
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source_dir` | string | No | Directory path (default: from env) |
| `recursive` | boolean | No | Search subdirectories (default: true) |
| `file_types` | array | No | File extensions (default: [".txt", ".md"]) |

**Response:**
```json
{
  "status": "completed",
  "documents_processed": 25,
  "chunks_created": 142,
  "duration_seconds": 12.5,
  "timestamp": "2025-01-08T10:30:00Z"
}
```

**Status Codes:**
- `200 OK` - Indexing completed
- `202 Accepted` - Indexing started (async)
- `400 Bad Request` - Invalid parameters
- `500 Internal Server Error` - Indexing failed

### 4. Get Session

Retrieve session information.

**Endpoint:** `GET /sessions/{session_id}`

**Path Parameters:**

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session identifier |

**Response:**
```json
{
  "session_id": "user-session-123",
  "created_at": "2025-01-08T10:00:00Z",
  "last_activity": "2025-01-08T10:30:00Z",
  "message_count": 5,
  "conversation_history": [
    {
      "role": "user",
      "content": "How do I reset my password?",
      "timestamp": "2025-01-08T10:00:00Z"
    },
    {
      "role": "assistant",
      "content": "To reset your password...",
      "timestamp": "2025-01-08T10:00:01Z"
    }
  ],
  "workflow_state": null
}
```

**Status Codes:**
- `200 OK` - Session found
- `404 Not Found` - Session not found

### 5. Delete Session

Clear a session's conversation history.

**Endpoint:** `DELETE /sessions/{session_id}`

**Response:**
```json
{
  "status": "deleted",
  "session_id": "user-session-123"
}
```

**Status Codes:**
- `200 OK` - Session deleted
- `404 Not Found` - Session not found

### 6. List Sessions

List all active sessions (admin).

**Endpoint:** `GET /sessions`

**Query Parameters:**

| Field | Type | Description |
|-------|------|-------------|
| `limit` | integer | Max sessions to return (default: 50) |
| `active_only` | boolean | Only active sessions (default: true) |

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "user-session-123",
      "last_activity": "2025-01-08T10:30:00Z",
      "message_count": 5
    }
  ],
  "total": 1
}
```

### 7. Search Knowledge Base

Direct search without conversation (testing).

**Endpoint:** `POST /search`

**Request Body:**
```json
{
  "query": "VPN troubleshooting",
  "top_k": 5
}
```

**Response:**
```json
{
  "results": [
    {
      "document": "VPN setup guide content...",
      "score": 0.92,
      "metadata": {
        "source_file": "vpn_guide.md",
        "doc_type": "guide"
      }
    }
  ],
  "total": 5
}
```

## Request/Response Models

### ChatRequest

```typescript
interface ChatRequest {
  message: string;        // Required: User's message
  session_id: string;     // Required: Session identifier
  user_id?: string;       // Optional: User identifier
}
```

### ChatResponse

```typescript
interface ChatResponse {
  response: string;       // Bot's response text
  session_id: string;     // Session identifier
  timestamp: string;      // ISO 8601 timestamp
  metadata: {
    intent: string;       // Detected intent
    action_taken: string; // Action performed
    confidence: number;   // Confidence score (0-1)
    sources?: Array<{     // Source documents
      title: string;
      score: number;
    }>;
  };
}
```

### ErrorResponse

```typescript
interface ErrorResponse {
  error: string;          // Error type
  message: string;        // Human-readable message
  detail?: string;        // Additional details
  timestamp: string;      // ISO 8601 timestamp
}
```

## Error Handling

### Error Response Format

```json
{
  "error": "ValidationError",
  "message": "Invalid request format",
  "detail": "Field 'message' is required",
  "timestamp": "2025-01-08T10:30:00Z"
}
```

### Common Error Codes

| Code | Error | Description |
|------|-------|-------------|
| `400` | `ValidationError` | Invalid request format |
| `401` | `AuthenticationError` | Invalid or missing API key |
| `404` | `NotFoundError` | Resource not found |
| `429` | `RateLimitError` | Too many requests |
| `500` | `InternalServerError` | Server error |
| `503` | `ServiceUnavailable` | Service is down |

## Rate Limiting

Default rate limits:
- **60 requests per minute** per session
- **1000 requests per hour** per API key

Rate limit headers:
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1609459200
```

When rate limited:
```json
{
  "error": "RateLimitError",
  "message": "Rate limit exceeded",
  "detail": "Try again in 30 seconds",
  "retry_after": 30
}
```

## Examples

### Example 1: Simple Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I reset my password?",
    "session_id": "session-123"
  }'
```

**Response:**
```json
{
  "response": "To reset your password, visit the self-service portal at portal.company.com...",
  "session_id": "session-123",
  "timestamp": "2025-01-08T10:30:00Z",
  "metadata": {
    "intent": "factual",
    "action_taken": "answer",
    "confidence": 0.95,
    "sources": [
      {
        "title": "Password FAQ",
        "score": 0.93
      }
    ]
  }
}
```

### Example 2: Multi-turn Conversation

**First Message:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need help",
    "session_id": "session-456"
  }'
```

**Response:**
```json
{
  "response": "I'm here to help! Could you please provide more details about what you need assistance with?",
  "metadata": {
    "action_taken": "clarify"
  }
}
```

**Follow-up:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "My printer is not working",
    "session_id": "session-456"
  }'
```

**Response:**
```json
{
  "response": "Let's troubleshoot your printer. First, can you check if the printer is powered on?",
  "metadata": {
    "action_taken": "troubleshoot"
  }
}
```

### Example 3: Index Documents

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "source_dir": "./data/docs",
    "recursive": true
  }'
```

**Response:**
```json
{
  "status": "completed",
  "documents_processed": 25,
  "chunks_created": 142,
  "duration_seconds": 12.5
}
```

### Example 4: With API Key

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "message": "VPN setup help",
    "session_id": "session-789",
    "user_id": "employee@company.com"
  }'
```

## Interactive API Documentation

FastAPI provides interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These interfaces allow you to:
- View all endpoints
- Test API calls
- See request/response schemas
- Download OpenAPI specification

## SDK Examples

### Python SDK

```python
import requests

class AgenticITClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    def chat(self, message, session_id, user_id=None):
        payload = {
            "message": message,
            "session_id": session_id
        }
        if user_id:
            payload["user_id"] = user_id
        
        response = requests.post(
            f"{self.base_url}/chat",
            json=payload,
            headers=self.headers
        )
        return response.json()

# Usage
client = AgenticITClient()
response = client.chat("How do I set up VPN?", "session-123")
print(response["response"])
```

### JavaScript SDK

```javascript
class AgenticITClient {
  constructor(baseUrl = 'http://localhost:8000', apiKey = null) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
  }

  async chat(message, sessionId, userId = null) {
    const payload = { message, session_id: sessionId };
    if (userId) payload.user_id = userId;

    const headers = { 'Content-Type': 'application/json' };
    if (this.apiKey) headers['X-API-Key'] = this.apiKey;

    const response = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload)
    });

    return response.json();
  }
}

// Usage
const client = new AgenticITClient();
const response = await client.chat('How do I reset my password?', 'session-123');
console.log(response.response);
```

## Webhooks (Future Feature)

Planned support for webhooks to receive events:

- User query received
- Response generated
- Ticket created
- Error occurred

## WebSocket Support (Future Feature)

Planned support for real-time streaming:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/session-123');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.text);
};

ws.send(JSON.stringify({
  message: 'How do I set up VPN?'
}));
```

For more information, see [Getting Started](Getting-Started.md) and [Configuration](Configuration.md).
