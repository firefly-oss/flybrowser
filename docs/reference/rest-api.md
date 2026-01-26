# REST API Reference

FlyBrowser provides a comprehensive REST API for browser automation through HTTP requests.

## Base URL

When running locally:
```
http://localhost:8000
```

## Interactive Documentation

The API includes built-in interactive documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Spec**: `http://localhost:8000/openapi.json`

## Health Endpoints

### GET /health

Health check endpoint. Returns service status.

**Response:**

```json
{
  "status": "healthy",
  "version": "1.26.1",
  "uptime_seconds": 3600.5,
  "active_sessions": 5,
  "system_info": {
    "sessions": 5
  }
}
```

### GET /metrics

Get detailed service metrics.

**Response:**

```json
{
  "total_requests": 1250,
  "active_sessions": 5,
  "cache_stats": {},
  "cost_stats": {},
  "rate_limit_stats": {}
}
```

## Session Endpoints

### POST /sessions

Create a new browser session.

**Request Body:**

```json
{
  "llm_provider": "openai",
  "llm_model": "gpt-4o",
  "api_key": "sk-...",
  "headless": true,
  "browser_type": "chromium"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `llm_provider` | string | Yes | LLM provider (openai, anthropic, ollama, gemini) |
| `llm_model` | string | No | Specific model (e.g., gpt-4o, claude-3-5-sonnet) |
| `api_key` | string | Depends | API key for the LLM provider |
| `headless` | boolean | No | Run browser in headless mode (default: true) |
| `browser_type` | string | No | Browser type: chromium, firefox, webkit (default: chromium) |

**Response:**

```json
{
  "session_id": "abc123",
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z",
  "metadata": {
    "browser_type": "chromium"
  }
}
```

### GET /sessions

List all active browser sessions.

**Response:**

```json
{
  "sessions": [
    {
      "session_id": "abc123",
      "status": "active",
      "created_at": "2024-01-15T10:30:00Z",
      "last_activity": "2024-01-15T10:35:00Z",
      "llm_provider": "openai",
      "browser_type": "chromium"
    }
  ],
  "total": 1
}
```

### GET /sessions/{session_id}

Get information about a specific session.

**Response:**

```json
{
  "session_id": "abc123",
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z",
  "last_activity": "2024-01-15T10:35:00Z",
  "llm_provider": "openai",
  "llm_model": "gpt-4o",
  "browser_type": "chromium"
}
```

### DELETE /sessions/{session_id}

Delete a browser session and release resources.

**Response:**

```json
{
  "status": "deleted",
  "session_id": "abc123"
}
```

## Navigation Endpoints

### POST /sessions/{session_id}/navigate

Navigate the browser to a URL.

**Request Body:**

```json
{
  "url": "https://example.com",
  "wait_until": "domcontentloaded"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | Yes | URL to navigate to |
| `wait_until` | string | No | Load state: load, domcontentloaded, networkidle |

**Response:**

```json
{
  "success": true,
  "url": "https://example.com",
  "title": "Example Domain",
  "duration_ms": 1250
}
```

### POST /sessions/{session_id}/navigate-nl

Navigate using natural language instructions.

**Request Body:**

```json
{
  "instruction": "go to the login page",
  "use_vision": true,
  "context": {
    "conditions": {
      "requires_login": false
    },
    "constraints": {
      "timeout_seconds": 30
    }
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `instruction` | string | Yes | Natural language navigation instruction |
| `use_vision` | boolean | No | Use vision for navigation (default: true) |
| `context` | object | No | Context with conditions, constraints |

**Response:**

```json
{
  "success": true,
  "url": "https://example.com/login",
  "title": "Login",
  "navigation_type": "react",
  "error": null
}
```

## Automation Endpoints

### POST /sessions/{session_id}/extract

Extract data from the current page using natural language.

**Request Body:**

```json
{
  "query": "Get all product names and prices",
  "use_vision": false,
  "schema": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "price": {"type": "string"}
      }
    }
  },
  "context": {
    "filters": {
      "price_max": 100,
      "category": "electronics"
    },
    "preferences": {
      "sort_by": "price",
      "limit": 10
    }
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Natural language extraction query |
| `use_vision` | boolean | No | Use vision for extraction (default: false) |
| `schema` | object | No | JSON Schema for structured extraction |
| `context` | object | No | Context with filters, preferences, etc. |

**Response:**

```json
{
  "success": true,
  "data": [
    {"name": "Product A", "price": "$19.99"},
    {"name": "Product B", "price": "$29.99"}
  ],
  "cached": false,
  "metadata": {},
  "llm_usage": {
    "prompt_tokens": 1500,
    "completion_tokens": 200,
    "total_tokens": 1700,
    "cost_usd": 0.0034,
    "model": "gpt-4o",
    "calls_count": 1,
    "cached_calls": 0
  },
  "page_metrics": {
    "url": "https://shop.example.com/products",
    "title": "Products",
    "html_size_bytes": 45000,
    "html_size_kb": 43.95,
    "element_count": 150,
    "interactive_element_count": 25,
    "obstacles_detected": 1,
    "obstacles_dismissed": 1
  },
  "timing": {
    "total_ms": 2500,
    "phases": {},
    "started_at": "2024-01-15T10:30:00Z",
    "ended_at": "2024-01-15T10:30:02Z"
  }
}
```

### POST /sessions/{session_id}/act

Perform an action on the page using natural language.

**Request Body:**

```json
{
  "instruction": "click the login button",
  "use_vision": true,
  "context": {
    "form_data": {
      "input[name=email]": "user@example.com",
      "input[name=password]": "secret123"
    },
    "files": [
      {
        "field": "resume",
        "path": "/path/to/resume.pdf",
        "mime_type": "application/pdf"
      }
    ]
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `instruction` | string | Yes | Natural language action instruction |
| `use_vision` | boolean | No | Use vision for action (default: true) |
| `context` | object | No | Context with form_data, files, etc. |

**Response:**

```json
{
  "success": true,
  "action_type": "act",
  "element_found": true,
  "duration_ms": 850,
  "metadata": {},
  "llm_usage": {...},
  "page_metrics": {...},
  "timing": {...}
}
```

### POST /sessions/{session_id}/agent

Execute a complex task using the intelligent agent. This is the recommended endpoint for multi-step automation.

**Request Body:**

```json
{
  "task": "Search for 'python programming' and extract the top 5 results",
  "context": {
    "form_data": {
      "input[name=company]": "Acme Corp"
    },
    "files": [
      {
        "field": "logo",
        "path": "/path/to/logo.png",
        "mime_type": "image/png"
      }
    ],
    "filters": {
      "site": "github.com",
      "language": "python"
    },
    "preferences": {
      "max_results": 5,
      "sort_by": "stars"
    },
    "conditions": {
      "requires_login": false
    },
    "constraints": {
      "timeout_seconds": 60
    },
    "metadata": {
      "request_id": "abc123"
    }
  },
  "max_iterations": 50,
  "max_time_seconds": 1800
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task` | string | Yes | Natural language task description |
| `context` | object | No | Structured context (form_data, files, filters, preferences, conditions, constraints, metadata) |
| `max_iterations` | integer | No | Maximum iterations (default: 50) |
| `max_time_seconds` | number | No | Maximum execution time (default: 1800) |

**Context Object Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `form_data` | object | Form field selectors and values for automated filling |
| `files` | array | File upload specifications (field, path, mime_type, name) |
| `filters` | object | Data filtering criteria (e.g., price_max, category, site, filetype) |
| `preferences` | object | User preferences (e.g., max_results, sort_by, safe_search) |
| `conditions` | object | Conditional logic (e.g., requires_login, max_redirects) |
| `constraints` | object | General constraints (e.g., timeout_seconds, max_retries) |
| `metadata` | object | Tool-specific metadata (e.g., request_id, user_agent) |

**Response:**

```json
{
  "success": true,
  "task": "Search for 'python programming' and extract the top 5 results",
  "result_data": [...],
  "iterations": 8,
  "duration_seconds": 15.5,
  "final_url": "https://google.com/search?q=python+programming",
  "error_message": null,
  "execution_history": [
    {"step": 1, "action": "navigate", "details": "..."},
    {"step": 2, "action": "type", "details": "..."}
  ],
  "llm_usage": null
}
```

### POST /sessions/{session_id}/observe

Find elements on the page matching a description.

**Request Body:**

```json
{
  "query": "find the search input",
  "return_selectors": true,
  "context": {
    "filters": {
      "type": "submit",
      "visible": true
    }
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Natural language element query |
| `return_selectors` | boolean | No | Return CSS selectors (default: true) |
| `context` | object | No | Context with filters, preferences |

**Response:**

```json
{
  "success": true,
  "elements": [
    {
      "selector": "#search-input",
      "description": "Search input field",
      "tag": "input",
      "text": "",
      "visible": true,
      "actionable": true
    }
  ],
  "page_url": "https://example.com",
  "error": null
}
```

### POST /sessions/{session_id}/workflow

Execute a multi-step workflow.

**Request Body:**

```json
{
  "workflow": {
    "steps": [
      {"action": "navigate", "url": "https://example.com"},
      {"action": "click", "selector": "#login"},
      {"action": "fill", "selector": "#email", "value": "{{email}}"}
    ]
  },
  "variables": {
    "email": "user@example.com"
  }
}
```

**Response:**

```json
{
  "success": true,
  "steps_completed": 3,
  "total_steps": 3,
  "error": null,
  "step_results": [...],
  "variables": {...}
}
```

### POST /sessions/{session_id}/monitor

Monitor the page for a condition.

**Request Body:**

```json
{
  "condition": "wait for the loading spinner to disappear",
  "timeout": 30.0,
  "poll_interval": 1.0
}
```

**Response:**

```json
{
  "success": true,
  "condition_met": true,
  "elapsed_time": 5.2,
  "error": null,
  "details": {}
}
```

## Screenshot Endpoints

### POST /sessions/{session_id}/screenshot

Capture a screenshot of the current page.

**Request Body:**

```json
{
  "full_page": false,
  "mask_pii": true
}
```

**Response:**

```json
{
  "success": true,
  "screenshot_id": "ss_abc123",
  "format": "png",
  "width": 1920,
  "height": 1080,
  "data_base64": "iVBORw0KGgo...",
  "url": "https://example.com",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Recording Endpoints

### POST /sessions/{session_id}/recording/start

Start recording the browser session.

**Request Body:**

```json
{
  "video_enabled": true
}
```

**Response:**

```json
{
  "success": true,
  "recording_id": "rec_abc123",
  "video_enabled": true
}
```

### POST /sessions/{session_id}/recording/stop

Stop recording and return recording data.

**Response:**

```json
{
  "success": true,
  "recording_id": "rec_abc123",
  "duration_seconds": 45.5,
  "screenshot_count": 12,
  "video_path": "/recordings/rec_abc123.webm",
  "video_size_bytes": 2500000
}
```

## Streaming Endpoints

### POST /sessions/{session_id}/stream/start

Start a live stream of the browser session.

**Request Body:**

```json
{
  "protocol": "hls",
  "quality": "high",
  "codec": "h264",
  "width": 1920,
  "height": 1080,
  "frame_rate": 30,
  "rtmp_url": null,
  "rtmp_key": null,
  "max_viewers": 10
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `protocol` | string | No | hls, dash, rtmp (default: hls) |
| `quality` | string | No | Quality profile (default: high) |
| `codec` | string | No | h264, h265, vp9 (default: h264) |
| `width` | integer | No | Video width (default: 1920) |
| `height` | integer | No | Video height (default: 1080) |
| `frame_rate` | integer | No | Frames per second (default: 30) |
| `rtmp_url` | string | No | RTMP destination URL |
| `rtmp_key` | string | No | RTMP stream key |
| `max_viewers` | integer | No | Maximum concurrent viewers |

**Response:**

```json
{
  "success": true,
  "stream_id": "stream_abc123",
  "hls_url": "http://localhost:8000/streams/stream_abc123/playlist.m3u8",
  "dash_url": null,
  "rtmp_url": null,
  "websocket_url": null,
  "player_url": "http://localhost:8000/streams/stream_abc123/player"
}
```

### GET /sessions/{session_id}/stream/status

Get stream status.

**Response:**

```json
{
  "active": true,
  "stream_id": "stream_abc123",
  "viewers": 2,
  "duration_seconds": 120.5
}
```

### POST /sessions/{session_id}/stream/stop

Stop the stream.

**Response:**

```json
{
  "success": true,
  "stream_id": "stream_abc123",
  "duration_seconds": 300.5
}
```

## PII Endpoints

### POST /sessions/{session_id}/credentials

Store a credential securely.

**Request Body:**

```json
{
  "name": "login_password",
  "value": "secret123",
  "pii_type": "password"
}
```

**Response:**

```json
{
  "success": true,
  "credential_id": "cred_abc123"
}
```

### POST /sessions/{session_id}/secure-fill

Securely fill a form field with a stored credential.

**Request Body:**

```json
{
  "selector": "#password",
  "credential_id": "cred_abc123",
  "clear_first": true
}
```

**Response:**

```json
{
  "success": true
}
```

### POST /sessions/{session_id}/mask-pii

Mask PII in text.

**Request Body:**

```json
{
  "text": "My email is user@example.com"
}
```

**Response:**

```json
{
  "masked_text": "My email is ****@****.***"
}
```

## Error Responses

All error responses follow this format:

```json
{
  "error": "ErrorType",
  "message": "Human-readable error message",
  "details": {
    "exception": "Technical details"
  }
}
```

### Common HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Session or resource not found |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Feature not enabled |

## Example: Complete Workflow

```bash
# 1. Create a session
SESSION_ID=$(curl -s -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "llm_provider": "openai",
    "api_key": "sk-...",
    "headless": true
  }' | jq -r '.session_id')

# 2. Navigate to a page
curl -X POST "http://localhost:8000/sessions/${SESSION_ID}/navigate" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# 3. Extract data
curl -X POST "http://localhost:8000/sessions/${SESSION_ID}/extract" \
  -H "Content-Type: application/json" \
  -d '{"query": "Get the page title"}'

# 4. Take a screenshot
curl -X POST "http://localhost:8000/sessions/${SESSION_ID}/screenshot" \
  -H "Content-Type: application/json" \
  -d '{"full_page": true}'

# 5. Delete the session
curl -X DELETE "http://localhost:8000/sessions/${SESSION_ID}"
```

## See Also

- [SDK Reference](sdk.md) - Python SDK documentation
- [Configuration](configuration.md) - Service configuration options
