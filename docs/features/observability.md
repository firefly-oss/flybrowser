# Observability

FlyBrowser provides comprehensive observability capabilities for monitoring, debugging, and analyzing browser automation sessions.

## Overview

The observability layer provides three key capabilities:

- **Command Logging**: Structured logging of all SDK operations, LLM calls, and tool executions
- **Source Capture**: HTML snapshots, DOM trees, and HAR (HTTP Archive) network logs
- **Live View**: Real-time browser streaming with iFrame embedding and optional user control

## Quick Start

```python
from flybrowser import FlyBrowser
from flybrowser.observability import ObservabilityConfig

async with FlyBrowser(
    llm_provider="openai",
    api_key="sk-...",
    observability_config=ObservabilityConfig(
        enable_command_logging=True,
        enable_source_capture=True,
        enable_live_view=True,
        live_view_port=8765,
    ),
) as browser:
    # Live view is available at ws://localhost:8765
    print(f"Live View: {browser.get_live_view_url()}")
    
    await browser.goto("https://example.com")
    result = await browser.extract("Get the data")
    
    # Export all captured data
    browser.export_observability("./output/")
```

## Command Logging

Track every operation, LLM call, and tool execution with structured logging.

### Features

- Full operation history with timing
- LLM prompt/response logging (optional)
- Tool execution tracking
- Error and exception capture
- Export to JSON, JSONL, or SQLite

### Configuration

```python
from flybrowser.observability import ObservabilityConfig

config = ObservabilityConfig(
    enable_command_logging=True,
    log_llm_prompts=True,        # Log LLM prompts
    log_llm_responses=True,      # Log LLM responses
    max_log_entries=10000,       # Max entries to keep
    auto_export_logs=True,       # Auto-export on session end
    log_export_path="./logs/",
)
```

### Direct Usage

```python
from flybrowser.observability import CommandLogger

logger = CommandLogger(session_id="my-session")

# Log a command with context manager
with logger.log_command("goto", {"url": "https://example.com"}) as entry:
    # Perform the operation
    await page.goto("https://example.com")
    entry.success = True
    entry.result = {"status": 200}

# Query history
history = logger.get_history()
errors = logger.query(status="error")
llm_calls = logger.query(command_type="llm")

# Export
logger.export("session_log.json")
logger.export("session_log.jsonl", format="jsonl")
```

### Log Entry Structure

Each command entry includes:

```python
{
    "id": "cmd_abc123",
    "timestamp": "2026-01-25T10:30:00Z",
    "command": "extract",
    "parameters": {"query": "Get the data"},
    "duration_ms": 1250,
    "success": True,
    "result": {"data": "..."},
    "error": None,
    "llm_usage": {
        "prompt_tokens": 450,
        "completion_tokens": 120,
        "total_tokens": 570,
        "cost_usd": 0.0023,
    },
    "page_state": {
        "url": "https://example.com",
        "title": "Example Page",
    },
}
```

### Querying History

```python
# Get all entries
all_entries = logger.get_history()

# Filter by command type
goto_commands = logger.query(command="goto")
llm_commands = logger.query(command_type="llm")

# Filter by status
errors = logger.query(status="error")
successes = logger.query(status="success")

# Filter by time range
recent = logger.query(since="2026-01-25T10:00:00Z")

# Combined filters
slow_llm = logger.query(
    command_type="llm",
    min_duration_ms=5000,
)
```

## Source Capture

Capture HTML snapshots, DOM trees, and network traffic for debugging and replay.

### Features

- Full HTML snapshots with resources
- DOM tree serialization
- HAR (HTTP Archive) network logging
- Screenshot integration
- Resource capture (images, CSS, JS)

### Configuration

```python
config = ObservabilityConfig(
    enable_source_capture=True,
    capture_resources=True,           # Capture page resources
    max_resource_size_bytes=5*1024*1024,  # 5MB max per resource
    max_snapshots=100,                # Max snapshots to keep
    auto_capture_on_navigation=True,  # Auto-capture on page load
    capture_har=True,                 # Enable HAR logging
)
```

### Direct Usage

```python
from flybrowser.observability import SourceCaptureManager

capture = SourceCaptureManager(session_id="my-session")

# Capture HTML snapshot
snapshot = await capture.capture_html(page)
print(f"Captured: {snapshot.url}")
print(f"Size: {len(snapshot.html)} bytes")

# Capture with resources
snapshot = await capture.capture_html(
    page,
    capture_resources=True,
    capture_screenshot=True,
)

# Start HAR capture
await capture.start_har_capture(page)
# ... perform operations ...
har = await capture.stop_har_capture()

# Export all captures
capture.export_all("./captures/")
```

### HTML Snapshot Structure

```python
{
    "id": "snap_abc123",
    "timestamp": "2026-01-25T10:30:00Z",
    "url": "https://example.com",
    "title": "Example Page",
    "html": "<!DOCTYPE html>...",
    "resources": [
        {
            "url": "https://example.com/style.css",
            "type": "stylesheet",
            "content": "...",
        },
    ],
    "screenshot_base64": "...",
}
```

### HAR Log Structure

HAR logs follow the HTTP Archive 1.2 specification:

```python
{
    "log": {
        "version": "1.2",
        "entries": [
            {
                "startedDateTime": "2026-01-25T10:30:00Z",
                "request": {
                    "method": "GET",
                    "url": "https://example.com/api/data",
                    "headers": [...],
                },
                "response": {
                    "status": 200,
                    "content": {...},
                },
                "time": 125,
            },
        ],
    },
}
```

## Live View

Stream browser sessions in real-time with WebSocket-based live view.

### Features

- Real-time browser streaming via WebSocket
- Embeddable iFrame for dashboards
- Optional viewer control (view-only or interactive)
- Multiple viewer support
- Authentication support

### Configuration

```python
from flybrowser.observability import ObservabilityConfig, StreamQuality, ControlMode

config = ObservabilityConfig(
    enable_live_view=True,
    live_view_host="0.0.0.0",
    live_view_port=8765,
    live_view_quality=StreamQuality.HIGH,
    live_view_control_mode=ControlMode.VIEW_ONLY,  # or INTERACT
    live_view_require_auth=True,
    live_view_auth_token="secret-token",
    live_view_max_viewers=10,
)
```

### Stream Quality Options

| Quality | Resolution | Frame Rate | Bandwidth |
|---------|------------|------------|-----------|
| `LOW` | 640x480 | 10 fps | ~500 kbps |
| `MEDIUM` | 1280x720 | 15 fps | ~1.5 Mbps |
| `HIGH` | 1920x1080 | 30 fps | ~3 Mbps |
| `ULTRA` | 1920x1080 | 60 fps | ~6 Mbps |

### Control Modes

| Mode | Description |
|------|-------------|
| `VIEW_ONLY` | Viewers can only watch (default) |
| `INTERACT` | Viewers can click and type |
| `ASSIST` | Viewers can assist but not override |

### Direct Usage

```python
from flybrowser.observability import LiveViewServer, StreamConfig, ControlMode

server = LiveViewServer(
    host="0.0.0.0",
    port=8765,
    default_control_mode=ControlMode.VIEW_ONLY,
)

# Start server
await server.start()

# Attach to browser page
await server.attach(page)

# Get embed URLs
ws_url = server.get_websocket_url()  # ws://localhost:8765/ws
embed_url = server.get_embed_url()   # http://localhost:8765/embed
iframe_html = server.get_embed_html()  # <iframe src="...">

# Stop server
await server.stop()
```

### Embedding Live View

Embed the live view in your dashboard:

```html
<!-- Using the embed URL -->
<iframe 
    src="http://localhost:8765/embed"
    width="1280" 
    height="720"
    frameborder="0">
</iframe>

<!-- Or with authentication -->
<iframe 
    src="http://localhost:8765/embed?token=secret-token"
    width="1280" 
    height="720"
    frameborder="0">
</iframe>
```

### WebSocket Protocol

Connect directly for custom integrations:

```javascript
const ws = new WebSocket("ws://localhost:8765/ws");

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    
    switch (msg.type) {
        case "frame":
            // Display frame
            img.src = "data:image/jpeg;base64," + msg.data;
            break;
        case "page_state":
            // Update page info
            console.log(msg.url, msg.title);
            break;
    }
};

// Send control messages (if INTERACT mode)
ws.send(JSON.stringify({
    type: "click",
    x: 100,
    y: 200,
}));
```

## ObservabilityManager

The `ObservabilityManager` provides unified access to all observability features:

```python
from flybrowser.observability import ObservabilityConfig, ObservabilityManager

config = ObservabilityConfig(
    enable_command_logging=True,
    enable_source_capture=True,
    enable_live_view=True,
    live_view_port=8765,
)

manager = ObservabilityManager(config)

# Attach to page
await manager.attach(page)

# Use context managers for operations
with manager.log_command("goto", {"url": "https://example.com"}):
    await page.goto("https://example.com")

# Capture snapshot
snapshot = await manager.capture_snapshot()

# Get live view URL
live_url = manager.get_live_view_url()

# Detach and export
await manager.detach()
manager.export_all("./output/")
```

## Integration with FlyBrowser SDK

When using `observability_config` with FlyBrowser, everything is automatically wired:

```python
from flybrowser import FlyBrowser
from flybrowser.observability import ObservabilityConfig

async with FlyBrowser(
    llm_provider="openai",
    api_key="sk-...",
    observability_config=ObservabilityConfig(
        enable_command_logging=True,
        log_llm_prompts=True,
        enable_source_capture=True,
        capture_har=True,
        enable_live_view=True,
        live_view_port=8765,
    ),
) as browser:
    # All operations are automatically logged
    await browser.goto("https://example.com")
    
    # HAR capture is automatic
    result = await browser.extract("Get the data")
    
    # Get live view URL
    live_url = browser.get_live_view_url()
    embed_html = browser.get_live_view_html()
    
    # Export everything
    browser.export_observability("./output/")
```

## Completion Page

In non-headless mode, FlyBrowser displays an **interactive Completion Page** after agent tasks complete. This page provides rich observability data:

**Displayed Information:**
- Execution metrics (duration, iterations, tokens, cost)
- LLM usage breakdown (model, provider, token counts, API calls, latency)
- Tools executed with individual timings and success/failure status
- Reasoning steps timeline (thought → action) for each ReAct cycle
- Interactive JSON tree for exploring complex nested result data
- Session metadata, reasoning strategy, and stop reason

**Error Display (on failure):**
- Error message explaining what went wrong
- Optional stack trace for debugging automation issues

**Data Robustness:**
- Handles both ReActStep objects and dictionary representations
- Gracefully handles missing or incomplete data with sensible defaults
- Normalizes LLM usage metrics (tokens, cost, latency)
- Never fails to render due to malformed data

See [Agent Documentation](agent.md#completion-page) for full details.

```python
# View completion page after agent execution
browser = FlyBrowser(
    llm_provider="openai",
    api_key="sk-...",
    headless=False,  # Required to see completion page
)

async with browser:
    result = await browser.agent("Complete the task")
    # Completion page visible with full execution details
```

## Use Cases

### Debugging Automation Failures

```python
config = ObservabilityConfig(
    enable_command_logging=True,
    log_llm_prompts=True,
    log_llm_responses=True,
    enable_source_capture=True,
    auto_capture_on_navigation=True,
)

async with FlyBrowser(observability_config=config) as browser:
    try:
        result = await browser.agent("Complete the checkout")
    except Exception as e:
        # All steps are logged for debugging
        browser.export_observability("./debug/")
        raise
```

### Monitoring Dashboard

```python
config = ObservabilityConfig(
    enable_live_view=True,
    live_view_port=8765,
    live_view_control_mode=ControlMode.VIEW_ONLY,
)

# Embed in your monitoring dashboard
# <iframe src="http://server:8765/embed"></iframe>
```

### Session Recording for Review

```python
config = ObservabilityConfig(
    enable_command_logging=True,
    enable_source_capture=True,
    capture_resources=True,
    capture_har=True,
)

async with FlyBrowser(observability_config=config) as browser:
    await browser.agent("Perform the workflow")
    
    # Export for review
    browser.export_observability("./session_recording/")
    # Creates:
    # - ./session_recording/commands.json
    # - ./session_recording/captures/
    # - ./session_recording/har.json
```

### Team Collaboration

```python
config = ObservabilityConfig(
    enable_live_view=True,
    live_view_port=8765,
    live_view_control_mode=ControlMode.ASSIST,  # Team can assist
    live_view_require_auth=True,
    live_view_auth_token=os.environ["TEAM_TOKEN"],
    live_view_max_viewers=5,
)
```

## Environment Variables

```bash
# Command logging
FLYBROWSER_OBSERVABILITY_LOGGING=true
FLYBROWSER_OBSERVABILITY_LOG_LLM=true
FLYBROWSER_OBSERVABILITY_MAX_ENTRIES=10000

# Source capture
FLYBROWSER_OBSERVABILITY_CAPTURE=true
FLYBROWSER_OBSERVABILITY_CAPTURE_HAR=true

# Live view
FLYBROWSER_LIVE_VIEW_ENABLED=true
FLYBROWSER_LIVE_VIEW_PORT=8765
FLYBROWSER_LIVE_VIEW_AUTH=true
FLYBROWSER_LIVE_VIEW_TOKEN=secret-token
```

## Best Practices

### 1. Enable Logging in Development

```python
config = ObservabilityConfig(
    enable_command_logging=True,
    log_llm_prompts=True,
    log_llm_responses=True,
)
```

### 2. Use HAR for API Debugging

```python
config = ObservabilityConfig(
    enable_source_capture=True,
    capture_har=True,
)
```

### 3. Secure Live View in Production

```python
config = ObservabilityConfig(
    enable_live_view=True,
    live_view_require_auth=True,
    live_view_auth_token=os.environ["SECURE_TOKEN"],
    live_view_control_mode=ControlMode.VIEW_ONLY,
)
```

### 4. Limit Resource Capture Size

```python
config = ObservabilityConfig(
    enable_source_capture=True,
    capture_resources=True,
    max_resource_size_bytes=1*1024*1024,  # 1MB limit
    max_snapshots=50,
)
```

## See Also

- [SDK Reference](../reference/sdk.md) - Complete API
- [Configuration](../reference/configuration.md) - All options
- [Live Streaming](streaming.md) - HLS/DASH/RTMP streaming
- [Screenshots and Recording](screenshots-recording.md) - Recording sessions
