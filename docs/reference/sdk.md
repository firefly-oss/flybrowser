# SDK Reference

Complete API reference for the FlyBrowser SDK.

## FlyBrowser Class

The main entry point for browser automation.

### Constructor

```python
FlyBrowser(
    endpoint: str | None = None,
    llm_provider: str = "openai",
    llm_model: str | None = None,
    api_key: str | None = None,
    vision_enabled: bool | None = None,
    model_config: dict | None = None,
    headless: bool = True,
    browser_type: str = "chromium",
    recording_enabled: bool = False,
    pii_masking_enabled: bool = True,
    timeout: float = 30.0,
    pretty_logs: bool = True,
    speed_preset: str = "balanced",
    log_verbosity: str = "normal",
    agent_config: AgentConfig | None = None,
    config_file: str | None = None,
    stealth_config: StealthConfig | None = None,
    observability_config: ObservabilityConfig | None = None,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endpoint` | `str \| None` | `None` | Server URL. `None` for embedded mode |
| `llm_provider` | `str` | `"openai"` | LLM provider: openai, anthropic, gemini, ollama |
| `llm_model` | `str \| None` | `None` | Model name. Uses provider default if not specified |
| `api_key` | `str \| None` | `None` | API key for LLM provider |
| `vision_enabled` | `bool \| None` | `None` | Override vision capability detection |
| `model_config` | `dict \| None` | `None` | Additional model configuration |
| `headless` | `bool` | `True` | Run browser without visible window |
| `browser_type` | `str` | `"chromium"` | Browser: chromium, firefox, webkit |
| `recording_enabled` | `bool` | `False` | Enable session recording |
| `pii_masking_enabled` | `bool` | `True` | Enable PII masking |
| `timeout` | `float` | `30.0` | Request timeout in seconds |
| `pretty_logs` | `bool` | `True` | Human-readable logs |
| `speed_preset` | `str` | `"balanced"` | fast, balanced, thorough |
| `log_verbosity` | `str` | `"normal"` | silent, minimal, normal, verbose, debug |
| `agent_config` | `AgentConfig \| None` | `None` | Custom agent configuration |
| `config_file` | `str \| None` | `None` | Path to YAML/JSON config file |
| `stealth_config` | `StealthConfig \| None` | `None` | Stealth configuration for fingerprint, CAPTCHA, proxy |
| `observability_config` | `ObservabilityConfig \| None` | `None` | Observability for logging, capture, live view |

#### Example

```python
# Embedded mode with OpenAI
browser = FlyBrowser(
    llm_provider="openai",
    api_key="sk-...",
    headless=False,
)

# Server mode
browser = FlyBrowser(
    endpoint="http://localhost:8000",
)

# With custom configuration
from flybrowser.agents.config import AgentConfig
config = AgentConfig(max_iterations=100)
browser = FlyBrowser(
    llm_provider="anthropic",
    api_key="sk-ant-...",
    agent_config=config,
)

# With stealth and observability
from flybrowser.stealth import StealthConfig
from flybrowser.observability import ObservabilityConfig

browser = FlyBrowser(
    llm_provider="openai",
    api_key="sk-...",
    stealth_config=StealthConfig(
        fingerprint_enabled=True,
        captcha_enabled=True,
        captcha_provider="2captcha",
        captcha_api_key="key",
    ),
    observability_config=ObservabilityConfig(
        enable_command_logging=True,
        enable_live_view=True,
    ),
)
```

---

## Lifecycle Methods

### start()

```python
async def start() -> None
```

Start the browser session. Called automatically when using context manager.

### stop()

```python
async def stop() -> None
```

Stop the browser session and clean up resources.

### Context Manager

```python
async with FlyBrowser(...) as browser:
    # Browser is started and ready
    await browser.goto("https://example.com")
# Browser is automatically stopped
```

---

## Navigation Methods

### goto()

```python
async def goto(
    url: str,
    wait_until: str = "domcontentloaded"
) -> None
```

Navigate directly to a URL.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | Required | URL to navigate to |
| `wait_until` | `str` | `"domcontentloaded"` | load, domcontentloaded, networkidle |

```python
await browser.goto("https://example.com")
await browser.goto("https://example.com", wait_until="networkidle")
```

### navigate()

```python
async def navigate(
    instruction: str,
    use_vision: bool = True,
    context: ActionContext | dict | None = None
) -> dict
```

Navigate using natural language.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instruction` | `str` | Required | Natural language navigation instruction |
| `use_vision` | `bool` | `True` | Use visual context |
| `context` | `ActionContext \| dict \| None` | `None` | Structured context (conditions, constraints) |

```python
result = await browser.navigate("go to the login page")
result = await browser.navigate("click on Products in the menu")

# With conditions context (future enhancement)
from flybrowser.agents.context import ContextBuilder

context = ContextBuilder()\
    .with_conditions({"requires_login": False})\
    .with_constraints({"timeout_seconds": 30})\
    .build()
result = await browser.navigate("go to dashboard", context=context)
```

---

## Core Methods

### act()

```python
async def act(
    instruction: str,
    use_vision: bool = True,
    context: ActionContext | dict | None = None,
    return_metadata: bool = True
) -> AgentRequestResponse | dict
```

Execute a browser action using natural language.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instruction` | `str` | Required | Action to perform |
| `use_vision` | `bool` | `True` | Use visual context |
| `context` | `ActionContext \| dict \| None` | `None` | Structured context (form data, files, etc.) |
| `return_metadata` | `bool` | `True` | Return full response object |

```python
# Click
result = await browser.act("click the Submit button")

# Type
result = await browser.act("type 'hello' in the search box")

# Multiple actions
result = await browser.act("scroll down and click Learn More")

# With form data context
from flybrowser.agents.context import ContextBuilder

context = ContextBuilder()\
    .with_form_data({
        "input[name=email]": "user@example.com",
        "input[name=password]": "secure_password"
    })\
    .build()
result = await browser.act("Fill and submit the login form", context=context)

# With file upload context
context = ContextBuilder()\
    .with_file("resume", "/path/to/resume.pdf", "application/pdf")\
    .build()
result = await browser.act("Upload resume to the application form", context=context)
```

### extract()

```python
async def extract(
    query: str,
    use_vision: bool = False,
    schema: dict | None = None,
    context: ActionContext | dict | None = None,
    return_metadata: bool = True
) -> AgentRequestResponse | dict
```

Extract data from the page.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | What to extract |
| `use_vision` | `bool` | `False` | Use visual context |
| `schema` | `dict \| None` | `None` | JSON Schema for structured data |
| `context` | `ActionContext \| dict \| None` | `None` | Structured context (filters, preferences) |
| `return_metadata` | `bool` | `True` | Return full response object |

```python
# Simple extraction
result = await browser.extract("get the page title")

# With schema
result = await browser.extract(
    "get product details",
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"}
        }
    }
)

# With filters and preferences context
from flybrowser.agents.context import ContextBuilder

context = ContextBuilder()\
    .with_filters({"price_max": 500, "category": "electronics"})\
    .with_preferences({"sort_by": "price", "limit": 10})\
    .build()

result = await browser.extract(
    "get product listings",
    context=context,
    schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"}
            }
        }
    }
)
```

### observe()

```python
async def observe(
    query: str,
    return_selectors: bool = True,
    context: ActionContext | dict | None = None,
    return_metadata: bool = True
) -> AgentRequestResponse | list
```

Find elements on the page.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | Description of elements to find |
| `return_selectors` | `bool` | `True` | Include CSS selectors |
| `context` | `ActionContext \| dict \| None` | `None` | Structured context (filters, preferences) |
| `return_metadata` | `bool` | `True` | Return full response object |

```python
result = await browser.observe("find all buttons")
result = await browser.observe("find the login form", return_selectors=True)

# With filters context
from flybrowser.agents.context import ContextBuilder

context = ContextBuilder()\
    .with_filters({"type": "submit", "visible": True})\
    .build()
result = await browser.observe("find buttons", context=context)
```

### agent()

```python
async def agent(
    task: str,
    context: ActionContext | dict | None = None,
    max_iterations: int = 50,
    max_time_seconds: float = 1800.0,
    return_metadata: bool = True
) -> AgentRequestResponse | dict
```

Execute autonomous multi-step tasks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | Required | Task description |
| `context` | `ActionContext \| dict \| None` | `None` | Structured context (form data, files, filters, etc.) |
| `max_iterations` | `int` | `50` | Max ReAct cycles |
| `max_time_seconds` | `float` | `1800.0` | Time limit |
| `return_metadata` | `bool` | `True` | Return full response |

**Completion Page:** In non-headless mode, displays an interactive completion page with:
- Metrics (duration, iterations, tokens, cost)
- LLM usage details (model, provider, token counts, API calls, latency)
- Tools executed with individual timings and success/failure status
- Reasoning steps visualization (thought → action timeline)
- Interactive JSON tree explorer for complex nested results
- Session metadata, strategy, and stop reason
- Error display with optional stack trace (on failure)

The completion page uses multi-layer data validation and will never fail to render due to missing or malformed data.

```python
result = await browser.agent(
    "Find the cheapest flight to Tokyo and extract the price"
)

# With simple dict context (legacy)
result = await browser.agent(
    "Complete the registration form",
    context={"email": "user@example.com", "name": "John"},
    max_iterations=30
)

# With ActionContext for form filling and file upload
from flybrowser.agents.context import ContextBuilder

context = ContextBuilder()\
    .with_form_data({
        "input[name=company]": "Acme Corp",
        "textarea[name=description]": "Enterprise solutions"
    })\
    .with_file("logo", "/path/to/logo.png", "image/png")\
    .build()

result = await browser.agent(
    "Fill out the company registration form and upload logo",
    context=context
)

# With filters for search and extraction
context = ContextBuilder()\
    .with_filters({"site": "github.com", "language": "python"})\
    .with_preferences({"max_results": 5, "sort_by": "stars"})\
    .build()

result = await browser.agent(
    "Search for Python web frameworks and extract the top repositories",
    context=context
)
```

### execute_task()

```python
async def execute_task(task: str) -> dict
```

Execute a complex task with ReAct reasoning.

```python
result = await browser.execute_task(
    "Go to google.com and search for 'python tutorials'"
)
```

---

## Search

### search()

```python
async def search(
    query: str,
    search_type: str = "auto",
    max_results: int = 10,
    ranking: str = "auto",
    return_metadata: bool = True
) -> AgentRequestResponse | dict
```

Perform a web search using the configured search provider.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | Search query or natural language instruction |
| `search_type` | `str` | `"auto"` | Type: auto, web, images, news, videos, places, shopping |
| `max_results` | `int` | `10` | Maximum results (1-50) |
| `ranking` | `str` | `"auto"` | Ranking: auto, balanced, relevance, freshness, authority |
| `return_metadata` | `bool` | `True` | Return full response object |

**Returns:** Search results with ranked items, answer boxes, and related searches.

```python
# Simple search
results = await browser.search("Python tutorials")

# News search with freshness ranking
news = await browser.search(
    "AI developments",
    search_type="news",
    ranking="freshness",
    max_results=20,
)

# Process results
for item in results.data["results"]:
    print(f"{item['title']}: {item['url']}")
```

### configure_search()

```python
def configure_search(
    provider: str | None = None,
    api_key: str | None = None,
    enable_ranking: bool | None = None,
    ranking_weights: dict[str, float] | None = None,
    cache_ttl_seconds: int | None = None
) -> None
```

Configure search settings at runtime.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | `None` | Provider: serper, google, bing, auto |
| `api_key` | `str` | `None` | API key for the provider |
| `enable_ranking` | `bool` | `None` | Enable intelligent result ranking |
| `ranking_weights` | `dict` | `None` | Custom ranking weights |
| `cache_ttl_seconds` | `int` | `None` | Result cache TTL in seconds |

```python
# Switch to Serper provider
browser.configure_search(
    provider="serper",
    api_key="your-api-key"
)

# Prioritize fresh results
browser.configure_search(
    ranking_weights={
        "bm25": 0.25,
        "freshness": 0.45,
        "domain_authority": 0.15,
        "position": 0.15,
    }
)
```

---

## Screenshot & Recording

### screenshot()

```python
async def screenshot(
    full_page: bool = False,
    mask_pii: bool = True
) -> dict
```

Capture a screenshot.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `full_page` | `bool` | `False` | Capture full scrollable page |
| `mask_pii` | `bool` | `True` | Apply PII masking |

**Returns:** Dictionary with `data_base64`, `width`, `height`, `format`

```python
screenshot = await browser.screenshot(full_page=True)
image_data = base64.b64decode(screenshot["data_base64"])
```

### start_recording() / stop_recording()

```python
async def start_recording() -> dict
async def stop_recording() -> dict
```

Record the browser session.

```python
await browser.start_recording()
# ... perform actions ...
recording = await browser.stop_recording()
```

---

## Streaming

### start_stream()

```python
async def start_stream(
    protocol: str = "hls",
    quality: str = "high",
    codec: str = "h264",
    width: int | None = None,
    height: int | None = None,
    frame_rate: int | None = None,
    rtmp_url: str | None = None,
    rtmp_key: str | None = None
) -> dict
```

Start live streaming the browser.

### stop_stream()

```python
async def stop_stream() -> dict
```

Stop the stream.

### get_stream_status()

```python
async def get_stream_status() -> dict
```

Get current stream status.

---

## Security

### store_credential()

```python
async def store_credential(
    name: str,
    value: str,
    pii_type: str = "sensitive"
) -> None
```

Store a credential securely (never logged).

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Credential identifier |
| `value` | `str` | The credential value |
| `pii_type` | `str` | Type: password, credit_card, ssn, sensitive |

```python
await browser.store_credential("my_password", "secret123", pii_type="password")
```

### secure_fill()

```python
async def secure_fill(
    selector: str,
    credential_id: str
) -> dict
```

Fill a field using a stored credential.

```python
await browser.secure_fill("#password", "my_password")
```

### mask_pii()

```python
async def mask_pii(text: str) -> str
```

Mask PII in text.

```python
masked = await browser.mask_pii("Call me at 555-1234")
# "Call me at [PHONE]"
```

---

## Usage Tracking

### get_usage_summary()

```python
async def get_usage_summary() -> dict
```

Get LLM usage statistics for the session.

```python
usage = await browser.get_usage_summary()
print(f"Total tokens: {usage['total_tokens']}")
print(f"Total cost: ${usage['total_cost_usd']:.4f}")
```

---

## Response Objects

### AgentRequestResponse

All core methods return `AgentRequestResponse` when `return_metadata=True`.

```python
class AgentRequestResponse:
    success: bool        # Operation succeeded
    data: Any           # Result data
    error: str | None   # Error message
    operation: str      # Method name
    query: str | None   # Original query
    
    # When metadata is included:
    execution: ExecutionInfo | None
    llm_usage: LLMUsageInfo | None
    metadata: dict | None
    
    def pprint() -> None  # Pretty print all details
    def to_dict() -> dict # Convert to dictionary
```

### ExecutionInfo

```python
class ExecutionInfo:
    iterations: int              # ReAct cycles used
    max_iterations: int          # Configured limit
    duration_seconds: float      # Total time
    pages_scraped: int          # Pages visited
    actions_taken: list[str]    # Tools invoked
    success: bool               # Final outcome
    summary: str                # Human-readable summary
    history: list[dict]         # Step-by-step details
```

### LLMUsageInfo

```python
class LLMUsageInfo:
    prompt_tokens: int          # Input tokens
    completion_tokens: int      # Output tokens
    total_tokens: int           # Combined
    cost_usd: float            # Estimated cost
    model: str                 # Model used
    calls_count: int           # Number of LLM calls
    cached_calls: int          # Cache hits
```

---

## Batch Operations

### batch_execute()

```python
async def batch_execute(
    tasks: list[str],
    parallel: bool = False,
    stop_on_failure: bool = False
) -> list[dict]
```

Execute multiple tasks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tasks` | `list[str]` | Required | Tasks to execute |
| `parallel` | `bool` | `False` | Run in parallel |
| `stop_on_failure` | `bool` | `False` | Stop on first failure |

```python
results = await browser.batch_execute([
    "Extract title from page A",
    "Extract title from page B",
], parallel=True)
```

---

## Properties

### Mode Information

```python
browser._mode        # "embedded" or "server"
browser._started     # Whether browser is started
browser._session_id  # Session ID (server mode)
```

---

## Supported LLM Providers

| Provider | Models | Vision |
|----------|--------|--------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo | Yes |
| Anthropic | claude-3-5-sonnet, claude-3-opus, claude-3-haiku | Yes |
| Google | gemini-1.5-pro, gemini-1.5-flash | Yes |
| Ollama | llama3, mistral, qwen, etc. | Varies |

---

## Error Handling

All methods may raise:

- `ValueError` - Invalid parameters or task
- `RuntimeError` - Browser not started or connection failed
- `asyncio.TimeoutError` - Operation timed out

```python
try:
    result = await browser.extract("get price")
    if not result.success:
        print(f"Extraction failed: {result.error}")
except ValueError as e:
    print(f"Invalid query: {e}")
except asyncio.TimeoutError:
    print("Operation timed out")
```

---

## See Also

- [Quickstart Guide](../getting-started/quickstart.md) - Get started quickly
- [Core Concepts](../getting-started/concepts.md) - Understand the framework
- [Configuration Reference](configuration.md) - Complete configuration options
