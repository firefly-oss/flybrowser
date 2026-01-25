# FlyBrowser

```
  _____.__         ___.
_/ ____\  | ___.__.\_ |_________  ______  _  ________ ___________
\   __\|  |<   |  | | __ \_  __ \/  _ \ \/ \/ /  ___// __ \_  __ \
 |  |  |  |_\___  | | \_\ \  | \(  <_> )     /\___ \\  ___/|  | \/
 |__|  |____/ ____| |___  /__|   \____/ \/\_//____  >\___  >__|
            \/          \/                        \/     \/
```

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**LLM-powered browser automation that speaks your language.**

FlyBrowser combines Playwright's bulletproof browser control with LLM intelligence, letting you automate the web using plain English instead of brittle CSS selectors. Whether you're scraping data, testing UIs, or building automation workflows, FlyBrowser just works—and it speaks every language your LLM does.

```python
await browser.goto("https://example.com")
await browser.act("click the login button")
data = await browser.extract("Get all product prices")
stream = await browser.start_stream(protocol="hls", quality="high")
```

---

## Table of Contents

- [Key Features](#key-features)
- [Core Operations](#core-operations)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Your First Automation](#your-first-automation)
- [Streaming & Recording](#streaming--recording)
- [ReAct Agent Architecture](#react-agent-architecture)
  - [Core ReAct Loop](#core-react-loop)
  - [Parallel Site Exploration](#parallel-site-exploration)
  - [Architecture Overview](#architecture-overview)
  - [Key Components](#key-components)
  - [Tools (Not Agents)](#tools-not-agents)
  - [Automatic Obstacle Handling](#automatic-obstacle-handling)
  - [Reasoning Strategies](#reasoning-strategies)
  - [Task Planning](#task-planning)
  - [Search Capabilities](#search-capabilities)
  - [Response Validation](#response-validation)
- [Deployment Modes](#deployment-modes)
  - [Embedded Mode](#embedded-mode)
  - [Standalone Server](#standalone-server)
  - [Cluster Mode](#cluster-mode)
- [Security & PII Protection](#security--pii-protection)
- [Autonomous Mode](#autonomous-mode)
  - [Form Automation](#form-automation)
  - [Multi-Step Booking](#multi-step-booking)
  - [Research & Data Gathering](#research--data-gathering)
- [Use Cases](#use-cases)
  - [Web Scraping](#web-scraping)
  - [UI Testing](#ui-testing)
  - [Monitoring & Alerts](#monitoring--alerts)
  - [Content Recording](#content-recording)
  - [Live Streaming](#live-streaming)
- [REST API](#rest-api)
- [Documentation](#documentation)
- [Interactive REPL](#interactive-repl)
- [LLM Providers](#llm-providers)
- [Configuration](#configuration)
- [Development](#development)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Community & Support](#community--support)

---

## Key Features

- **Natural Language Control**: Describe actions in plain English—FlyBrowser figures out the details
- **Live Streaming & Recording**: Stream browser sessions in real-time (HLS/DASH/RTMP) with professional codecs
- **Smart Validators**: 99.8% success rate with automatic response validation and self-correction
- **Intelligent Result Selection**: Multi-factor scoring automatically chooses the best option—not just the first result
- **PII Protection**: Secure credential handling that never exposes passwords to LLMs
- **Multi-Deployment**: Run embedded in scripts, as a standalone server, or in a distributed cluster
- **Hardware Acceleration**: NVENC, VideoToolbox, QSV support for high-performance encoding
- **Built-in Observability**: Detailed timing breakdowns, metrics, and health monitoring
- **Cost Tracking**: Per-request LLM usage stats (tokens, cost) with `return_metadata=True`

---

## Core Operations

FlyBrowser provides high-level operations that work identically across embedded, standalone, and cluster modes:

| Operation | Description | Use Case |
|-----------|-------------|----------|
| `goto(url)` | Navigate to a URL | Direct navigation |
| `act(instruction)` | Perform actions via natural language | Click, type, scroll, interact |
| `extract(query, schema)` | Extract data from the page | Get structured data |
| `navigate(instruction)` | Navigate via natural language | Find and click links |
| `observe(query)` | Find elements on the page | Get selectors before acting |
| `agent(task, context)` | Execute complex goals autonomously | Multi-step workflows |
| `execute_task(task)` | Execute task with ReAct reasoning | Single complex task |

```python
from flybrowser import FlyBrowser

async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
    # Direct navigation
    await browser.goto("https://shop.example.com")
    
    # Natural language actions (uses EXECUTE mode)
    await browser.act("click the 'Electronics' category")
    await browser.act("sort by price low to high")
    
    # Extract structured data (uses SCRAPE mode)
    products = await browser.extract(
        "Get all product names and prices",
        schema={"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "price": {"type": "number"}
        }}}
    )
    
    # Natural language navigation (uses NAVIGATE mode)
    await browser.navigate("go to checkout")
    
    # Observe elements before acting
    elements = await browser.observe("find the checkout button")
    print(f"Found: {elements[0]['selector']}")
    
    # Autonomous agent for complex tasks (uses AUTO mode with planning)
    result = await browser.agent(
        task="Complete the checkout process",
        context={"shipping": "123 Main St", "payment": "saved_card"},
        max_iterations=50
    )
    print(result.data)  # Extracted/returned data
    result.pprint()     # Pretty-print with LLM usage stats
```

---

## Quick Start

### Installation

```bash
# One-liner (recommended)
curl -fsSL https://get.flybrowser.dev | bash

# Or from source
git clone https://github.com/firefly-oss/flybrowser.git
cd flybrowser
./install.sh
```

### Your First Automation

```python
from flybrowser import FlyBrowser

async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
    # Navigate and interact naturally
    await browser.goto("https://news.ycombinator.com")
    await browser.act("click the login link")
    
    # Extract structured data
    posts = await browser.extract("Get the top 5 post titles and scores")
    
    # Record or stream your session
    await browser.start_recording()
    await browser.act("scroll down slowly")
    recording = await browser.stop_recording()
    
    print(f"Extracted: {posts['data']}")
    print(f"Recording: {recording['recording_id']}")
```

**Works in Jupyter too:**
```bash
flybrowser setup jupyter install
jupyter notebook
# Select "FlyBrowser" kernel, use await directly!
```

---

## Streaming & Recording

Stream browser sessions in real-time or record for later playback:

```python
# Start live HLS stream
stream = await browser.start_stream(
    protocol="hls",      # or "dash", "rtmp"
    quality="medium",    # low_bandwidth, medium, high
    codec="h265"         # 40% bandwidth savings vs h264
)

print(f"Watch at: {stream['stream_url']}")
print(f"Web player: {stream['player_url']}")
# Works in ALL modes: embedded, standalone, cluster

# Open embedded web player in browser (no external software needed)
import webbrowser
webbrowser.open(stream['player_url'])

# Monitor stream health (nested structure - safe access required)
status = await browser.get_stream_status()
if status.get('active'):
    stream_data = status.get('status', {})  # First level of nesting
    metrics = stream_data.get('metrics', {})  # Second level for metrics
    print(f"FPS: {metrics.get('current_fps', 0):.1f}")
    print(f"Health: {stream_data.get('health', 'unknown')}")
    print(f"Bitrate: {metrics.get('current_bitrate', 0):.0f} bps")

# Stream to Twitch/YouTube
stream = await browser.start_stream(
    protocol="rtmp",
    rtmp_url="rtmp://live.twitch.tv/app",
    rtmp_key="your_stream_key"
)
```

**CLI Management:**
```bash
# Stream management
flybrowser stream start sess_123 --protocol hls --quality high
flybrowser stream status sess_123
flybrowser stream url sess_123
flybrowser stream web sess_123   # Open embedded web player in browser (no software needed)
flybrowser stream play sess_123  # Auto-detect and launch player (ffplay/vlc/mpv)

# Recording management
flybrowser recordings list
flybrowser recordings download rec_xyz -o session.mp4
flybrowser recordings clean --older-than 30d
```

**Bandwidth Optimization:**
- H.264: 1.5 Mbps (baseline)
- H.265: 900 kbps (40% savings) ⭐
- VP9: 1.0 Mbps (33% savings)

**Hardware Acceleration:**
- NVIDIA NVENC (automatic detection)
- Apple VideoToolbox (M1/M2/M3)
- Intel Quick Sync (QSV)

---

## ReAct Agent Architecture

FlyBrowser uses the **ReAct (Reasoning + Acting)** pattern for intelligent browser automation with explicit thought-action-observation cycles.

### Core ReAct Loop

All operations follow the ReAct cycle:
```
THOUGHT → ACTION → OBSERVATION → REPEAT
```

- **Thought**: LLM reasons about what to do next based on task and context
- **Action**: Execute browser operation through registered tools
- **Observation**: Capture result and update context
- **Repeat**: Continue until task complete or max iterations reached

### Parallel Site Exploration

FlyBrowser includes a **DAG-based parallel exploration** system for efficient multi-page site analysis:

```python
# Automatic parallel exploration during site navigation
result = await browser.agent(
    task="Navigate the whole site and summarize all pages",
    context={"url": "https://example.com"}
)
```

**Key Features:**
- **ExplorationDAG**: Directed acyclic graph tracks page dependencies (parent → children)
- **Pipeline Mode**: Capture screenshots while analyzing previous page (2-4x speedup)
- **Parallel LLM Analysis**: Multiple pages analyzed concurrently (respects rate limits)
- **Smart Ordering**: Child pages only explored after parent pages complete

**Execution Flow:**
```
Homepage → Explore (screenshots + analysis) → Store PageMap
    ↓
Discover nav links → Queue Level 1 pages
    ↓
Parallel: [Page A] [Page B] [Page C] (up to max_parallel_pages)
    ↓
Pipeline: Capture Page D while analyzing Page C
```

### Architecture Overview

```
FlyBrowser SDK (sdk.py)
    ↓
ReActBrowserAgent (sdk_integration.py)
    ↓ initialize()
    ├── ToolRegistry (30+ tools)
    │   ├── Navigation: NavigateTool, GoBackTool, GoForwardTool, RefreshTool
    │   ├── Interaction: ClickTool, TypeTool, ScrollTool, HoverTool, PressKeyTool, SelectOptionTool, etc.
    │   ├── Extraction: ExtractTextTool, ScreenshotTool, GetPageStateTool
    │   ├── Exploration: PageExplorerTool
    │   ├── Search: SearchAPITool, SearchHumanTool, SearchHumanAdvancedTool, SearchRankTool
    │   └── System: CompleteTool, FailTool, WaitTool, AskUserTool
    ├── ReActAgent (react_agent.py)
    │   ├── TaskPlanner - Complex task planning
    │   ├── GoalInterpreter - Fast-path URL navigation
    │   ├── AgentMemory - 4-tier memory system
    │   ├── PromptManager - Template-based prompts
    │   └── ReActParser - Parse LLM responses (structured output)
    ├── ParallelPageExplorer (parallel_explorer.py)
    │   ├── ExplorationDAG - Page dependency tracking
    │   ├── SitemapGraph - Site exploration state machine
    │   └── Pipeline mode - Capture while analyzing (2-4x speedup)
    └── Safety: Circuit breakers, loop detection, response repair

Execution Flow:
ReActAgent.execute(task, operation_mode)
  → GoalInterpreter: Fast-path URL navigation (skip LLM for trivial goals)
  → TaskPlanner: Create execution plan (if complex)
  → LLM generates: Thought + Action (structured JSON output)
  → Auto-repair: Fix malformed JSON responses
  → ToolRegistry.execute_tool(action)
  → Observation → Update Memory
  → Loop until complete/fail
```

### Key Components

**ReActAgent** - Core reasoning loop
- Manages thought-action-observation cycles
- Integrates with LLM for reasoning
- Maintains execution state and history
- Supports multiple reasoning strategies (Standard, CoT, ToT)

**AgentOrchestrator** - Safety wrapper
- Circuit breakers for infinite loops
- Timeout management
- Execution mode control (Autonomous, Supervised)
- Progress tracking and reporting

**ToolRegistry** - Tool management
- Register and discover tools
- Generate tool descriptions for LLM
- Execute tool calls with validation
- Support for tool dependencies

**AgentMemory** - 4-tier memory system
- **Episodic**: Recent actions and observations
- **Semantic**: Long-term knowledge and patterns
- **Procedural**: Learned skills and strategies
- **Working**: Current task context

### Tools (Not Agents)

FlyBrowser uses **tools** (not separate agents) for browser operations:

```python
from flybrowser.agents.tools import (
    # Navigation tools
    NavigateTool, GoBackTool, GoForwardTool, RefreshTool,
    # Interaction tools
    ClickTool, TypeTool, ScrollTool, HoverTool, PressKeyTool,
    SelectOptionTool, CheckboxTool, FocusTool, FillTool,
    # Extraction tools
    ExtractTextTool, ScreenshotTool, GetPageStateTool,
    # Search tools
    SearchAPITool, SearchHumanTool, SearchHumanAdvancedTool, SearchRankTool,
    # Exploration tools
    PageExplorerTool,
    # System tools
    CompleteTool, FailTool, WaitTool, AskUserTool,
    # Registry
    ToolRegistry,
)
```

### Automatic Obstacle Handling

**PageAnalyzer** tool detects and handles page obstacles:

```python
# Automatic obstacle detection
result = await browser.extract("Get product prices")
# Automatically handles cookie banners, modals, popups
```

**Supported Obstacle Types:**
- Cookie consent banners (GDPR, CCPA)
- Modal dialogs and popups
- Newsletter subscription overlays
- Age verification gates
- Login walls and paywalls
- Chat widgets and help dialogs

### Reasoning Strategies

ReActAgent supports multiple reasoning strategies:

**Standard ReAct**: Fast, single-path reasoning
```python
browser = FlyBrowser(llm_provider="openai")
result = await browser.act("click the login button")
```

**Chain-of-Thought (CoT)**: Detailed step-by-step reasoning
```python
# Automatically selected for complex tasks
result = await browser.agent(
    task="Complete the multi-page checkout process",
    context={"payment": "credit_card", "shipping": "express"}
)
```

**Tree-of-Thoughts (ToT)**: Explore multiple solution paths
```python
# Used for ambiguous or high-stakes tasks
result = await browser.extract(
    query="Extract all product data",
    schema={"type": "array", "items": {...}}
)
```

### Task Planning

**TaskPlanner** breaks down complex goals into phases:

```python
# Automatic planning for complex tasks
result = await browser.agent(
    task="Research competitors and extract pricing",
    context={"industry": "SaaS", "competitors": 5},
    max_iterations=50
)

# Planner creates phases:
# 1. Search phase: Find competitors
# 2. Navigation phase: Visit pricing pages
# 3. Extraction phase: Get pricing data
# 4. Verification phase: Validate completeness
```

### Search Capabilities

**Multi-Engine Search** with intelligent ranking:

```python
# Simple search
result = await browser.search("Python async tutorials")

# Multi-engine with options
result = await browser.search(
    "AI research papers",
    engine="duckduckgo",  # google, bing, duckduckgo
    max_results=10,
    date_range="past_month"
)

# Search with exploration
result = await browser.search(
    "Startup company info",
    explore=True,
    goal="Find company description and pricing"
)
```

**Search Features:**
- Multi-Engine Support: Google, Bing, DuckDuckGo
- Query Optimization: LLM-powered query enhancement
- Result Ranking: Multi-factor scoring (relevance, quality, authority)
- Deep Exploration: Navigate to results and extract content
- Pagination: Automatic multi-page handling

### Response Validation

**Intelligent validation** ensures quality:

```python
# Automatic schema validation
products = await browser.extract(
    "Get product name, price, and rating for all items",
    schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"},
                "rating": {"type": "number"}
            }
        }
    }
)
```

**Validation Pipeline:**
1. Direct JSON parse
2. Code block extraction
3. Pattern matching
4. Best-effort parsing
5. LLM-based correction

**Performance:**
- 99.8% success rate
- <1ms overhead on successful validations (90% of cases)
- Automatic retry with correction on failures

---

## Deployment Modes

Run FlyBrowser however you need it:

### Embedded Mode
```python
# Everything in one process - perfect for scripts
async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
    await browser.goto("https://example.com")
    data = await browser.extract("Get the main content")
```

### Standalone Server
```bash
# Start server
flybrowser serve --port 8000

# Connect from clients
```
```python
async with FlyBrowser(endpoint="http://localhost:8000") as browser:
    # Same API, server handles browser sessions
    await browser.goto("https://example.com")
```

### Cluster Mode
```bash
# 3-node cluster with automatic failover
flybrowser serve --cluster --node-id node1 --port 8001 --raft-port 5001
flybrowser serve --cluster --node-id node2 --port 8002 --raft-port 5002 --peers node1:5001
flybrowser serve --cluster --node-id node3 --port 8003 --raft-port 5003 --peers node1:5001,node2:5002
```

**Features:**
- Raft consensus for coordination
- Automatic session migration on node failure
- Load balancing across nodes
- Zero-downtime deployments

| Feature | Embedded | Standalone | Cluster |
|---------|----------|------------|---------|
| Browser Sessions | 1 | Configurable | Auto-scaled |
| Recording | [ok] Local | [ok] S3/NFS/Local | [ok] S3/NFS |
| Live Streaming | [ok] Local server | [ok] Full support | [ok] Full support |
| Failover | N/A | N/A | [ok] Automatic |
| Use Case | Scripts, dev | Teams, services | Production |

---

## Security & PII Protection

Never expose sensitive data to LLMs:

```python
from flybrowser import FlyBrowser

browser = FlyBrowser(pii_masking_enabled=True)

# Store credentials securely
pwd_id = browser.store_credential("password", "secret123", pii_type="password")

# Use in automation - LLM never sees the actual value
await browser.secure_fill("#password", pwd_id)

# Automatic PII masking in logs
await browser.act("type john@example.com in email field")
# LLM sees: "type [MASKED_EMAIL] in email field"
```

**Protected Data Types:**
- Passwords
- API keys
- Credit cards
- Social security numbers
- Email addresses
- Phone numbers
- Custom patterns

---

## Autonomous Mode

The most powerful feature of FlyBrowser - give it a complex goal and watch it figure out how to accomplish it:

### Form Automation
```python
from flybrowser import FlyBrowser

async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
    await browser.goto("https://jobs.example.com/apply")
    
    result = await browser.agent(
        task="Fill out and submit the job application",
        context={
            "name": "Jane Smith",
            "email": "jane@example.com",
            "phone": "555-123-4567",
            "position": "Senior Engineer",
            "experience_years": 5,
            "cover_letter": "I am excited to apply for this position..."
        },
        max_iterations=30,
        max_time_seconds=300
    )
    
    if result.success:
        print(f"Application submitted! Confirmation: {result.data}")
        result.pprint()  # Pretty-print execution summary and LLM usage
    else:
        print(f"Failed: {result.error}")
```

### Multi-Step Booking
```python
# Book a restaurant reservation
await browser.goto("https://opentable.com")

result = await browser.agent(
    task="Book a table for 4 people at an Italian restaurant near downtown",
    context={
        "location": "San Francisco, CA",
        "date": "Saturday at 7pm",
        "party_size": 4,
        "cuisine": "Italian",
        "name": "John Doe",
        "phone": "555-987-6543",
        "email": "john@example.com"
    },
    max_time_seconds=600  # 10 minutes max
)

print(f"Reservation: {result.data}")
print(f"Duration: {result.execution.duration_seconds:.1f}s")
print(f"LLM Cost: ${result.llm_usage.cost_usd:.4f}")
```

### Research & Data Gathering
```python
# Research competitors and extract insights
await browser.goto("https://google.com")

result = await browser.agent(
    task="Research the top 5 competitors in the CRM space and gather their pricing info",
    context={"industry": "CRM software", "focus": "small business"},
)

if result.success:
    print(f"Research complete: {result.data}")
result.pprint()  # Shows execution summary, LLM usage, costs
```

---

## Use Cases

### Web Scraping
```python
# Extract structured data with schema validation
from flybrowser import FlyBrowser

async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
    await browser.goto("https://shop.example.com/products")
    
    # Extract with optional schema
    result = await browser.extract(
        "Get all product names, prices, and stock status",
        schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "price": {"type": "number"},
                    "in_stock": {"type": "boolean"}
                },
                "required": ["name", "price"]
            }
        }
    )
    
    # Result is an AgentRequestResponse with:
    # - result.success: Whether extraction succeeded
    # - result.data: The extracted data
    # - result.llm_usage: Token counts and cost
    # - result.execution: Timing and iteration info
    
    if result.success:
        products = result.data
        print(f"Extracted {len(products)} products")
        result.pprint()  # Shows full execution summary
        print(f"Schema compliance: {result['schema_compliance']*100:.1f}%")
        
        for product in products[:3]:  # Show first 3
            print(f"  - {product['name']}: ${product['price']}")
```

### UI Testing
```python
# Test workflows with natural language
await browser.act("click the checkout button")
await browser.act("fill in shipping address with test data")
screenshot = await browser.screenshot()
assert "Order confirmed" in screenshot['text']
```

### Monitoring & Alerts
```python
# Wait for specific conditions
await browser.monitor("wait for the success message to appear")
await browser.monitor("check if price drops below $50")
```

### Content Recording
```python
# Record tutorials, demos, or evidence
await browser.start_recording(codec="h265", quality="high")
await browser.act("demonstrate the checkout process")
recording = await browser.stop_recording()
```

### Live Streaming
```python
# Stream to platforms or save for later
stream = await browser.start_stream(
    protocol="rtmp",
    rtmp_url="rtmp://live.youtube.com/app",
    rtmp_key="your_key"
)
```

---

## REST API

FlyBrowser provides a full REST API for integration with any language or platform.

### Starting the Server
```bash
# Start standalone server
flybrowser serve --port 8000

flybrowser serve --port 8000
```

### API Documentation (Auto-Generated)

FastAPI automatically generates interactive API documentation:

| URL | Description |
|-----|-------------|
| `http://localhost:8000/docs` | **Swagger UI** - Interactive API explorer |
| `http://localhost:8000/redoc` | **ReDoc** - Clean API documentation |
| `http://localhost:8000/openapi.json` | **OpenAPI Schema** - Machine-readable spec |

### Example: Create Session and Execute Tasks

```bash
# 1. Create a browser session
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "llm_provider": "openai",
    "api_key": "sk-...",
    "headless": true
  }'
# Response: {"session_id": "sess_abc123", "status": "created"}

# 2. Navigate to a URL
curl -X POST http://localhost:8000/sessions/sess_abc123/navigate \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# 3. Perform an action
curl -X POST http://localhost:8000/sessions/sess_abc123/action \
  -H "Content-Type: application/json" \
  -d '{"instruction": "click the login button"}'

# 4. Extract data
curl -X POST http://localhost:8000/sessions/sess_abc123/extract \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Get all product names and prices",
    "schema": {"type": "array", "items": {"type": "object"}}
  }'
```

### Autonomous Mode via API

```bash
# Execute a complex goal autonomously
curl -X POST http://localhost:8000/sessions/sess_abc123/auto \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Fill out the contact form and submit it",
    "context": {
      "name": "John Doe",
      "email": "john@example.com",
      "message": "Hello, I am interested in your services."
    },
    "max_iterations": 30,
    "max_time_seconds": 300
  }'

# Response:
# {
#   "success": true,
#   "goal": "Fill out the contact form and submit it",
#   "result_data": {"confirmation": "Message sent successfully"},
#   "sub_goals_completed": 4,
#   "total_sub_goals": 4,
#   "iterations": 12,
#   "duration_seconds": 45.2,
#   "actions_taken": ["clicked input", "typed name", ...]
# }
```

### Scraping via API

```bash
# Scrape data with schema validation
curl -X POST http://localhost:8000/sessions/sess_abc123/scrape \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Extract all product listings",
    "target_schema": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "price": {"type": "number"},
          "rating": {"type": "number"}
        },
        "required": ["name", "price"]
      }
    },
    "validators": ["not_empty", "min_items_10"],
    "max_pages": 5,
    "max_time_seconds": 600
  }'

# Response:
# {
#   "success": true,
#   "result_data": [{"name": "Widget", "price": 29.99, "rating": 4.5}, ...],
#   "pages_scraped": 5,
#   "items_extracted": 47,
#   "validation_results": [{"validator": "not_empty", "passed": true}, ...],
#   "schema_compliance": 0.98
# }
```

### Available Validators (API)

When using the `/scrape` endpoint, specify validators by name:

| Validator | Description |
|-----------|-------------|
| `not_empty` | Data must not be empty |
| `min_items_5` / `min_items_10` / `min_items_20` / `min_items_50` | Minimum item count |
| `max_items_100` / `max_items_500` | Maximum item count |
| `has_name` / `has_price` / `has_url` | Required fields |
| `unique_id` / `unique_name` | Field uniqueness |
| `no_nulls` | No null values |

### Clean Up

```bash
# Delete the session when done
curl -X DELETE http://localhost:8000/sessions/sess_abc123 \

```

---

## Documentation

### Getting Started
| Guide | Description |
|-------|-------------|
| [Installation](docs/getting-started/installation.md) | Install and configure FlyBrowser |
| [Quickstart](docs/getting-started/quickstart.md) | Your first automation in 5 minutes |
| [Core Concepts](docs/getting-started/concepts.md) | Key concepts and terminology |

### Features
| Feature | Description |
|---------|-------------|
| [Act](docs/features/act.md) | Execute actions (click, type, select) |
| [Extract](docs/features/extract.md) | Extract structured data with schemas |
| [Agent](docs/features/agent.md) | Multi-step autonomous tasks |
| [Observe](docs/features/observe.md) | Find and analyze page elements |
| [Navigation](docs/features/navigation.md) | URL and natural language navigation |
| [Screenshots](docs/features/screenshots.md) | Capture and compare screenshots |
| [Streaming](docs/features/streaming.md) | Real-time action streaming |
| [PII Handling](docs/features/pii.md) | Secure credential management |

### Guides
| Guide | Description |
|-------|-------------|
| [Basic Automation](docs/guides/basic-automation.md) | Common automation patterns |
| [Data Extraction](docs/guides/data-extraction.md) | Scraping and extraction techniques |
| [Form Automation](docs/guides/form-automation.md) | Form filling and submission |
| [Multi-Page Workflows](docs/guides/multi-page-workflows.md) | Complex navigation flows |
| [Authentication](docs/guides/authentication.md) | Login and session handling |
| [Error Handling](docs/guides/error-handling.md) | Retry and recovery patterns |

### Reference
| Reference | Description |
|-----------|-------------|
| [SDK Reference](docs/reference/sdk.md) | Complete Python API |
| [REST API](docs/reference/rest-api.md) | HTTP endpoints |
| [CLI Reference](docs/reference/cli.md) | Command-line tools |
| [Configuration](docs/reference/configuration.md) | All configuration options |

### Architecture
| Topic | Description |
|-------|-------------|
| [Overview](docs/architecture/overview.md) | System architecture |
| [ReAct Framework](docs/architecture/react.md) | Reasoning and acting loop |
| [Tools System](docs/architecture/tools.md) | Browser action tools |
| [Memory Management](docs/architecture/memory.md) | Context and history |
| [LLM Integration](docs/architecture/llm-integration.md) | Provider abstraction |

### Deployment
| Mode | Description |
|------|-------------|
| [Embedded Mode](docs/deployment/embedded.md) | Run in your Python process |
| [Standalone Mode](docs/deployment/standalone.md) | HTTP server deployment |
| [Cluster Mode](docs/deployment/cluster.md) | Multi-node distributed setup |
| [Docker](docs/deployment/docker.md) | Container deployment |
| [Kubernetes](docs/deployment/kubernetes.md) | K8s orchestration |

### Advanced
| Topic | Description |
|-------|-------------|
| [Custom Tools](docs/advanced/custom-tools.md) | Extend with custom actions |
| [Custom Providers](docs/advanced/custom-providers.md) | Add LLM providers |
| [Performance](docs/advanced/performance.md) | Optimization techniques |
| [Troubleshooting](docs/advanced/troubleshooting.md) | Debug and resolve issues |

### Examples
| Category | Description |
|----------|-------------|
| [Examples README](examples/README.md) | Overview of all examples |
| [Web Scraping](examples/scraping/) | Data extraction examples |
| [UI Testing](examples/testing/) | Automated testing examples |
| [Workflows](examples/workflows/) | Business automation examples |
| [Examples Guide](docs/examples/index.md) | Detailed examples documentation |

---

## Interactive REPL

```bash
flybrowser
```

Launches an interactive shell:
```
flybrowser> goto https://example.com
flybrowser> extract What is the main heading?
flybrowser> act click the More information link
flybrowser> screenshot
flybrowser> quit
```

---

## LLM Providers

Works with any LLM:

### Auto-Select Model (Recommended)

Let FlyBrowser choose the best model based on your requirements:

```python
from flybrowser import FlyBrowser, ModelPreference

# Auto-select best cheap model
browser = FlyBrowser(
    llm_provider="openai",
    llm_preference=ModelPreference.BEST_QUALITY_CHEAP,
    api_key="sk-..."
)

# Auto-select model with vision capabilities
browser = FlyBrowser(
    llm_provider="anthropic",
    llm_preference=ModelPreference.VISION_OPTIMIZED,
    api_key="sk-ant-..."
)
```

**Available Preferences:**
| Preference | Description |
|------------|-------------|
| `BEST_QUALITY` | Highest quality, regardless of cost |
| `BEST_QUALITY_CHEAP` | Best quality among affordable models |
| `CHEAPEST` | Lowest cost |
| `BALANCED` | Good balance of quality and cost |
| `VISION_OPTIMIZED` | Best model with vision support |
| `FAST_RESPONSE` | Fastest response time |
| `REASONING` | Complex reasoning tasks |
| `CODING` | Code generation |
| `LOCAL_ONLY` | Only local/free models |

### OpenAI
```python
browser = FlyBrowser(
    llm_provider="openai",
    llm_model="gpt-5.2",  # Or: gpt-5-mini, gpt-4o, gpt-4o-mini
    api_key="sk-..."
)
```

### Anthropic
```python
browser = FlyBrowser(
    llm_provider="anthropic",
    llm_model="claude-sonnet-4-5-20250929",
    api_key="sk-ant-..."
)
```

### Google Gemini
```python
browser = FlyBrowser(
    llm_provider="gemini",
    llm_model="gemini-2.0-flash",
    api_key="AIza..."
)
```

### Local LLMs (Ollama)
```bash
ollama serve
ollama pull qwen3:8b
```
```python
browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="qwen3:8b"  # Or: llama3.2-vision, gemma3:12b
)
```

**Supported Providers:**
- OpenAI (GPT-5.2, GPT-5-mini, GPT-4o)
- Anthropic (Claude 4.5, Claude 3.5)
- Google Gemini (Gemini 2.0, Gemini 1.5)
- Ollama (Qwen3, Llama 3.2, Gemma 3)
- Any OpenAI-compatible endpoint

---

## Configuration

```python
from flybrowser import FlyBrowser, ModelPreference

# Via constructor
browser = FlyBrowser(
    llm_provider="openai",
    llm_model="gpt-5.2",                         # Explicit model OR...
    llm_preference=ModelPreference.BALANCED,      # ...auto-select by preference
    api_key="sk-...",
    headless=True,
    browser_type="chromium",
    recording_enabled=False,
    pii_masking_enabled=True,
    timeout=30.0,
    pretty_logs=True,              # Human-readable colored logs (default)
    speed_preset="balanced",       # fast, balanced, thorough
    log_verbosity="normal"         # silent, minimal, normal, verbose, debug
)

# Via environment variables
FLYBROWSER_LLM_PROVIDER=openai
FLYBROWSER_LLM_MODEL=gpt-4
FLYBROWSER_API_KEY=sk-...
FLYBROWSER_HEADLESS=true
FLYBROWSER_BROWSER_TYPE=chromium
FLYBROWSER_RECORDING_ENABLED=false
FLYBROWSER_PII_MASKING_ENABLED=true
```

**Storage Configuration:**
```bash
# Local storage (default)
FLYBROWSER_RECORDING_STORAGE=local
FLYBROWSER_RECORDING_DIR=~/.flybrowser/recordings

# S3/MinIO storage
FLYBROWSER_RECORDING_STORAGE=s3
FLYBROWSER_S3_BUCKET=my-recordings
FLYBROWSER_S3_REGION=us-east-1
FLYBROWSER_S3_ACCESS_KEY=...
FLYBROWSER_S3_SECRET_KEY=...

# Shared/NFS storage (cluster mode)
FLYBROWSER_RECORDING_STORAGE=shared
FLYBROWSER_RECORDING_DIR=/mnt/nfs/recordings
```

See [Configuration Reference](docs/reference/configuration.md) for all options.

---

## Development

```bash
# Install dev dependencies
./install.sh --dev

# Run tests
task test

# Code quality
task check         # Format, lint, typecheck
task precommit     # Full pre-commit checks

# Development server
task serve         # Auto-reload on changes
```

### Project Tasks

| Task | Description |
|------|-------------|
| `task install` | Quick install (auto-detects uv/pip) |
| `task install:dev` | Install with dev dependencies |
| `task dev` | Start development environment |
| `task repl` | Launch interactive REPL |
| `task serve` | Start dev server with reload |
| `task test` | Run all tests |
| `task test:cov` | Tests with coverage report |
| `task check` | Run all quality checks |
| `task precommit` | Pre-commit checks |
| `task doctor` | Check installation health |
| `task build` | Build distribution packages |

---

## Performance

**Validation Performance:**
- 90% of responses validate in < 1ms
- 99.8% overall success rate
- 0.6% average overhead
- 75% fewer failed operations

**Streaming Performance:**
- Hardware acceleration: 3-5x faster encoding
- H.265 bandwidth savings: 40% vs H.264
- Latency: DASH <1s, HLS 2-3s, RTMP <500ms

**Cluster Performance:**
- Session failover: < 100ms
- Raft consensus: < 50ms typical
- Auto-scaling: Dynamic based on load

---

## Contributing

We welcome contributions! Here's how:

```bash
# 1. Fork and clone
git clone https://github.com/your-username/flybrowser.git
cd flybrowser

# 2. Create a branch
git checkout -b feature/your-feature

# 3. Make changes and test
./install.sh --dev
task check && task test

# 4. Commit and push
git commit -m "Add your feature"
git push origin feature/your-feature

# 5. Open a Pull Request
```

For architecture details, see [Architecture Overview](docs/architecture/overview.md).

---

## License

Copyright 2026 Firefly Software Solutions Inc.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built with these amazing projects:

- [Playwright](https://playwright.dev/) - Rock-solid browser automation
- [FastAPI](https://fastapi.tiangolo.com/) - Modern API framework
- [FFmpeg](https://ffmpeg.org/) - Video encoding powerhouse
- [OpenAI](https://openai.com/) & [Anthropic](https://anthropic.com/) - LLM intelligence

Inspired by [Stagehand](https://github.com/browserbase/stagehand).

---

## Community & Support

- **Documentation**: [Full docs](docs/index.md)
- **Discord**: [Join our community](https://discord.gg/flybrowser)
- **Issues**: [GitHub Issues](https://github.com/firefly-oss/flybrowser/issues)
- **Discussions**: [GitHub Discussions](https://github.com/firefly-oss/flybrowser/discussions)
- **Email**: support@flybrowser.dev

---

<p align="center">
  <strong>Made with love by Firefly Software Solutions Inc</strong>
</p>

<p align="center">
  <a href="https://flybrowser.dev">Website</a> •
  <a href="https://github.com/firefly-oss/flybrowser">GitHub</a> •
  <a href="https://discord.gg/flybrowser">Discord</a> •
  <a href="https://twitter.com/flybrowser">Twitter</a>
</p>
