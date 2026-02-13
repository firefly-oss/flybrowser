# FlyBrowser Documentation

**LLM-powered browser automation that speaks your language.**

FlyBrowser combines [Playwright](https://playwright.dev/) browser control with LLM intelligence, letting you automate the web using plain English. Built on [fireflyframework-genai](https://github.com/fireflyframework/fireflyframework-genai) -- Firefly's open-source agent framework -- it inherits production-grade ReAct reasoning, multi-provider LLM support, and a composable toolkit system.

- **Version**: 26.02.01 (CalVer)
- **Python**: >= 3.13
- **License**: Apache 2.0
- **Author**: Firefly Software Solutions Inc

```python
from flybrowser import FlyBrowser

async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
    await browser.goto("https://news.ycombinator.com")
    stories = await browser.extract("Get the top 5 post titles and scores")
    await browser.agent("Click into the top story and summarize the comments")
```

---

## Getting Started

New to FlyBrowser? Start here.

| Guide | Description |
|-------|-------------|
| [Installation](getting-started/installation.md) | System requirements, installation methods, and Playwright setup |
| [Quickstart](getting-started/quickstart.md) | Your first automation in five minutes |
| [Core Concepts](getting-started/concepts.md) | Sessions, the ReAct loop, actions vs. agents, and result objects |
| [Setup Wizard](getting-started/setup-wizard.md) | Interactive `flybrowser setup` configuration guide |

---

## Features

Detailed documentation for every FlyBrowser capability.

| Feature | Description |
|---------|-------------|
| [Act](features/act.md) | Execute browser actions via natural language -- click, type, scroll, select, hover |
| [Extract](features/extract.md) | Extract structured data from pages with optional JSON Schema validation |
| [Agent](features/agent.md) | Autonomous multi-step task execution with configurable iteration and time limits |
| [Observe](features/observe.md) | Find and analyze page elements, returning selectors for programmatic use |
| [Navigation](features/navigation.md) | Direct URL navigation and natural language link-following |
| [Search](features/search.md) | Multi-provider web search (Serper, Google, Bing) with intelligent ranking and browser fallback |
| [Obstacle Detection](features/obstacle-detection.md) | Automatic dismissal of popups, modals, cookie banners, and newsletter overlays |
| [Screenshots](features/screenshots.md) | Page capture with optional PII masking and full-page mode |
| [Streaming](features/streaming.md) | Live browser streaming via HLS, DASH, or RTMP with H.264/H.265/VP9 codecs |
| [PII Handling](features/pii.md) | Encrypted credential storage, placeholder-based LLM injection, and auto-masking |
| [Stealth Mode](features/stealth.md) | Fingerprint generation, CAPTCHA solving, proxy rotation, and human-like behavior |
| [Observability](features/observability.md) | Command logging, live view iframe, completion page, OpenTelemetry tracing, Prometheus metrics |

---

## User Guides

Step-by-step guides for common workflows.

| Guide | Description |
|-------|-------------|
| [Basic Automation](guides/basic-automation.md) | Navigation, clicking, typing, and scrolling patterns |
| [Data Extraction](guides/data-extraction.md) | Scraping techniques, schema validation, and pagination |
| [Form Automation](guides/form-automation.md) | Filling and submitting forms, including multi-step wizards |
| [Multi-Page Workflows](guides/multi-page-workflows.md) | Complex tasks that span multiple pages |
| [Authentication](guides/authentication.md) | Login flows, session persistence, and secure credential handling |
| [Error Handling](guides/error-handling.md) | Retry strategies, recovery patterns, and debugging |
| [Context Usage](guides/context-usage.md) | Passing structured context to operations for better results |

---

## CLI Reference

FlyBrowser includes a full command-line interface for interactive use, scripting, and server management.

| Reference | Description |
|-----------|-------------|
| [CLI Reference](reference/cli.md) | Complete command-line reference |
| [Direct Commands](cli/direct-commands.md) | `goto`, `act`, `extract`, `agent`, `screenshot`, and `run` |
| [Session Management](cli/session-management.md) | `session create`, `list`, `info`, `connect`, `exec`, `close`, `close-all` |
| [Pipelines](cli/pipelines.md) | YAML workflow pipelines with `flybrowser run` |

**Quick reference:**

```
flybrowser repl                             Interactive REPL
flybrowser goto <url>                       Navigate to URL
flybrowser act <instruction>                Perform action
flybrowser extract <query>                  Extract data
flybrowser agent <task>                     Autonomous agent
flybrowser screenshot                       Capture screenshot
flybrowser run <workflow.yaml>              Execute YAML pipeline
flybrowser serve                            Start REST API server
flybrowser setup [quick|llm|server|...]     Guided configuration
flybrowser doctor                           Check installation health
```

---

## API Reference

| Reference | Description |
|-----------|-------------|
| [SDK Reference](reference/sdk.md) | Complete Python API -- `FlyBrowser` class, all methods, parameters, and return types |
| [REST API Reference](reference/rest-api.md) | HTTP endpoints for standalone and cluster modes |
| [Configuration](reference/configuration.md) | All constructor options, environment variables, and storage configuration |

---

## Architecture

How FlyBrowser works under the hood.

| Topic | Description |
|-------|-------------|
| [System Overview](architecture/overview.md) | High-level architecture diagram and component relationships |
| [Framework Integration](architecture/framework-integration.md) | How FlyBrowser builds on fireflyframework-genai |
| [ReAct Framework](architecture/react.md) | The Thought-Action-Observation reasoning loop |
| [Tools System](architecture/tools.md) | 6 ToolKits with 32 browser tools, `@firefly_tool` decorator |
| [Memory System](architecture/memory.md) | BrowserMemoryManager, dual-write sync, page history, navigation graphs |
| [LLM Integration](architecture/llm-integration.md) | Provider delegation, model selection, and API call management |
| [Response Validation](architecture/validation.md) | OutputReviewer and the 5-stage validation pipeline |
| [Context System](architecture/context.md) | Context building, prompt construction, and state management |
| [Security](architecture/security.md) | RBAC roles, JWT authentication, PII encryption, memory zeroing |

---

## Deployment

Run FlyBrowser however your infrastructure requires. The Python SDK API is identical across all modes.

| Mode | Description |
|------|-------------|
| [Embedded](deployment/embedded.md) | In-process -- scripts, notebooks, development |
| [Standalone Server](deployment/standalone.md) | FastAPI HTTP server for multi-client access |
| [Cluster](deployment/cluster.md) | Multi-node with Raft consensus, auto-failover, and load balancing |
| [Docker](deployment/docker.md) | Container images and Docker Compose configurations |
| [Kubernetes](deployment/kubernetes.md) | Helm charts and K8s deployment patterns |

---

## LLM Providers

LLM orchestration is handled by [fireflyframework-genai](https://github.com/fireflyframework/fireflyframework-genai). FlyBrowser works with:

| Provider | Models | Vision |
|----------|--------|--------|
| **OpenAI** | gpt-5.2, gpt-5-mini, gpt-4o, gpt-4o-mini | Yes |
| **Anthropic** | claude-sonnet-4-5-20250929, claude-3-5-sonnet-20241022 | Yes |
| **Google Gemini** | gemini-2.0-flash, gemini-1.5-pro | Yes |
| **Qwen (Alibaba)** | qwen3, qwen-plus, qwen-vl | Yes (qwen-vl) |
| **Ollama (local)** | qwen3, llama3.2, gemma3, etc. | Model-dependent |
| **Custom** | Any OpenAI-compatible endpoint | Varies |

See the [LLM Integration](architecture/llm-integration.md) page for configuration details.

---

## Advanced Topics

| Topic | Description |
|-------|-------------|
| [Custom Tools](advanced/custom-tools.md) | Create your own browser tools using the `@firefly_tool` decorator and `ToolKit` base class |
| [Custom Providers](advanced/custom-providers.md) | Add support for additional LLM providers |
| [Performance](advanced/performance.md) | Speed presets, hardware acceleration, parallel execution, and caching |
| [Troubleshooting](advanced/troubleshooting.md) | Common issues, diagnostics with `flybrowser doctor`, and debug logging |

---

## Examples

| Category | Description |
|----------|-------------|
| [Examples Overview](examples/index.md) | Guide to all example code with descriptions |
| [Web Scraping](examples/web-scraping.md) | Structured data extraction from real websites |
| [UI Testing](examples/ui-testing.md) | Automated testing with natural language assertions |
| [Workflow Automation](examples/workflow-automation.md) | End-to-end business workflows: forms, bookings, research |

---

## Core SDK Methods

A quick reference for the primary `FlyBrowser` methods. See the [SDK Reference](reference/sdk.md) for full documentation.

| Method | Description |
|--------|-------------|
| `goto(url, wait_until)` | Navigate to a URL |
| `navigate(instruction, context, use_vision)` | Navigate via natural language |
| `act(instruction, context, use_vision, return_metadata, max_iterations)` | Perform browser actions |
| `extract(query, context, use_vision, schema, return_metadata, max_iterations)` | Extract structured data |
| `observe(query, context, return_selectors, return_metadata, max_iterations)` | Find page elements |
| `agent(task, context, max_iterations, max_time_seconds, return_metadata)` | Autonomous multi-step execution |
| `execute_task(task)` | Single task with ReAct reasoning |
| `search(query, search_type, max_results, ranking, return_metadata)` | Web search |
| `batch_execute(tasks, parallel, stop_on_failure)` | Run multiple tasks |
| `screenshot(full_page, mask_pii)` | Capture page screenshot |
| `start_stream(protocol, quality, ...)` | Start live stream |
| `start_recording()` / `stop_recording()` | Record session |
| `store_credential(name, value, pii_type)` | Store encrypted credential |
| `secure_fill(selector, credential_id, clear_first)` | Fill field without LLM exposure |
| `mask_pii(text)` | Mask PII patterns in text |
| `get_usage_summary()` | Token counts, cost, and API stats |

---

## License

Copyright 2026 Firefly Software Solutions Inc. Licensed under the [Apache License 2.0](https://opensource.org/licenses/Apache-2.0).
