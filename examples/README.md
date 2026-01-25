# Examples

Learn FlyBrowser by example. Each file is self-contained and runnable.

## Quick Start

```bash
# Install
pip install flybrowser
playwright install chromium

# Set your API key
export OPENAI_API_KEY="sk-..."

# Run an example
python examples/scraping/hackernews.py
```

## Examples by Category

### Basic Examples (`basic/`)

Quickstart and introductory examples.

| File | Description |
|------|-------------|
| `quickstart.py` | Your first automation - navigate, interact, extract |

### Web Scraping (`scraping/`)

Data extraction and scraping patterns.

| File | Description |
|------|-------------|
| `hackernews.py` | Hacker News scraper with structured extraction |
| `product_extraction.py` | E-commerce product data extraction |
| `pagination.py` | Multi-page scraping with pagination |
| `price_monitor.py` | Price tracking with history |

### UI Testing (`testing/`)

Automated testing patterns.

| File | Description |
|------|-------------|
| `form_validation.py` | Form validation and error testing |
| `navigation_testing.py` | Navigation link verification |
| `visual_testing.py` | Screenshot capture and visual checks |
| `e2e_checkout.py` | End-to-end checkout flow testing |

### Streaming & Recording (`streaming/`)

Live streaming and session recording.

| File | Description |
|------|-------------|
| `basic_streaming.py` | HLS/DASH streaming with web player |
| `rtmp_streaming.py` | Stream to Twitch/YouTube via RTMP |
| `recording.py` | Record browser sessions for tutorials |

### Workflow Automation (`workflows/`)

Business process automation.

| File | Description |
|------|-------------|
| `job_application.py` | Autonomous job application form filling |
| `booking.py` | Restaurant reservation automation |
| `research.py` | Competitor research and data gathering |
| `invoice_processing.py` | Invoice download and extraction |
| `report_generation.py` | Multi-source report compilation |
| `data_sync.py` | Data synchronization between systems |
| `monitoring.py` | Website monitoring and alerting |

## SDK Methods Reference

| Method | Purpose | Use Case |
|--------|---------|----------|
| `goto(url)` | Direct navigation | Navigate to a specific URL |
| `act(instruction)` | Single action | Click, type, select |
| `extract(query)` | Data extraction | Get structured data |
| `observe(query)` | Find elements | Locate page elements |
| `agent(task)` | Complex tasks | Multi-step automation |
| `screenshot()` | Capture page | Visual verification |

## Basic Pattern

```python
import asyncio
from flybrowser import FlyBrowser

async def main():
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        await browser.goto("https://example.com")
        result = await browser.extract("Get the main heading")
        print(result.data)

asyncio.run(main())
```

## Structured Extraction

```python
data = await browser.extract(
    "Get all products",
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

## Secure Credential Handling

```python
# Store credential (never sent to LLM)
await browser.store_credential("password", "secret123")

# Securely fill into form
await browser.secure_fill(
    "password_field", 
    "password",
    selector="input[type='password']"
)
```

## Configuration Options

```python
browser = FlyBrowser(
    llm_provider="openai",        # openai, anthropic, gemini, ollama
    api_key="sk-...",             # API key for provider
    headless=True,                # Run without visible browser
    log_verbosity="normal",       # silent, minimal, normal, verbose, debug
    speed_preset="balanced",      # fast, balanced, thorough
)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Missing API key | `export OPENAI_API_KEY="sk-..."` |
| Playwright not installed | `playwright install chromium` |
| Timeout errors | Use `speed_preset="thorough"` |
| Element not found | Use `observe()` to debug |

## Documentation

For complete documentation, see:

- [Getting Started](../docs/getting-started/)
- [Features](../docs/features/)
- [Guides](../docs/guides/)
- [API Reference](../docs/reference/)

---

<p align="center">
  <em>Questions? <a href="https://discord.gg/flybrowser">Join our Discord</a></em>
</p>
