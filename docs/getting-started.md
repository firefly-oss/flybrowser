# Getting Started with FlyBrowser

This guide covers the installation, configuration, and basic usage of FlyBrowser. By the end of this guide, you will have a working FlyBrowser setup and understand how to perform common automation tasks.

## Prerequisites

Before installing FlyBrowser, ensure your system meets the following requirements:

- Python 3.9 or higher
- pip package manager
- Access to an LLM provider:
  - **Cloud providers**: OpenAI, Anthropic, or Google Gemini (requires API key)
  - **Local providers**: Ollama, LM Studio, LocalAI, or vLLM (no API key needed)

## Installation

### One-Liner Installation (Recommended)

The fastest way to install FlyBrowser:

```bash path=null start=null
curl -fsSL https://get.flybrowser.dev | bash
```

This automatically:
- Checks Python 3.9+ is installed
- Creates a virtual environment at `~/.flybrowser/venv`
- Installs FlyBrowser and all dependencies
- Installs Playwright browsers
- Creates the `flybrowser` CLI command

### Alternative Installation Methods

**From Source:**
```bash path=null start=null
git clone https://github.com/firefly-oss/flybrowsers.git
cd flybrowsers
./install.sh
```

**For Development (with REPL support):**
```bash path=null start=null
git clone https://github.com/firefly-oss/flybrowsers.git
cd flybrowsers
./install.sh --dev  # or: task install:dev
```

### Verify Installation

Verify that FlyBrowser is correctly installed:

```bash path=null start=null
flybrowser doctor
```

## Configuration

### LLM Provider Setup

FlyBrowser requires an LLM provider for natural language understanding. Choose any of the supported providers below.

#### OpenAI (Recommended for Cloud)

OpenAI's GPT models offer excellent performance for browser automation tasks.

**Setup:**
```bash path=null start=null
export OPENAI_API_KEY="sk-proj-your-openai-api-key"
```

**Usage:**
```python path=null start=null
browser = FlyBrowser(
    llm_provider="openai",
    llm_model="gpt-5.2"  # Default model; also: gpt-5-mini, gpt-5-nano, gpt-4.1, gpt-4o
)
```

#### Anthropic

Anthropic's Claude models excel at careful, nuanced automation tasks.

**Setup:**
```bash path=null start=null
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key"
```

**Usage:**
```python path=null start=null
browser = FlyBrowser(
    llm_provider="anthropic",
    llm_model="claude-sonnet-4-5-20250929"  # Default; also: claude-haiku-4-5-20251001, claude-opus-4-5-20251101
)
```

#### Google Gemini

Google's Gemini models offer excellent multimodal capabilities and massive context windows (up to 1M tokens), making them ideal for vision-heavy automation.

**Setup:**
```bash path=null start=null
export GOOGLE_API_KEY="AIza-your-google-api-key"
```

**Usage:**
```python path=null start=null
browser = FlyBrowser(
    llm_provider="gemini",  # or "google" - both work
    llm_model="gemini-2.0-flash"  # Default; also: gemini-2.0-pro, gemini-1.5-pro, gemini-1.5-flash
)
```

#### Ollama (Local - Recommended for Privacy)

Ollama enables running open-source LLMs locally without API costs or data leaving your machine.

**Setup:**
```bash path=null start=null
# Install Ollama from https://ollama.ai, then:
ollama serve
ollama pull qwen3:8b  # Download a model
```

**Usage:**
```python path=null start=null
browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="qwen3:8b"  # Default; also: gemma3:12b, llama3.2:3b, deepseek-r1:8b, phi4
)
```

No API key required. Ollama runs on `http://localhost:11434` by default.

#### Other Local Providers

FlyBrowser also supports these local inference solutions:

**LM Studio** (port 1234):
```python path=null start=null
browser = FlyBrowser(
    llm_provider="lm_studio",
    llm_model="local-model",
    base_url="http://localhost:1234"
)
```

**LocalAI** (port 8080):
```python path=null start=null
browser = FlyBrowser(
    llm_provider="localai",
    llm_model="your-model",
    base_url="http://localhost:8080"
)
```

**vLLM** (port 8000) - for high-throughput inference:
```python path=null start=null
browser = FlyBrowser(
    llm_provider="vllm",
    llm_model="meta-llama/Llama-2-7b-chat-hf",
    base_url="http://localhost:8000"
)
```

## Quick Start: Interactive REPL

The fastest way to try FlyBrowser is the interactive REPL:

```bash path=null start=null
flybrowser
```

This opens an interactive shell:

```
flybrowser> goto example.com
[OK] Navigated to https://example.com

flybrowser> extract What is the main heading?
--- Extracted Data ---
{"heading": "Example Domain"}
---

flybrowser> act click the More information link
[OK] Action completed

flybrowser> screenshot
[OK] Screenshot saved to screenshot_20260121_094620.png

flybrowser> help
(shows all available commands)

flybrowser> exit
Goodbye!
```

## Basic Usage

### Creating a Browser Instance

The `FlyBrowser` class is the main entry point for embedded usage:

```python path=null start=null
import asyncio
from flybrowser import FlyBrowser

async def main():
    # Create browser instance with configuration
    browser = FlyBrowser(
        llm_provider="openai",
        llm_model="gpt-5.2",     # Or use any supported model
        headless=True,           # Run without visible browser window
        browser_type="chromium", # chromium, firefox, or webkit
        timeout=30.0             # Operation timeout in seconds
    )
    
    # Start the browser
    await browser.start()
    
    try:
        # Perform automation tasks
        await browser.goto("https://example.com")
        
    finally:
        # Always stop the browser to clean up resources
        await browser.stop()

asyncio.run(main())
```

### Navigation

Navigate to URLs using the `goto` method:

```python path=null start=null
# Navigate to a URL
await browser.goto("https://example.com")

# Navigate with natural language (uses LLM to determine action)
await browser.navigate("Go to the login page")
```

### Data Extraction

Extract data from pages using natural language queries:

```python path=null start=null
# Extract specific data
title = await browser.extract("What is the main heading on this page?")

# Extract structured data
products = await browser.extract(
    "Extract all product names and prices as a list"
)

# Extract with specific format instructions
data = await browser.extract(
    "Extract the article title, author, and publication date"
)
```

### Actions

Perform actions on the page using natural language commands:

```python path=null start=null
# Click elements
await browser.act("Click the 'Sign In' button")

# Fill forms
await browser.act("Type 'john@example.com' in the email field")

# Complex interactions
await browser.act("Select 'United States' from the country dropdown")

# Multiple actions
await browser.act("Fill in the search box with 'FlyBrowser' and press Enter")
```

### Screenshots

Capture screenshots of the current page:

```python path=null start=null
# Capture full page screenshot
screenshot_bytes = await browser.screenshot()

# Save to file
with open("screenshot.png", "wb") as f:
    f.write(screenshot_bytes)
```

## Complete Example

The following example demonstrates a complete automation workflow:

```python path=null start=null
import asyncio
from flybrowser import FlyBrowser

async def search_and_extract():
    browser = FlyBrowser(
        llm_provider="openai",
        llm_model="gpt-5.2",
        headless=True
    )
    
    await browser.start()
    
    try:
        # Navigate to a search engine
        await browser.goto("https://www.google.com")
        
        # Perform a search
        await browser.act("Type 'Python web automation' in the search box")
        await browser.act("Click the search button")
        
        # Extract results
        results = await browser.extract(
            "Extract the titles and URLs of the first 5 search results"
        )
        
        print("Search Results:")
        print(results)
        
        # Take a screenshot
        screenshot = await browser.screenshot()
        with open("search_results.png", "wb") as f:
            f.write(screenshot)
            
    finally:
        await browser.stop()

if __name__ == "__main__":
    asyncio.run(search_and_extract())
```

## Workflows

Workflows allow you to define reusable automation sequences:

```python path=null start=null
workflow = {
    "name": "login_workflow",
    "steps": [
        {"action": "goto", "url": "https://example.com/login"},
        {"action": "act", "command": "Enter 'user@example.com' in email field"},
        {"action": "act", "command": "Enter password in password field"},
        {"action": "act", "command": "Click the login button"},
        {"action": "extract", "query": "Confirm login was successful"}
    ]
}

result = await browser.run_workflow(workflow)
```

## Security Features

### PII Masking

FlyBrowser can automatically mask personally identifiable information:

```python path=null start=null
browser = FlyBrowser(
    llm_provider="openai",
    llm_model="gpt-5.2",
    pii_masking_enabled=True  # Enabled by default
)
```

Manually mask specific text:

```python path=null start=null
masked = await browser.mask_pii("Contact: john@example.com, 555-123-4567")
# Result: "Contact: [EMAIL], [PHONE]"
```

### Credential Storage

Store credentials securely for form filling:

```python path=null start=null
# Store a credential
await browser.store_credential("login_email", "user@example.com")

# Use stored credential for secure form filling
await browser.secure_fill("email", "login_email")
```

## Session Recording

Enable session recording to capture automation runs:

```python path=null start=null
browser = FlyBrowser(
    llm_provider="openai",
    llm_model="gpt-5.2",
    recording_enabled=True
)
```

## Error Handling

Implement proper error handling for production use:

```python path=null start=null
import asyncio
from flybrowser import FlyBrowser

async def robust_automation():
    browser = FlyBrowser(
        llm_provider="openai",
        llm_model="gpt-5.2",
        timeout=30.0
    )
    
    try:
        await browser.start()
        await browser.goto("https://example.com")
        
        # Operations that might fail
        result = await browser.extract("Extract data from the page")
        
    except TimeoutError:
        print("Operation timed out")
    except ConnectionError:
        print("Failed to connect to the page")
    except Exception as e:
        print(f"Automation error: {e}")
    finally:
        await browser.stop()

asyncio.run(robust_automation())
```

## Using in Jupyter Notebooks

Jupyter notebooks require special handling due to their built-in event loop. You have two options:

### Option 1: Direct await (Recommended)

Jupyter supports top-level `await` - just use it directly:

```python path=null start=null
# In a Jupyter cell
from flybrowser import FlyBrowser

browser = FlyBrowser(
    llm_provider="openai",
    llm_model="gpt-5.2"
)

await browser.start()
await browser.goto("https://example.com")
data = await browser.extract("What is the main heading?")
await browser.stop()
```

### Option 2: Using nest_asyncio

For compatibility with `asyncio.run()`, install and use `nest_asyncio`:

```bash path=null start=null
pip install flybrowser[jupyter]
```

```python path=null start=null
from flybrowser.utils.jupyter import setup_jupyter
import asyncio

setup_jupyter()

# Now asyncio.run() works in Jupyter
async def main():
    async with FlyBrowser(
        llm_provider="openai",
        llm_model="gpt-5.2"
    ) as browser:
        await browser.goto("https://example.com")
        return await browser.extract("What is the main heading?")

result = asyncio.run(main())
```

For complete Jupyter documentation, see the [Jupyter Notebooks Guide](jupyter-notebooks.md).

## Monitoring

Monitor browser operations in real-time:

```python path=null start=null
status = await browser.monitor()
print(f"Current URL: {status.get('url')}")
print(f"Page title: {status.get('title')}")
```

## Next Steps

Now that you understand the basics of FlyBrowser, explore the following resources:

- [Embedded Mode Guide](deployment/embedded.md) - Detailed embedded deployment documentation
- [Standalone Mode Guide](deployment/standalone.md) - Run FlyBrowser as an HTTP service
- [Cluster Mode Guide](deployment/cluster.md) - Deploy a distributed cluster
- [SDK Reference](reference/sdk.md) - Complete SDK documentation
- [REST API Reference](reference/api.md) - HTTP API documentation
- [CLI Reference](reference/cli.md) - Command-line interface documentation
