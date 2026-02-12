# Installation

This guide covers installing FlyBrowser and its dependencies on your system.

## System Requirements

FlyBrowser requires:

- **Python 3.9 or later** - Python 3.10, 3.11, and 3.12 are fully supported
- **Operating System** - macOS, Linux, or Windows
- **Memory** - At least 4GB RAM recommended (browsers consume significant memory)
- **Disk Space** - Approximately 500MB for the package and browser binaries

## Installation Methods

### Using pip (Recommended)

The simplest way to install FlyBrowser is via pip:

```bash
pip install flybrowser
```

After installation, install the browser binaries that Playwright requires:

```bash
playwright install chromium
```

You can also install Firefox or WebKit if needed:

```bash
playwright install firefox webkit
```

### Using uv (Fast Alternative)

If you use uv for Python package management:

```bash
uv pip install flybrowser
uv run playwright install chromium
```

### From Source

To install from source for development or to get the latest changes:

```bash
git clone https://github.com/fireflyresearch/flybrowser.git
cd flybrowser
pip install -e .
playwright install chromium
```

For development with testing and linting tools:

```bash
pip install -e ".[dev]"
```

### Using the Install Script

FlyBrowser provides an installation script that handles all setup:

```bash
curl -fsSL https://get.flybrowser.dev | bash
```

Or if you have cloned the repository:

```bash
./install.sh
```

The install script:
- Detects your Python version
- Creates a virtual environment if needed
- Installs FlyBrowser and dependencies
- Installs Playwright browsers
- Verifies the installation

## Dependencies

FlyBrowser automatically installs these dependencies:

**Core Dependencies**

- `playwright` - Browser automation engine
- `pydantic` - Data validation and settings
- `aiohttp` - Async HTTP client
- `beautifulsoup4` - HTML parsing
- `pillow` - Image processing

**LLM Provider Libraries**

- `openai` - OpenAI API client
- `anthropic` - Anthropic API client

**Server Dependencies**

- `fastapi` - REST API framework
- `uvicorn` - ASGI server
- `redis` - Session storage (optional)

**Optional Dependencies**

For Jupyter notebook support:

```bash
pip install flybrowser[jupyter]
```

This adds:
- `nest_asyncio` - Allows nested event loops in notebooks
- `ipython` - Enhanced interactive Python

For an enhanced REPL experience:

```bash
pip install flybrowser[repl]
```

This adds:
- `prompt_toolkit` - Better command line editing

## LLM Provider Setup

FlyBrowser requires access to an LLM provider. Set up at least one of the following:

### OpenAI

1. Create an account at [platform.openai.com](https://platform.openai.com)
2. Generate an API key from the API keys section
3. Either set the environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

Or pass the key when creating a FlyBrowser instance:

```python
browser = FlyBrowser(llm_provider="openai", api_key="sk-...")
```

### Anthropic

1. Create an account at [console.anthropic.com](https://console.anthropic.com)
2. Generate an API key
3. Set the environment variable or pass directly:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

```python
browser = FlyBrowser(llm_provider="anthropic", api_key="sk-ant-...")
```

### Google Gemini

1. Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set the environment variable or pass directly:

```bash
export GOOGLE_API_KEY="AIza..."
```

```python
browser = FlyBrowser(llm_provider="gemini", api_key="AIza...")
```

### Ollama (Local Models)

For running models locally without API costs:

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Start the Ollama server:

```bash
ollama serve
```

3. Pull a model:

```bash
ollama pull qwen3:8b
```

4. Use with FlyBrowser (no API key needed):

```python
browser = FlyBrowser(llm_provider="ollama", llm_model="qwen3:8b")
```

### Qwen (Alibaba Cloud DashScope)

Qwen models are available through Alibaba Cloud's DashScope service:

1. Create an account at [DashScope Console](https://dashscope.console.aliyun.com/)
2. Generate an API key from the API Keys section
3. Set the environment variable or pass directly:

```bash
export DASHSCOPE_API_KEY="sk-..."  # or QWEN_API_KEY
```

```python
browser = FlyBrowser(llm_provider="qwen", api_key="sk-...")
```

Qwen supports multiple regions. For international access:

```python
browser = FlyBrowser(
    llm_provider="qwen",
    llm_model="qwen-plus",
    region="international",  # or "us" for US region
)
```

Available models include: `qwen-plus`, `qwen-turbo`, `qwen-max`, `qwen3-235b-a22b`, and vision models `qwen-vl-max`, `qwen-vl-plus`.

## Verifying Installation

After installation, verify everything works:

```bash
flybrowser doctor
```

This command checks:
- Python version compatibility
- Package installation status
- Playwright browser availability
- LLM provider configuration

You can also run a quick test:

```python
import asyncio
from flybrowser import FlyBrowser

async def test():
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        await browser.goto("https://example.com")
        print("Installation successful!")

asyncio.run(test())
```

## Platform-Specific Notes

### macOS

On macOS, you may need to allow Playwright browsers through security settings the first time they run. If you see a security warning, go to System Preferences > Security & Privacy and allow the browser.

For Apple Silicon (M1/M2/M3), FlyBrowser and Playwright work natively without Rosetta.

### Linux

On Ubuntu/Debian, Playwright may need additional system dependencies:

```bash
sudo apt-get install libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libdbus-1-3 libxkbcommon0 libxcomposite1 \
    libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2
```

Playwright provides a command to install all dependencies:

```bash
playwright install-deps chromium
```

### Windows

On Windows, ensure you have the Visual C++ Redistributable installed. Playwright browsers should work without additional setup.

### Docker

When running in Docker, use the Playwright-provided base images:

```dockerfile
FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

WORKDIR /app
COPY . .
RUN pip install flybrowser

CMD ["python", "your_script.py"]
```

## Troubleshooting

### Browser fails to launch

If Playwright browsers fail to launch:

1. Ensure browsers are installed: `playwright install chromium`
2. Check for missing system dependencies: `playwright install-deps chromium`
3. Try running in headed mode first to see any error dialogs

### Import errors

If you get import errors after installation:

1. Verify you are using the correct Python environment
2. Check that all dependencies installed: `pip check`
3. Try reinstalling: `pip install --force-reinstall flybrowser`

### LLM connection issues

If LLM calls fail:

1. Verify your API key is correct
2. Check network connectivity to the provider
3. Ensure you have API credits/quota available
4. Try a different model or provider

## Next Steps

With FlyBrowser installed, proceed to the [Quickstart Guide](quickstart.md) to run your first automation.
