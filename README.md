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

**Browser automation that speaks your language.** FlyBrowser pairs Playwright's rock-solid browser control with LLM intelligence, letting you automate the web using plain English instead of brittle selectors.

---

## Documentation

For comprehensive documentation, see the [Documentation Index](docs/index.md).

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, configuration, and basic usage |
| [Embedded Mode](docs/deployment/embedded.md) | Direct Python library integration |
| [Standalone Mode](docs/deployment/standalone.md) | HTTP service deployment |
| [Cluster Mode](docs/deployment/cluster.md) | Distributed high-availability deployment |
| [SDK Reference](docs/reference/sdk.md) | Complete Python SDK documentation |
| [REST API Reference](docs/reference/api.md) | HTTP endpoints and schemas |
| [CLI Reference](docs/reference/cli.md) | Command-line tools |
| [Configuration](docs/reference/configuration.md) | Environment variables and config files |

---

## Quick Start

### Installation

**One-liner (recommended):**
```bash
curl -fsSL https://get.flybrowser.dev | bash
```

**Or from source:**
```bash
git clone https://github.com/firefly-oss/flybrowsers.git
cd flybrowsers
./install.sh
```

**Installation Modes:**
```bash
# Virtual environment (default, recommended)
./install.sh --install-mode venv

# System-wide (direct access, may conflict)
./install.sh --install-mode system

# User installation (no sudo required)
./install.sh --install-mode user
```

**For development:**
```bash
git clone https://github.com/firefly-oss/flybrowsers.git
cd flybrowsers
./install.sh --dev  # or: task install:dev
```

**Verify installation:**
```bash
flybrowser doctor
```

See [INSTALL_GUIDE.md](INSTALL_GUIDE.md) for detailed installation options and activation guides.

### Embedded Usage

```python
import asyncio
from flybrowser import FlyBrowser

async def main():
    browser = FlyBrowser(
        llm_provider="openai",
        llm_model="gpt-4",
        api_key="your-api-key"
    )
    await browser.start()
    
    try:
        await browser.goto("https://example.com")
        data = await browser.extract("What is the main heading?")
        print(data)
        await browser.act("click the More information link")
    finally:
        await browser.stop()

asyncio.run(main())
```

### Jupyter Notebooks

**Setup (one-time):**
```bash
flybrowser setup jupyter install
```

Then in Jupyter, select the **FlyBrowser** kernel and use await directly:

```python
from flybrowser import FlyBrowser

browser = FlyBrowser(
    llm_provider="openai",
    llm_model="gpt-4"
)

await browser.start()
await browser.goto("https://example.com")
data = await browser.extract("What is the main heading?")
await browser.stop()
```

**Management commands:**
```bash
flybrowser setup jupyter status     # Check installation
flybrowser setup jupyter fix        # Fix issues
flybrowser setup jupyter uninstall  # Remove kernel
```

See the [Jupyter Notebooks Guide](docs/jupyter-notebooks.md) for complete documentation.

### Interactive REPL

```bash
flybrowser
```

This launches an interactive shell where you can:
- Navigate: `goto https://example.com`
- Extract data: `extract What is the main heading?`
- Perform actions: `act click the login button`
- Take screenshots: `screenshot`

### Standalone Server

```bash
flybrowser serve --port 8080
# or legacy: flybrowser-serve --host 0.0.0.0 --port 8080
```

```python
from flybrowser import FlyBrowserClient

client = FlyBrowserClient(endpoint="http://localhost:8080")
session = await client.create_session(llm_provider="openai", llm_model="gpt-4")
await client.navigate(session["session_id"], "https://example.com")
result = await client.extract(session["session_id"], "Get the page title")
await client.close_session(session["session_id"])
```

### Cluster Deployment

```bash
# Start a 3-node cluster
flybrowser serve --cluster --node-id node1 --port 8001 --raft-port 5001
flybrowser serve --cluster --node-id node2 --port 8002 --raft-port 5002 --peers node1:5001
flybrowser serve --cluster --node-id node3 --port 8003 --raft-port 5003 --peers node1:5001,node2:5002
```

See the [Cluster Deployment Guide](docs/deployment/cluster.md) for detailed setup instructions.

### Local LLMs with Ollama

```bash
ollama serve
ollama pull llama2
```

```python
browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="llama2"
)
```

---

## Architecture

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

FlyBrowser supports three deployment modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Embedded** | Direct Python library integration | Scripts, testing, single applications |
| **Standalone** | HTTP service with REST API | Microservices, multi-client access |
| **Cluster** | Distributed deployment with Raft consensus | High availability, production workloads |

---

## Development

```bash
# Install dev dependencies
./install.sh --dev
# or: task install:dev

# Run tests
task test

# Code quality
task check    # Format, lint, typecheck
task precommit # Full pre-commit checks
```

### Available Tasks

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
| `task docker:build` | Build Docker image |

### Common Issues

**"Playwright not installed"**
```bash
playwright install chromium
```

**"API key not found"**
```bash
export OPENAI_API_KEY="sk-..."
# or create .env file
```

**"Ollama connection refused"**
```bash
ollama serve  # Start Ollama first
```

**"Session timeout"**
```python
browser = FlyBrowser(timeout=120)  # Increase timeout
```

### Getting Help

- **Docs**: [flybrowser.dev/docs](https://flybrowser.dev/docs)
- **Discord**: [discord.gg/flybrowser](https://discord.gg/flybrowser)
- **Issues**: [GitHub Issues](https://github.com/firefly-oss/flybrowsers/issues)
- **Email**: support@flybrowser.dev

---

## Examples

The `examples/` directory contains working code for common scenarios:

| Example | What It Shows |
|---------|---------------|
| `01_basic_usage.py` | Navigation, extraction, screenshots |
| `02_action_agent.py` | Clicks, typing, multi-step actions |
| `03_navigation_agent.py` | Natural language navigation |
| `04_workflow_agent.py` | Multi-step workflows with variables |
| `05_monitoring_agent.py` | Waiting for conditions |
| `06_pii_safe_automation.py` | Secure credential handling |
| `07_server_mode.py` | Connecting to FlyBrowser server |
| `08_rest_api_client.py` | Direct REST API usage |
| `09_llm_providers.py` | OpenAI, Anthropic, Ollama |
| `10_integrated_example.py` | All agents working together |
| `11_jupyter_notebook.py` | Using FlyBrowser in Jupyter notebooks |

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development workflow
git checkout -b feature/your-feature
# Make changes
task check && task test
git commit -m "Add your feature"
git push origin feature/your-feature
# Open a Pull Request
```

---

## License

Copyright 2026 Firefly Software Solutions Inc. Licensed under Apache 2.0. See [LICENSE](LICENSE).

---

## Acknowledgments

Built on the shoulders of giants:
- [Playwright](https://playwright.dev/) for browser automation
- [FastAPI](https://fastapi.tiangolo.com/) for the REST API
- [OpenAI](https://openai.com/) and [Anthropic](https://anthropic.com/) for LLM APIs
- Inspired by [Stagehand](https://github.com/browserbase/stagehand)

---

<p align="center">
  <strong>Made with love by Firefly Software Solutions Inc</strong>
</p>