# Examples

Learn FlyBrowser by example. Each file is self-contained and runnable.

---

## Quick Start

```bash
# Install
pip install -e '.[dev]'
playwright install chromium

# Set your API key
export OPENAI_API_KEY="sk-..."

# Run an example
python examples/01_basic_usage.py
```

---

## Examples by Topic

### Getting Started

| File | What You'll Learn |
|------|-------------------|
| `01_basic_usage.py` | Navigation, extraction, screenshots |
| `02_action_agent.py` | Clicking, typing, form filling |
| `03_navigation_agent.py` | Natural language navigation |

### Advanced Patterns

| File | What You'll Learn |
|------|-------------------|
| `04_workflow_agent.py` | Multi-step workflows with variables |
| `05_monitoring_agent.py` | Waiting for conditions |
| `06_pii_safe_automation.py` | Secure credential handling |

### Deployment

| File | What You'll Learn |
|------|-------------------|
| `07_server_mode.py` | Connecting to a FlyBrowser server |
| `08_rest_api_client.py` | Using the REST API directly |
| `09_llm_providers.py` | OpenAI, Anthropic, Gemini, Ollama |
| `10_integrated_example.py` | All agents working together |
| `11_jupyter_notebook.py` | Using FlyBrowser in Jupyter notebooks |

---

## Running in Different Modes

### Embedded Mode (Default)

The browser runs in your Python process:

```bash
python examples/01_basic_usage.py
```

### Server Mode

Connect to a running FlyBrowser server:

```bash
# Terminal 1: Start server
task serve

# Terminal 2: Run example
python examples/07_server_mode.py
```

### Cluster Mode

Connect to a multi-node cluster:

```bash
cd docker
docker-compose -f docker-compose.cluster.yml up -d
python examples/07_server_mode.py
```

---

## Key Patterns

### Secure Credential Handling

Credentials never touch the LLM:

```python
# Store credential locally
email_id = browser.store_credential("email", "user@example.com")

# LLM sees: "Fill {{CREDENTIAL:email}} into the field"
# Browser fills: "user@example.com"
await browser.secure_fill("#email", email_id)
```

### Structured Extraction

Get data in the shape you need:

```python
products = await browser.extract(
    "Get all products",
    schema={"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"},
        "price": {"type": "number"}
    }}}
)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Missing API key | `export OPENAI_API_KEY="sk-..."` |
| Playwright not installed | `playwright install chromium` |
| Server not running | `task serve` |

---

<p align="center">
  <em>Questions? <a href="https://discord.gg/flybrowser">Join our Discord</a></em>
</p>
