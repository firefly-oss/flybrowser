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

## ✨ Key Features

- **🤖 Natural Language Control**: Describe actions in plain English—FlyBrowser figures out the details
- **🎥 Live Streaming & Recording**: Stream browser sessions in real-time (HLS/DASH/RTMP) with professional codecs
- **🧠 Smart Validators**: 99.8% success rate with automatic response validation and self-correction
- **🔒 PII Protection**: Secure credential handling that never exposes passwords to LLMs
- **🌍 Multi-Deployment**: Run embedded in scripts, as a standalone server, or in a distributed cluster
- **⚡ Hardware Acceleration**: NVENC, VideoToolbox, QSV support for high-performance encoding
- **📊 Built-in Observability**: Detailed timing breakdowns, metrics, and health monitoring

---

## 🚀 Quick Start

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

## 🎥 Streaming & Recording

Stream browser sessions in real-time or record for later playback:

```python
# Start live HLS stream
stream = await browser.start_stream(
    protocol="hls",      # or "dash", "rtmp"
    quality="medium",    # low_bandwidth, medium, high
    codec="h265"         # 40% bandwidth savings vs h264
)

print(f"Watch at: {stream['stream_url']}")
# Works in ALL modes: embedded, standalone, cluster

# Monitor stream health (nested structure)
status = await browser.get_stream_status()
if status.get('active'):
    stream_data = status.get('status', {})
    metrics = stream_data.get('metrics', {})
    print(f"FPS: {metrics.get('current_fps')}, Health: {stream_data.get('health')}")

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

## 🧠 Intelligent Agents

FlyBrowser's agent system handles the complexity of web automation:

### ActionAgent
```python
# Multi-step actions with automatic validation
await browser.act("fill out the contact form with name John Doe and email john@example.com")

# Smart element detection
await browser.act("click the blue submit button on the right side")
```

### ExtractionAgent
```python
# Structured data extraction
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

# Returns validated JSON, guaranteed structure
```

### NavigationAgent
```python
# Natural language navigation
await browser.navigate("go to the pricing page")
await browser.navigate("find the documentation link in the footer")
```

### WorkflowAgent
```python
# Multi-step workflows with variables
workflow = {
    "steps": [
        {"action": "goto", "url": "https://example.com"},
        {"action": "type", "selector": "#search", "value": "${query}"},
        {"action": "click", "selector": "button[type=submit]"},
        {"action": "extract", "query": "Get search results"}
    ]
}

result = await browser.run_workflow(workflow, variables={"query": "python"})
```

### ResponseValidator
99.8% success rate with automatic validation and self-correction:

- 5-stage validation strategy
- Schema enforcement
- LLM self-correction on failures
- < 1ms overhead on successful validations

---

## 🌍 Deployment Modes

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
| Recording | ✓ Local | ✓ S3/NFS/Local | ✓ S3/NFS |
| Live Streaming | ✓ Local server | ✓ Full support | ✓ Full support |
| Failover | N/A | N/A | ✓ Automatic |
| Use Case | Scripts, dev | Teams, services | Production |

---

## 🔒 Security & PII Protection

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

## 🎯 Use Cases

### Web Scraping
```python
# Extract structured data from any website
products = await browser.extract(
    "Get all product names, prices, and availability status",
    schema=product_schema
)
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

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Installation and basic usage |
| [Streaming & Recording](docs/features/streaming-recording.md) | Video recording and live streaming |
| [Validator Agents](docs/features/validator-agents.md) | Response validation and timing |
| [SDK Reference](docs/reference/sdk.md) | Complete Python API |
| [REST API](docs/reference/api.md) | HTTP endpoints |
| [CLI Reference](docs/reference/cli.md) | Command-line tools |
| [Embedded Mode](docs/deployment/embedded.md) | Direct integration |
| [Standalone Mode](docs/deployment/standalone.md) | Server deployment |
| [Cluster Mode](docs/deployment/cluster.md) | Distributed deployment |
| [Jupyter Notebooks](docs/jupyter-notebooks.md) | Notebook integration |
| [Examples](examples/) | Working code samples |

---

## 🎮 Interactive REPL

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

## 🌐 LLM Providers

Works with any LLM:

### OpenAI
```python
browser = FlyBrowser(
    llm_provider="openai",
    llm_model="gpt-4",
    api_key="sk-..."
)
```

### Anthropic
```python
browser = FlyBrowser(
    llm_provider="anthropic",
    llm_model="claude-3-opus-20240229",
    api_key="sk-ant-..."
)
```

### Local LLMs (Ollama)
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

**Supported Providers:**
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3)
- Ollama (Llama 2, Mistral, Mixtral)
- Any OpenAI-compatible endpoint

---

## ⚙️ Configuration

```python
# Via constructor
browser = FlyBrowser(
    llm_provider="openai",
    llm_model="gpt-4",
    api_key="sk-...",
    headless=True,
    browser_type="chromium",
    recording_enabled=False,
    pii_masking_enabled=True,
    timeout=30.0,
    log_level="INFO"
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

## 🧪 Development

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

## 📊 Performance

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

## 🤝 Contributing

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📝 License

Copyright 2026 Firefly Software Solutions Inc.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built with these amazing projects:

- [Playwright](https://playwright.dev/) - Rock-solid browser automation
- [FastAPI](https://fastapi.tiangolo.com/) - Modern API framework
- [FFmpeg](https://ffmpeg.org/) - Video encoding powerhouse
- [OpenAI](https://openai.com/) & [Anthropic](https://anthropic.com/) - LLM intelligence

Inspired by [Stagehand](https://github.com/browserbase/stagehand).

---

## 💬 Community & Support

- **Documentation**: [Full docs](docs/index.md)
- **Discord**: [Join our community](https://discord.gg/flybrowser)
- **Issues**: [GitHub Issues](https://github.com/firefly-oss/flybrowser/issues)
- **Discussions**: [GitHub Discussions](https://github.com/firefly-oss/flybrowser/discussions)
- **Email**: support@flybrowser.dev

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=firefly-oss/flybrowser&type=Date)](https://star-history.com/#firefly-oss/flybrowser&Date)

---

<p align="center">
  <strong>Made with ❤️ by Firefly Software Solutions Inc</strong>
</p>

<p align="center">
  <a href="https://flybrowser.dev">Website</a> •
  <a href="https://github.com/firefly-oss/flybrowser">GitHub</a> •
  <a href="https://discord.gg/flybrowser">Discord</a> •
  <a href="https://twitter.com/flybrowser">Twitter</a>
</p>
