# FlyBrowser Documentation

FlyBrowser is an LLM-powered browser automation framework that provides intelligent web interaction capabilities through natural language commands. It supports three deployment modes to accommodate different use cases: embedded library integration, standalone server deployment, and distributed cluster configuration.

## Documentation Overview

### Getting Started

- [Getting Started Guide](getting-started.md) - Installation, configuration, and first steps with FlyBrowser
- [Jupyter Notebooks Guide](jupyter-notebooks.md) - Using FlyBrowser in Jupyter, JupyterLab, and Google Colab

### Deployment Guides

Step-by-step instructions for each deployment mode:

- [Embedded Mode](deployment/embedded.md) - Integrate FlyBrowser directly into your Python application
- [Standalone Mode](deployment/standalone.md) - Run FlyBrowser as a standalone HTTP service
- [Cluster Mode](deployment/cluster.md) - Deploy a distributed cluster with high availability and horizontal scaling

### Reference Documentation

Complete technical reference for all FlyBrowser interfaces:

- [SDK Reference](reference/sdk.md) - Python SDK classes, methods, and parameters
- [REST API Reference](reference/api.md) - HTTP endpoints, request/response schemas, and authentication
- [CLI Reference](reference/cli.md) - Command-line tools and options
- [Configuration Reference](reference/configuration.md) - Environment variables and configuration options

### Architecture

- [Architecture Overview](../ARCHITECTURE.md) - System design, components, and internal structure

## Deployment Mode Comparison

### Embedded Mode

Direct Python library integration within your application process.

**Use Cases:**
- Scripts and automation tools
- Testing frameworks
- Applications requiring tight integration
- Single-machine deployments

**Characteristics:**
- No network overhead
- Synchronous and asynchronous APIs
- Direct access to all features
- Browser lifecycle managed by your application

### Standalone Mode

HTTP service running as a separate process or container.

**Use Cases:**
- Microservice architectures
- Language-agnostic integration
- Containerized deployments
- Development and testing environments

**Characteristics:**
- REST API interface
- Session management
- Process isolation
- Single-node deployment

### Cluster Mode

Distributed deployment with Raft consensus for high availability.

**Use Cases:**
- Production workloads requiring high availability
- Horizontal scaling for increased throughput
- Multi-tenant environments
- Mission-critical automation

**Characteristics:**
- Automatic leader election
- Session replication across nodes
- Fault tolerance (survives minority node failures)
- Dynamic cluster scaling

## Quick Start

### Installation

```bash path=null start=null
curl -fsSL https://get.flybrowser.dev | bash
# Or from source:
git clone https://github.com/firefly-oss/flybrowsers.git
cd flybrowsers && ./install.sh
```

### Embedded Usage

```python path=null start=null
from flybrowser import FlyBrowser

async def main():
    browser = FlyBrowser(
        llm_provider="openai",     # or: anthropic, gemini, ollama
        llm_model="gpt-5.2"        # uses provider default if not set
    )
    await browser.start()
    
    await browser.goto("https://example.com")
    data = await browser.extract("Extract the main heading")
    
    await browser.stop()
```

### Standalone Server

```bash path=null start=null
flybrowser-serve --host 0.0.0.0 --port 8000
```

### Cluster Deployment

```bash path=null start=null
# Node 1 (initial leader)
flybrowser-serve --cluster --node-id node1 --raft-port 4321 --port 8001

# Node 2
flybrowser-serve --cluster --node-id node2 --raft-port 4322 --port 8002 \
    --peers node1:4321

# Node 3
flybrowser-serve --cluster --node-id node3 --raft-port 4323 --port 8003 \
    --peers node1:4321,node2:4322
```

## Requirements

- Python 3.9 or higher
- Playwright browsers (installed automatically)
- LLM provider access:
  - **Cloud providers**: OpenAI, Anthropic, or Google Gemini (API key required)
  - **Local providers**: Ollama, LM Studio, LocalAI, or vLLM (no API key needed)

## License

FlyBrowser is released under the MIT License.
