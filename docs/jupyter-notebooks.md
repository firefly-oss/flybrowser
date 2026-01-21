# Using FlyBrowser in Jupyter Notebooks

This guide explains how to use FlyBrowser in Jupyter notebooks, Jupyter Lab, Google Colab, and other IPython-based environments.

## Quick Start

### 1. Install Jupyter Kernel (One-Time Setup)

```bash
flybrowser setup jupyter install
```

This command will:
- [OK] Install jupyter and ipykernel if needed
- [OK] Register the FlyBrowser kernel
- [OK] Show you how to use it

**Management Commands:**
```bash
flybrowser setup jupyter status     # Check installation
flybrowser setup jupyter fix        # Fix broken installation
flybrowser setup jupyter uninstall  # Remove kernel
```

### 2. Launch Jupyter

```bash
jupyter notebook
```

### 3. Select FlyBrowser Kernel

In your notebook:
- **Kernel** -> **Change Kernel** -> **FlyBrowser**

### 4. Use FlyBrowser

```python
from flybrowser import FlyBrowser

browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="llama2"
)

await browser.start()
await browser.goto("https://example.com")
data = await browser.extract("What is the main heading?")
print(data)
await browser.stop()
```

**That's it!** *

---

## Table of Contents

- [Quick Start](#quick-start)
- [Understanding the Problem](#understanding-the-problem)
- [Installation](#installation)
- [Two Approaches](#two-approaches)
  - [Approach 1: Direct await (Recommended)](#approach-1-direct-await-recommended)
  - [Approach 2: nest_asyncio](#approach-2-nest_asyncio)
- [Complete Examples](#complete-examples)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)

---

## Understanding the Problem

When you try to use `asyncio.run()` in a Jupyter notebook, you'll encounter this error:

```python path=null start=null
RuntimeError: asyncio.run() cannot be called from a running event loop
```

This happens because:
1. Jupyter notebooks already run an event loop in the background
2. `asyncio.run()` tries to create a new event loop
3. Python doesn't allow nested event loops by default

FlyBrowser provides two solutions to this problem.

---

## Installation

### Basic Installation

```bash path=null start=null
pip install flybrowser
playwright install chromium
```

### With Jupyter Support

For the full Jupyter experience with `nest_asyncio` support:

```bash path=null start=null
pip install flybrowser[jupyter]
playwright install chromium
```

This installs:
- `nest_asyncio` - Allows nested event loops
- `ipython` - Enhanced interactive Python

### Setting Up Jupyter Kernel

**Automatic Setup (Recommended):**
```bash
flybrowser setup jupyter install
```

**Alternative - Manual Setup:**

If you installed FlyBrowser using `./install.sh` and prefer manual setup:

```bash
# Activate venv
source ~/.flybrowser/venv/bin/activate

# Install and register kernel
pip install jupyter ipykernel nest_asyncio
python -m ipykernel install --user --name=flybrowser --display-name="FlyBrowser"
```

Then in your Jupyter notebook:
1. Go to **Kernel -> Change Kernel -> FlyBrowser**
2. Restart the kernel
3. Your code should now work!

**Troubleshooting:**
If the kernel isn't working:
```bash
flybrowser setup jupyter status  # Check what's wrong
flybrowser setup jupyter fix     # Fix it automatically
```

---

## Two Approaches

### Approach 1: Direct await (Recommended)

Jupyter notebooks support top-level `await` directly in cells. This is the cleanest and most natural approach.

#### Cell 1: Imports

```python path=null start=null
from flybrowser import FlyBrowser
```

#### Cell 2: Create and Start Browser

```python path=null start=null
browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="llama2",
    headless=True
)

await browser.start()
```

#### Cell 3: Navigate and Extract

```python path=null start=null
await browser.goto("https://example.com")
data = await browser.extract("What is the main heading?")
print(data)
```

#### Cell 4: Perform Actions

```python path=null start=null
await browser.act("click the More information link")
```

#### Cell 5: Clean Up

```python path=null start=null
await browser.stop()
```

**Advantages:**
- [OK] Clean and simple
- [OK] No additional setup required
- [OK] Native Jupyter support
- [OK] Works in all modern Jupyter environments

**When to Use:**
- Default choice for new notebooks
- When working exclusively in Jupyter
- When you want the simplest solution

---

### Approach 2: nest_asyncio

This approach allows you to use `asyncio.run()` in Jupyter, which is useful for porting existing scripts.

#### Cell 1: Setup

```python path=null start=null
from flybrowser import FlyBrowser
from flybrowser.utils.jupyter import setup_jupyter
import asyncio

# Configure nest_asyncio
setup_jupyter()
```

#### Cell 2: Define Your Async Function

```python path=null start=null
async def main():
    async with FlyBrowser(
        llm_provider="ollama",
        llm_model="llama2",
        headless=True
    ) as browser:
        await browser.goto("https://example.com")
        data = await browser.extract("What is the main heading?")
        print(data)
        
        await browser.act("click the More information link")
    
    return "Complete!"
```

#### Cell 3: Run It

```python path=null start=null
result = asyncio.run(main())
print(result)
```

**Advantages:**
- [OK] Allows using `asyncio.run()`
- [OK] Easy to port existing scripts
- [OK] Familiar pattern for Python developers

**When to Use:**
- Porting existing Python scripts to Jupyter
- When you prefer the `asyncio.run()` pattern
- When sharing code between scripts and notebooks

---

## Complete Examples

### Example 1: Web Scraping in Jupyter

```python path=null start=null
# Cell 1: Setup
from flybrowser import FlyBrowser

browser = FlyBrowser(
    llm_provider="openai",
    llm_model="gpt-5.2",
    headless=True
)

await browser.start()
```

```python path=null start=null
# Cell 2: Scrape Product Information
await browser.goto("https://example-store.com")

products = await browser.extract(
    "Extract all product names and prices",
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

print(f"Found {len(products)} products")
for product in products:
    print(f"- {product['name']}: ${product['price']}")
```

```python path=null start=null
# Cell 3: Take Screenshot
screenshot = await browser.screenshot(full_page=True)
print(f"Screenshot: {screenshot['width']}x{screenshot['height']} pixels")

# Display in Jupyter
from IPython.display import Image
import base64

Image(data=base64.b64decode(screenshot['data_base64']))
```

```python path=null start=null
# Cell 4: Cleanup
await browser.stop()
```

### Example 2: Automated Form Filling

```python path=null start=null
# Cell 1: Setup
from flybrowser import FlyBrowser

browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="qwen3:8b",
    headless=False  # Show browser window
)

await browser.start()
```

```python path=null start=null
# Cell 2: Navigate to Form
await browser.goto("https://example.com/contact")
```

```python path=null start=null
# Cell 3: Fill Form
await browser.act("Type 'John Doe' in the name field")
await browser.act("Type 'john@example.com' in the email field")
await browser.act("Type 'Hello from FlyBrowser!' in the message field")
```

```python path=null start=null
# Cell 4: Submit and Verify
await browser.act("Click the submit button")

result = await browser.extract("Was the form submitted successfully?")
print(result)
```

```python path=null start=null
# Cell 5: Cleanup
await browser.stop()
```

### Example 3: Using with Context Manager

```python path=null start=null
# Single cell - context manager handles start/stop automatically
from flybrowser import FlyBrowser

async with FlyBrowser(
    llm_provider="anthropic",
    llm_model="claude-sonnet-4-5-20250929",
    headless=True
) as browser:
    await browser.goto("https://news.ycombinator.com")
    
    headlines = await browser.extract(
        "Extract the top 5 story titles and scores",
        schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "score": {"type": "integer"}
                }
            }
        }
    )
    
    for i, story in enumerate(headlines[:5], 1):
        print(f"{i}. {story['title']} ({story['score']} points)")

# Browser automatically stopped here
```

---

## Common Patterns

### Pattern 1: Interactive Development

Split operations across cells for iterative development:

```python path=null start=null
# Cell 1: Start browser once
browser = FlyBrowser(llm_provider="ollama", llm_model="llama2")
await browser.start()
```

```python path=null start=null
# Cell 2: Navigate (run multiple times if needed)
await browser.goto("https://example.com")
```

```python path=null start=null
# Cell 3: Experiment with extraction (iterate on query)
data = await browser.extract("What information is on this page?")
print(data)
```

```python path=null start=null
# Cell 4: Try different actions
await browser.act("scroll down")
```

```python path=null start=null
# Cell 5: Final cleanup
await browser.stop()
```

### Pattern 2: Reusable Functions

Define reusable async functions in Jupyter:

```python path=null start=null
async def scrape_page(url: str, query: str):
    """Scrape data from a URL using a natural language query."""
    async with FlyBrowser(
        llm_provider="openai",
        llm_model="gpt-5.2"
    ) as browser:
        await browser.goto(url)
        return await browser.extract(query)

# Use it
result = await scrape_page(
    "https://example.com",
    "Extract the main content"
)
print(result)
```

### Pattern 3: Batch Processing

Process multiple URLs in sequence:

```python path=null start=null
urls = [
    "https://example1.com",
    "https://example2.com",
    "https://example3.com"
]

results = []

async with FlyBrowser(
    llm_provider="ollama",
    llm_model="qwen3:8b"
) as browser:
    for url in urls:
        print(f"Processing {url}...")
        await browser.goto(url)
        data = await browser.extract("Extract the page title")
        results.append({"url": url, "title": data})

for result in results:
    print(f"{result['url']}: {result['title']}")
```

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'flybrowser'"

**Problem:** Your Jupyter notebook is using a different Python kernel than the one where FlyBrowser is installed.

**Solution 1 (If installed via install.sh):** Register the FlyBrowser venv as a Jupyter kernel:

```bash path=null start=null
# Run the helper script
cd /path/to/flybrowser
./setup_jupyter_kernel.sh

# Then in Jupyter: Kernel -> Change Kernel -> Python (FlyBrowser)
```

**Solution 2:** Install FlyBrowser in your current Python environment:

```bash path=null start=null
pip install -e /path/to/flybrowser[jupyter]
```

**Solution 3:** Start Jupyter from the FlyBrowser venv:

```bash path=null start=null
source ~/.flybrowser/venv/bin/activate
jupyter notebook
```

### Error: "RuntimeError: asyncio.run() cannot be called from a running event loop"

**Solution 1 (Recommended):** Use `await` directly instead of `asyncio.run()`

```python path=null start=null
# [X] Don't do this in Jupyter
asyncio.run(main())

# [OK] Do this instead
await main()
```

**Solution 2:** Use `nest_asyncio`

```python path=null start=null
from flybrowser.utils.jupyter import setup_jupyter
setup_jupyter()

# Now asyncio.run() works
asyncio.run(main())
```

### Error: "nest_asyncio not installed"

Install Jupyter extras:

```bash path=null start=null
pip install flybrowser[jupyter]
```

### Error: "Playwright not installed"

Install Playwright browsers:

```bash path=null start=null
playwright install chromium
```

### Browser Doesn't Start

Check your LLM configuration:

```python path=null start=null
# For OpenAI
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

# For Ollama (must be running)
# In terminal: ollama serve
```

### Kernel Crashes or Hangs

1. **Restart the kernel:** Kernel -> Restart Kernel
2. **Ensure browser.stop() is called:**
   ```python path=null start=null
   try:
       await browser.start()
       # ... your code ...
   finally:
       await browser.stop()
   ```

3. **Use context manager for automatic cleanup:**
   ```python path=null start=null
   async with FlyBrowser(...) as browser:
       # ... your code ...
   # Automatically cleaned up
   ```

### Display Screenshots in Jupyter

```python path=null start=null
from IPython.display import Image
import base64

screenshot = await browser.screenshot()
Image(data=base64.b64decode(screenshot['data_base64']))
```

### Memory Issues with Long Sessions

Restart the browser periodically in long-running notebooks:

```python path=null start=null
# After many operations
await browser.stop()
await browser.start()
```

---

## Best Practices

1. **Use `async with` for automatic cleanup:**
   ```python path=null start=null
   async with FlyBrowser(...) as browser:
       # Your code
   # No need to call stop()
   ```

2. **Set `headless=False` during development:**
   ```python path=null start=null
   browser = FlyBrowser(headless=False)  # See what's happening
   ```

3. **Handle errors gracefully:**
   ```python path=null start=null
   try:
       await browser.goto(url)
   except Exception as e:
       print(f"Navigation failed: {e}")
   ```

4. **Save important results:**
   ```python path=null start=null
   import json
   
   data = await browser.extract("...")
   with open("results.json", "w") as f:
       json.dump(data, f, indent=2)
   ```

5. **Use environment variables for API keys:**
   ```python path=null start=null
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   
   browser = FlyBrowser(
       llm_provider="openai",
       api_key=os.getenv("OPENAI_API_KEY")
   )
   ```

---

## Google Colab

FlyBrowser works in Google Colab with some additional setup:

```python path=null start=null
# Cell 1: Install dependencies
!pip install flybrowser[jupyter]
!playwright install-deps
!playwright install chromium
```

```python path=null start=null
# Cell 2: Use FlyBrowser (use await directly)
from flybrowser import FlyBrowser

async with FlyBrowser(
    llm_provider="openai",
    api_key="sk-...",
    headless=True
) as browser:
    await browser.goto("https://example.com")
    data = await browser.extract("What is on this page?")
    print(data)
```

---

## Next Steps

- Explore the [complete example notebook](../examples/11_jupyter_notebook.py)
- Check out [other examples](../examples/README.md)
- Read the [full SDK reference](reference/sdk.md)
- Join our [Discord community](https://discord.gg/flybrowser)

---

**Need help?** Open an issue on [GitHub](https://github.com/firefly-oss/flybrowsers/issues) or ask in [Discord](https://discord.gg/flybrowser).
