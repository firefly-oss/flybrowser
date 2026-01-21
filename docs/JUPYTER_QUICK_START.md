# Jupyter Notebook Quick Start

## [OK] One-Time Setup

```bash
# Activate the FlyBrowser virtual environment
source ~/.flybrowser/venv/bin/activate

# Install Jupyter and kernel support
pip install jupyter ipykernel nest_asyncio

# Register the FlyBrowser kernel
python -m ipykernel install --user --name=flybrowser --display-name="FlyBrowser"
```

**Done!** The kernel is now permanently installed.

---

## *Start* Using Jupyter with FlyBrowser

### 1. Start Jupyter Notebook

```bash
# From any directory
jupyter notebook
```

###  2. Select the FlyBrowser Kernel

**In your Jupyter notebook:**
- Click: **Kernel** -> **Change Kernel** -> **FlyBrowser**

**Or when creating a new notebook:**
- Click: **New** -> **FlyBrowser**

### 3. Write Your Code

```python
from flybrowser import FlyBrowser

browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="llama2",
    pretty_logs=True,
    log_level="INFO"
)

await browser.start()
await browser.goto("https://example.com")

# Extract data
result = await browser.extract("What is the main heading?")
if result["success"]:
    print(result["data"])
else:
    print(f"Error: {result['error']}")

await browser.stop()
```

---

## *Troubleshoot* Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'flybrowser'"

**Solution**: You're using the wrong kernel.

1. Check current kernel: Look at top-right of Jupyter (should say "FlyBrowser")
2. Change kernel: **Kernel** -> **Change Kernel** -> **FlyBrowser**
3. Restart: **Kernel** -> **Restart Kernel**

### Problem: Kernel not in list

**Re-install the kernel**:
```bash
source ~/.flybrowser/venv/bin/activate
python -m ipykernel install --user --name=flybrowser --display-name="FlyBrowser"
```

**Verify installation**:
```bash
jupyter kernelspec list
```

You should see:
```
flybrowser    /Users/your username/Library/Jupyter/kernels/flybrowser
```

### Problem: Async/await doesn't work

**Solution**: Add at the top of your notebook:
```python
import nest_asyncio
nest_asyncio.apply()
```

Or use the helper:
```python
from flybrowser.utils.jupyter import auto_setup_jupyter
auto_setup_jupyter()
```

---

## *Example* Complete Example

```python
# Cell 1: Setup (run once per notebook session)
import nest_asyncio
nest_asyncio.apply()

from flybrowser import FlyBrowser

# Cell 2: Initialize browser
browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="llama2",
    headless=True,
    pretty_logs=True
)

await browser.start()

# Cell 3: Navigate and extract
await browser.goto("https://news.ycombinator.com")

result = await browser.extract(
    "Get the titles of the top 5 stories",
    schema={
        "type": "object",
        "properties": {
            "stories": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }
)

if result["success"]:
    for i, title in enumerate(result["data"]["stories"], 1):
        print(f"{i}. {title}")
else:
    print(f"Error: {result['error']}")

# Cell 4: Cleanup
await browser.stop()
```

---

## *Commands* Management Commands

### Check if kernel is installed
```bash
jupyter kernelspec list | grep flybrowser
```

### Remove kernel
```bash
jupyter kernelspec uninstall flybrowser
```

### Reinstall kernel
```bash
source ~/.flybrowser/venv/bin/activate
python -m ipykernel install --user --name=flybrowser --display-name="FlyBrowser" --force
```

---

## *Tips* Tips

1. **Always use the FlyBrowser kernel** - This ensures the venv's Python is used
2. **Use `nest_asyncio`** - Required for async/await in Jupyter
3. **Check `result["success"]`** - All operations return error dicts, not exceptions
4. **One browser per notebook** - Create one browser instance and reuse it
5. **Clean up** - Always `await browser.stop()` when done

---

## *Links* More Information

- Full Guide: `docs/jupyter-notebooks.md`
- Examples: `examples/11_jupyter_notebook.py`
- Installation: `INSTALL_GUIDE.md`
- Error Handling: `docs/LOGGING.md`

---

## *Quick* Ultra-Quick Reference

```bash
# Setup (once)
source ~/.flybrowser/venv/bin/activate && pip install jupyter ipykernel nest_asyncio && python -m ipykernel install --user --name=flybrowser --display-name="FlyBrowser"

# Use (every time)
jupyter notebook
# Select FlyBrowser kernel
# from flybrowser import FlyBrowser
# ...
```

**That's it!** *
