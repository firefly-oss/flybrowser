# FlyBrowser Logging Guide

This guide explains how to configure and use FlyBrowser's logging system to understand what your automation is doing.

## Quick Start

### Human-Readable Logs (Default, Recommended)

```python
from flybrowser import FlyBrowser

browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="llama2",
    pretty_logs=True,   # Human-readable colored logs (default)
    log_level="INFO"    # INFO, DEBUG, WARNING, ERROR
)
```

**Output:**
```
10:15:23 [    INFO] Initializing FlyBrowser in embedded mode
10:15:23 [    INFO] Starting chromium browser (headless=True)
10:15:24 [    INFO] Browser started successfully
10:15:24 [    INFO] Navigating to https://example.com
10:15:25 [    INFO] Successfully navigated to https://example.com
10:15:25 [    INFO] Extracting data: What is the main heading?
10:15:30 [    INFO] Data extracted successfully
```

### JSON Logs (For Log Aggregation)

```python
browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="llama2",
    pretty_logs=False,  # JSON format
    log_level="INFO"
)
```

**Output:**
```json
{"timestamp": "2026-01-21T10:15:23Z", "level": "info", "logger": "flybrowser", "message": "Initializing FlyBrowser in embedded mode"}
{"timestamp": "2026-01-21T10:15:23Z", "level": "info", "logger": "flybrowser", "message": "Starting chromium browser (headless=True)"}
```

---

## Log Levels

### ERROR
Only shows errors - when things go wrong.

```python
browser = FlyBrowser(..., log_level="ERROR")
```

**When to use**: Production environments where you only want to know about failures.

### WARNING (Quiet)
Shows warnings and errors.

```python
browser = FlyBrowser(..., log_level="WARNING")
```

**When to use**: Production environments, when you want quiet operation but still want to see important issues.

### INFO (Default, Recommended)
Shows normal operations, warnings, and errors.

```python
browser = FlyBrowser(..., log_level="INFO")  # Default
```

**What you'll see**:
- Browser start/stop
- Navigation (goto, navigate)
- Data extraction progress
- Action execution ("Executing action: click button")
- Success/failure messages

**When to use**: Development, debugging, understanding what FlyBrowser is doing. **This is the default and recommended for most users.**

### DEBUG (Verbose)
Shows everything including internal details.

```python
browser = FlyBrowser(..., log_level="DEBUG")
```

**Additional information you'll see**:
- LLM prompts sent
- LLM responses received
- Element detection attempts
- Retry logic
- Internal state changes
- Performance metrics

**When to use**: Deep debugging, troubleshooting issues, understanding LLM interactions.

---

## Disabling Logs Completely

Not recommended, but if you need silence:

```python
import logging

# Before creating FlyBrowser
logging.getLogger("flybrowser").setLevel(logging.CRITICAL)

browser = FlyBrowser(...)
```

---

## Understanding Log Output

### Normal Operation Flow

With `log_level="INFO"` (default), a typical session looks like:

```
10:15:23 [    INFO] Initializing FlyBrowser in embedded mode
10:15:23 [    INFO] Starting chromium browser (headless=True)
10:15:24 [    INFO] Browser started successfully
10:15:24 [    INFO] FlyBrowser started in embedded mode
10:15:24 [    INFO] Navigating to https://example.com
10:15:25 [    INFO] Successfully navigated to https://example.com
10:15:25 [    INFO] Extracting data: What is the main heading?
10:15:30 [    INFO] Data extracted successfully
10:15:30 [    INFO] Executing action: click the login button
10:15:32 [    INFO] Planned 3 action steps
10:15:32 [    INFO] Finding element: login button
10:15:33 [    INFO] Action completed: 3/3 steps
```

### When Things Go Wrong

Errors are clearly marked:

```
10:15:33 [   ERROR] Element detection failed: Could not find element matching "nonexistent button"
10:15:34 [   ERROR] Action execution failed: Failed to execute action 'click the nonexistent button'
```

### Warnings

```
10:15:33 [ WARNING] Step 1 failed (attempt 1), retrying...
10:15:34 [ WARNING] Skipping Playwright browser installation
```

---

## Environment Variables

You can also configure logging via environment variables:

```bash
# Set log level
export FLYBROWSER_LOG_LEVEL=DEBUG

# Set log format
export FLYBROWSER_LOG_FORMAT=human  # or json, text

# Then use FlyBrowser normally
python your_script.py
```

---

## Common Scenarios

### Scenario 1: Development - See Everything

```python
browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="llama2",
    pretty_logs=True,
    log_level="INFO"  # or DEBUG for even more details
)
```

### Scenario 2: Production - Quiet Operation

```python
browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="llama2",
    pretty_logs=False,  # JSON for log aggregation
    log_level="WARNING"  # Only warnings and errors
)
```

### Scenario 3: Debugging Issues

```python
browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="llama2",
    pretty_logs=True,
    log_level="DEBUG"  # See everything
)
```

### Scenario 4: Jupyter Notebooks - Clean Output

```python
# In Jupyter, you might want less verbose output
browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="llama2",
    pretty_logs=True,
    log_level="WARNING"  # Only show issues
)
```

---

## Programmatic Log Control

### Get the Logger

```python
import logging

# Get FlyBrowser logger
logger = logging.getLogger("flybrowser")

# Change level dynamically
logger.setLevel(logging.DEBUG)

# Silence it temporarily
logger.setLevel(logging.CRITICAL)

# Restore
logger.setLevel(logging.INFO)
```

### Custom Log Handlers

```python
import logging

logger = logging.getLogger("flybrowser")

# Add file handler
file_handler = logging.FileHandler("flybrowser.log")
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Now logs go to both console and file
browser = FlyBrowser(...)
```

---

## Error Handling Best Practices

FlyBrowser uses a **consistent error handling pattern** across all agents. Instead of raising exceptions for operational failures, agents return error information in a dictionary with a `success` field. This allows your automation to gracefully handle failures and continue execution.

### Consistent Error Dictionary Pattern

All agent operations return dictionaries with the following structure:

**Success Response:**
```python
{
    "success": True,
    "data": {...},        # Actual result data (varies by agent)
    # ... other agent-specific fields
}
```

**Error Response:**
```python
{
    "success": False,
    "error": "Error message",
    "exception_type": "ExceptionClass",
    # ... other agent-specific fields with default values
}
```

### Examples for Each Agent

#### 1. ActionAgent - Actions and Form Filling

```python
# Execute an action
result = await browser.act("click the login button")

if result["success"]:
    print(f"Action completed: {result['steps_completed']}/{result['total_steps']} steps")
else:
    print(f"Action failed: {result['error']}")
    print(f"Completed {result['steps_completed']}/{result['total_steps']} steps before failing")
    # Your automation continues...
```

#### 2. ExtractionAgent - Data Extraction

```python
# Extract data from page
result = await browser.extract("What is the price?")

if result["success"]:
    price = result["data"]  # Extracted data
    print(f"Price: {price}")
else:
    print(f"Extraction failed: {result['error']}")
    # Use default value or skip
    price = None
```

#### 3. NavigationAgent - Page Navigation

```python
# Navigate using natural language
result = await browser.navigate("go to the products page")

if result["success"]:
    print(f"Navigated to: {result['url']}")
else:
    print(f"Navigation failed: {result['error']}")
    print(f"Current URL: {result['url']}")
    # Try alternative navigation
```

#### 4. WorkflowAgent - Multi-Step Workflows

```python
# Execute complex workflow
result = await browser.run_workflow(
    "Login, search for laptop, add first result to cart"
)

if result["success"]:
    print(f"Workflow completed: {result['steps_completed']} steps")
else:
    print(f"Workflow failed: {result['error']}")
    print(f"Completed {result['steps_completed']}/{result['total_steps']} steps")
    # Examine step_results for details
    for step in result.get('step_results', []):
        if not step.get('success'):
            print(f"Failed step: {step}")
```

#### 5. MonitoringAgent - Page Monitoring

```python
# Monitor for changes
result = await browser.monitor(
    "wait for the loading spinner to disappear",
    timeout=30
)

if result["success"]:
    print(f"Condition met after {result['monitoring_duration']}s")
else:
    print(f"Monitoring failed: {result['error']}")
    # Continue with timeout handling
```

### Graceful Degradation Example

Here's a real-world example showing how to handle failures gracefully:

```python
async def scrape_product_with_fallback(browser, product_url):
    """Scrape product with fallback strategies."""
    
    # Try to navigate
    nav_result = await browser.navigate(product_url)
    if not nav_result["success"]:
        logger.error(f"Navigation failed: {nav_result['error']}")
        return None
    
    # Try structured extraction first
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"},
            "in_stock": {"type": "boolean"}
        }
    }
    
    result = await browser.extract("Extract product info", schema=schema)
    
    if result["success"]:
        return result["data"]
    else:
        # Fallback: Try individual extractions
        logger.warning(f"Structured extraction failed: {result['error']}")
        logger.info("Trying individual field extraction...")
        
        product = {}
        
        # Extract name
        name_result = await browser.extract("What is the product name?")
        if name_result["success"]:
            product["name"] = name_result["data"].get("extracted_text", "Unknown")
        
        # Extract price
        price_result = await browser.extract("What is the price?")
        if price_result["success"]:
            product["price"] = price_result["data"].get("extracted_text", "N/A")
        
        return product if product else None

# Usage
product = await scrape_product_with_fallback(browser, "https://example.com/product")
if product:
    print(f"Successfully scraped: {product}")
else:
    print("All extraction methods failed")
```

### Best Practices

1. **Always Check `success` Field**: Never assume operations succeeded.

```python
# Good [OK]
result = await browser.act("click button")
if result["success"]:
    # Continue
    pass

# Bad [X] - Don't assume success
result = await browser.act("click button")
data = result["data"]  # Might not exist if failed!
```

2. **Log Errors for Debugging**: Use the error information for troubleshooting.

```python
result = await browser.extract("Get data")
if not result["success"]:
    logger.error(
        f"Extraction failed: {result['error']} "
        f"(Type: {result.get('exception_type')})"
    )
```

3. **Implement Fallback Strategies**: Have alternative approaches when primary methods fail.

```python
# Try vision-based extraction first
result = await browser.extract("Get price", use_vision=True)

if not result["success"]:
    # Fallback to text-based
    result = await browser.extract("Get price", use_vision=False)
```

4. **Aggregate Results**: Continue collecting data even if some operations fail.

```python
products = []
for url in product_urls:
    result = await extract_product(browser, url)
    if result["success"]:
        products.append(result["data"])
    else:
        logger.warning(f"Skipping {url}: {result['error']}")
        # Continue to next product

print(f"Successfully extracted {len(products)}/{len(product_urls)} products")
```

5. **Use Error Info for Retry Logic**: Implement smart retries based on error type.

```python
max_retries = 3
for attempt in range(max_retries):
    result = await browser.act("click button")
    
    if result["success"]:
        break
    
    if "Element not found" in result["error"]:
        # Wait for element to appear
        await asyncio.sleep(1)
    else:
        # Different error, don't retry
        break
```

---

## Tips

1. **Start with INFO**: It's the default for a reason - gives you enough detail without overwhelming you.

2. **Use DEBUG when stuck**: If something isn't working, switch to DEBUG to see exactly what the LLM is doing.

3. **Pretty logs in development**: Human-readable logs are much easier to scan during development.

4. **JSON logs in production**: Structured logs are better for log aggregation tools like ELK, Splunk, etc.

5. **Watch for warnings**: Warnings often indicate issues that aren't fatal but should be addressed.

6. **Check error messages**: FlyBrowser provides detailed error messages to help you understand what went wrong.

---

## Troubleshooting

### "Too many logs!"

```python
# Reduce to WARNING
browser = FlyBrowser(..., log_level="WARNING")
```

### "Not enough detail to debug"

```python
# Increase to DEBUG
browser = FlyBrowser(..., log_level="DEBUG")
```

### "Logs aren't colored in my terminal"

Colors are automatically disabled if:
- Output is not a TTY (e.g., piped to a file)
- Terminal doesn't support colors

Force colors:
```python
from flybrowser.utils.logger import HumanFormatter
import logging

logger = logging.getLogger("flybrowser")
handler = logging.StreamHandler()
handler.setFormatter(HumanFormatter(use_colors=True))
logger.handlers = [handler]
```

### "Want logs in a file"

```python
import logging

logger = logging.getLogger("flybrowser")
file_handler = logging.FileHandler("flybrowser.log")
file_handler.setLevel(logging.DEBUG)

from flybrowser.utils.logger import JsonFormatter
file_handler.setFormatter(JsonFormatter())

logger.addHandler(file_handler)
```

---

For more information, see:
- [Getting Started Guide](getting-started.md)
- [Jupyter Notebooks Guide](jupyter-notebooks.md)
- [SDK Reference](reference/sdk.md)
