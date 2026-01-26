# Quickstart Guide

This guide walks you through your first FlyBrowser automation in about five minutes.

## Prerequisites

Before starting, ensure you have:

1. FlyBrowser installed (see [Installation](installation.md))
2. An API key from an LLM provider (OpenAI, Anthropic, or Google)
3. Python 3.9 or later

## Your First Script

Create a file called `first_automation.py`:

```python
import asyncio
from flybrowser import FlyBrowser

async def main():
    # Create a browser instance with your LLM provider
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",  # Replace with your actual API key
        headless=False,    # Set to True to run without visible browser
    ) as browser:
        
        # Navigate to a website
        await browser.goto("https://news.ycombinator.com")
        print("Navigated to Hacker News")
        
        # Extract data using natural language
        result = await browser.extract("Get the title of the top story")
        
        if result.success:
            print(f"Top story: {result.data}")
        else:
            print(f"Extraction failed: {result.error}")

# Run the async function
asyncio.run(main())
```

Run it:

```bash
python first_automation.py
```

You should see a browser window open, navigate to Hacker News, and the script will print the title of the top story.

## Understanding the Code

Let us break down what happens in this script:

### Creating a Browser Instance

```python
async with FlyBrowser(
    llm_provider="openai",
    api_key="sk-...",
    headless=False,
) as browser:
```

The `async with` statement creates a FlyBrowser instance and ensures it is properly closed when done. Key parameters:

- `llm_provider` - Which LLM service to use ("openai", "anthropic", "gemini", or "ollama")
- `api_key` - Your API key for the LLM provider
- `headless` - Whether to show the browser window (False) or run invisibly (True)

### Navigating to a Page

```python
await browser.goto("https://news.ycombinator.com")
```

The `goto()` method navigates directly to a URL. It waits for the page to load before returning.

### Extracting Data

```python
result = await browser.extract("Get the title of the top story")
```

The `extract()` method uses the LLM to understand your query and extract the relevant data from the page. It returns an `AgentRequestResponse` object with:

- `result.success` - Boolean indicating if extraction succeeded
- `result.data` - The extracted data
- `result.error` - Error message if something went wrong

## Performing Actions

FlyBrowser can interact with pages using natural language:

```python
import asyncio
from flybrowser import FlyBrowser

async def main():
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        headless=False,
    ) as browser:
        
        await browser.goto("https://www.google.com")
        
        # Type in the search box
        await browser.act("type 'python web automation' in the search box")
        
        # Click the search button
        await browser.act("click the Google Search button")
        
        # Wait a moment for results to load
        await asyncio.sleep(2)
        
        # Extract search results
        result = await browser.extract(
            "Get the titles of the first 5 search results"
        )
        
        if result.success:
            for i, title in enumerate(result.data, 1):
                print(f"{i}. {title}")

asyncio.run(main())
```

The `act()` method interprets your instruction and performs the appropriate browser action - clicking, typing, scrolling, hovering, and more.

## Extracting Structured Data

For more complex extractions, provide a JSON Schema to get structured data:

```python
import asyncio
from flybrowser import FlyBrowser

async def main():
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        
        await browser.goto("https://news.ycombinator.com")
        
        # Extract structured data with a schema
        result = await browser.extract(
            "Get the top 5 stories with their titles and scores",
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "score": {"type": "integer"},
                        "comments": {"type": "integer"}
                    },
                    "required": ["title", "score"]
                }
            }
        )
        
        if result.success:
            for story in result.data:
                print(f"{story['title']} - {story['score']} points")
            
            # Show detailed execution info
            result.pprint()

asyncio.run(main())
```

The schema ensures the extracted data matches the expected structure and types.

## Using the Autonomous Agent

For complex, multi-step tasks, use the `agent()` method:

```python
import asyncio
from flybrowser import FlyBrowser

async def main():
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        headless=False,
    ) as browser:
        
        # Let the agent figure out the steps
        result = await browser.agent(
            task="Go to Hacker News, find the top story, click on it, "
                 "and extract the main content of the article",
            max_iterations=20,
        )
        
        if result.success:
            print("Task completed!")
            print(f"Result: {result.data}")
        else:
            print(f"Task failed: {result.error}")
        
        # See what the agent did
        result.pprint()

asyncio.run(main())
```

The agent uses ReAct (Reasoning and Acting) to plan and execute multi-step workflows, adapting to obstacles along the way.

## Taking Screenshots

Capture the current page state:

```python
import asyncio
from flybrowser import FlyBrowser
import base64

async def main():
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        
        await browser.goto("https://example.com")
        
        # Take a screenshot
        screenshot = await browser.screenshot(full_page=True)
        
        # Save to file
        image_data = base64.b64decode(screenshot["data_base64"])
        with open("screenshot.png", "wb") as f:
            f.write(image_data)
        
        print(f"Screenshot saved: {screenshot['width']}x{screenshot['height']}")

asyncio.run(main())
```

## Running in Jupyter Notebooks

FlyBrowser works in Jupyter notebooks. First, install notebook support:

```bash
pip install flybrowser[jupyter]
```

Then in a notebook cell:

```python
from flybrowser import FlyBrowser

# In Jupyter, you can use await directly at the top level
browser = FlyBrowser(llm_provider="openai", api_key="sk-...")
await browser.start()

await browser.goto("https://example.com")
result = await browser.extract("Get the page title")
print(result.data)

await browser.stop()
```

## Using Environment Variables

Instead of hardcoding API keys, use environment variables:

```bash
export OPENAI_API_KEY="sk-..."
```

Then in Python:

```python
import os
from flybrowser import FlyBrowser

browser = FlyBrowser(
    llm_provider="openai",
    api_key=os.environ.get("OPENAI_API_KEY"),
)
```

Or let FlyBrowser read the key automatically (it checks standard environment variables):

```python
from flybrowser import FlyBrowser

# Will automatically use OPENAI_API_KEY from environment
browser = FlyBrowser(llm_provider="openai")
```

## Configuration Options

Common configuration parameters:

```python
browser = FlyBrowser(
    # LLM Configuration
    llm_provider="openai",           # openai, anthropic, gemini, ollama
    llm_model="gpt-4o",              # Specific model (optional)
    api_key="sk-...",                # API key
    
    # Browser Configuration
    headless=True,                   # Run without visible window
    browser_type="chromium",         # chromium, firefox, webkit
    
    # Performance
    speed_preset="balanced",         # fast, balanced, thorough
    timeout=30.0,                    # Request timeout in seconds
    
    # Features
    recording_enabled=False,         # Enable session recording
    pii_masking_enabled=True,        # Mask sensitive data
    
    # Logging
    log_verbosity="normal",          # silent, minimal, normal, verbose, debug
    pretty_logs=True,                # Human-readable logs
)
```

## Using Context for Structured Data

For complex forms, file uploads, and filtered extraction, use the context system:

### Form Filling with Context

```python
import asyncio
from flybrowser import FlyBrowser
from flybrowser.agents.context import ContextBuilder

async def main():
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
        headless=False,
    ) as browser:
        
        await browser.goto("https://example.com/login")
        
        # Create context with form data
        context = ContextBuilder()\
            .with_form_data({
                "input[name=email]": "user@example.com",
                "input[name=password]": "secure_password",
                "input[type=checkbox]": True
            })\
            .build()
        
        # Fill and submit the form with context
        await browser.act("Fill and submit the login form", context=context)

asyncio.run(main())
```

### File Upload with Context

```python
import asyncio
from flybrowser import FlyBrowser
from flybrowser.agents.context import ContextBuilder

async def main():
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        
        await browser.goto("https://example.com/upload")
        
        # Create context for file upload
        context = ContextBuilder()\
            .with_file("resume", "/path/to/resume.pdf", "application/pdf")\
            .with_file("cover_letter", "/path/to/cover.docx")\
            .build()
        
        await browser.act("Upload the documents", context=context)

asyncio.run(main())
```

### Filtered Extraction with Context

```python
import asyncio
from flybrowser import FlyBrowser
from flybrowser.agents.context import ContextBuilder

async def main():
    async with FlyBrowser(
        llm_provider="openai",
        api_key="sk-...",
    ) as browser:
        
        await browser.goto("https://shop.example.com/products")
        
        # Create context with filters and preferences
        context = ContextBuilder()\
            .with_filters({"price_max": 100, "category": "electronics"})\
            .with_preferences({"sort_by": "price", "limit": 10})\
            .build()
        
        # Extract with context
        result = await browser.extract(
            "Get product listings",
            context=context,
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
        
        if result.success:
            for product in result.data:
                print(f"{product['name']}: ${product['price']}")

asyncio.run(main())
```

The context system provides type-safe, structured data passing for:
- **form_data**: Automated form filling
- **files**: File uploads with metadata
- **filters**: Data filtering criteria
- **preferences**: User preferences
- **conditions**: Conditional logic
- **constraints**: General constraints
- **metadata**: Tool-specific metadata

See the [Context Usage Guide](../guides/context-usage.md) for more examples.

## Common Patterns

### Retry on Failure

```python
async def extract_with_retry(browser, query, max_retries=3):
    for attempt in range(max_retries):
        result = await browser.extract(query)
        if result.success:
            return result
        print(f"Attempt {attempt + 1} failed, retrying...")
    return result  # Return last failed result
```

### Waiting for Elements

```python
# The act() method automatically waits for elements
# But you can add explicit waits if needed
import asyncio

await browser.goto("https://example.com")
await asyncio.sleep(2)  # Wait for dynamic content
await browser.act("click the button that just appeared")
```

### Error Handling

```python
try:
    result = await browser.extract("Get the price")
    if not result.success:
        print(f"Extraction returned error: {result.error}")
except Exception as e:
    print(f"Exception occurred: {e}")
```

## Next Steps

Now that you have the basics, explore:

- [Core Concepts](concepts.md) - Understand how FlyBrowser works
- [Basic Automation Guide](../guides/basic-automation.md) - More detailed action examples
- [Data Extraction Guide](../guides/data-extraction.md) - Advanced extraction techniques
- [SDK Reference](../reference/sdk.md) - Complete API documentation
