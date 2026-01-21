#!/usr/bin/env python3
# Copyright 2026 Firefly Software Solutions Inc
# Licensed under the Apache License, Version 2.0
"""
FlyBrowser Jupyter Notebook Example

This example demonstrates how to use FlyBrowser in Jupyter notebooks.
Jupyter notebooks have a running event loop, which requires special handling.

This file can be run as a Python script or converted to a Jupyter notebook:
  jupyter nbconvert --to notebook --execute examples/11_jupyter_notebook.py

When using in Jupyter notebooks, you can copy cells from this file.

## Two Approaches:

### Approach 1: Direct await (Recommended)
Use top-level await in Jupyter cells without asyncio.run()

### Approach 2: nest_asyncio
Use nest_asyncio to allow asyncio.run() in notebooks
"""

import os

# ==============================================================================
# JUPYTER CELL 1: Setup and Imports
# ==============================================================================
print("=" * 70)
print("FlyBrowser Jupyter Notebook Example")
print("=" * 70)

from flybrowser import FlyBrowser

# ==============================================================================
# JUPYTER CELL 2: Approach 1 - Direct await (RECOMMENDED)
# ==============================================================================
print("\n[Approach 1] Using 'await' directly (recommended for Jupyter)")
print("-" * 70)

# In Jupyter, you can use 'await' directly at the top level
# This is the recommended approach as it's cleaner and more natural

# Create browser instance
browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="llama2",
    headless=True,
    pretty_logs=True,  # Human-readable logs (default)
    log_level="INFO"   # Change to DEBUG for more details
)

# Start the browser (in Jupyter, just use await directly)
await browser.start()

print("✓ Browser started successfully")

# ==============================================================================
# JUPYTER CELL 3: Navigate to a website
# ==============================================================================
print("\n[Step 1] Navigating to example.com...")

await browser.goto("https://example.com")

print("✓ Navigation complete")

# ==============================================================================
# JUPYTER CELL 4: Extract data
# ==============================================================================
print("\n[Step 2] Extracting page information...")

data = await browser.extract("What is the main heading on this page?")

print(f"✓ Extracted data: {data}")

# ==============================================================================
# JUPYTER CELL 5: Take a screenshot
# ==============================================================================
print("\n[Step 3] Taking a screenshot...")

screenshot = await browser.screenshot(full_page=True)

print(f"✓ Screenshot captured: {screenshot['width']}x{screenshot['height']} pixels")

# You can save the screenshot to a file
import base64
with open("jupyter_screenshot.png", "wb") as f:
    f.write(base64.b64decode(screenshot['data_base64']))

print("✓ Screenshot saved to jupyter_screenshot.png")

# ==============================================================================
# JUPYTER CELL 6: Perform an action
# ==============================================================================
print("\n[Step 4] Performing an action...")

try:
    result = await browser.act("click the More information link")
    print(f"✓ Action result: {result}")
except Exception as e:
    print(f"⚠ Action skipped: {e}")

# ==============================================================================
# JUPYTER CELL 7: Clean up
# ==============================================================================
print("\n[Step 5] Stopping browser...")

await browser.stop()

print("✓ Browser stopped successfully")

# ==============================================================================
# JUPYTER CELL 8: Approach 2 - Using nest_asyncio (Alternative)
# ==============================================================================
print("\n" + "=" * 70)
print("[Approach 2] Using asyncio.run() with nest_asyncio")
print("-" * 70)

# This approach allows you to use asyncio.run() in Jupyter
# Install with: pip install flybrowser[jupyter]

from flybrowser.utils.jupyter import setup_jupyter
import asyncio

# Apply nest_asyncio patch
setup_jupyter()

print("✓ nest_asyncio configured")


# Define an async function
async def main_with_asyncio_run():
    """Example using asyncio.run() approach."""
    print("\nRunning with asyncio.run()...")
    
    async with FlyBrowser(
        llm_provider="ollama",
        llm_model="llama2",
        headless=True
    ) as browser:
        print("✓ Browser started")
        
        await browser.goto("https://example.com")
        print("✓ Navigated to example.com")
        
        data = await browser.extract("What is the main heading?")
        print(f"✓ Extracted: {data}")
        
        print("✓ Browser will stop automatically")
    
    return "Complete!"


# Now you can use asyncio.run() in Jupyter
result = asyncio.run(main_with_asyncio_run())
print(f"Result: {result}")

# ==============================================================================
# JUPYTER CELL 9: Summary and Tips
# ==============================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print("""
✓ Successfully demonstrated both approaches for using FlyBrowser in Jupyter!

Key Takeaways:
1. Approach 1 (Direct await) is recommended:
   - Simpler and more natural in Jupyter
   - Just use 'await' directly in cells
   - No additional setup required

2. Approach 2 (nest_asyncio) is for compatibility:
   - Allows using asyncio.run() in notebooks
   - Requires: pip install flybrowser[jupyter]
   - Good for porting existing scripts

Tips:
- Remember to call browser.start() and browser.stop()
- Or use 'async with' context manager for automatic cleanup
- Set headless=False to see the browser in action
- Check the full documentation at docs/jupyter-notebooks.md

Next Steps:
- Try modifying the examples above
- Explore other FlyBrowser features (workflows, monitoring, etc.)
- Check out the other examples in the examples/ directory
""")

print("=" * 70)
