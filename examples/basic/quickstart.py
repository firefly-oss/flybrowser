"""
Example: Your First Automation

The simplest FlyBrowser example - navigate, interact, and extract data.
This matches the "Your First Automation" example from the README.

Prerequisites:
- pip install flybrowser
- export OPENAI_API_KEY="sk-..."
"""

import asyncio
import os
from flybrowser import FlyBrowser


async def main():
    """Run the quickstart example."""
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
    ) as browser:
        # Navigate and interact naturally
        await browser.goto("https://news.ycombinator.com")
        
        # Extract structured data
        posts = await browser.extract("Get the top 5 post titles and scores")
        
        # Record or stream your session
        await browser.start_recording()
        await browser.act("scroll down slowly")
        recording = await browser.stop_recording()
        
        print(f"Extracted: {posts['data']}")
        print(f"Recording: {recording['recording_id']}")


if __name__ == "__main__":
    asyncio.run(main())
