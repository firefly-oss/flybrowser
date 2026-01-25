"""
Example: Hacker News Scraper

A complete scraper for Hacker News front page stories.
Demonstrates structured data extraction with JSON schema.

Prerequisites:
- pip install flybrowser
- export OPENAI_API_KEY="sk-..."
"""

import asyncio
import json
import os
from datetime import datetime
from flybrowser import FlyBrowser

# Schema for structured extraction
STORY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "rank": {"type": "integer"},
            "title": {"type": "string"},
            "url": {"type": "string"},
            "points": {"type": "integer"},
            "author": {"type": "string"},
            "comments": {"type": "integer"},
            "time_ago": {"type": "string"}
        },
        "required": ["title"]
    }
}


async def scrape_hackernews(num_pages: int = 1):
    """
    Scrape Hacker News stories.
    
    Args:
        num_pages: Number of pages to scrape (default: 1)
    
    Returns:
        List of story dictionaries
    """
    all_stories = []
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
        log_verbosity="normal",
    ) as browser:
        for page in range(1, num_pages + 1):
            # Navigate to page
            url = f"https://news.ycombinator.com/news?p={page}"
            print(f"Scraping page {page}: {url}")
            await browser.goto(url)
            
            # Extract stories with schema
            result = await browser.extract(
                "Extract all stories with their rank, title, URL, points, "
                "author, number of comments, and time posted",
                schema=STORY_SCHEMA,
                max_iterations=20
            )
            
            if result.success and result.data:
                # Add page info to each story
                for story in result.data:
                    story['page'] = page
                
                all_stories.extend(result.data)
                print(f"  Extracted {len(result.data)} stories")
            else:
                print(f"  Failed: {result.error}")
                break
        
        # Save to file
        output = {
            "scraped_at": datetime.now().isoformat(),
            "total_stories": len(all_stories),
            "stories": all_stories
        }
        
        output_file = f"hackernews_stories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\nSaved {len(all_stories)} stories to {output_file}")
        
        # Show usage stats
        usage = browser.get_usage_summary()
        print(f"Total tokens: {usage['total_tokens']:,}")
        print(f"Estimated cost: ${usage['cost_usd']:.4f}")
        
        return all_stories


async def main():
    """Main entry point."""
    print("=" * 60)
    print("Hacker News Scraper")
    print("=" * 60)
    
    stories = await scrape_hackernews(num_pages=2)
    
    # Display sample results
    if stories:
        print("\nTop 5 Stories:")
        print("-" * 60)
        for story in stories[:5]:
            title = story.get('title', 'Unknown')[:50]
            points = story.get('points', 0)
            print(f"  {story.get('rank', '?'):>3}. {title}... ({points} points)")


if __name__ == "__main__":
    asyncio.run(main())
