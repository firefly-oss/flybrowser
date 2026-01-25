"""
Example: Scrape Paginated Results

Demonstrates handling pagination to collect data across multiple pages.
Shows different pagination strategies: next button, page numbers, infinite scroll.

Prerequisites:
- pip install flybrowser
- export OPENAI_API_KEY="sk-..."
"""

import asyncio
import os
from flybrowser import FlyBrowser


async def scrape_with_next_button(start_url: str, max_pages: int = 5):
    """
    Scrape paginated results using a "Next" button.
    
    Args:
        start_url: URL of the first page
        max_pages: Maximum number of pages to scrape
        
    Returns:
        List of all collected items
    """
    all_items = []
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
        log_verbosity="minimal",
    ) as browser:
        await browser.goto(start_url)
        
        for page_num in range(1, max_pages + 1):
            print(f"Scraping page {page_num}...")
            
            # Extract items from current page
            result = await browser.extract(
                "Extract all item names and details from this page",
                schema={
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"}
                        }
                    }
                }
            )
            
            if result.success and result.data:
                all_items.extend(result.data)
                print(f"  Found {len(result.data)} items (total: {len(all_items)})")
            else:
                print(f"  No items found or error: {result.error}")
                break
            
            # Click next page button
            if page_num < max_pages:
                nav_result = await browser.act("Click the 'Next' button or next page link")
                if not nav_result.success:
                    print("  No more pages available")
                    break
                
                # Wait for page to load
                await asyncio.sleep(1)
        
        print(f"\nTotal items collected: {len(all_items)}")
        return all_items


async def scrape_with_page_numbers(base_url: str, max_pages: int = 5):
    """
    Scrape by navigating to specific page numbers.
    
    Args:
        base_url: URL template with {page} placeholder
        max_pages: Maximum number of pages to scrape
        
    Returns:
        List of all collected items
    """
    all_items = []
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
        log_verbosity="minimal",
    ) as browser:
        for page_num in range(1, max_pages + 1):
            # Navigate directly to page
            url = base_url.format(page=page_num)
            print(f"Scraping page {page_num}: {url}")
            await browser.goto(url)
            
            # Extract items
            result = await browser.extract(
                "Extract all items with their names and details"
            )
            
            if result.success and result.data:
                items = result.data if isinstance(result.data, list) else [result.data]
                all_items.extend(items)
                print(f"  Found {len(items)} items")
            else:
                print(f"  Failed or empty: {result.error}")
                # Stop if no results (probably no more pages)
                if not result.data:
                    break
        
        print(f"\nTotal items collected: {len(all_items)}")
        return all_items


async def scrape_infinite_scroll(url: str, max_scrolls: int = 10):
    """
    Scrape pages with infinite scroll loading.
    
    Args:
        url: Page URL with infinite scroll
        max_scrolls: Maximum number of scroll actions
        
    Returns:
        List of all unique items collected
    """
    all_items = []
    scroll_count = 0
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
        speed_preset="thorough",  # More patient with dynamic content
    ) as browser:
        await browser.goto(url)
        
        while scroll_count < max_scrolls:
            # Extract currently visible items
            result = await browser.extract(
                "Extract all post titles and authors visible on the page"
            )
            
            if result.success and result.data:
                # Track new items (avoid duplicates by comparing)
                current_items = result.data if isinstance(result.data, list) else [result.data]
                new_count = 0
                for item in current_items:
                    # Simple deduplication
                    if item not in all_items:
                        all_items.append(item)
                        new_count += 1
                
                print(f"Scroll {scroll_count + 1}: Found {new_count} new items (total: {len(all_items)})")
                
                # Check if we've stopped loading new content
                if new_count == 0:
                    print("No new content loaded, stopping")
                    break
            
            # Scroll to load more content
            await browser.act("Scroll down to load more content")
            await asyncio.sleep(2)  # Wait for content to load
            
            scroll_count += 1
        
        print(f"\nTotal unique items: {len(all_items)}")
        return all_items


async def scrape_search_results(search_url: str, query: str, max_pages: int = 3):
    """
    Search and scrape results across multiple pages.
    
    Args:
        search_url: URL of search page
        query: Search query
        max_pages: Maximum pages of results
        
    Returns:
        List of search results
    """
    all_results = []
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto(search_url)
        
        # Perform search
        print(f"Searching for: {query}")
        await browser.act(f"Type '{query}' in the search box and press Enter")
        await asyncio.sleep(2)
        
        for page_num in range(1, max_pages + 1):
            print(f"Extracting page {page_num} results...")
            
            # Extract search results
            result = await browser.extract(
                "Get all search results with title, description, and URL",
                schema={
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "url": {"type": "string"}
                        }
                    }
                }
            )
            
            if result.success and result.data:
                all_results.extend(result.data)
                print(f"  Found {len(result.data)} results")
            else:
                print(f"  No results or error")
                break
            
            # Navigate to next page
            if page_num < max_pages:
                nav = await browser.act("Click the next page button or link")
                if not nav.success:
                    print("  No more pages")
                    break
                await asyncio.sleep(1)
        
        print(f"\nTotal search results: {len(all_results)}")
        return all_results


async def main():
    """Main entry point demonstrating pagination patterns."""
    print("=" * 60)
    print("Pagination Scraping Examples")
    print("=" * 60)
    
    # Example 1: Scrape using page numbers (Hacker News)
    print("\n--- Example 1: Page Number Navigation ---")
    items = await scrape_with_page_numbers(
        "https://news.ycombinator.com/news?p={page}",
        max_pages=2
    )
    
    if items:
        print(f"\nFirst 3 items:")
        for item in items[:3]:
            print(f"  - {item}")


if __name__ == "__main__":
    asyncio.run(main())
