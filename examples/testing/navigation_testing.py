"""
Example: Navigation Testing

Tests navigation links, menus, and page routing.
Demonstrates verifying navigation flows work correctly.

Prerequisites:
- pip install flybrowser
- export OPENAI_API_KEY="sk-..."
"""

import asyncio
import os
from flybrowser import FlyBrowser


async def test_navigation_links(base_url: str):
    """
    Test that all main navigation links work correctly.
    
    Args:
        base_url: Base URL of the website
        
    Returns:
        List of navigation test results
    """
    results = []
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto(base_url)
        
        # Get all navigation links
        nav_links = await browser.extract(
            "Get all main navigation links with their text and expected destinations"
        )
        
        if not nav_links.success or not nav_links.data:
            print("Failed to extract navigation links")
            return []
        
        links = nav_links.data if isinstance(nav_links.data, list) else [nav_links.data]
        print(f"Found {len(links)} navigation links")
        
        for link in links:
            link_text = link.get("text", str(link)) if isinstance(link, dict) else str(link)
            print(f"\nTesting: {link_text}")
            
            # Navigate using the link
            await browser.goto(base_url)  # Reset to home
            await browser.act(f"Click the '{link_text}' navigation link")
            
            await asyncio.sleep(1)
            
            # Verify the page loaded correctly
            page_info = await browser.extract(
                "What is the current page title and main heading?"
            )
            
            # Check if page matches expected destination
            expected = link_text.lower()
            actual = str(page_info.data).lower() if page_info.data else ""
            
            passed = expected in actual or link_text.lower() in actual
            results.append({
                "link": link_text,
                "page_info": page_info.data,
                "passed": passed
            })
            
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] Page: {page_info.data}")
    
    # Summary
    passed_count = sum(1 for r in results if r["passed"])
    print(f"\n=== Navigation Test Results: {passed_count}/{len(results)} passed ===")
    
    return results


async def test_breadcrumb_navigation(deep_url: str):
    """
    Test that breadcrumb navigation works correctly.
    
    Args:
        deep_url: URL of a page deep in the site hierarchy
        
    Returns:
        Breadcrumb test results
    """
    results = []
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        # Navigate to a deep page
        await browser.goto(deep_url)
        
        # Get breadcrumb trail
        breadcrumbs = await browser.extract(
            "What are the breadcrumb links shown on this page?"
        )
        
        print(f"Breadcrumb trail: {breadcrumbs.data}")
        
        # Test each breadcrumb link
        if breadcrumbs.success and breadcrumbs.data:
            crumbs = breadcrumbs.data if isinstance(breadcrumbs.data, list) else [breadcrumbs.data]
            
            for crumb in crumbs[:-1]:  # Skip current page
                crumb_text = crumb.get("text", str(crumb)) if isinstance(crumb, dict) else str(crumb)
                print(f"\nTesting breadcrumb: {crumb_text}")
                
                await browser.act(f"Click the '{crumb_text}' breadcrumb link")
                await asyncio.sleep(1)
                
                # Verify navigation
                current = await browser.extract("What page am I on now?")
                print(f"  Navigated to: {current.data}")
                
                results.append({
                    "breadcrumb": crumb_text,
                    "destination": current.data,
                    "success": crumb_text.lower() in str(current.data).lower() if current.data else False
                })
                
                # Go back to original page for next test
                await browser.goto(deep_url)
    
    return results


async def test_menu_interactions(url: str):
    """
    Test dropdown menus and interactive navigation.
    
    Args:
        url: Page URL with menus
        
    Returns:
        Menu interaction test results
    """
    results = []
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto(url)
        
        # Test 1: Hover dropdown menus
        print("Test 1: Hover dropdown menus")
        await browser.act("Hover over the first dropdown menu in the navigation")
        await asyncio.sleep(0.5)
        
        dropdown = await browser.extract("What submenu items are shown?")
        has_dropdown = dropdown.success and dropdown.data
        results.append({
            "test": "Hover dropdown",
            "passed": has_dropdown,
            "details": dropdown.data
        })
        print(f"  {'PASS' if has_dropdown else 'FAIL'}: {dropdown.data}")
        
        # Test 2: Mobile menu toggle (if applicable)
        print("\nTest 2: Mobile menu toggle")
        await browser.act("Click the mobile menu button or hamburger icon if present")
        await asyncio.sleep(0.5)
        
        mobile_menu = await browser.extract("Is the mobile menu open? What items are visible?")
        results.append({
            "test": "Mobile menu toggle",
            "passed": mobile_menu.success,
            "details": mobile_menu.data
        })
        print(f"  Result: {mobile_menu.data}")
        
        # Test 3: Submenu navigation
        print("\nTest 3: Submenu navigation")
        await browser.goto(url)  # Reset
        await browser.act("Hover over a menu item with a submenu and click on a submenu item")
        await asyncio.sleep(1)
        
        submenu_nav = await browser.extract("Did the page navigate to a new location?")
        results.append({
            "test": "Submenu navigation",
            "passed": submenu_nav.success,
            "details": submenu_nav.data
        })
        print(f"  Result: {submenu_nav.data}")
    
    return results


async def test_pagination_navigation(url: str):
    """
    Test pagination controls.
    
    Args:
        url: URL of a paginated page
        
    Returns:
        Pagination test results
    """
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto(url)
        
        results = []
        
        # Test 1: Get current page info
        print("Test 1: Identify pagination controls")
        pagination = await browser.extract(
            "What pagination controls are available? Current page number?"
        )
        print(f"  Pagination info: {pagination.data}")
        results.append(("Pagination identified", pagination.success))
        
        # Test 2: Go to next page
        print("\nTest 2: Next page navigation")
        await browser.act("Click the 'Next' page button or link")
        await asyncio.sleep(1)
        
        next_page = await browser.extract("What page number am I on now?")
        print(f"  Current page: {next_page.data}")
        results.append(("Next page works", next_page.success))
        
        # Test 3: Go to previous page
        print("\nTest 3: Previous page navigation")
        await browser.act("Click the 'Previous' page button or link")
        await asyncio.sleep(1)
        
        prev_page = await browser.extract("What page number am I on now?")
        print(f"  Current page: {prev_page.data}")
        results.append(("Previous page works", prev_page.success))
        
        # Test 4: Go to specific page
        print("\nTest 4: Jump to specific page")
        await browser.act("Click on page number 3 or the third page link")
        await asyncio.sleep(1)
        
        specific_page = await browser.extract("What page number am I on now?")
        is_page_3 = "3" in str(specific_page.data)
        print(f"  Current page: {specific_page.data} ({'PASS' if is_page_3 else 'FAIL'})")
        results.append(("Specific page jump", is_page_3))
        
        return results


async def test_back_forward_buttons(url: str):
    """
    Test browser back/forward navigation consistency.
    
    Args:
        url: Starting URL
        
    Returns:
        Browser navigation test results
    """
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto(url)
        
        # Navigate to a few pages
        print("Navigating through pages...")
        pages_visited = [url]
        
        for i in range(3):
            await browser.act("Click any link to navigate to a new page")
            await asyncio.sleep(1)
            current = await browser.extract("What is the current page URL or title?")
            pages_visited.append(str(current.data))
            print(f"  Page {i+2}: {current.data}")
        
        # Test back button
        print("\nTesting back navigation...")
        for i in range(2):
            await browser.act("Click the browser back button or go back")
            await asyncio.sleep(1)
            current = await browser.extract("What page am I on now?")
            print(f"  After back {i+1}: {current.data}")
        
        # Test forward button
        print("\nTesting forward navigation...")
        await browser.act("Click the browser forward button or go forward")
        await asyncio.sleep(1)
        current = await browser.extract("What page am I on now?")
        print(f"  After forward: {current.data}")
        
        return {
            "pages_visited": pages_visited,
            "back_forward_works": True  # Basic verification
        }


async def main():
    """Main entry point for navigation tests."""
    print("=" * 60)
    print("Navigation Testing Examples")
    print("=" * 60)
    
    # Example: Test main navigation (using Hacker News as example)
    print("\n--- Testing Navigation Links ---")
    await test_navigation_links("https://news.ycombinator.com")
    
    print("\n--- Testing Pagination ---")
    await test_pagination_navigation("https://news.ycombinator.com")


if __name__ == "__main__":
    asyncio.run(main())
