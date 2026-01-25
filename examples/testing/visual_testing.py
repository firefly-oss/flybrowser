"""
Example: Visual Testing

Captures screenshots for visual comparison and verification.
Demonstrates screenshot capture, element verification, and responsive testing.

Prerequisites:
- pip install flybrowser
- export OPENAI_API_KEY="sk-..."
"""

import asyncio
import base64
import os
from pathlib import Path
from datetime import datetime
from flybrowser import FlyBrowser


async def capture_visual_states(url: str, test_name: str):
    """
    Capture screenshots at different viewport sizes and states.
    
    Args:
        url: Page URL to capture
        test_name: Name for the test run
        
    Returns:
        Path to screenshots directory
    """
    output_dir = Path(f"screenshots/{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto(url)
        
        # Capture initial state
        screenshot = await browser.screenshot(full_page=True)
        save_screenshot(screenshot, output_dir / "initial.png")
        print("Captured: initial state")
        
        # Capture after scrolling
        await browser.act("Scroll down to the middle of the page")
        screenshot = await browser.screenshot()
        save_screenshot(screenshot, output_dir / "middle_viewport.png")
        print("Captured: middle viewport")
        
        # Capture footer area
        await browser.act("Scroll to the bottom of the page")
        screenshot = await browser.screenshot()
        save_screenshot(screenshot, output_dir / "footer.png")
        print("Captured: footer")
        
        # Capture with modal open (if applicable)
        try:
            await browser.act("Click a button that opens a modal or popup")
            await asyncio.sleep(1)
            screenshot = await browser.screenshot()
            save_screenshot(screenshot, output_dir / "modal_open.png")
            print("Captured: modal state")
        except:
            print("Skipped: modal state (no modal found)")
        
        print(f"\nScreenshots saved to: {output_dir}")
        return str(output_dir)


def save_screenshot(screenshot_data: dict, path: Path):
    """Save screenshot from base64 data."""
    image_data = base64.b64decode(screenshot_data["data_base64"])
    with open(path, "wb") as f:
        f.write(image_data)


async def verify_visual_elements(url: str, elements: list[str]):
    """
    Verify key visual elements using vision.
    
    Args:
        url: Page URL to check
        elements: List of element descriptions to verify
        
    Returns:
        Dictionary of verification results
    """
    results = []
    
    async with FlyBrowser(
        llm_provider="openai",
        llm_model="gpt-4o",  # Vision model
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto(url)
        
        for element in elements:
            # Use vision to check if element is visible
            check = await browser.extract(
                f"Is the {element} visible and properly displayed on the page? "
                "Describe what you see.",
                use_vision=True
            )
            
            visible = check.success and check.data and "yes" in str(check.data).lower()
            
            results.append({
                "element": element,
                "visible": visible,
                "description": check.data
            })
            
            status = "PASS" if visible else "FAIL"
            print(f"[{status}] {element}")
            if check.data:
                desc = str(check.data)[:100]
                print(f"       {desc}...")
        
        # Summary
        visible_count = sum(1 for r in results if r["visible"])
        print(f"\n=== Visual Check: {visible_count}/{len(results)} elements verified ===")
        
        return results


async def test_responsive_design(url: str):
    """
    Test responsive behavior across viewport sizes.
    
    Args:
        url: Page URL to test
        
    Returns:
        List of viewport test results
    """
    # Common viewport sizes
    viewports = [
        {"name": "Mobile", "width": 375, "height": 667},
        {"name": "Tablet", "width": 768, "height": 1024},
        {"name": "Desktop", "width": 1920, "height": 1080},
    ]
    
    results = []
    
    for viewport in viewports:
        print(f"\n=== Testing {viewport['name']} ({viewport['width']}x{viewport['height']}) ===")
        
        async with FlyBrowser(
            llm_provider="openai",
            llm_model="gpt-4o",  # Vision model
            api_key=os.getenv("OPENAI_API_KEY"),
            headless=True,
        ) as browser:
            await browser.goto(url)
            
            # Set viewport through Playwright
            page = browser.browser_manager.page
            await page.set_viewport_size({
                "width": viewport["width"],
                "height": viewport["height"]
            })
            
            await asyncio.sleep(1)  # Wait for reflow
            
            # Take screenshot
            screenshot = await browser.screenshot()
            
            # Check layout with vision
            layout_check = await browser.extract(
                f"Analyze the current layout. Is it properly adapted for a "
                f"{viewport['name'].lower()} viewport? Check for: "
                f"1) Content not overflowing "
                f"2) Text is readable "
                f"3) Navigation is accessible "
                f"4) Images are properly sized",
                use_vision=True
            )
            
            # Check specific responsive elements
            nav_check = await browser.extract(
                "Is the navigation menu visible? Is it a hamburger menu or full menu?",
                use_vision=True
            )
            
            results.append({
                "viewport": viewport["name"],
                "dimensions": f"{viewport['width']}x{viewport['height']}",
                "layout": layout_check.data,
                "navigation": nav_check.data,
                "screenshot_id": screenshot.get("screenshot_id")
            })
            
            print(f"  Layout: {str(layout_check.data)[:100]}...")
            print(f"  Navigation: {nav_check.data}")
    
    return results


async def compare_before_after(url: str, action: str):
    """
    Compare page before and after an action.
    
    Args:
        url: Starting URL
        action: Action to perform
        
    Returns:
        Before and after screenshot comparison
    """
    output_dir = Path("screenshots/comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto(url)
        
        # Capture before state
        before = await browser.screenshot()
        save_screenshot(before, output_dir / f"before_{timestamp}.png")
        print("Captured: before state")
        
        # Perform action
        print(f"Performing: {action}")
        await browser.act(action)
        await asyncio.sleep(1)
        
        # Capture after state
        after = await browser.screenshot()
        save_screenshot(after, output_dir / f"after_{timestamp}.png")
        print("Captured: after state")
        
        # Analyze differences with vision
        diff_analysis = await browser.extract(
            "What changed on the page compared to before the action? "
            "Describe any visible differences.",
            use_vision=True
        )
        
        print(f"\nChanges detected: {diff_analysis.data}")
        
        return {
            "before_path": str(output_dir / f"before_{timestamp}.png"),
            "after_path": str(output_dir / f"after_{timestamp}.png"),
            "changes": diff_analysis.data
        }


async def verify_color_contrast(url: str):
    """
    Check color contrast for accessibility.
    
    Args:
        url: Page URL to check
        
    Returns:
        Color contrast analysis
    """
    async with FlyBrowser(
        llm_provider="openai",
        llm_model="gpt-4o",  # Vision model
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto(url)
        
        # Analyze color contrast
        analysis = await browser.extract(
            "Analyze the color contrast on this page for accessibility. "
            "Are there any text elements that are hard to read due to "
            "low contrast between text and background? "
            "List any accessibility concerns.",
            use_vision=True
        )
        
        print("Color Contrast Analysis:")
        print(analysis.data)
        
        return analysis.data


async def main():
    """Main entry point for visual testing."""
    print("=" * 60)
    print("Visual Testing Examples")
    print("=" * 60)
    
    # Example 1: Capture visual states
    print("\n--- Capturing Visual States ---")
    await capture_visual_states("https://news.ycombinator.com", "hackernews")
    
    # Example 2: Verify visual elements
    print("\n--- Verifying Visual Elements ---")
    await verify_visual_elements(
        "https://news.ycombinator.com",
        [
            "logo or site title",
            "navigation links",
            "main content area",
            "footer section"
        ]
    )
    
    # Example 3: Test responsive design
    print("\n--- Testing Responsive Design ---")
    await test_responsive_design("https://news.ycombinator.com")


if __name__ == "__main__":
    asyncio.run(main())
