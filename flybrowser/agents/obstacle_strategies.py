# Copyright 2026 Firefly Software Solutions Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Obstacle dismissal strategies for world-class obstacle handler.

Each strategy is an independent async function that attempts to dismiss obstacles
using different techniques. Strategies are executed in priority order determined by AI.
"""

import asyncio
import logging
from typing import Optional
from playwright.async_api import Page, TimeoutError as PlaywrightTimeout

logger = logging.getLogger(__name__)


async def try_css_selector(page: Page, selector: str, timeout: float = 2.0) -> bool:
    """
    Try to click element using CSS selector.
    
    Args:
        page: Playwright page
        selector: CSS selector (e.g., ".accept-btn", "#cookie-accept")
        timeout: Max wait time in seconds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.debug(f"[Strategy:CSS] Attempting selector: {selector}")
        element = await page.wait_for_selector(selector, state="visible", timeout=timeout * 1000)
        if element:
            await element.click()
            await page.wait_for_timeout(800)  # Wait for dismissal animation
            logger.info(f" [Strategy:CSS] Successfully clicked: {selector}")
            return True
    except (PlaywrightTimeout, Exception) as e:
        logger.debug(f"[Strategy:CSS] Failed: {e}")
    return False


async def try_xpath(page: Page, xpath: str, timeout: float = 2.0) -> bool:
    """
    Try to click element using XPath expression.
    
    Args:
        page: Playwright page
        xpath: XPath expression (e.g., "//button[contains(text(), 'Accept')]")
        timeout: Max wait time in seconds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.debug(f"[Strategy:XPath] Attempting xpath: {xpath}")
        element = await page.wait_for_selector(f"xpath={xpath}", state="visible", timeout=timeout * 1000)
        if element:
            await element.click()
            await page.wait_for_timeout(500)
            logger.info(f" [Strategy:XPath] Successfully clicked: {xpath}")
            return True
    except (PlaywrightTimeout, Exception) as e:
        logger.debug(f"[Strategy:XPath] Failed: {e}")
    return False


async def try_text_contains(page: Page, text: str, timeout: float = 2.0) -> bool:
    """
    Try to click element containing specific text (language-agnostic).
    
    Uses multiple Playwright locator strategies for maximum reliability.
    
    Args:
        page: Playwright page
        text: Text to search for (e.g., "Accept", "Aceptar", "Accepter")
        timeout: Max wait time in seconds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.debug(f"[Strategy:Text] Attempting to find text: '{text}'")
        
        # Method 1: Use Playwright's get_by_role with name (MOST RELIABLE for buttons)
        try:
            button = page.get_by_role("button", name=text)
            if await button.count() > 0:
                logger.debug(f"[Strategy:Text] Found button with role API: '{text}'")
                await button.first.click(timeout=timeout * 1000)
                await page.wait_for_timeout(1500)
                logger.info(f" [Strategy:Text] Successfully clicked button: '{text}'")
                return True
        except Exception as e:
            logger.debug(f"[Strategy:Text] get_by_role failed: {e}")
        
        # Method 2: Use Playwright's get_by_text (exact match first, then partial)
        try:
            # Try exact match first
            element = page.get_by_text(text, exact=True)
            if await element.count() > 0:
                logger.debug(f"[Strategy:Text] Found exact text match: '{text}'")
                await element.first.click(timeout=timeout * 1000)
                await page.wait_for_timeout(1500)
                logger.info(f" [Strategy:Text] Successfully clicked exact text: '{text}'")
                return True
        except Exception as e:
            logger.debug(f"[Strategy:Text] get_by_text exact failed: {e}")
        
        try:
            # Try partial match
            element = page.get_by_text(text, exact=False)
            if await element.count() > 0:
                # Get the first visible one
                for i in range(min(3, await element.count())):
                    el = element.nth(i)
                    if await el.is_visible():
                        logger.debug(f"[Strategy:Text] Found partial text match: '{text}'")
                        await el.click(timeout=timeout * 1000)
                        await page.wait_for_timeout(1500)
                        logger.info(f" [Strategy:Text] Successfully clicked partial text: '{text}'")
                        return True
        except Exception as e:
            logger.debug(f"[Strategy:Text] get_by_text partial failed: {e}")
        
        # Method 3: Traditional selector-based approach (fallback)
        selectors = [
            f"button:has-text('{text}')",
            f"div[role='button']:has-text('{text}')",
            f"a:has-text('{text}')",
            f"span:has-text('{text}')",
        ]
        
        for selector in selectors:
            try:
                element = await page.wait_for_selector(selector, state="visible", timeout=timeout * 1000)
                if element:
                    await element.click()
                    await page.wait_for_timeout(1500)
                    logger.info(f" [Strategy:Text] Successfully clicked via selector: {selector}")
                    return True
            except PlaywrightTimeout:
                continue
            except Exception as e:
                logger.debug(f"[Strategy:Text] Selector '{selector}' failed: {e}")
                continue
                
    except Exception as e:
        logger.debug(f"[Strategy:Text] Failed: {e}")
    return False


async def try_coordinates(page: Page, x: int, y: int, timeout: float = 2.0) -> bool:
    """
    Try to click at specific screen coordinates (useful for VLM-detected buttons).
    
    This strategy clicks at coordinates but DOES NOT verify success.
    Verification is handled by the ObstacleDetector using VLM analysis.
    
    Args:
        page: Playwright page
        x: X coordinate
        y: Y coordinate
        timeout: Max wait time in seconds
        
    Returns:
        True if click was performed, False if click failed
    """
    try:
        logger.debug(f"[Strategy:Coords] Attempting click at ({x}, {y})")
        
        # Get element at coordinates to log what we're clicking
        element_info = await page.evaluate(f"""
            () => {{
                const element = document.elementFromPoint({x}, {y});
                if (!element) return null;
                return {{
                    tagName: element.tagName,
                    text: (element.innerText || element.textContent || '').slice(0, 100).trim(),
                    isButton: element.tagName === 'BUTTON' || element.getAttribute('role') === 'button',
                    isClickable: element.tagName === 'BUTTON' || element.tagName === 'A' || 
                                 element.getAttribute('role') === 'button' || 
                                 window.getComputedStyle(element).cursor === 'pointer'
                }};
            }}
        """)
        
        if element_info:
            element_text = element_info.get('text', '')[:50]
            is_clickable = element_info.get('isClickable', False)
            logger.debug(f"[Strategy:Coords] Element at ({x}, {y}): {element_info.get('tagName')} '{element_text}' (clickable: {is_clickable})")
            
            # Warn if we're not clicking something that looks clickable
            if not is_clickable:
                logger.warning(f"[Strategy:Coords] Element at ({x}, {y}) may not be clickable")
        else:
            logger.warning(f"[Strategy:Coords] No element found at ({x}, {y}) - coordinates may be wrong")
            return False  # Don't even try if no element there
        
        # Perform the click
        try:
            await page.mouse.click(x, y)
            logger.debug(f"[Strategy:Coords] Mouse click performed at ({x}, {y})")
        except Exception as e:
            logger.warning(f"[Strategy:Coords] Mouse click failed: {e}, trying JS click...")
            try:
                clicked = await page.evaluate(f"""
                    () => {{
                        const element = document.elementFromPoint({x}, {y});
                        if (element) {{
                            element.click();
                            return true;
                        }}
                        return false;
                    }}
                """)
                if not clicked:
                    logger.warning(f"[Strategy:Coords] JS click failed - no element at coordinates")
                    return False
            except Exception as e2:
                logger.warning(f"[Strategy:Coords] All click methods failed: {e2}")
                return False
        
        # Wait for any animations/transitions
        await page.wait_for_timeout(1500)
        
        # We performed the click - actual verification will be done by ObstacleDetector
        # using VLM to check if the obstacle is gone (no hardcoded selectors!)
        logger.info(f"[Strategy:Coords] Click performed at ({x}, {y}) on '{element_info.get('text', '')[:30]}' - verification pending")
        return True  # Click was performed, verification happens at higher level
            
    except Exception as e:
        logger.debug(f"[Strategy:Coords] Failed: {e}")
    return False


async def try_escape_key(page: Page, timeout: float = 2.0) -> bool:
    """
    Try pressing Escape key to dismiss modal/overlay.
    
    Args:
        page: Playwright page
        timeout: Max wait time in seconds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.debug("[Strategy:Escape] Attempting Escape key")
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(500)
        logger.info(" [Strategy:Escape] Pressed Escape key")
        return True
    except Exception as e:
        logger.debug(f"[Strategy:Escape] Failed: {e}")
    return False


async def try_javascript_removal(page: Page, selector: str, timeout: float = 2.0) -> bool:
    """
    Try to forcefully remove obstacle element using JavaScript.
    
    Args:
        page: Playwright page
        selector: CSS selector of element to remove
        timeout: Max wait time in seconds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.debug(f"[Strategy:JS] Attempting JS removal: {selector}")
        result = await page.evaluate(f"""
            () => {{
                const elements = document.querySelectorAll('{selector}');
                if (elements.length > 0) {{
                    elements.forEach(el => el.remove());
                    // Also remove any backdrop/overlay
                    const backdrops = document.querySelectorAll('.modal-backdrop, .overlay, [class*="backdrop"]');
                    backdrops.forEach(el => el.remove());
                    // Re-enable body scroll
                    document.body.style.overflow = '';
                    return true;
                }}
                return false;
            }}
        """)
        if result:
            await page.wait_for_timeout(500)
            logger.info(f" [Strategy:JS] Successfully removed: {selector}")
            return True
    except Exception as e:
        logger.debug(f"[Strategy:JS] Failed: {e}")
    return False


async def try_tab_navigation(page: Page, max_tabs: int = 10, timeout: float = 2.0) -> bool:
    """
    Try to use Tab key navigation to find and activate dismiss button.
    
    Args:
        page: Playwright page
        max_tabs: Maximum number of Tab presses to try
        timeout: Max wait time in seconds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.debug(f"[Strategy:Tab] Attempting tab navigation (max {max_tabs} tabs)")
        for i in range(max_tabs):
            await page.keyboard.press("Tab")
            await page.wait_for_timeout(100)
            
            # Check if focused element looks like a dismiss button
            is_dismiss_button = await page.evaluate("""
                () => {
                    const focused = document.activeElement;
                    if (!focused) return false;
                    
                    const text = focused.textContent?.toLowerCase() || '';
                    const dismissKeywords = ['accept', 'agree', 'ok', 'close', 'dismiss', 
                                           'aceptar', 'accepter', 'akzeptieren', 'schließen'];
                    
                    return dismissKeywords.some(kw => text.includes(kw));
                }
            """)
            
            if is_dismiss_button:
                await page.keyboard.press("Enter")
                await page.wait_for_timeout(500)
                logger.info(f" [Strategy:Tab] Found and activated dismiss button after {i+1} tabs")
                return True
                
    except Exception as e:
        logger.debug(f"[Strategy:Tab] Failed: {e}")
    return False


async def try_event_simulation(page: Page, selector: str, timeout: float = 2.0) -> bool:
    """
    Try to trigger click event programmatically (bypasses visibility checks).
    
    Args:
        page: Playwright page
        selector: CSS selector of element to click
        timeout: Max wait time in seconds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.debug(f"[Strategy:Event] Attempting event simulation: {selector}")
        result = await page.evaluate(f"""
            () => {{
                const element = document.querySelector('{selector}');
                if (element) {{
                    // Try multiple event types
                    element.click();
                    element.dispatchEvent(new MouseEvent('click', {{bubbles: true}}));
                    element.dispatchEvent(new MouseEvent('mousedown', {{bubbles: true}}));
                    element.dispatchEvent(new MouseEvent('mouseup', {{bubbles: true}}));
                    return true;
                }}
                return false;
            }}
        """)
        if result:
            await page.wait_for_timeout(500)
            logger.info(f" [Strategy:Event] Successfully simulated click: {selector}")
            return True
    except Exception as e:
        logger.debug(f"[Strategy:Event] Failed: {e}")
    return False


# Strategy registry for easy lookup
STRATEGIES = {
    "css": try_css_selector,
    "xpath": try_xpath,
    "text": try_text_contains,
    "coordinates": try_coordinates,
    "escape": try_escape_key,
    "javascript": try_javascript_removal,
    "tab": try_tab_navigation,
    "event": try_event_simulation,
}
