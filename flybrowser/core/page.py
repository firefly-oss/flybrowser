# Copyright 2026 Firefly Software Solutions Inc
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
Page controller for browser interactions.

This module provides the PageController class which manages page-level
operations such as navigation, content extraction, screenshots, and
page state management.

The PageController wraps Playwright's Page API with error handling
and logging for production use.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from playwright.async_api import Page

from flybrowser.exceptions import NavigationError, PageError
from flybrowser.utils.logger import logger


class PageController:
    """
    Controls page interactions and state management.

    This class provides high-level methods for common page operations:
    - Navigation with configurable wait conditions
    - Screenshot capture
    - HTML content extraction
    - Page metadata retrieval
    - Scroll operations

    Attributes:
        page: The underlying Playwright Page instance

    Example:
        >>> controller = PageController(page)
        >>> await controller.goto("https://example.com")
        >>> html = await controller.get_html()
        >>> screenshot = await controller.screenshot(full_page=True)
    """

    def __init__(self, page: Page) -> None:
        """
        Initialize the page controller.

        Args:
            page: Playwright Page instance to control

        Example:
            >>> from playwright.async_api import async_playwright
            >>> playwright = await async_playwright().start()
            >>> browser = await playwright.chromium.launch()
            >>> page = await browser.new_page()
            >>> controller = PageController(page)
        """
        self.page = page

    async def goto(self, url: str, wait_until: str = "domcontentloaded", timeout: int = 30000) -> None:
        """
        Navigate to a URL with configurable wait conditions.

        Args:
            url: URL to navigate to (must include protocol, e.g., https://)
            wait_until: When to consider navigation succeeded. Options:
                - "load": Wait for the load event (default)
                - "domcontentloaded": Wait for DOMContentLoaded event
                - "networkidle": Wait for network to be idle (no requests for 500ms)
                - "commit": Wait for navigation to commit
            timeout: Maximum time to wait for navigation in milliseconds.
                Default: 30000 (30 seconds)

        Raises:
            NavigationError: If navigation fails or times out

        Example:
            >>> await controller.goto("https://example.com")
            >>> await controller.goto("https://example.com", wait_until="networkidle")
            >>> await controller.goto("https://example.com", timeout=60000)
        """
        try:
            logger.info(f"Navigating to {url}")
            await self.page.goto(url, wait_until=wait_until, timeout=timeout)
            logger.info(f"Successfully navigated to {url}")
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            raise NavigationError(f"Failed to navigate to {url}: {e}") from e

    async def screenshot(self, full_page: bool = False) -> bytes:
        """
        Take a screenshot of the current page.

        Args:
            full_page: Whether to capture the full scrollable page.
                - True: Captures entire page including content below the fold
                - False: Captures only the visible viewport (default)

        Returns:
            Screenshot as PNG bytes

        Raises:
            PageError: If screenshot capture fails

        Example:
            >>> screenshot_bytes = await controller.screenshot()
            >>> with open("screenshot.png", "wb") as f:
            ...     f.write(screenshot_bytes)

            >>> full_screenshot = await controller.screenshot(full_page=True)
        """
        try:
            return await self.page.screenshot(full_page=full_page, type="png")
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            raise PageError(f"Failed to take screenshot: {e}") from e

    async def get_html(self) -> str:
        """
        Get the complete HTML content of the page.

        Returns:
            Full HTML content as string including DOCTYPE and all elements

        Raises:
            PageError: If HTML extraction fails

        Example:
            >>> html = await controller.get_html()
            >>> print(html[:100])  # Print first 100 characters
        """
        try:
            return await self.page.content()
        except Exception as e:
            logger.error(f"Failed to get HTML: {e}")
            raise PageError(f"Failed to get page HTML: {e}") from e

    async def get_title(self) -> str:
        """
        Get the page title.

        Returns:
            Page title
        """
        try:
            return await self.page.title()
        except Exception as e:
            logger.error(f"Failed to get title: {e}")
            raise PageError(f"Failed to get page title: {e}") from e

    async def get_url(self) -> str:
        """
        Get the current page URL.

        Returns:
            Current URL
        """
        return self.page.url

    async def wait_for_selector(
        self, selector: str, timeout: int = 30000, state: str = "visible"
    ) -> None:
        """
        Wait for a selector to be in a specific state.

        Args:
            selector: CSS selector
            timeout: Timeout in milliseconds
            state: Element state to wait for (visible, hidden, attached, detached)
        """
        try:
            await self.page.wait_for_selector(selector, timeout=timeout, state=state)
        except Exception as e:
            logger.error(f"Wait for selector failed: {e}")
            raise PageError(f"Failed to wait for selector {selector}: {e}") from e

    async def evaluate(self, script: str) -> Any:
        """
        Execute JavaScript in the page context.

        Args:
            script: JavaScript code to execute

        Returns:
            Result of the script execution
        """
        try:
            return await self.page.evaluate(script)
        except Exception as e:
            logger.error(f"Script evaluation failed: {e}")
            raise PageError(f"Failed to evaluate script: {e}") from e

    async def get_page_state(self) -> Dict[str, Any]:
        """
        Get comprehensive page state information.

        Returns:
            Dictionary containing page state
        """
        return {
            "url": await self.get_url(),
            "title": await self.get_title(),
            "viewport": self.page.viewport_size,
        }

