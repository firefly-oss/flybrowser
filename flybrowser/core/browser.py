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
Browser management for FlyBrowser.

This module provides the BrowserManager class which handles the lifecycle
of Playwright browser instances. It manages browser launching, context creation,
page management, and cleanup.

The BrowserManager supports multiple browser types (Chromium, Firefox, WebKit)
and provides a clean interface for browser operations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from flybrowser.exceptions import BrowserError
from flybrowser.utils.logger import logger


class BrowserManager:
    """
    Manages Playwright browser instances and their lifecycle.

    This class handles:
    - Browser launching with configurable options
    - Browser context creation
    - Page management
    - Resource cleanup

    Attributes:
        headless: Whether browser runs in headless mode (no visible window)
        browser_type: Type of browser (chromium, firefox, webkit)
        launch_options: Additional Playwright launch options
        page: The current active page instance

    Example:
        >>> manager = BrowserManager(headless=True, browser_type="chromium")
        >>> await manager.start()
        >>> page = manager.page
        >>> await page.goto("https://example.com")
        >>> await manager.stop()
    """

    def __init__(
        self,
        headless: bool = True,
        browser_type: str = "chromium",
        **launch_options: Any,
    ) -> None:
        """
        Initialize the browser manager with configuration.

        Args:
            headless: Whether to run browser in headless mode (no visible window).
                Default: True
            browser_type: Browser type to use. Supported values:
                - "chromium": Chromium-based browser (default)
                - "firefox": Mozilla Firefox
                - "webkit": WebKit (Safari engine)
            **launch_options: Additional Playwright launch options such as:
                - args: List of command-line arguments
                - downloads_path: Path for downloads
                - proxy: Proxy configuration
                - slow_mo: Slow down operations by specified milliseconds

        Example:
            >>> manager = BrowserManager(
            ...     headless=True,
            ...     browser_type="chromium",
            ...     args=["--disable-blink-features=AutomationControlled"]
            ... )
        """
        self.headless = headless
        self.browser_type = browser_type
        self.launch_options = launch_options
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    async def start(self) -> None:
        """
        Start the browser and initialize all components.

        This method:
        1. Initializes Playwright
        2. Launches the browser with configured options
        3. Creates a new browser context
        4. Opens an initial page

        Raises:
            BrowserError: If browser fails to start or unsupported browser type

        Example:
            >>> manager = BrowserManager(headless=True)
            >>> await manager.start()
        """
        try:
            logger.info(f"Starting {self.browser_type} browser (headless={self.headless})")
            self._playwright = await async_playwright().start()

            # Get the appropriate browser type
            if self.browser_type == "chromium":
                browser_launcher = self._playwright.chromium
            elif self.browser_type == "firefox":
                browser_launcher = self._playwright.firefox
            elif self.browser_type == "webkit":
                browser_launcher = self._playwright.webkit
            else:
                raise BrowserError(f"Unsupported browser type: {self.browser_type}")

            # Launch browser with configured options
            self._browser = await browser_launcher.launch(
                headless=self.headless, **self.launch_options
            )

            # Create browser context (isolated session)
            self._context = await self._browser.new_context()

            # Create initial page
            self._page = await self._context.new_page()

            logger.info("Browser started successfully")
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            raise BrowserError(f"Failed to start browser: {e}") from e

    async def stop(self) -> None:
        """
        Stop the browser and cleanup all resources.

        This method gracefully closes:
        1. All open pages
        2. The browser context
        3. The browser instance
        4. Playwright resources

        Raises:
            BrowserError: If cleanup fails

        Example:
            >>> await manager.stop()
        """
        try:
            logger.info("Stopping browser")
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
            logger.info("Browser stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping browser: {e}")
            raise BrowserError(f"Failed to stop browser: {e}") from e

    async def new_page(self) -> Page:
        """
        Create a new page in the current context.

        Returns:
            New Page instance
        """
        if not self._context:
            raise BrowserError("Browser context not initialized. Call start() first.")
        return await self._context.new_page()

    @property
    def page(self) -> Page:
        """Get the current page."""
        if not self._page:
            raise BrowserError("No active page. Call start() first.")
        return self._page

    @property
    def context(self) -> BrowserContext:
        """Get the browser context."""
        if not self._context:
            raise BrowserError("Browser context not initialized. Call start() first.")
        return self._context

    @property
    def browser(self) -> Browser:
        """Get the browser instance."""
        if not self._browser:
            raise BrowserError("Browser not initialized. Call start() first.")
        return self._browser

    async def __aenter__(self) -> "BrowserManager":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

