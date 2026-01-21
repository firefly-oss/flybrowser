# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BrowserManager."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flybrowser.core.browser import BrowserManager
from flybrowser.exceptions import BrowserError


class TestBrowserManagerInit:
    """Tests for BrowserManager initialization."""

    def test_default_init(self):
        """Test default initialization values."""
        manager = BrowserManager()
        
        assert manager.headless is True
        assert manager.browser_type == "chromium"
        assert manager.launch_options == {}
        assert manager._playwright is None
        assert manager._browser is None
        assert manager._context is None
        assert manager._page is None

    def test_custom_headless(self):
        """Test initialization with headless=False."""
        manager = BrowserManager(headless=False)
        assert manager.headless is False

    def test_custom_browser_type(self):
        """Test initialization with different browser types."""
        for browser_type in ["chromium", "firefox", "webkit"]:
            manager = BrowserManager(browser_type=browser_type)
            assert manager.browser_type == browser_type

    def test_launch_options(self):
        """Test initialization with custom launch options."""
        options = {"args": ["--disable-gpu"], "slow_mo": 100}
        manager = BrowserManager(**options)
        assert manager.launch_options == options


class TestBrowserManagerStart:
    """Tests for BrowserManager.start()."""

    @pytest.mark.asyncio
    async def test_start_chromium(self, mock_playwright):
        """Test starting a Chromium browser."""
        manager = BrowserManager(browser_type="chromium")
        
        with patch("flybrowser.core.browser.async_playwright") as mock_pw:
            mock_pw_instance = MagicMock()
            mock_pw_instance.start = AsyncMock(return_value=mock_playwright)
            mock_pw.return_value = mock_pw_instance
            
            await manager.start()
            
            mock_playwright.chromium.launch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_firefox(self, mock_playwright):
        """Test starting a Firefox browser."""
        manager = BrowserManager(browser_type="firefox")
        
        with patch("flybrowser.core.browser.async_playwright") as mock_pw:
            mock_pw_instance = MagicMock()
            mock_pw_instance.start = AsyncMock(return_value=mock_playwright)
            mock_pw.return_value = mock_pw_instance
            
            await manager.start()
            
            mock_playwright.firefox.launch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_webkit(self, mock_playwright):
        """Test starting a WebKit browser."""
        manager = BrowserManager(browser_type="webkit")
        
        with patch("flybrowser.core.browser.async_playwright") as mock_pw:
            mock_pw_instance = MagicMock()
            mock_pw_instance.start = AsyncMock(return_value=mock_playwright)
            mock_pw.return_value = mock_pw_instance
            
            await manager.start()
            
            mock_playwright.webkit.launch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_unsupported_browser(self):
        """Test starting with unsupported browser type."""
        manager = BrowserManager(browser_type="invalid")
        
        with patch("flybrowser.core.browser.async_playwright") as mock_pw:
            mock_pw_instance = MagicMock()
            mock_pw_instance.start = AsyncMock()
            mock_pw.return_value = mock_pw_instance
            
            with pytest.raises(BrowserError, match="Unsupported browser type"):
                await manager.start()

    @pytest.mark.asyncio
    async def test_start_creates_context_and_page(self, mock_playwright):
        """Test that start creates context and page."""
        manager = BrowserManager()
        
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = MagicMock()
        
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
        
        with patch("flybrowser.core.browser.async_playwright") as mock_pw:
            mock_pw_instance = MagicMock()
            mock_pw_instance.start = AsyncMock(return_value=mock_playwright)
            mock_pw.return_value = mock_pw_instance
            
            await manager.start()
            
            mock_browser.new_context.assert_awaited_once()
            mock_context.new_page.assert_awaited_once()


class TestBrowserManagerStop:
    """Tests for BrowserManager.stop()."""

    @pytest.mark.asyncio
    async def test_stop_closes_all_resources(self, mock_playwright):
        """Test that stop closes page, context, browser, and playwright."""
        manager = BrowserManager()
        
        mock_page = AsyncMock()
        mock_context = AsyncMock()
        mock_browser = AsyncMock()
        mock_pw = AsyncMock()
        
        manager._page = mock_page
        manager._context = mock_context
        manager._browser = mock_browser
        manager._playwright = mock_pw
        
        await manager.stop()
        
        mock_page.close.assert_awaited_once()
        mock_context.close.assert_awaited_once()
        mock_browser.close.assert_awaited_once()
        mock_pw.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_handles_none_resources(self):
        """Test stop handles None resources gracefully."""
        manager = BrowserManager()
        
        # Should not raise
        await manager.stop()

    @pytest.mark.asyncio
    async def test_stop_raises_on_error(self):
        """Test stop raises BrowserError on failure."""
        manager = BrowserManager()
        mock_page = AsyncMock()
        mock_page.close.side_effect = Exception("Close failed")
        manager._page = mock_page
        
        with pytest.raises(BrowserError, match="Failed to stop browser"):
            await manager.stop()


class TestBrowserManagerProperties:
    """Tests for BrowserManager properties."""

    def test_page_property_raises_when_not_started(self):
        """Test page property raises when browser not started."""
        manager = BrowserManager()
        
        with pytest.raises(BrowserError, match="No active page"):
            _ = manager.page

    def test_page_property_returns_page(self):
        """Test page property returns the page."""
        manager = BrowserManager()
        mock_page = MagicMock()
        manager._page = mock_page
        
        assert manager.page is mock_page

    def test_context_property_raises_when_not_started(self):
        """Test context property raises when browser not started."""
        manager = BrowserManager()
        
        with pytest.raises(BrowserError, match="Browser context not initialized"):
            _ = manager.context

    def test_context_property_returns_context(self):
        """Test context property returns the context."""
        manager = BrowserManager()
        mock_context = MagicMock()
        manager._context = mock_context
        
        assert manager.context is mock_context

    def test_browser_property_raises_when_not_started(self):
        """Test browser property raises when browser not started."""
        manager = BrowserManager()
        
        with pytest.raises(BrowserError, match="Browser not initialized"):
            _ = manager.browser

    def test_browser_property_returns_browser(self):
        """Test browser property returns the browser."""
        manager = BrowserManager()
        mock_browser = MagicMock()
        manager._browser = mock_browser
        
        assert manager.browser is mock_browser


class TestBrowserManagerNewPage:
    """Tests for BrowserManager.new_page()."""

    @pytest.mark.asyncio
    async def test_new_page_raises_when_not_started(self):
        """Test new_page raises when context not initialized."""
        manager = BrowserManager()
        
        with pytest.raises(BrowserError, match="Browser context not initialized"):
            await manager.new_page()

    @pytest.mark.asyncio
    async def test_new_page_creates_page(self):
        """Test new_page creates a new page."""
        manager = BrowserManager()
        mock_context = AsyncMock()
        mock_page = MagicMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        manager._context = mock_context
        
        page = await manager.new_page()
        
        assert page is mock_page
        mock_context.new_page.assert_awaited_once()


class TestBrowserManagerContextManager:
    """Tests for BrowserManager async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_starts_and_stops(self, mock_playwright):
        """Test async context manager starts and stops browser."""
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
        
        with patch("flybrowser.core.browser.async_playwright") as mock_pw:
            mock_pw_instance = MagicMock()
            mock_pw_instance.start = AsyncMock(return_value=mock_playwright)
            mock_pw_instance.stop = AsyncMock()
            mock_pw.return_value = mock_pw_instance
            
            async with BrowserManager() as manager:
                assert manager._page is mock_page
            
            # Verify stop was called
            mock_page.close.assert_awaited_once()
            mock_context.close.assert_awaited_once()
            mock_browser.close.assert_awaited_once()
