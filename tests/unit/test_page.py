# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PageController."""

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from flybrowser.core.page import PageController
from flybrowser.exceptions import NavigationError, PageError


class TestPageControllerInit:
    """Tests for PageController initialization."""

    def test_init_with_page(self):
        """Test initialization with a page."""
        mock_page = MagicMock()
        controller = PageController(mock_page)
        assert controller.page is mock_page


class TestPageControllerGoto:
    """Tests for PageController.goto()."""

    @pytest.mark.asyncio
    async def test_goto_navigates_to_url(self):
        """Test goto navigates to URL."""
        mock_page = MagicMock()
        mock_page.goto = AsyncMock()
        controller = PageController(mock_page)
        
        await controller.goto("https://example.com")
        
        mock_page.goto.assert_awaited_once_with(
            "https://example.com",
            wait_until="domcontentloaded",
            timeout=30000
        )

    @pytest.mark.asyncio
    async def test_goto_with_custom_wait_until(self):
        """Test goto with custom wait_until."""
        mock_page = MagicMock()
        mock_page.goto = AsyncMock()
        controller = PageController(mock_page)
        
        await controller.goto("https://example.com", wait_until="networkidle")
        
        mock_page.goto.assert_awaited_once_with(
            "https://example.com",
            wait_until="networkidle",
            timeout=30000
        )

    @pytest.mark.asyncio
    async def test_goto_with_custom_timeout(self):
        """Test goto with custom timeout."""
        mock_page = MagicMock()
        mock_page.goto = AsyncMock()
        controller = PageController(mock_page)
        
        await controller.goto("https://example.com", timeout=60000)
        
        mock_page.goto.assert_awaited_once_with(
            "https://example.com",
            wait_until="domcontentloaded",
            timeout=60000
        )

    @pytest.mark.asyncio
    async def test_goto_raises_navigation_error(self):
        """Test goto raises NavigationError on failure."""
        mock_page = MagicMock()
        mock_page.goto = AsyncMock(side_effect=Exception("Navigation failed"))
        controller = PageController(mock_page)
        
        with pytest.raises(NavigationError, match="Failed to navigate"):
            await controller.goto("https://example.com")


class TestPageControllerScreenshot:
    """Tests for PageController.screenshot()."""

    @pytest.mark.asyncio
    async def test_screenshot_default(self):
        """Test screenshot with default options."""
        mock_page = MagicMock()
        mock_page.screenshot = AsyncMock(return_value=b"png_data")
        controller = PageController(mock_page)
        
        result = await controller.screenshot()
        
        assert result == b"png_data"
        mock_page.screenshot.assert_awaited_once_with(full_page=False, type="png")

    @pytest.mark.asyncio
    async def test_screenshot_full_page(self):
        """Test full page screenshot."""
        mock_page = MagicMock()
        mock_page.screenshot = AsyncMock(return_value=b"png_data")
        controller = PageController(mock_page)
        
        result = await controller.screenshot(full_page=True)
        
        assert result == b"png_data"
        mock_page.screenshot.assert_awaited_once_with(full_page=True, type="png")

    @pytest.mark.asyncio
    async def test_screenshot_raises_page_error(self):
        """Test screenshot raises PageError on failure."""
        mock_page = MagicMock()
        mock_page.screenshot = AsyncMock(side_effect=Exception("Screenshot failed"))
        controller = PageController(mock_page)
        
        with pytest.raises(PageError, match="Failed to take screenshot"):
            await controller.screenshot()


class TestPageControllerGetHtml:
    """Tests for PageController.get_html()."""

    @pytest.mark.asyncio
    async def test_get_html_returns_content(self):
        """Test get_html returns page content."""
        mock_page = MagicMock()
        mock_page.content = AsyncMock(return_value="<html><body>Test</body></html>")
        controller = PageController(mock_page)
        
        result = await controller.get_html()
        
        assert result == "<html><body>Test</body></html>"
        mock_page.content.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_html_raises_page_error(self):
        """Test get_html raises PageError on failure."""
        mock_page = MagicMock()
        mock_page.content = AsyncMock(side_effect=Exception("Content failed"))
        controller = PageController(mock_page)
        
        with pytest.raises(PageError, match="Failed to get page HTML"):
            await controller.get_html()


class TestPageControllerGetTitle:
    """Tests for PageController.get_title()."""

    @pytest.mark.asyncio
    async def test_get_title_returns_title(self):
        """Test get_title returns page title."""
        mock_page = MagicMock()
        mock_page.title = AsyncMock(return_value="Test Page")
        controller = PageController(mock_page)
        
        result = await controller.get_title()
        
        assert result == "Test Page"
        mock_page.title.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_title_raises_page_error(self):
        """Test get_title raises PageError on failure."""
        mock_page = MagicMock()
        mock_page.title = AsyncMock(side_effect=Exception("Title failed"))
        controller = PageController(mock_page)
        
        with pytest.raises(PageError, match="Failed to get page title"):
            await controller.get_title()


class TestPageControllerGetUrl:
    """Tests for PageController.get_url()."""

    @pytest.mark.asyncio
    async def test_get_url_returns_url(self):
        """Test get_url returns current URL."""
        mock_page = MagicMock()
        mock_page.url = "https://example.com/page"
        controller = PageController(mock_page)
        
        result = await controller.get_url()
        
        assert result == "https://example.com/page"


class TestPageControllerWaitForSelector:
    """Tests for PageController.wait_for_selector()."""

    @pytest.mark.asyncio
    async def test_wait_for_selector_default(self):
        """Test wait_for_selector with default options."""
        mock_page = MagicMock()
        mock_page.wait_for_selector = AsyncMock()
        controller = PageController(mock_page)
        
        await controller.wait_for_selector("#element")
        
        mock_page.wait_for_selector.assert_awaited_once_with(
            "#element", timeout=30000, state="visible"
        )

    @pytest.mark.asyncio
    async def test_wait_for_selector_with_options(self):
        """Test wait_for_selector with custom options."""
        mock_page = MagicMock()
        mock_page.wait_for_selector = AsyncMock()
        controller = PageController(mock_page)
        
        await controller.wait_for_selector("#element", timeout=5000, state="hidden")
        
        mock_page.wait_for_selector.assert_awaited_once_with(
            "#element", timeout=5000, state="hidden"
        )

    @pytest.mark.asyncio
    async def test_wait_for_selector_raises_page_error(self):
        """Test wait_for_selector raises PageError on failure."""
        mock_page = MagicMock()
        mock_page.wait_for_selector = AsyncMock(side_effect=Exception("Timeout"))
        controller = PageController(mock_page)
        
        with pytest.raises(PageError, match="Failed to wait for selector"):
            await controller.wait_for_selector("#element")


class TestPageControllerEvaluate:
    """Tests for PageController.evaluate()."""

    @pytest.mark.asyncio
    async def test_evaluate_executes_script(self):
        """Test evaluate executes JavaScript."""
        mock_page = MagicMock()
        mock_page.evaluate = AsyncMock(return_value=42)
        controller = PageController(mock_page)
        
        result = await controller.evaluate("1 + 1")
        
        assert result == 42
        mock_page.evaluate.assert_awaited_once_with("1 + 1")

    @pytest.mark.asyncio
    async def test_evaluate_raises_page_error(self):
        """Test evaluate raises PageError on failure."""
        mock_page = MagicMock()
        mock_page.evaluate = AsyncMock(side_effect=Exception("Script error"))
        controller = PageController(mock_page)
        
        with pytest.raises(PageError, match="Failed to evaluate script"):
            await controller.evaluate("invalid script")


class TestPageControllerGetPageState:
    """Tests for PageController.get_page_state()."""

    @pytest.mark.asyncio
    async def test_get_page_state(self):
        """Test get_page_state returns comprehensive state."""
        mock_page = MagicMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.viewport_size = {"width": 1920, "height": 1080}
        controller = PageController(mock_page)
        
        result = await controller.get_page_state()
        
        assert result["url"] == "https://example.com"
        assert result["title"] == "Example"
        assert result["viewport"] == {"width": 1920, "height": 1080}
