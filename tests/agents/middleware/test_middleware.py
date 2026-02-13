"""Tests for custom browser middleware."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from flybrowser.agents.middleware.obstacle import ObstacleDetectionMiddleware
from flybrowser.agents.middleware.screenshot import ScreenshotOnErrorMiddleware


class TestObstacleDetectionMiddleware:
    def test_creation(self):
        page = MagicMock()
        mw = ObstacleDetectionMiddleware(page)
        assert mw._page is page

    @pytest.mark.asyncio
    async def test_before_run_no_obstacle(self):
        page = MagicMock()
        page.page = AsyncMock()
        page.page.evaluate = AsyncMock(
            return_value={"hasCookieBanner": False, "hasPopup": False}
        )
        ctx = MagicMock()
        ctx.metadata = {}

        mw = ObstacleDetectionMiddleware(page)
        await mw.before_run(ctx)

        assert "obstacle_detected" not in ctx.metadata

    @pytest.mark.asyncio
    async def test_before_run_with_cookie_banner(self):
        page = MagicMock()
        page.page = AsyncMock()
        page.page.evaluate = AsyncMock(
            return_value={"hasCookieBanner": True, "hasPopup": False}
        )
        ctx = MagicMock()
        ctx.metadata = {}

        mw = ObstacleDetectionMiddleware(page)
        await mw.before_run(ctx)

        assert ctx.metadata["obstacle_detected"] == "cookie_banner"


class TestScreenshotOnErrorMiddleware:
    def test_creation(self):
        page = MagicMock()
        mw = ScreenshotOnErrorMiddleware(page)
        assert mw._page is page

    @pytest.mark.asyncio
    async def test_after_run_no_error(self):
        page = MagicMock()
        page.screenshot = AsyncMock()
        ctx = MagicMock()
        ctx.metadata = {}
        result = MagicMock()
        result.error = None

        mw = ScreenshotOnErrorMiddleware(page)
        await mw.after_run(ctx, result)

        page.screenshot.assert_not_called()

    @pytest.mark.asyncio
    async def test_after_run_with_error_takes_screenshot(self):
        page = MagicMock()
        page.screenshot = AsyncMock(return_value=b"\x89PNG\r\n\x1a\nfakedata")
        ctx = MagicMock()
        ctx.metadata = {}
        result = MagicMock()
        result.error = "Something went wrong"

        mw = ScreenshotOnErrorMiddleware(page)
        await mw.after_run(ctx, result)

        page.screenshot.assert_called_once()
        assert "error_screenshot_b64" in ctx.metadata
