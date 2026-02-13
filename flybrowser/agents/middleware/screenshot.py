"""ScreenshotOnErrorMiddleware — capture screenshot when agent errors."""
import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ScreenshotOnErrorMiddleware:
    def __init__(self, page) -> None:
        self._page = page

    async def before_run(self, ctx: Any) -> None:
        pass

    async def after_run(self, ctx: Any, result: Any) -> None:
        if getattr(result, "error", None):
            try:
                img_bytes = await self._page.screenshot()
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                ctx.metadata["error_screenshot_b64"] = b64[:100]
            except Exception as e:
                logger.debug(f"Error screenshot failed: {e}")
