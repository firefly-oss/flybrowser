"""ObstacleDetectionMiddleware — detect popups/captchas before agent actions."""
import logging
from typing import Any

logger = logging.getLogger(__name__)

_DETECT_OBSTACLES_JS = """() => {
    const hasCookieBanner = !!(
        document.querySelector('[class*="cookie"]') ||
        document.querySelector('[id*="cookie"]') ||
        document.querySelector('[class*="consent"]') ||
        document.querySelector('[id*="consent"]')
    );
    const hasPopup = !!(
        document.querySelector('[class*="modal"].show') ||
        document.querySelector('[role="dialog"][aria-modal="true"]')
    );
    return { hasCookieBanner, hasPopup };
}"""


class ObstacleDetectionMiddleware:
    def __init__(self, page) -> None:
        self._page = page

    async def before_run(self, ctx: Any) -> None:
        try:
            result = await self._page.page.evaluate(_DETECT_OBSTACLES_JS)
            if result.get("hasCookieBanner"):
                ctx.metadata["obstacle_detected"] = "cookie_banner"
            elif result.get("hasPopup"):
                ctx.metadata["obstacle_detected"] = "popup"
        except Exception as e:
            logger.debug(f"Obstacle detection failed: {e}")

    async def after_run(self, ctx: Any, result: Any) -> None:
        pass
