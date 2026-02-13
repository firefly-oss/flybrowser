"""Custom browser middleware for fireflyframework-genai agents."""
from flybrowser.agents.middleware.obstacle import ObstacleDetectionMiddleware
from flybrowser.agents.middleware.screenshot import ScreenshotOnErrorMiddleware
__all__ = ["ObstacleDetectionMiddleware", "ScreenshotOnErrorMiddleware"]
