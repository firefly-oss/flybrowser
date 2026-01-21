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
Browser Monitoring and Recording for FlyBrowser.

This module provides screenshot capture and video recording capabilities
for browser sessions. It includes:
- ScreenshotCapture: Capture screenshots at various points
- VideoRecorder: Record the entire navigation process
- RecordingManager: Manage recording sessions and storage
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from playwright.async_api import Page

from flybrowser.exceptions import BrowserError
from flybrowser.utils.logger import logger


class RecordingState(str, Enum):
    """State of a recording session."""

    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class ScreenshotFormat(str, Enum):
    """Supported screenshot formats."""

    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"


@dataclass
class RecordingConfig:
    """Configuration for recording sessions."""

    enabled: bool = True
    output_dir: str = "./recordings"
    screenshot_format: ScreenshotFormat = ScreenshotFormat.PNG
    screenshot_quality: int = 80  # For JPEG/WebP
    screenshot_full_page: bool = False
    video_enabled: bool = True
    video_size: Dict[str, int] = field(default_factory=lambda: {"width": 1280, "height": 720})
    video_frame_rate: int = 25
    max_screenshots: int = 1000
    auto_screenshot_on_navigation: bool = True
    auto_screenshot_on_action: bool = False
    screenshot_interval_seconds: Optional[float] = None  # For periodic screenshots


@dataclass
class Screenshot:
    """Represents a captured screenshot."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    data: bytes = field(default_factory=bytes)
    format: ScreenshotFormat = ScreenshotFormat.PNG
    url: str = ""
    title: str = ""
    width: int = 0
    height: int = 0
    full_page: bool = False
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_base64(self) -> str:
        """Convert screenshot data to base64 string."""
        return base64.b64encode(self.data).decode("utf-8")

    def to_data_url(self) -> str:
        """Convert screenshot to data URL."""
        mime_type = f"image/{self.format.value}"
        return f"data:{mime_type};base64,{self.to_base64()}"


@dataclass
class VideoRecording:
    """Represents a video recording."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    file_path: Optional[str] = None
    size_bytes: int = 0
    duration_seconds: float = 0.0
    width: int = 1280
    height: int = 720
    frame_rate: int = 25
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScreenshotCapture:
    """
    Captures screenshots from browser pages.

    Supports various capture modes including on-demand, periodic,
    and event-triggered screenshots.

    Example:
        >>> capture = ScreenshotCapture(config)
        >>> screenshot = await capture.take(page)
        >>> print(screenshot.to_data_url())
    """

    def __init__(self, config: Optional[RecordingConfig] = None) -> None:
        """Initialize screenshot capture."""
        self.config = config or RecordingConfig()
        self._screenshots: List[Screenshot] = []
        self._lock = asyncio.Lock()
        self._periodic_task: Optional[asyncio.Task] = None

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    async def take(
        self,
        page: Page,
        full_page: Optional[bool] = None,
        save_to_file: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Screenshot:
        """
        Take a screenshot of the current page.

        Args:
            page: Playwright page to capture
            full_page: Whether to capture full page (overrides config)
            save_to_file: Whether to save to file
            metadata: Additional metadata to attach

        Returns:
            Screenshot object with captured data
        """
        try:
            full_page = full_page if full_page is not None else self.config.screenshot_full_page

            # Capture screenshot
            screenshot_options: Dict[str, Any] = {
                "full_page": full_page,
                "type": self.config.screenshot_format.value,
            }

            if self.config.screenshot_format in (ScreenshotFormat.JPEG, ScreenshotFormat.WEBP):
                screenshot_options["quality"] = self.config.screenshot_quality

            data = await page.screenshot(**screenshot_options)

            # Get page info
            viewport = page.viewport_size or {"width": 0, "height": 0}

            screenshot = Screenshot(
                data=data,
                format=self.config.screenshot_format,
                url=page.url,
                title=await page.title(),
                width=viewport.get("width", 0),
                height=viewport.get("height", 0),
                full_page=full_page,
                metadata=metadata or {},
            )

            # Save to file if requested
            if save_to_file:
                file_path = self._get_screenshot_path(screenshot)
                with open(file_path, "wb") as f:
                    f.write(data)
                screenshot.file_path = file_path

            # Store screenshot
            async with self._lock:
                self._screenshots.append(screenshot)

                # Enforce max screenshots limit
                while len(self._screenshots) > self.config.max_screenshots:
                    self._screenshots.pop(0)

            logger.debug(f"Screenshot captured: {screenshot.id}")
            return screenshot

        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            raise BrowserError(f"Screenshot capture failed: {e}") from e

    def _get_screenshot_path(self, screenshot: Screenshot) -> str:
        """Generate file path for screenshot."""
        timestamp = int(screenshot.timestamp * 1000)
        filename = f"screenshot_{timestamp}_{screenshot.id[:8]}.{screenshot.format.value}"
        return os.path.join(self.config.output_dir, filename)

    async def start_periodic(self, page: Page, interval: Optional[float] = None) -> None:
        """Start periodic screenshot capture."""
        interval = interval or self.config.screenshot_interval_seconds
        if not interval:
            raise ValueError("Screenshot interval not specified")

        if self._periodic_task:
            self._periodic_task.cancel()

        async def periodic_capture():
            while True:
                try:
                    await self.take(page, save_to_file=True)
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Periodic screenshot error: {e}")
                    await asyncio.sleep(interval)

        self._periodic_task = asyncio.create_task(periodic_capture())

    async def stop_periodic(self) -> None:
        """Stop periodic screenshot capture."""
        if self._periodic_task:
            self._periodic_task.cancel()
            self._periodic_task = None

    def get_screenshots(self) -> List[Screenshot]:
        """Get all captured screenshots."""
        return list(self._screenshots)

    def get_screenshot(self, screenshot_id: str) -> Optional[Screenshot]:
        """Get a specific screenshot by ID."""
        for screenshot in self._screenshots:
            if screenshot.id == screenshot_id:
                return screenshot
        return None

    def clear(self) -> None:
        """Clear all stored screenshots."""
        self._screenshots.clear()


class VideoRecorder:
    """
    Records video of browser sessions using Playwright's built-in recording.

    Example:
        >>> recorder = VideoRecorder(config)
        >>> context = await recorder.create_recording_context(browser)
        >>> page = await context.new_page()
        >>> # ... perform actions ...
        >>> video = await recorder.stop()
    """

    def __init__(self, config: Optional[RecordingConfig] = None) -> None:
        """Initialize video recorder."""
        self.config = config or RecordingConfig()
        self._state = RecordingState.IDLE
        self._current_recording: Optional[VideoRecording] = None
        self._context = None
        self._page = None

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    async def create_recording_context(self, browser: Any) -> Any:
        """
        Create a browser context with video recording enabled.

        Args:
            browser: Playwright browser instance

        Returns:
            Browser context with recording enabled
        """
        if not self.config.video_enabled:
            return await browser.new_context()

        self._current_recording = VideoRecording(
            width=self.config.video_size.get("width", 1280),
            height=self.config.video_size.get("height", 720),
            frame_rate=self.config.video_frame_rate,
        )

        self._context = await browser.new_context(
            record_video_dir=self.config.output_dir,
            record_video_size=self.config.video_size,
        )

        self._state = RecordingState.RECORDING
        logger.info(f"Video recording started: {self._current_recording.id}")

        return self._context

    async def stop(self) -> Optional[VideoRecording]:
        """
        Stop recording and return the video recording info.

        Returns:
            VideoRecording object with file path and metadata
        """
        if self._state != RecordingState.RECORDING:
            return None

        self._state = RecordingState.STOPPED

        if self._context:
            # Get video path from pages
            for page in self._context.pages:
                video = page.video
                if video:
                    try:
                        video_path = await video.path()
                        if self._current_recording:
                            self._current_recording.file_path = video_path
                            self._current_recording.ended_at = time.time()
                            self._current_recording.duration_seconds = (
                                self._current_recording.ended_at - self._current_recording.started_at
                            )

                            # Get file size
                            if os.path.exists(video_path):
                                self._current_recording.size_bytes = os.path.getsize(video_path)
                    except Exception as e:
                        logger.error(f"Error getting video path: {e}")

            await self._context.close()

        logger.info(f"Video recording stopped: {self._current_recording.id if self._current_recording else 'N/A'}")
        return self._current_recording

    @property
    def state(self) -> RecordingState:
        """Get current recording state."""
        return self._state

    @property
    def current_recording(self) -> Optional[VideoRecording]:
        """Get current recording info."""
        return self._current_recording



class RecordingManager:
    """
    Unified manager for browser recording (screenshots and video).

    Provides a high-level API for managing recording sessions,
    including automatic screenshot capture on navigation and actions.

    Example:
        >>> manager = RecordingManager(config)
        >>> await manager.start_session(browser)
        >>> page = manager.page
        >>> await page.goto("https://example.com")
        >>> await manager.capture_screenshot()
        >>> session = await manager.stop_session()
    """

    def __init__(self, config: Optional[RecordingConfig] = None) -> None:
        """Initialize recording manager."""
        self.config = config or RecordingConfig()
        self._screenshot_capture = ScreenshotCapture(self.config)
        self._video_recorder = VideoRecorder(self.config)
        self._context = None
        self._page: Optional[Page] = None
        self._session_id: Optional[str] = None
        self._navigation_handler = None

    async def start_session(self, browser: Any) -> Page:
        """
        Start a recording session.

        Args:
            browser: Playwright browser instance

        Returns:
            Page with recording enabled
        """
        self._session_id = str(uuid.uuid4())

        # Create context with video recording if enabled
        self._context = await self._video_recorder.create_recording_context(browser)
        self._page = await self._context.new_page()

        # Set up auto-screenshot on navigation if enabled
        if self.config.auto_screenshot_on_navigation:
            async def on_navigation(page: Page):
                try:
                    await self._screenshot_capture.take(
                        page,
                        save_to_file=True,
                        metadata={"trigger": "navigation"},
                    )
                except Exception as e:
                    logger.error(f"Auto-screenshot on navigation failed: {e}")

            self._page.on("load", lambda: asyncio.create_task(on_navigation(self._page)))

        logger.info(f"Recording session started: {self._session_id}")
        return self._page

    async def stop_session(self) -> Dict[str, Any]:
        """
        Stop the recording session and return all recordings.

        Returns:
            Dictionary with screenshots and video info
        """
        # Stop periodic screenshots if running
        await self._screenshot_capture.stop_periodic()

        # Stop video recording
        video = await self._video_recorder.stop()

        # Get all screenshots
        screenshots = self._screenshot_capture.get_screenshots()

        result = {
            "session_id": self._session_id,
            "screenshots": [
                {
                    "id": s.id,
                    "timestamp": s.timestamp,
                    "url": s.url,
                    "title": s.title,
                    "file_path": s.file_path,
                    "format": s.format.value,
                }
                for s in screenshots
            ],
            "video": None,
        }

        if video:
            result["video"] = {
                "id": video.id,
                "file_path": video.file_path,
                "duration_seconds": video.duration_seconds,
                "size_bytes": video.size_bytes,
            }

        logger.info(f"Recording session stopped: {self._session_id}")
        return result

    async def capture_screenshot(
        self,
        full_page: Optional[bool] = None,
        save_to_file: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Screenshot:
        """Capture a screenshot of the current page."""
        if not self._page:
            raise BrowserError("No active recording session")

        return await self._screenshot_capture.take(
            self._page,
            full_page=full_page,
            save_to_file=save_to_file,
            metadata=metadata,
        )

    async def start_periodic_screenshots(self, interval: float) -> None:
        """Start periodic screenshot capture."""
        if not self._page:
            raise BrowserError("No active recording session")

        await self._screenshot_capture.start_periodic(self._page, interval)

    async def stop_periodic_screenshots(self) -> None:
        """Stop periodic screenshot capture."""
        await self._screenshot_capture.stop_periodic()

    def get_screenshots(self) -> List[Screenshot]:
        """Get all captured screenshots."""
        return self._screenshot_capture.get_screenshots()

    def get_screenshot(self, screenshot_id: str) -> Optional[Screenshot]:
        """Get a specific screenshot by ID."""
        return self._screenshot_capture.get_screenshot(screenshot_id)

    def get_screenshot_data(self, screenshot_id: str) -> Optional[bytes]:
        """Get screenshot data by ID."""
        screenshot = self.get_screenshot(screenshot_id)
        return screenshot.data if screenshot else None

    def get_screenshot_base64(self, screenshot_id: str) -> Optional[str]:
        """Get screenshot as base64 string."""
        screenshot = self.get_screenshot(screenshot_id)
        return screenshot.to_base64() if screenshot else None

    @property
    def page(self) -> Optional[Page]:
        """Get the current page."""
        return self._page

    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self._session_id

    @property
    def video_state(self) -> RecordingState:
        """Get video recording state."""
        return self._video_recorder.state


