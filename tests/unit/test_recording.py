# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Recording module."""

import asyncio
import base64
import os
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flybrowser.core.recording import (
    RecordingConfig,
    RecordingManager,
    RecordingState,
    Screenshot,
    ScreenshotCapture,
    ScreenshotFormat,
    VideoRecorder,
    VideoRecording,
)
from flybrowser.exceptions import BrowserError


class TestRecordingConfig:
    """Tests for RecordingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RecordingConfig()
        
        assert config.enabled is True
        assert config.output_dir == "./recordings"
        assert config.screenshot_format == ScreenshotFormat.PNG
        assert config.screenshot_quality == 80
        assert config.screenshot_full_page is False
        assert config.video_enabled is True
        assert config.max_screenshots == 1000
        assert config.auto_screenshot_on_navigation is True
        assert config.auto_screenshot_on_action is False

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = RecordingConfig(
            enabled=False,
            output_dir="/tmp/recordings",
            screenshot_format=ScreenshotFormat.JPEG,
            screenshot_quality=90,
            max_screenshots=500
        )
        
        assert config.enabled is False
        assert config.output_dir == "/tmp/recordings"
        assert config.screenshot_format == ScreenshotFormat.JPEG
        assert config.screenshot_quality == 90
        assert config.max_screenshots == 500


class TestScreenshot:
    """Tests for Screenshot dataclass."""

    def test_default_values(self):
        """Test default screenshot values."""
        screenshot = Screenshot()
        
        assert screenshot.id is not None
        assert screenshot.timestamp > 0
        assert screenshot.data == b""
        assert screenshot.format == ScreenshotFormat.PNG
        assert screenshot.url == ""
        assert screenshot.title == ""
        assert screenshot.full_page is False
        assert screenshot.metadata == {}

    def test_to_base64(self):
        """Test converting screenshot to base64."""
        screenshot = Screenshot(data=b"test image data")
        
        result = screenshot.to_base64()
        
        expected = base64.b64encode(b"test image data").decode("utf-8")
        assert result == expected

    def test_to_data_url_png(self):
        """Test converting PNG screenshot to data URL."""
        screenshot = Screenshot(
            data=b"test image data",
            format=ScreenshotFormat.PNG
        )
        
        result = screenshot.to_data_url()
        
        b64 = base64.b64encode(b"test image data").decode("utf-8")
        assert result == f"data:image/png;base64,{b64}"

    def test_to_data_url_jpeg(self):
        """Test converting JPEG screenshot to data URL."""
        screenshot = Screenshot(
            data=b"test image data",
            format=ScreenshotFormat.JPEG
        )
        
        result = screenshot.to_data_url()
        
        b64 = base64.b64encode(b"test image data").decode("utf-8")
        assert result == f"data:image/jpeg;base64,{b64}"


class TestVideoRecording:
    """Tests for VideoRecording dataclass."""

    def test_default_values(self):
        """Test default video recording values (Full HD)."""
        recording = VideoRecording()
        
        assert recording.id is not None
        assert recording.started_at > 0
        assert recording.ended_at is None
        assert recording.file_path is None
        assert recording.size_bytes == 0
        assert recording.duration_seconds == 0.0
        assert recording.width == 1920  # Full HD
        assert recording.height == 1080  # Full HD
        assert recording.frame_rate == 30


class TestScreenshotCapture:
    """Tests for ScreenshotCapture."""

    def test_init_creates_output_dir(self, temp_dir):
        """Test initialization creates output directory."""
        config = RecordingConfig(output_dir=os.path.join(temp_dir, "screenshots"))
        capture = ScreenshotCapture(config)
        
        assert os.path.exists(config.output_dir)

    @pytest.mark.asyncio
    async def test_take_screenshot(self, temp_dir):
        """Test taking a screenshot."""
        config = RecordingConfig(output_dir=temp_dir)
        capture = ScreenshotCapture(config)
        
        mock_page = MagicMock()
        mock_page.screenshot = AsyncMock(return_value=b"png_data")
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.viewport_size = {"width": 1920, "height": 1080}
        
        screenshot = await capture.take(mock_page)
        
        assert screenshot.data == b"png_data"
        assert screenshot.url == "https://example.com"
        assert screenshot.title == "Example"
        assert screenshot.width == 1920
        assert screenshot.height == 1080

    @pytest.mark.asyncio
    async def test_take_screenshot_full_page(self, temp_dir):
        """Test taking a full page screenshot."""
        config = RecordingConfig(output_dir=temp_dir)
        capture = ScreenshotCapture(config)
        
        mock_page = MagicMock()
        mock_page.screenshot = AsyncMock(return_value=b"png_data")
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.viewport_size = {"width": 1920, "height": 1080}
        
        screenshot = await capture.take(mock_page, full_page=True)
        
        assert screenshot.full_page is True
        mock_page.screenshot.assert_awaited_once()
        call_args = mock_page.screenshot.await_args
        assert call_args.kwargs["full_page"] is True

    @pytest.mark.asyncio
    async def test_take_screenshot_with_jpeg_quality(self, temp_dir):
        """Test screenshot with JPEG format and quality."""
        config = RecordingConfig(
            output_dir=temp_dir,
            screenshot_format=ScreenshotFormat.JPEG,
            screenshot_quality=75
        )
        capture = ScreenshotCapture(config)
        
        mock_page = MagicMock()
        mock_page.screenshot = AsyncMock(return_value=b"jpeg_data")
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.viewport_size = {}
        
        await capture.take(mock_page)
        
        call_args = mock_page.screenshot.await_args
        assert call_args.kwargs["quality"] == 75
        assert call_args.kwargs["type"] == "jpeg"

    @pytest.mark.asyncio
    async def test_take_screenshot_save_to_file(self, temp_dir):
        """Test saving screenshot to file."""
        config = RecordingConfig(output_dir=temp_dir)
        capture = ScreenshotCapture(config)
        
        mock_page = MagicMock()
        mock_page.screenshot = AsyncMock(return_value=b"png_data")
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.viewport_size = {}
        
        screenshot = await capture.take(mock_page, save_to_file=True)
        
        assert screenshot.file_path is not None
        assert os.path.exists(screenshot.file_path)
        
        with open(screenshot.file_path, "rb") as f:
            assert f.read() == b"png_data"

    @pytest.mark.asyncio
    async def test_take_screenshot_enforces_max_limit(self, temp_dir):
        """Test max screenshots limit is enforced."""
        config = RecordingConfig(output_dir=temp_dir, max_screenshots=3)
        capture = ScreenshotCapture(config)
        
        mock_page = MagicMock()
        mock_page.screenshot = AsyncMock(return_value=b"png_data")
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.viewport_size = {}
        
        # Take 5 screenshots
        for _ in range(5):
            await capture.take(mock_page)
        
        # Should only keep max_screenshots
        assert len(capture.get_screenshots()) == 3

    @pytest.mark.asyncio
    async def test_take_screenshot_raises_on_error(self, temp_dir):
        """Test screenshot raises BrowserError on failure."""
        config = RecordingConfig(output_dir=temp_dir)
        capture = ScreenshotCapture(config)
        
        mock_page = MagicMock()
        mock_page.screenshot = AsyncMock(side_effect=Exception("Screenshot failed"))
        
        with pytest.raises(BrowserError, match="Screenshot capture failed"):
            await capture.take(mock_page)

    def test_get_screenshots(self, temp_dir):
        """Test getting all screenshots."""
        config = RecordingConfig(output_dir=temp_dir)
        capture = ScreenshotCapture(config)
        
        screenshot1 = Screenshot(id="s1", data=b"data1")
        screenshot2 = Screenshot(id="s2", data=b"data2")
        capture._screenshots = [screenshot1, screenshot2]
        
        result = capture.get_screenshots()
        
        assert len(result) == 2
        assert result[0].id == "s1"
        assert result[1].id == "s2"

    def test_get_screenshot_by_id(self, temp_dir):
        """Test getting screenshot by ID."""
        config = RecordingConfig(output_dir=temp_dir)
        capture = ScreenshotCapture(config)
        
        screenshot = Screenshot(id="test-id", data=b"data")
        capture._screenshots = [screenshot]
        
        result = capture.get_screenshot("test-id")
        assert result is screenshot
        
        # Non-existent ID
        assert capture.get_screenshot("invalid") is None

    def test_clear(self, temp_dir):
        """Test clearing screenshots."""
        config = RecordingConfig(output_dir=temp_dir)
        capture = ScreenshotCapture(config)
        
        capture._screenshots = [Screenshot(), Screenshot()]
        capture.clear()
        
        assert len(capture.get_screenshots()) == 0


class TestVideoRecorder:
    """Tests for VideoRecorder."""

    def test_init_creates_output_dir(self, temp_dir):
        """Test initialization creates output directory."""
        config = RecordingConfig(output_dir=os.path.join(temp_dir, "videos"))
        recorder = VideoRecorder(config)
        
        assert os.path.exists(config.output_dir)

    def test_initial_state(self, temp_dir):
        """Test initial recorder state."""
        config = RecordingConfig(output_dir=temp_dir)
        recorder = VideoRecorder(config)
        
        assert recorder.state == RecordingState.IDLE
        assert recorder.current_recording is None

    @pytest.mark.asyncio
    async def test_create_recording_context_disabled(self, temp_dir):
        """Test creating context when video disabled."""
        config = RecordingConfig(output_dir=temp_dir, video_enabled=False)
        recorder = VideoRecorder(config)
        
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        
        result = await recorder.create_recording_context(mock_browser)
        
        assert result is mock_context
        mock_browser.new_context.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_create_recording_context_enabled(self, temp_dir):
        """Test creating context with video enabled."""
        config = RecordingConfig(
            output_dir=temp_dir,
            video_enabled=True,
            video_size={"width": 1920, "height": 1080}
        )
        recorder = VideoRecorder(config)
        
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        
        result = await recorder.create_recording_context(mock_browser)
        
        assert result is mock_context
        assert recorder.state == RecordingState.RECORDING
        assert recorder.current_recording is not None
        assert recorder.current_recording.width == 1920
        assert recorder.current_recording.height == 1080

    @pytest.mark.asyncio
    async def test_stop_when_not_recording(self, temp_dir):
        """Test stop returns None when not recording."""
        config = RecordingConfig(output_dir=temp_dir)
        recorder = VideoRecorder(config)
        
        result = await recorder.stop()
        
        assert result is None

    @pytest.mark.asyncio
    async def test_stop_recording(self, temp_dir):
        """Test stopping a recording."""
        config = RecordingConfig(output_dir=temp_dir)
        recorder = VideoRecorder(config)
        
        # Simulate active recording
        recorder._state = RecordingState.RECORDING
        recorder._current_recording = VideoRecording()
        
        mock_context = MagicMock()
        mock_context.pages = []
        mock_context.close = AsyncMock()
        recorder._context = mock_context
        
        result = await recorder.stop()
        
        assert recorder.state == RecordingState.STOPPED
        assert result is recorder._current_recording
        mock_context.close.assert_awaited_once()


class TestRecordingManager:
    """Tests for RecordingManager."""

    def test_init(self, temp_dir):
        """Test RecordingManager initialization."""
        config = RecordingConfig(output_dir=temp_dir)
        manager = RecordingManager(config)
        
        assert manager.config is config
        assert manager._page is None
        assert manager._session_id is None

    @pytest.mark.asyncio
    async def test_start_session(self, temp_dir):
        """Test starting a recording session."""
        config = RecordingConfig(
            output_dir=temp_dir,
            auto_screenshot_on_navigation=False
        )
        manager = RecordingManager(config)
        
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        
        with patch.object(
            manager._video_recorder,
            "create_recording_context",
            new_callable=AsyncMock,
            return_value=mock_context
        ):
            page = await manager.start_session(mock_browser)
            
            assert page is mock_page
            assert manager.page is mock_page
            assert manager.session_id is not None

    @pytest.mark.asyncio
    async def test_stop_session(self, temp_dir):
        """Test stopping a recording session."""
        config = RecordingConfig(output_dir=temp_dir)
        manager = RecordingManager(config)
        
        manager._session_id = "test-session"
        manager._page = MagicMock()
        
        with patch.object(
            manager._screenshot_capture,
            "stop_periodic",
            new_callable=AsyncMock
        ):
            with patch.object(
                manager._video_recorder,
                "stop",
                new_callable=AsyncMock,
                return_value=None
            ):
                result = await manager.stop_session()
                
                assert result["session_id"] == "test-session"
                assert "screenshots" in result
                assert "video" in result

    @pytest.mark.asyncio
    async def test_capture_screenshot(self, temp_dir):
        """Test capturing screenshot via manager."""
        config = RecordingConfig(output_dir=temp_dir)
        manager = RecordingManager(config)
        
        mock_page = MagicMock()
        mock_page.screenshot = AsyncMock(return_value=b"png_data")
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.viewport_size = {}
        
        manager._page = mock_page
        
        screenshot = await manager.capture_screenshot()
        
        assert screenshot.data == b"png_data"

    @pytest.mark.asyncio
    async def test_capture_screenshot_no_session(self, temp_dir):
        """Test capture screenshot raises when no session."""
        config = RecordingConfig(output_dir=temp_dir)
        manager = RecordingManager(config)
        
        with pytest.raises(BrowserError, match="No active recording session"):
            await manager.capture_screenshot()

    def test_get_screenshots(self, temp_dir):
        """Test getting screenshots from manager."""
        config = RecordingConfig(output_dir=temp_dir)
        manager = RecordingManager(config)
        
        screenshot = Screenshot(id="test", data=b"data")
        manager._screenshot_capture._screenshots = [screenshot]
        
        result = manager.get_screenshots()
        
        assert len(result) == 1
        assert result[0].id == "test"

    def test_get_screenshot_by_id(self, temp_dir):
        """Test getting screenshot by ID."""
        config = RecordingConfig(output_dir=temp_dir)
        manager = RecordingManager(config)
        
        screenshot = Screenshot(id="test-id", data=b"data")
        manager._screenshot_capture._screenshots = [screenshot]
        
        result = manager.get_screenshot("test-id")
        assert result is screenshot

    def test_get_screenshot_data(self, temp_dir):
        """Test getting screenshot data."""
        config = RecordingConfig(output_dir=temp_dir)
        manager = RecordingManager(config)
        
        screenshot = Screenshot(id="test-id", data=b"test_data")
        manager._screenshot_capture._screenshots = [screenshot]
        
        result = manager.get_screenshot_data("test-id")
        assert result == b"test_data"
        
        # Non-existent ID
        assert manager.get_screenshot_data("invalid") is None

    def test_get_screenshot_base64(self, temp_dir):
        """Test getting screenshot as base64."""
        config = RecordingConfig(output_dir=temp_dir)
        manager = RecordingManager(config)
        
        screenshot = Screenshot(id="test-id", data=b"test_data")
        manager._screenshot_capture._screenshots = [screenshot]
        
        result = manager.get_screenshot_base64("test-id")
        expected = base64.b64encode(b"test_data").decode("utf-8")
        assert result == expected

    def test_video_state_property(self, temp_dir):
        """Test video_state property."""
        config = RecordingConfig(output_dir=temp_dir)
        manager = RecordingManager(config)
        
        assert manager.video_state == RecordingState.IDLE


class TestRecordingState:
    """Tests for RecordingState enum."""

    def test_states(self):
        """Test all recording states exist."""
        assert RecordingState.IDLE == "idle"
        assert RecordingState.RECORDING == "recording"
        assert RecordingState.PAUSED == "paused"
        assert RecordingState.STOPPED == "stopped"
        assert RecordingState.ERROR == "error"


class TestScreenshotFormat:
    """Tests for ScreenshotFormat enum."""

    def test_formats(self):
        """Test all screenshot formats exist."""
        assert ScreenshotFormat.PNG == "png"
        assert ScreenshotFormat.JPEG == "jpeg"
        assert ScreenshotFormat.WEBP == "webp"
