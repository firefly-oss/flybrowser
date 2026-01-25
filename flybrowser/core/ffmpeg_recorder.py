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

"""FFmpeg-based Video Recorder for FlyBrowser.

This module provides advanced video recording capabilities using ffmpeg:
- Modern codecs (H.264, H.265, VP9, VP8, AV1)
- Streaming protocols (RTMP, HLS, DASH)
- Hardware acceleration support
- Bandwidth-optimized encoding profiles
- Real-time frame capture from Playwright
- Adaptive frame rate and quality
- Low-latency streaming optimizations
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
import uuid
import errno
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from playwright.async_api import Page

from flybrowser.exceptions import BrowserError
from flybrowser.utils.logger import logger


# =============================================================================
# Error Recovery Infrastructure
# =============================================================================

class CircuitBreakerState(str, Enum):
    """State of a circuit breaker."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failures exceeded threshold, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures.

    When failures exceed the threshold, the circuit opens and blocks
    further attempts for a cooldown period. After cooldown, it enters
    half-open state to test if the service recovered.
    """

    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # seconds
    half_open_max_calls: int = 3

    # State tracking
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    half_open_calls: int = 0

    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                # Recovered - close the circuit
                logger.info(f"[CIRCUIT_BREAKER] {self.name}: Recovered, closing circuit")
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.half_open_calls = 0
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Failed during recovery test - reopen circuit
            logger.warning(f"[CIRCUIT_BREAKER] {self.name}: Failed during recovery, reopening")
            self.state = CircuitBreakerState.OPEN
            self.half_open_calls = 0
            self.success_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                logger.warning(
                    f"[CIRCUIT_BREAKER] {self.name}: Threshold exceeded "
                    f"({self.failure_count} failures), opening circuit"
                )
                self.state = CircuitBreakerState.OPEN

    def can_execute(self) -> bool:
        """Check if an operation can be executed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True

        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                logger.info(f"[CIRCUIT_BREAKER] {self.name}: Recovery timeout passed, entering half-open")
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                self.success_count = 0
                return True
            return False

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_calls += 1
            return True

        return False

    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0


@dataclass
class RecoveryMetrics:
    """Metrics for tracking recovery attempts and outcomes."""

    total_recovery_attempts: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    ffmpeg_restarts: int = 0
    capture_restarts: int = 0
    last_recovery_time: float = 0.0
    consecutive_failures: int = 0
    uptime_seconds: float = 0.0

    def record_recovery_attempt(self, success: bool, recovery_type: str) -> None:
        """Record a recovery attempt."""
        self.total_recovery_attempts += 1
        self.last_recovery_time = time.time()

        if success:
            self.successful_recoveries += 1
            self.consecutive_failures = 0
        else:
            self.failed_recoveries += 1
            self.consecutive_failures += 1

        if recovery_type == "ffmpeg":
            self.ffmpeg_restarts += 1
        elif recovery_type == "capture":
            self.capture_restarts += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_recovery_attempts": self.total_recovery_attempts,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "ffmpeg_restarts": self.ffmpeg_restarts,
            "capture_restarts": self.capture_restarts,
            "last_recovery_time": self.last_recovery_time,
            "consecutive_failures": self.consecutive_failures,
            "uptime_seconds": self.uptime_seconds,
        }


class StreamingError(BrowserError):
    """Exception for streaming-specific errors."""
    pass


class FFmpegCrashError(StreamingError):
    """Exception raised when FFmpeg process crashes."""
    pass


class FrameCaptureError(StreamingError):
    """Exception raised when frame capture fails."""
    pass


class DiskSpaceError(StreamingError):
    """Exception raised when disk space is insufficient."""
    pass


class RecoverableError(StreamingError):
    """Exception for errors that can be recovered from."""
    pass


class UnrecoverableError(StreamingError):
    """Exception for errors that cannot be recovered from."""
    pass


def check_disk_space(path: str, min_bytes: int = 100 * 1024 * 1024) -> bool:
    """Check if there's sufficient disk space.

    Args:
        path: Path to check
        min_bytes: Minimum required bytes (default 100MB)

    Returns:
        True if sufficient space available
    """
    try:
        stat = os.statvfs(path)
        available = stat.f_bavail * stat.f_frsize
        return available >= min_bytes
    except (OSError, AttributeError):
        # On Windows or if statvfs fails, assume OK
        return True


def is_recoverable_error(error: Exception) -> bool:
    """Determine if an error is recoverable.

    Args:
        error: The exception to check

    Returns:
        True if the error can potentially be recovered from
    """
    # Unrecoverable errors
    unrecoverable_patterns = [
        "disk full",
        "no space left",
        "permission denied",
        "browser closed",
        "context closed",
        "target closed",
    ]

    error_str = str(error).lower()
    for pattern in unrecoverable_patterns:
        if pattern in error_str:
            return False

    # Check for specific error types
    if isinstance(error, (PermissionError, UnrecoverableError)):
        return False

    if isinstance(error, OSError):
        if error.errno in (errno.ENOSPC, errno.EACCES, errno.EPERM):
            return False

    return True


class VideoCodec(str, Enum):
    """Supported video codecs."""
    
    H264 = "h264"           # Most compatible, good compression
    H265 = "h265"           # Better compression than H.264
    VP9 = "vp9"             # Google's open codec
    VP8 = "vp8"             # Older Google codec
    AV1 = "av1"             # Latest open codec, best compression
    
    def get_encoder(self, use_hw_accel: bool = False) -> str:
        """Get ffmpeg encoder name."""
        if use_hw_accel:
            hw_encoders = {
                VideoCodec.H264: ["h264_nvenc", "h264_videotoolbox", "h264_qsv"],
                VideoCodec.H265: ["hevc_nvenc", "hevc_videotoolbox", "hevc_qsv"],
            }
            if self in hw_encoders:
                # Return first available hardware encoder
                for encoder in hw_encoders[self]:
                    if self._is_encoder_available(encoder):
                        return encoder
        
        # Software encoders (fallback)
        software_encoders = {
            VideoCodec.H264: "libx264",
            VideoCodec.H265: "libx265",
            VideoCodec.VP9: "libvpx-vp9",
            VideoCodec.VP8: "libvpx",
            VideoCodec.AV1: "libaom-av1",
        }
        return software_encoders[self]
    
    @staticmethod
    def _is_encoder_available(encoder: str) -> bool:
        """Check if an encoder is available in ffmpeg."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return encoder in result.stdout
        except Exception:
            return False


class StreamingProtocol(str, Enum):
    """Supported streaming protocols."""
    
    HLS = "hls"             # HTTP Live Streaming (Apple)
    DASH = "dash"           # Dynamic Adaptive Streaming (MPEG)
    RTMP = "rtmp"           # Real-Time Messaging Protocol
    FILE = "file"           # File output only


class QualityProfile(str, Enum):
    """Pre-configured quality profiles for common use cases."""
    
    ULTRA_LOW_LATENCY = "ultra_low_latency"  # Minimal latency, lower quality
    LOW_BANDWIDTH = "low_bandwidth"          # 500kbps, optimized for slow connections
    MEDIUM = "medium"                        # 1.5Mbps, balanced quality/size
    HIGH = "high"                            # 3Mbps, high quality
    ULTRA_HIGH = "ultra_high"                # 6Mbps, maximum quality
    LOSSLESS = "lossless"                    # Lossless encoding
    # New high-quality profiles for local/LAN streaming
    LOCAL_HIGH = "local_high"                # 12Mbps, optimized for localhost/LAN
    LOCAL_4K = "local_4k"                    # 25Mbps, 4K quality for localhost/LAN
    STUDIO = "studio"                        # 50Mbps, near-lossless for production


@dataclass
class FFmpegConfig:
    """Configuration for FFmpeg video recording.
    
    Attributes:
        codec: Video codec to use
        quality_profile: Pre-configured quality profile
        output_path: Output file path (for FILE protocol)
        streaming_protocol: Streaming protocol (HLS, DASH, RTMP, FILE)
        streaming_url: Destination URL for streaming (RTMP only)
        width: Video width in pixels
        height: Video height in pixels
        frame_rate: Frames per second
        bitrate: Target bitrate (e.g., "1.5M", "500k")
        crf: Constant Rate Factor (0-51, lower is better quality)
        preset: Encoding preset (ultrafast, veryfast, fast, medium, slow, veryslow)
        enable_hw_accel: Enable hardware acceleration if available
        ffmpeg_path: Path to ffmpeg binary (auto-detected if None)
        pixel_format: Pixel format for encoding
        tune: Encoding tune (film, animation, grain, stillimage, zerolatency)
        keyframe_interval: Keyframe interval in seconds (for streaming)
        enable_adaptive_fps: Dynamically adjust FPS based on capture performance
        target_latency_ms: Target latency for streaming in milliseconds
    """
    
    codec: VideoCodec = VideoCodec.H264
    quality_profile: QualityProfile = QualityProfile.MEDIUM
    output_path: Optional[str] = None
    streaming_protocol: StreamingProtocol = StreamingProtocol.FILE
    streaming_url: Optional[str] = None
    
    # Video settings (720p by default for better performance)
    width: int = 1280
    height: int = 720
    frame_rate: int = 30
    
    # Encoding settings (overridden by quality_profile)
    bitrate: Optional[str] = None
    crf: Optional[int] = None
    preset: str = "ultrafast"  # Changed default for lower latency
    
    # Advanced options
    enable_hw_accel: bool = True
    ffmpeg_path: Optional[str] = None
    pixel_format: str = "yuv420p"
    tune: Optional[str] = "zerolatency"  # Default to low latency
    
    # Buffer settings - larger buffer for smoother streaming
    # For 4K at 30fps, each frame is ~24MB raw RGB, so 60 frames = ~1.4GB
    # We use smaller buffer and let FFmpeg handle backpressure
    frame_buffer_size: int = 60  # ~2 seconds at 30fps - smaller to reduce memory
    
    # Streaming optimizations
    keyframe_interval: int = 1  # 1s GOP - faster segment generation, better for live
    enable_adaptive_fps: bool = True  # Adapt FPS based on performance
    target_latency_ms: int = 2000  # 2 second target latency (more buffer room)
    
    # Logging control
    verbose_logging: bool = False  # Set to True for debug logging
    
    def __post_init__(self) -> None:
        """Apply quality profile settings."""
        self._apply_quality_profile()
        
        if not self.ffmpeg_path:
            self.ffmpeg_path = self._find_ffmpeg()
        
        if not self.output_path and self.streaming_protocol == StreamingProtocol.FILE:
            self.output_path = f"recording_{int(time.time())}.mp4"
    
    def _apply_quality_profile(self) -> None:
        """Apply pre-configured quality profile settings.
        
        Industry-standard settings based on:
        - YouTube/Twitch recommended encoding settings
        - Apple HLS authoring specification
        - FFmpeg wiki best practices
        - Netflix/Amazon professional encoding guidelines
        
        Bitrate formulas (for H.264):
        - SD (480p):  ~1.5-4 Mbps
        - HD (720p):  ~2.5-5 Mbps  
        - FHD (1080p): ~4-8 Mbps
        - 4K (2160p): ~15-45 Mbps
        
        CRF values (H.264/x264):
        - 18: Visually lossless
        - 20-22: High quality (recommended)
        - 23: Default, good balance
        - 28+: Lower quality, smaller files
        """
        profiles = {
            # === STREAMING PROFILES (optimized for real-time delivery) ===
            
            # Ultra Low Latency: For interactive/real-time applications
            # Use case: Remote desktop, live gaming, interactive broadcasts
            QualityProfile.ULTRA_LOW_LATENCY: {
                "bitrate": "1500k",     # 1.5 Mbps - adequate for screen content
                "crf": 28,               # Lower quality trade-off for speed
                "preset": "ultrafast",   # Fastest encoding, minimal latency
                "tune": "zerolatency",   # Remove lookahead, faster encoding
                "frame_rate": 30,        # 30fps for smooth motion
                "keyframe_interval": 1,  # 1s GOP for fast seek/recovery
            },
            
            # Low Bandwidth: For slow/mobile connections (3G, weak WiFi)
            # Use case: Mobile viewing, bandwidth-constrained environments
            QualityProfile.LOW_BANDWIDTH: {
                "bitrate": "800k",       # 800 Kbps - works on 1 Mbps connections
                "crf": 26,               # Acceptable quality at low bitrate
                "preset": "veryfast",    # Fast but allows some optimization
                "tune": "zerolatency",
                "keyframe_interval": 2,  # 2s GOP for efficiency
            },
            
            # Medium: Standard streaming quality (YouTube 720p equivalent)
            # Use case: General web streaming, video calls, screen sharing
            QualityProfile.MEDIUM: {
                "bitrate": "2500k",      # 2.5 Mbps - YouTube 720p standard
                "crf": 23,               # FFmpeg default, good balance
                "preset": "veryfast",    # Good speed/quality trade-off
                "tune": "zerolatency",
                "keyframe_interval": 2,
            },
            
            # High: HD streaming quality (YouTube 1080p equivalent)
            # Use case: High quality streams, presentations, demos
            QualityProfile.HIGH: {
                "bitrate": "5000k",      # 5 Mbps - YouTube 1080p standard
                "crf": 21,               # High quality
                "preset": "fast",        # Better compression, still real-time
                "tune": "zerolatency",
                "keyframe_interval": 2,
            },
            
            # Ultra High: Premium streaming (YouTube 1080p60 / 1440p equivalent)
            # Use case: High-motion content, gaming streams, professional demos
            QualityProfile.ULTRA_HIGH: {
                "bitrate": "8000k",      # 8 Mbps - YouTube 1080p60 recommended
                "crf": 19,               # Near-reference quality
                "preset": "fast",
                "tune": "zerolatency",
                "keyframe_interval": 2,
            },
            
            # === LOCAL/LAN PROFILES (optimized for localhost/LAN, no bandwidth limit) ===
            
            # Local High: Premium quality for localhost/LAN (1080p)
            # Use case: Local demos, LAN streaming, development
            QualityProfile.LOCAL_HIGH: {
                "bitrate": "15M",        # 15 Mbps - Netflix 1080p equivalent
                "crf": 17,               # Excellent quality
                "preset": "fast",        # Good compression with speed
                "tune": "zerolatency",
                "keyframe_interval": 1,  # Fast segment generation
            },
            
            # Local 4K: Full 4K quality for localhost/LAN
            # Use case: 4K demos, high-res screen recording, professional preview
            QualityProfile.LOCAL_4K: {
                "bitrate": "35M",        # 35 Mbps - Netflix 4K HDR equivalent
                "crf": 15,               # Reference quality
                "preset": "medium",      # Better compression for 4K
                "tune": "zerolatency",
                "keyframe_interval": 1,
            },
            
            # === PRODUCTION PROFILES (highest quality) ===
            
            # Studio: Near-lossless for production/archival
            # Use case: Master recordings, archival, post-production source
            QualityProfile.STUDIO: {
                "bitrate": "60M",        # 60 Mbps - broadcast quality
                "crf": 12,               # Near-lossless
                "preset": "slow",        # Maximum compression efficiency
                "tune": None,            # No tune for maximum quality
                "keyframe_interval": 1,
            },
            
            # Lossless: Mathematically lossless encoding
            # Use case: Source preservation, professional archival
            QualityProfile.LOSSLESS: {
                "bitrate": None,         # VBR, no bitrate limit
                "crf": 0,                # Lossless mode
                "preset": "ultrafast",   # Fast for lossless (preset doesn't affect quality)
                "tune": None,
                "keyframe_interval": 1,
            },
        }
        
        if self.quality_profile in profiles:
            profile = profiles[self.quality_profile]
            if self.bitrate is None:
                self.bitrate = profile["bitrate"]
            if self.crf is None:
                self.crf = profile["crf"]
            if not hasattr(self, "_preset_set"):
                self.preset = profile["preset"]
            if self.tune is None:
                self.tune = profile.get("tune")
            if "frame_rate" in profile and not hasattr(self, "_frame_rate_set"):
                self.frame_rate = profile["frame_rate"]
            if "keyframe_interval" in profile:
                self.keyframe_interval = profile["keyframe_interval"]
    
    @staticmethod
    def _find_ffmpeg() -> str:
        """Find ffmpeg binary in system PATH."""
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise BrowserError(
                "ffmpeg not found in PATH. Please install ffmpeg or provide ffmpeg_path."
            )
        return ffmpeg_path


@dataclass
class RecordingMetadata:
    """Metadata for a recording session."""
    
    recording_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    file_path: Optional[str] = None
    streaming_url: Optional[str] = None
    codec: str = ""
    width: int = 0
    height: int = 0
    frame_rate: int = 0
    total_frames: int = 0
    file_size_bytes: int = 0
    duration_seconds: float = 0.0


class BrowserCaptureStrategy:
    """Base class for browser-specific capture strategies.
    
    Each browser (Chromium, Firefox, WebKit) has different optimal capture methods:
    - Chromium: CDP Page.startScreencast (best - direct from render engine)
    - Firefox: Optimized screenshot loop (CDP removed in Firefox 141+)
    - WebKit: Optimized screenshot loop with WebKit-specific tuning
    """
    
    def __init__(self, recorder: 'FFmpegRecorder'):
        self.recorder = recorder
        self._running = False
    
    async def start(self) -> None:
        """Start capturing frames."""
        raise NotImplementedError
    
    async def stop(self) -> None:
        """Stop capturing frames."""
        self._running = False
    
    def get_latest_frame(self) -> Optional[bytes]:
        """Get the most recent captured frame."""
        with self.recorder._frame_lock:
            return self.recorder._latest_frame


class ChromiumCDPCapture(BrowserCaptureStrategy):
    """High-performance capture using Chrome DevTools Protocol screencast.
    
    CDP Page.startScreencast captures frames directly from Chrome's rendering
    engine, providing the best possible quality and performance:
    - Frames delivered as they're rendered (not screenshot latency)
    - Can achieve 60+ fps on fast systems
    - Minimal CPU overhead compared to screenshots
    - JPEG compression happens in browser (hardware accelerated on some systems)
    """
    
    def __init__(self, recorder: 'FFmpegRecorder'):
        super().__init__(recorder)
        self._cdp_session = None
    
    async def start(self) -> None:
        """Start CDP screencast with robust initialization."""
        self._running = True
        page = self.recorder._page
        config = self.recorder.config

        logger.info(
            f"[CAPTURE] Starting CDP screencast (Chromium)\n"
            f"  Resolution: {config.width}x{config.height}\n"
            f"  Frame Rate: {config.frame_rate} fps\n"
            f"  Method: Page.startScreencast (direct from render engine)"
        )

        # Initialize state variables
        self._last_frame_time = time.time()
        self._last_log_time = time.time()
        self._frame_count = 0
        self._consecutive_restarts = 0
        self._recovery_in_progress = False
        self._navigation_in_progress = False

        # Create CDP session
        self._cdp_session = await page.context.new_cdp_session(page)
        logger.debug("[CAPTURE] CDP session created")

        # Handle incoming frames
        self._cdp_session.on("Page.screencastFrame", self._on_frame)

        # CRITICAL FIX: Listen for navigation events
        # Chrome's Page.startScreencast STOPS sending frames after navigation
        # We must restart it after each navigation completes
        page.on("load", self._on_page_load)
        logger.debug("[CAPTURE] Navigation event listener registered")

        # Start screencast with optimal settings
        await self._cdp_session.send("Page.startScreencast", {
            "format": "jpeg",
            "quality": 92,  # High quality JPEG
            "maxWidth": config.width,
            "maxHeight": config.height,
            "everyNthFrame": 1,  # Every rendered frame
        })
        logger.debug("[CAPTURE] Screencast command sent")

        # Wait briefly for first frame to confirm it's working
        await asyncio.sleep(0.5)

        # Start watchdog
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())

        logger.info("[CAPTURE] CDP screencast started - frames will arrive asynchronously")

    def _on_page_load(self, *args) -> None:
        """Handle page load event - restart screencast after navigation.

        CRITICAL: Chrome's Page.startScreencast automatically stops when the page
        navigates. We must restart it after each navigation to keep frames flowing.
        """
        if not self._running:
            return

        try:
            page = self.recorder._page
            url = page.url if page else "unknown"
            logger.info(f"[CAPTURE] Page navigation detected: {url[:80]}")
            logger.info("[CAPTURE] Restarting screencast after navigation...")

            # Mark that we're handling navigation
            self._navigation_in_progress = True

            # Schedule screencast restart (must be async)
            loop = asyncio.get_event_loop()
            task = loop.create_task(self._restart_after_navigation())
            task.add_done_callback(self._navigation_restart_done)

        except Exception as e:
            logger.error(f"[CAPTURE] Error handling page load: {e}")
            self._navigation_in_progress = False

    async def _restart_after_navigation(self) -> None:
        """Restart screencast after navigation completes."""
        try:
            # Wait for page to stabilize after navigation
            await asyncio.sleep(0.3)

            # Restart the screencast
            await self._restart_screencast()
            logger.info("[CAPTURE] Screencast restarted after navigation")

        except Exception as e:
            logger.error(f"[CAPTURE] Failed to restart after navigation: {e}")

    def _navigation_restart_done(self, task: asyncio.Task) -> None:
        """Callback when navigation restart completes."""
        self._navigation_in_progress = False
        try:
            exc = task.exception()
            if exc:
                logger.error(f"[CAPTURE] Navigation restart task error: {exc}")
        except Exception:
            pass

    async def _watchdog_loop(self) -> None:
        """Monitor frame reception and restart screencast if stalled."""
        logger.debug("[CAPTURE] Watchdog started")
        # Reset counter on start
        self._consecutive_restarts = 0
        self._recovery_in_progress = False

        while self._running:
            try:
                await asyncio.sleep(0.5)  # Check more frequently for faster detection
                if not self._running:
                    break

                # Check if page is still open
                if self.recorder._page.is_closed():
                    logger.warning("[CAPTURE] Page closed, stopping capture")
                    self._running = False
                    break

                # Skip check if recovery or navigation is in progress
                if self._recovery_in_progress or self._navigation_in_progress:
                    continue

                # Check if we're receiving frames
                time_since_frame = time.time() - self._last_frame_time

                # If no frames for 2 seconds, attempt recovery (faster detection)
                if time_since_frame > 2.0:
                    self._recovery_in_progress = True
                    logger.warning(f"[CAPTURE] No frames for {time_since_frame:.1f}s, attempting recovery...")

                    try:
                        # Exponential backoff based on consecutive failures
                        if self._consecutive_restarts > 0:
                            backoff_delay = min(2 ** self._consecutive_restarts, 10)
                            logger.info(f"[CAPTURE] Waiting {backoff_delay}s before recovery attempt...")
                            await asyncio.sleep(backoff_delay)

                        # Try simple restart first
                        if self._consecutive_restarts < 2:
                            await self._restart_screencast()
                            self._consecutive_restarts += 1
                            logger.info(f"[CAPTURE] Restart attempt {self._consecutive_restarts}")
                        else:
                            # After 2 failed restarts, recreate the entire CDP session
                            logger.warning("[CAPTURE] Multiple restart failures, recreating CDP session...")
                            await self._recreate_session()
                            self._consecutive_restarts = 0

                        # Reset frame time after recovery attempt to avoid immediate re-trigger
                        self._last_frame_time = time.time()

                        # Wait for frames to arrive before declaring success
                        # Record the frame count before waiting
                        frames_before = self.recorder._metadata.total_frames
                        await asyncio.sleep(1.0)  # Shorter wait for faster feedback
                        frames_after = self.recorder._metadata.total_frames

                        # Check if frames actually started flowing (not just time-based)
                        if frames_after > frames_before:
                            logger.info(f"[CAPTURE] Recovery successful, frames flowing ({frames_after - frames_before} frames received)")
                            self._consecutive_restarts = 0
                        else:
                            logger.warning(f"[CAPTURE] Recovery attempt completed but no new frames received (before: {frames_before}, after: {frames_after})")
                            # Don't reset _last_frame_time on failure - let watchdog trigger again quickly

                    except Exception as e:
                        logger.error(f"[CAPTURE] Recovery failed: {e}")
                    finally:
                        self._recovery_in_progress = False

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[CAPTURE] Watchdog error: {e}")
                await asyncio.sleep(1.0)

    async def _restart_screencast(self) -> None:
        """Restart the screencast command with proper stop/start cycle."""
        if not self._cdp_session:
            return

        config = self.recorder.config
        try:
            # Properly stop the existing screencast first
            try:
                await self._cdp_session.send("Page.stopScreencast")
                logger.debug("[CAPTURE] Stopped existing screencast")
                # Brief pause to let the stop command take effect
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.debug(f"[CAPTURE] Stop screencast error (may not be running): {e}")

            # Start fresh screencast
            await self._cdp_session.send("Page.startScreencast", {
                "format": "jpeg",
                "quality": 92,
                "maxWidth": config.width,
                "maxHeight": config.height,
                "everyNthFrame": 1,
            })
            logger.info("[CAPTURE] Screencast restarted successfully")
        except Exception as e:
            logger.warning(f"[CAPTURE] Failed to restart screencast: {e}")
            raise

    async def _recreate_session(self) -> None:
        """Recreate the entire CDP session for deep recovery."""
        try:
            # Remove old navigation listener
            page = self.recorder._page
            try:
                if page and not page.is_closed():
                    page.remove_listener("load", self._on_page_load)
            except Exception:
                pass

            # Detach old session
            if self._cdp_session:
                try:
                    await self._cdp_session.send("Page.stopScreencast")
                except Exception:
                    pass
                try:
                    await self._cdp_session.detach()
                except Exception:
                    pass
                self._cdp_session = None

            # Brief pause to ensure cleanup
            await asyncio.sleep(0.2)

            # Create fresh CDP session
            self._cdp_session = await page.context.new_cdp_session(page)
            self._cdp_session.on("Page.screencastFrame", self._on_frame)
            logger.debug("[CAPTURE] New CDP session created")

            # Re-register navigation listener
            page.on("load", self._on_page_load)
            logger.debug("[CAPTURE] Navigation event listener re-registered")

            # Start screencast on new session
            config = self.recorder.config
            await self._cdp_session.send("Page.startScreencast", {
                "format": "jpeg",
                "quality": 92,
                "maxWidth": config.width,
                "maxHeight": config.height,
                "everyNthFrame": 1,
            })
            logger.info("[CAPTURE] CDP session recreated and screencast started")

        except Exception as e:
            logger.error(f"[CAPTURE] Failed to recreate session: {e}")
            raise
    
    def _on_frame(self, params: dict) -> None:
        """Process incoming screencast frame."""
        if not self._running:
            return

        try:
            r = self.recorder

            # Track last frame time for debugging
            if not hasattr(self, '_last_frame_time'):
                self._last_frame_time = time.time()
                self._frame_count = 0

            self._frame_count += 1
            current_time = time.time()
            self._last_frame_time = current_time

            # Successfully received frame - reset error counter and recovery flag
            if self._consecutive_restarts > 0:
                logger.info(f"[CAPTURE] Frame flow restored after {self._consecutive_restarts} recovery attempts")
            self._consecutive_restarts = 0
            if hasattr(self, '_recovery_in_progress'):
                self._recovery_in_progress = False

            # Log every 30 seconds to monitor health
            if current_time - self._last_log_time > 30:
                fps = self._frame_count / 30.0
                logger.info(f"[CAPTURE] Stream health: {self._frame_count} frames in 30s ({fps:.1f} fps)")
                self._last_log_time = current_time
                self._frame_count = 0

            # CRITICAL: Acknowledge FIRST before heavy processing
            # Chrome will stop sending frames if ack is delayed
            session_id = params.get("sessionId")
            if self._cdp_session and self._running and session_id is not None:
                try:
                    loop = asyncio.get_event_loop()
                    task = loop.create_task(self._ack(session_id))
                    task.add_done_callback(self._ack_done)
                except Exception as e:
                    logger.warning(f"[CAPTURE] Failed to schedule frame ack: {e}")

            # Decode JPEG
            jpeg_data = r._base64_module.b64decode(params["data"])
            img = r._pil_image.open(r._io_module.BytesIO(jpeg_data))

            # Resize if needed (usually not, CDP respects maxWidth/maxHeight)
            if img.size != (r.config.width, r.config.height):
                img = img.resize(
                    (r.config.width, r.config.height),
                    r._pil_image.Resampling.BILINEAR
                )

            # Convert to RGB24
            if img.mode != 'RGB':
                img = img.convert('RGB')

            raw_frame = img.tobytes()

            # Store frame
            with r._frame_lock:
                r._latest_frame = raw_frame
            r._metadata.total_frames += 1

        except Exception as e:
            logger.warning(f"[CAPTURE] Frame processing error: {e}")
    
    async def _ack(self, session_id: int) -> None:
        """Acknowledge frame receipt to Chrome."""
        try:
            if self._cdp_session and self._running:
                await self._cdp_session.send("Page.screencastFrameAck", {
                    "sessionId": session_id
                })
        except Exception as e:
            # Log but don't raise - ack failures shouldn't crash the stream
            logger.debug(f"[CAPTURE] Frame ack failed for session {session_id}: {e}")

    def _ack_done(self, task: asyncio.Task) -> None:
        """Callback for ack task completion."""
        try:
            # Check if task raised an exception
            exc = task.exception()
            if exc:
                logger.debug(f"[CAPTURE] Ack task error: {exc}")
        except Exception:
            pass
    
    async def stop(self) -> None:
        """Stop CDP screencast."""
        self._running = False

        # Remove navigation event listener
        try:
            page = self.recorder._page
            if page and not page.is_closed():
                page.remove_listener("load", self._on_page_load)
                logger.debug("[CAPTURE] Navigation event listener removed")
        except Exception as e:
            logger.debug(f"[CAPTURE] Error removing navigation listener: {e}")

        if hasattr(self, '_watchdog_task') and self._watchdog_task:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass

        if self._cdp_session:
            try:
                await self._cdp_session.send("Page.stopScreencast")
                await self._cdp_session.detach()
            except Exception:
                pass
            self._cdp_session = None


class FirefoxOptimizedCapture(BrowserCaptureStrategy):
    """Optimized capture for Firefox using high-speed screenshot loop.
    
    Firefox removed CDP support in version 141+, so we use an optimized
    screenshot approach with Firefox-specific tuning:
    - Parallel screenshot requests for higher throughput
    - JPEG format for speed (Firefox's JPEG encoder is fast)
    - Aggressive timing to maximize frame rate
    - Frame interpolation ready (future enhancement)
    """
    
    def __init__(self, recorder: 'FFmpegRecorder'):
        super().__init__(recorder)
        self._capture_task = None
    
    async def start(self) -> None:
        """Start optimized Firefox capture."""
        self._running = True
        config = self.recorder.config
        
        logger.info(
            f"[CAPTURE] Starting Firefox optimized capture\n"
            f"  Resolution: {config.width}x{config.height}\n"
            f"  Frame Rate: {config.frame_rate} fps\n"
            f"  Method: High-speed screenshot loop (CDP not available in Firefox 141+)"
        )
        
        self._capture_task = asyncio.create_task(self._capture_loop())
        logger.info("[CAPTURE] Firefox capture loop started")
    
    async def _capture_loop(self) -> None:
        """High-speed capture loop optimized for Firefox with robust error handling."""
        r = self.recorder
        page = r._page
        config = r.config

        # Firefox performs well with slightly lower JPEG quality
        # This significantly speeds up encoding
        jpeg_quality = 85

        consecutive_errors = 0
        last_success_time = time.time()
        recovery_attempts = 0
        max_recovery_attempts = 5

        try:
            while self._running:
                # Check if recorder is stopping
                if r._stopping:
                    logger.debug("[FIREFOX] Recorder stopping, exiting capture loop")
                    break

                # Check circuit breaker
                if hasattr(r, '_capture_circuit_breaker') and not r._capture_circuit_breaker.can_execute():
                    logger.warning("[FIREFOX] Circuit breaker open, waiting for recovery window...")
                    await asyncio.sleep(r._capture_circuit_breaker.recovery_timeout)
                    continue

                try:
                    # Check if page is still valid
                    if page.is_closed():
                        logger.warning("[FIREFOX] Page closed, attempting recovery...")
                        if hasattr(r, '_recovery_metrics'):
                            r._recovery_metrics.record_recovery_attempt(success=False)
                        # Wait for page to potentially be restored
                        await asyncio.sleep(1.0)
                        if page.is_closed():
                            recovery_attempts += 1
                            if recovery_attempts >= max_recovery_attempts:
                                logger.error("[FIREFOX] Page closed and recovery failed after max attempts")
                                break
                            # Exponential backoff
                            backoff = min(2 ** recovery_attempts, 30)
                            logger.info(f"[FIREFOX] Waiting {backoff}s before next recovery attempt...")
                            await asyncio.sleep(backoff)
                        continue

                    # Capture screenshot
                    screenshot_bytes = await page.screenshot(
                        type="jpeg",
                        quality=jpeg_quality,
                        full_page=False,
                    )

                    # Process frame
                    img = r._pil_image.open(r._io_module.BytesIO(screenshot_bytes))

                    if img.size != (config.width, config.height):
                        img = img.resize(
                            (config.width, config.height),
                            r._pil_image.Resampling.BILINEAR
                        )

                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    raw_frame = img.tobytes()

                    with r._frame_lock:
                        r._latest_frame = raw_frame
                    r._metadata.total_frames += 1

                    # Success - reset error counters
                    if consecutive_errors > 0:
                        logger.info(f"[FIREFOX] Capture recovered after {consecutive_errors} errors")
                        if hasattr(r, '_recovery_metrics'):
                            r._recovery_metrics.record_recovery_attempt(success=True)
                    consecutive_errors = 0
                    recovery_attempts = 0
                    last_success_time = time.time()

                    # Record success with circuit breaker
                    if hasattr(r, '_capture_circuit_breaker'):
                        r._capture_circuit_breaker.record_success()

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    consecutive_errors += 1

                    # Track failure in recovery metrics
                    if hasattr(r, '_recovery_metrics'):
                        r._recovery_metrics.record_failure()

                    # Record failure with circuit breaker
                    if hasattr(r, '_capture_circuit_breaker'):
                        r._capture_circuit_breaker.record_failure()

                    # Log with appropriate severity based on error count
                    if consecutive_errors == 1:
                        logger.debug(f"[FIREFOX] Capture error (transient): {e}")
                    elif consecutive_errors == 10:
                        logger.warning(f"[FIREFOX] Multiple capture errors ({consecutive_errors}): {e}")
                    elif consecutive_errors >= 20:
                        # Check if we've been failing for too long
                        time_since_success = time.time() - last_success_time
                        if time_since_success > 60:
                            logger.error(f"[FIREFOX] Capture failed for {time_since_success:.0f}s, attempting deep recovery...")
                            recovery_attempts += 1
                            if recovery_attempts >= max_recovery_attempts:
                                logger.error(f"[FIREFOX] Max recovery attempts reached, stopping capture")
                                if hasattr(r, '_recovery_metrics'):
                                    r._recovery_metrics.record_recovery_attempt(success=False)
                                break
                            # Reset error counter and try again with backoff
                            consecutive_errors = 0
                            backoff = min(2 ** recovery_attempts, 30)
                            logger.info(f"[FIREFOX] Recovery attempt {recovery_attempts}/{max_recovery_attempts}, waiting {backoff}s...")
                            await asyncio.sleep(backoff)
                            continue

                    # Exponential backoff for transient errors
                    backoff = min(0.05 * (2 ** min(consecutive_errors, 6)), 2.0)
                    await asyncio.sleep(backoff)

        except asyncio.CancelledError:
            logger.debug("[FIREFOX] Capture loop cancelled")
        except Exception as e:
            logger.error(f"[FIREFOX] Unexpected error in capture loop: {e}")
            if hasattr(r, '_recovery_metrics'):
                r._recovery_metrics.record_failure()
    
    async def stop(self) -> None:
        """Stop Firefox capture."""
        self._running = False
        if self._capture_task:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass


class WebKitOptimizedCapture(BrowserCaptureStrategy):
    """Optimized capture for WebKit (Safari engine).
    
    WebKit has unique characteristics:
    - No CDP support (uses Playwright's internal protocol)
    - PNG screenshots can be faster than JPEG on some WebKit builds
    - Requires careful timing to avoid frame drops
    """
    
    def __init__(self, recorder: 'FFmpegRecorder'):
        super().__init__(recorder)
        self._capture_task = None
    
    async def start(self) -> None:
        """Start optimized WebKit capture."""
        self._running = True
        config = self.recorder.config
        
        logger.info(
            f"[CAPTURE] Starting WebKit optimized capture\n"
            f"  Resolution: {config.width}x{config.height}\n"
            f"  Frame Rate: {config.frame_rate} fps\n"
            f"  Method: Screenshot loop with WebKit tuning"
        )
        
        self._capture_task = asyncio.create_task(self._capture_loop())
        logger.info("[CAPTURE] WebKit capture loop started")
    
    async def _capture_loop(self) -> None:
        """Capture loop optimized for WebKit with robust error handling."""
        r = self.recorder
        page = r._page
        config = r.config

        # WebKit sometimes performs better with PNG for certain content
        # But JPEG is generally faster for real-time capture
        use_jpeg = True
        jpeg_quality = 88

        consecutive_errors = 0
        recovery_attempts = 0
        max_recovery_attempts = 10
        base_backoff = 0.1
        max_backoff = 5.0

        try:
            while self._running:
                # Check circuit breaker before attempting capture
                if not r._capture_circuit_breaker.can_proceed():
                    logger.warning("[CAPTURE] WebKit capture circuit breaker is open, waiting for recovery")
                    await asyncio.sleep(r._capture_circuit_breaker.recovery_timeout)
                    continue

                try:
                    # Capture screenshot
                    if use_jpeg:
                        screenshot_bytes = await page.screenshot(
                            type="jpeg",
                            quality=jpeg_quality,
                            full_page=False,
                        )
                    else:
                        screenshot_bytes = await page.screenshot(
                            type="png",
                            full_page=False,
                        )

                    # Process frame
                    img = r._pil_image.open(r._io_module.BytesIO(screenshot_bytes))

                    if img.size != (config.width, config.height):
                        img = img.resize(
                            (config.width, config.height),
                            r._pil_image.Resampling.BILINEAR
                        )

                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    raw_frame = img.tobytes()

                    with r._frame_lock:
                        r._latest_frame = raw_frame
                    r._metadata.total_frames += 1

                    # Success - reset error counters
                    consecutive_errors = 0
                    recovery_attempts = 0
                    r._capture_circuit_breaker.record_success()

                except asyncio.CancelledError:
                    raise

                except Exception as e:
                    error_str = str(e).lower()
                    consecutive_errors += 1
                    r._recovery_metrics.record_failure()
                    r._capture_circuit_breaker.record_failure()

                    # Check for page closed/disconnected errors
                    is_page_closed = any(x in error_str for x in [
                        'target closed', 'page closed', 'context closed',
                        'browser closed', 'connection closed', 'target crashed',
                        'session closed', 'execution context was destroyed'
                    ])

                    if is_page_closed:
                        logger.error(f"[CAPTURE] WebKit page appears closed: {e}")
                        recovery_attempts += 1

                        if recovery_attempts >= max_recovery_attempts:
                            logger.error(f"[CAPTURE] WebKit max recovery attempts ({max_recovery_attempts}) reached, stopping capture")
                            break

                        # Exponential backoff
                        backoff = min(base_backoff * (2 ** recovery_attempts), max_backoff)
                        logger.info(f"[CAPTURE] WebKit waiting {backoff:.1f}s before recovery attempt {recovery_attempts}/{max_recovery_attempts}")
                        await asyncio.sleep(backoff)

                        # Try to get a fresh page reference
                        try:
                            if hasattr(r, '_page') and r._page:
                                page = r._page
                                logger.info("[CAPTURE] WebKit refreshed page reference")
                                consecutive_errors = 0
                                r._recovery_metrics.record_recovery()
                        except Exception as refresh_error:
                            logger.warning(f"[CAPTURE] WebKit failed to refresh page: {refresh_error}")

                        continue

                    # For other errors, use backoff strategy
                    if consecutive_errors >= 50:
                        logger.error(f"[CAPTURE] WebKit capture failed after {consecutive_errors} consecutive errors: {e}")
                        break
                    elif consecutive_errors >= 20:
                        # Log every 10 errors after 20
                        if consecutive_errors % 10 == 0:
                            logger.warning(f"[CAPTURE] WebKit capture error #{consecutive_errors}: {e}")
                        await asyncio.sleep(min(0.5, base_backoff * consecutive_errors))
                    elif consecutive_errors >= 5:
                        logger.debug(f"[CAPTURE] WebKit transient error #{consecutive_errors}: {e}")
                        await asyncio.sleep(0.1)
                    else:
                        await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            logger.debug("[CAPTURE] WebKit capture loop cancelled")
        except Exception as e:
            logger.error(f"[CAPTURE] WebKit capture loop failed unexpectedly: {e}")
            r._recovery_metrics.record_failure()
        finally:
            logger.info(f"[CAPTURE] WebKit capture loop ended. Total frames: {r._metadata.total_frames}")
    
    async def stop(self) -> None:
        """Stop WebKit capture."""
        self._running = False
        if self._capture_task:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass


class FFmpegRecorder:
    """Professional video recorder with browser-specific optimizations.
    
    Automatically selects the best capture strategy based on browser type:
    - Chromium: CDP Page.startScreencast (highest performance)
    - Firefox: Optimized screenshot loop (CDP removed in Firefox 141+)  
    - WebKit: Optimized screenshot loop with WebKit tuning
    
    Architecture:
    - Browser-specific capture strategy pushes frames asynchronously
    - Thread-based writer outputs frames at constant 30fps (or configured rate)
    - Frame duplication ensures smooth playback when capture is slower
    - FFmpeg encodes to HLS/DASH/RTMP with minimal latency
    
    Industry Best Practices:
    - Constant frame rate output (duplicates frames if needed)
    - Separate capture and encoding threads
    - Lock-free frame handoff where possible
    - Graceful degradation on slower systems
    
    Example:
        >>> config = FFmpegConfig(
        ...     quality_profile=QualityProfile.LOCAL_HIGH,
        ...     streaming_protocol=StreamingProtocol.HLS,
        ...     output_path="stream/playlist.m3u8"
        ... )
        >>> recorder = FFmpegRecorder(config)
        >>> await recorder.start(page)  # Auto-detects browser type
        >>> # ... browser actions ...
        >>> metadata = await recorder.stop()
    """
    
    def __init__(self, config: FFmpegConfig) -> None:
        """Initialize the recorder."""
        import queue
        import threading

        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._recording = False
        self._stopping = False  # Flag to prevent recovery during shutdown
        self._capture_strategy: Optional[BrowserCaptureStrategy] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._page: Optional[Page] = None
        self._browser_type: str = "unknown"
        self._metadata = RecordingMetadata(
            codec=config.codec.value,
            width=config.width,
            height=config.height,
            frame_rate=config.frame_rate,
        )

        # Thread-safe frame storage
        self._latest_frame: Optional[bytes] = None
        self._frame_lock = threading.Lock()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=config.frame_buffer_size)
        self._dropped_frames = 0

        # Pre-import modules (populated in start())
        self._pil_image = None
        self._io_module = None
        self._base64_module = None

        # Recovery infrastructure
        self._recovery_metrics = RecoveryMetrics()
        self._ffmpeg_circuit_breaker = CircuitBreaker(
            name="ffmpeg",
            failure_threshold=3,
            recovery_timeout=10.0,
        )
        self._capture_circuit_breaker = CircuitBreaker(
            name="capture",
            failure_threshold=5,
            recovery_timeout=5.0,
        )

        # FFmpeg monitoring
        self._ffmpeg_monitor_task: Optional[asyncio.Task] = None
        self._ffmpeg_stderr_lines: List[str] = []
        self._last_ffmpeg_health_check: float = 0.0
        self._ffmpeg_restart_in_progress = False
        self._start_time: float = 0.0

        # Graceful degradation
        self._degraded_mode = False
        self._skip_frames_count = 0
        self._max_skip_frames = 10  # Skip up to 10 frames before recovery attempt
    
    def _detect_browser_type(self) -> str:
        """Detect the browser type from the page context."""
        try:
            # Playwright stores browser info in the context
            browser = self._page.context.browser
            if browser:
                browser_type = browser.browser_type.name
                return browser_type.lower()
        except Exception:
            pass
        return "chromium"  # Default fallback
    
    def _get_capture_strategy_name(self) -> str:
        """Get the name of capture strategy for the detected browser."""
        browser = self._browser_type
        if browser == "chromium":
            return "ChromiumCDPCapture (Page.startScreencast)"
        elif browser == "firefox":
            return "FirefoxOptimizedCapture (screenshot loop)"
        elif browser == "webkit":
            return "WebKitOptimizedCapture (screenshot loop)"
        else:
            return f"ChromiumCDPCapture (fallback for {browser})"
    
    def _create_capture_strategy(self) -> BrowserCaptureStrategy:
        """Create the optimal capture strategy for the detected browser."""
        browser = self._browser_type
        
        if browser == "chromium":
            logger.info("[RECORDER] Using ChromiumCDPCapture (best performance)")
            return ChromiumCDPCapture(self)
        elif browser == "firefox":
            logger.info("[RECORDER] Using FirefoxOptimizedCapture (CDP not available)")
            return FirefoxOptimizedCapture(self)
        elif browser == "webkit":
            logger.info("[RECORDER] Using WebKitOptimizedCapture")
            return WebKitOptimizedCapture(self)
        else:
            logger.warning(f"[RECORDER] Unknown browser '{browser}', trying CDP capture")
            return ChromiumCDPCapture(self)
    
    async def start(self, page: Page) -> str:
        """Start recording with automatic browser detection.

        Args:
            page: Playwright page to record

        Returns:
            Recording ID
        """
        import threading
        import base64

        if self._recording:
            raise BrowserError("Recording already in progress")

        self._page = page
        self._recording = True
        self._stopping = False
        self._start_time = time.time()

        # Check disk space before starting
        if self.config.output_path:
            output_dir = os.path.dirname(self.config.output_path) or "."
            if not check_disk_space(output_dir):
                self._recording = False
                raise DiskSpaceError(f"Insufficient disk space in {output_dir}")

        # Pre-import modules
        from PIL import Image
        import io
        self._pil_image = Image
        self._io_module = io
        self._base64_module = base64

        # Detect browser type
        self._browser_type = self._detect_browser_type()
        logger.info(
            f"[RECORDER] Browser detected: {self._browser_type}\n"
            f"  Capture strategy: {self._get_capture_strategy_name()}"
        )

        try:
            # Start FFmpeg process
            await self._start_ffmpeg_process()

            # Start FFmpeg health monitor
            self._ffmpeg_monitor_task = asyncio.create_task(self._monitor_ffmpeg_health())

            # Create and start browser-specific capture strategy
            self._capture_strategy = self._create_capture_strategy()
            try:
                await self._capture_strategy.start()
            except Exception as e:
                # If preferred strategy fails, fallback to Firefox strategy
                # (works for all browsers via screenshots)
                logger.warning(f"Primary capture failed ({e}), using fallback")
                self._capture_strategy = FirefoxOptimizedCapture(self)
                await self._capture_strategy.start()

            logger.info(
                f"Recording started: {self._metadata.recording_id} "
                f"({self.config.width}x{self.config.height}@{self.config.frame_rate}fps, "
                f"{self._browser_type})"
            )
            return self._metadata.recording_id

        except Exception as e:
            self._recording = False
            self._stopping = True
            if self._ffmpeg_monitor_task:
                self._ffmpeg_monitor_task.cancel()
            if self._process:
                self._process.kill()
            raise BrowserError(f"Failed to start recording: {e}") from e

    async def _start_ffmpeg_process(self) -> None:
        """Start the FFmpeg process with error handling."""
        import threading

        # Build and start FFmpeg
        cmd = self._build_ffmpeg_command()
        logger.info(f"[RECORDER] FFmpeg command: {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError:
            raise BrowserError("FFmpeg not found. Please install FFmpeg.")
        except PermissionError as e:
            raise UnrecoverableError(f"Permission denied starting FFmpeg: {e}")
        except OSError as e:
            if e.errno == errno.ENOSPC:
                raise DiskSpaceError("No space left on device")
            raise BrowserError(f"Failed to start FFmpeg: {e}")

        # Start writer thread
        self._writer_thread = threading.Thread(
            target=self._writer_thread_func,
            name="ffmpeg-writer",
            daemon=True
        )
        self._writer_thread.start()

        # Reset circuit breaker on successful start
        self._ffmpeg_circuit_breaker.reset()
        logger.debug("[RECORDER] FFmpeg process started successfully")

    async def _restart_ffmpeg_process(self) -> bool:
        """Restart FFmpeg process after a crash.

        Returns:
            True if restart was successful
        """
        if self._stopping or not self._recording:
            return False

        if self._ffmpeg_restart_in_progress:
            logger.debug("[RECORDER] FFmpeg restart already in progress")
            return False

        if not self._ffmpeg_circuit_breaker.can_execute():
            logger.error("[RECORDER] FFmpeg circuit breaker is open, cannot restart")
            return False

        self._ffmpeg_restart_in_progress = True
        logger.warning("[RECORDER] Attempting to restart FFmpeg process...")

        try:
            # Clean up old process
            if self._process:
                try:
                    if self._process.stdin:
                        self._process.stdin.close()
                    self._process.kill()
                    self._process.wait(timeout=2)
                except Exception:
                    pass
                self._process = None

            # Wait for writer thread to finish
            if self._writer_thread and self._writer_thread.is_alive():
                # Give it a moment to notice the process is gone
                await asyncio.sleep(0.5)

            # Start new FFmpeg process
            await self._start_ffmpeg_process()

            self._ffmpeg_circuit_breaker.record_success()
            self._recovery_metrics.record_recovery_attempt(True, "ffmpeg")
            logger.info("[RECORDER] FFmpeg process restarted successfully")
            return True

        except Exception as e:
            self._ffmpeg_circuit_breaker.record_failure()
            self._recovery_metrics.record_recovery_attempt(False, "ffmpeg")
            logger.error(f"[RECORDER] Failed to restart FFmpeg: {e}")
            return False
        finally:
            self._ffmpeg_restart_in_progress = False

    async def _monitor_ffmpeg_health(self) -> None:
        """Monitor FFmpeg process health and trigger recovery if needed."""
        logger.debug("[RECORDER] FFmpeg health monitor started")

        try:
            while self._recording and not self._stopping:
                await asyncio.sleep(2.0)  # Check every 2 seconds

                if not self._recording or self._stopping:
                    break

                # Check if FFmpeg process is still running
                if self._process:
                    poll_result = self._process.poll()
                    if poll_result is not None:
                        # Process has terminated
                        exit_code = poll_result

                        # Read stderr for error details
                        stderr_output = ""
                        try:
                            if self._process.stderr:
                                stderr_output = self._process.stderr.read().decode('utf-8', errors='ignore')
                                if stderr_output:
                                    self._ffmpeg_stderr_lines.extend(stderr_output.split('\n')[-20:])
                        except Exception:
                            pass

                        logger.error(
                            f"[RECORDER] FFmpeg process terminated unexpectedly "
                            f"(exit code: {exit_code})"
                        )
                        if stderr_output:
                            logger.error(f"[RECORDER] FFmpeg stderr: {stderr_output[-500:]}")

                        # Check if error is recoverable
                        if "No space left" in stderr_output or "disk full" in stderr_output.lower():
                            logger.error("[RECORDER] Disk full - unrecoverable error")
                            self._recording = False
                            break

                        # Attempt restart
                        if self._recording and not self._stopping:
                            success = await self._restart_ffmpeg_process()
                            if not success:
                                logger.error("[RECORDER] FFmpeg restart failed, stopping recording")
                                self._recording = False
                                break

                # Check disk space periodically
                if self.config.output_path:
                    output_dir = os.path.dirname(self.config.output_path) or "."
                    if not check_disk_space(output_dir, min_bytes=50 * 1024 * 1024):
                        logger.error("[RECORDER] Low disk space, stopping recording")
                        self._recording = False
                        break

                # Update uptime
                self._recovery_metrics.uptime_seconds = time.time() - self._start_time

        except asyncio.CancelledError:
            logger.debug("[RECORDER] FFmpeg health monitor cancelled")
        except Exception as e:
            logger.error(f"[RECORDER] FFmpeg health monitor error: {e}")
    
    async def stop(self) -> RecordingMetadata:
        """Stop recording and return metadata."""
        if not self._recording and self._stopping:
            return self._metadata

        # Signal stop - prevent recovery attempts during shutdown
        self._stopping = True
        self._recording = False

        logger.info("[RECORDER] Stopping recording...")

        # Stop FFmpeg health monitor
        if self._ffmpeg_monitor_task:
            self._ffmpeg_monitor_task.cancel()
            try:
                await asyncio.wait_for(self._ffmpeg_monitor_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            self._ffmpeg_monitor_task = None

        # Stop capture strategy
        if self._capture_strategy:
            try:
                await self._capture_strategy.stop()
            except Exception as e:
                logger.debug(f"[RECORDER] Error stopping capture strategy: {e}")
            self._capture_strategy = None

        # Legacy: Wait for capture task (fallback mode)
        if hasattr(self, '_capture_task') and self._capture_task:
            try:
                await asyncio.wait_for(self._capture_task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._capture_task.cancel()

        # Wait for writer thread
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=5.0)

        # Close FFmpeg gracefully
        if self._process:
            try:
                if self._process.stdin:
                    self._process.stdin.close()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("[RECORDER] FFmpeg did not exit gracefully, killing...")
                self._process.kill()
                self._process.wait()
            except Exception as e:
                logger.debug(f"[RECORDER] Error closing FFmpeg: {e}")

        # Update metadata
        self._metadata.ended_at = time.time()
        self._metadata.duration_seconds = self._metadata.ended_at - self._metadata.started_at
        self._recovery_metrics.uptime_seconds = self._metadata.duration_seconds

        if self.config.output_path and os.path.exists(self.config.output_path):
            self._metadata.file_path = self.config.output_path
            self._metadata.file_size_bytes = os.path.getsize(self.config.output_path)

        actual_fps = self._metadata.total_frames / max(self._metadata.duration_seconds, 0.1)

        logger.info(
            f"Recording stopped: {self._metadata.recording_id} "
            f"({self._metadata.total_frames} frames, {self._metadata.duration_seconds:.1f}s, "
            f"{actual_fps:.1f} captured fps, {self._dropped_frames} dropped, "
            f"recoveries: {self._recovery_metrics.successful_recoveries}/{self._recovery_metrics.total_recovery_attempts})"
        )

        return self._metadata

    def get_recovery_metrics(self) -> Dict[str, Any]:
        """Get recovery metrics for monitoring.

        Returns:
            Dictionary with recovery statistics
        """
        return self._recovery_metrics.to_dict()
    
    def _build_ffmpeg_command(self) -> List[str]:
        """Build FFmpeg command optimized per quality profile.
        
        LOCAL profiles (LOCAL_HIGH, LOCAL_4K, STUDIO) get ultra-low latency settings.
        Other profiles get standard streaming settings for internet delivery.
        """
        cmd = [self.config.ffmpeg_path, "-y"]
        
        # Determine if this is a local/LAN profile requiring ultra-low latency
        is_local_profile = self.config.quality_profile in [
            QualityProfile.LOCAL_HIGH,
            QualityProfile.LOCAL_4K,
            QualityProfile.STUDIO,
        ]
        
        # Input: raw RGB24 video from pipe
        cmd.extend([
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.config.width}x{self.config.height}",
            "-r", str(self.config.frame_rate),
            "-i", "pipe:0",
        ])
        
        # Encoding: H.264 settings
        encoder = self.config.codec.get_encoder(self.config.enable_hw_accel)
        preset = self.config.preset
        tune = self.config.tune
        
        cmd.extend([
            "-c:v", encoder,
            "-preset", preset,
            "-pix_fmt", self.config.pixel_format,
        ])
        
        # Add tune if specified
        if tune:
            cmd.extend(["-tune", tune])
        
        # Scene detection and keyframes - LOCAL profiles get more aggressive settings
        if is_local_profile:
            # Ultra-low latency: frequent keyframes, no scene detection
            keyframe_interval = 0.5  # 500ms
            cmd.extend([
                "-sc_threshold", "0",  # Disable scene detection
                "-force_key_frames", f"expr:gte(t,n_forced*{keyframe_interval})",
            ])
        else:
            # Standard streaming: normal keyframe interval
            keyframe_interval = self.config.keyframe_interval  # Usually 1-2s
            cmd.extend([
                "-force_key_frames", f"expr:gte(t,n_forced*{keyframe_interval})",
            ])
        
        # Quality: use CRF for VBR or bitrate for CBR
        if self.config.crf is not None:
            cmd.extend(["-crf", str(self.config.crf)])
        if self.config.bitrate:
            cmd.extend(["-b:v", self.config.bitrate])
        
        # GOP size based on keyframe interval
        gop = max(int(self.config.frame_rate * keyframe_interval), 15)
        cmd.extend(["-g", str(gop)])
        
        # Output format
        if self.config.streaming_protocol == StreamingProtocol.HLS:
            output_dir = os.path.dirname(self.config.output_path) if self.config.output_path else "."
            segment_pattern = os.path.join(output_dir, "segment%d.ts") if output_dir else "segment%d.ts"

            # HLS segment duration: LOCAL profiles get 1s for better stability, others get 2s
            hls_time = "1" if is_local_profile else "2"
            # Use delete_segments to prevent disk fill, independent_segments for better seeking
            hls_flags = "independent_segments+delete_segments+split_by_time" if is_local_profile else "independent_segments+delete_segments"

            cmd.extend([
                "-f", "hls",
                "-hls_time", hls_time,
                "-hls_list_size", "10",          # Keep last 10 segments (rolling window)
                "-hls_flags", hls_flags,
                "-hls_segment_type", "mpegts",   # Use MPEG-TS
                "-hls_segment_filename", segment_pattern,
                "-hls_allow_cache", "0",         # Disable caching for live streams
                "-start_number", "0",            # Start segment numbering at 0
            ])
            output = self.config.output_path or "playlist.m3u8"
            
        elif self.config.streaming_protocol == StreamingProtocol.DASH:
            cmd.extend([
                "-f", "dash",
                "-seg_duration", "1",
                "-window_size", "5",
            ])
            output = self.config.output_path or "manifest.mpd"
            
        elif self.config.streaming_protocol == StreamingProtocol.RTMP:
            cmd.extend(["-f", "flv", "-an"])
            output = self.config.streaming_url or "rtmp://localhost/live/stream"
            
        else:  # FILE
            cmd.extend(["-movflags", "+faststart"])
            output = self.config.output_path or "output.mp4"
        
        cmd.append(output)
        return cmd
    
    def _writer_thread_func(self) -> None:
        """Write frames to FFmpeg at constant frame rate.

        This ensures FFmpeg receives exactly 30fps (or configured rate)
        by duplicating the latest frame when no new frame is available.
        This is the industry-standard approach for variable-rate input.

        Enhanced with recovery support:
        - Waits for FFmpeg restart instead of stopping on pipe errors
        - Supports graceful degradation (skip frames)
        - Tracks write failures for metrics
        """
        frames_written = 0
        frame_interval = 1.0 / self.config.frame_rate
        next_write_time = time.monotonic()
        consecutive_write_errors = 0
        max_consecutive_errors = 10  # Allow some errors before degrading

        # Wait for first frame
        while self._recording and not self._stopping and self._latest_frame is None:
            time.sleep(0.01)

        while (self._recording or frames_written < 5) and not self._stopping:
            try:
                # Check if we're in the middle of an FFmpeg restart
                if self._ffmpeg_restart_in_progress:
                    # Wait for restart to complete, but keep the thread alive
                    logger.debug("[WRITER] Waiting for FFmpeg restart...")
                    time.sleep(0.1)
                    # Reset timing after restart
                    next_write_time = time.monotonic()
                    consecutive_write_errors = 0
                    continue

                # Get latest frame (thread-safe)
                with self._frame_lock:
                    frame_data = self._latest_frame

                if frame_data is None:
                    time.sleep(0.01)
                    continue

                # Check if we should skip frames (graceful degradation)
                if self._degraded_mode and self._skip_frames_count > 0:
                    self._skip_frames_count -= 1
                    next_write_time += frame_interval
                    continue

                # Precise timing for constant frame rate
                now = time.monotonic()
                sleep_time = next_write_time - now
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Write frame to FFmpeg
                if self._process and self._process.stdin:
                    try:
                        self._process.stdin.write(frame_data)
                        self._process.stdin.flush()
                        frames_written += 1
                        consecutive_write_errors = 0  # Reset on success

                        # Exit degraded mode after consistent success
                        if self._degraded_mode and frames_written % 100 == 0:
                            self._degraded_mode = False
                            logger.info("[WRITER] Exiting degraded mode after stable writes")

                    except (BrokenPipeError, OSError) as e:
                        consecutive_write_errors += 1
                        self._recovery_metrics.record_failure("ffmpeg_write")

                        if consecutive_write_errors == 1:
                            logger.warning(f"[WRITER] FFmpeg pipe error: {e}")

                        # Check if we should stop trying
                        if self._stopping:
                            break

                        # Enter degraded mode after multiple errors
                        if consecutive_write_errors >= max_consecutive_errors:
                            if not self._degraded_mode:
                                self._degraded_mode = True
                                self._skip_frames_count = 5  # Skip next 5 frames
                                logger.warning("[WRITER] Entering degraded mode due to write errors")

                        # Wait a bit for recovery - the monitor task will restart FFmpeg
                        if consecutive_write_errors >= 3:
                            time.sleep(0.1)

                        # Don't break - wait for FFmpeg restart from monitor task
                        continue
                else:
                    # No process available - wait for restart
                    if not self._stopping and self._recording:
                        time.sleep(0.1)
                        continue

                # Schedule next frame
                next_write_time += frame_interval

                # If we fell behind, reset timing (don't try to catch up)
                if time.monotonic() > next_write_time + frame_interval * 2:
                    next_write_time = time.monotonic()

            except Exception as e:
                logger.error(f"[WRITER] Thread error: {e}")
                if self._stopping:
                    break
                # Don't break on transient errors - try to continue
                time.sleep(0.1)
                continue

        # Final flush
        if self._process and self._process.stdin:
            try:
                self._process.stdin.flush()
            except Exception:
                pass

        logger.debug(f"[WRITER] Thread finished: {frames_written} frames written")
    
    async def _capture_frames_fallback(self) -> None:
        """Fallback capture using page.screenshot() when CDP screencast unavailable.
        
        Captures as fast as possible - writer thread handles constant output rate.
        """
        Image = self._pil_image
        io_module = self._io_module
        consecutive_errors = 0
        
        try:
            while self._recording:
                try:
                    # Capture screenshot (JPEG for speed)
                    screenshot_bytes = await self._page.screenshot(
                        type="jpeg",
                        quality=85,
                        full_page=False
                    )
                    
                    # Convert to raw RGB24
                    img = Image.open(io_module.BytesIO(screenshot_bytes))
                    
                    # Resize if needed
                    if img.size != (self.config.width, self.config.height):
                        img = img.resize(
                            (self.config.width, self.config.height),
                            Image.Resampling.BILINEAR
                        )
                    
                    # Convert to RGB
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    raw_frame = img.tobytes()
                    
                    # Store as latest frame
                    with self._frame_lock:
                        self._latest_frame = raw_frame
                    
                    self._metadata.total_frames += 1
                    consecutive_errors = 0
                    
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors >= 15:
                        logger.error(f"Too many capture errors: {e}")
                        break
                    await asyncio.sleep(0.05)
                    continue
                    
        except asyncio.CancelledError:
            pass
    
    @property
    def is_recording(self) -> bool:
        """Check if recording is in progress."""
        return self._recording
    
    @property
    def metadata(self) -> RecordingMetadata:
        """Get current recording metadata."""
        return self._metadata

    @property
    def dropped_frames(self) -> int:
        """Get count of dropped frames.

        Returns:
            Number of frames that were dropped due to queue overflow
        """
        return self._dropped_frames

    @property
    def frame_queue_size(self) -> int:
        """Get current frame queue size.

        Returns:
            Number of frames currently in the queue
        """
        return self._frame_queue.qsize() if self._frame_queue else 0

    @property
    def frame_buffer_capacity(self) -> int:
        """Get frame buffer capacity.

        Returns:
            Maximum number of frames the buffer can hold
        """
        return self._config.frame_buffer_size

    @property
    def buffer_health_percent(self) -> float:
        """Get buffer health as a percentage.

        Returns:
            Buffer health from 0-100, where 100 means buffer is empty (healthy)
            and 0 means buffer is full (unhealthy, frames being dropped)
        """
        if not self._frame_queue or self._config.frame_buffer_size == 0:
            return 100.0
        # Health is inverse of fill level - empty buffer = 100% health
        fill_ratio = self._frame_queue.qsize() / self._config.frame_buffer_size
        return max(0.0, min(100.0, (1.0 - fill_ratio) * 100.0))

    @property
    def is_degraded(self) -> bool:
        """Check if recorder is in degraded mode.

        Degraded mode means the recorder is skipping frames to recover
        from write errors or performance issues.

        Returns:
            True if recorder is in degraded mode
        """
        return self._degraded_mode

    @property
    def degradation_info(self) -> Dict[str, Any]:
        """Get detailed degradation status information.

        Returns:
            Dictionary with degradation details including mode, skip count,
            and recovery metrics
        """
        return {
            "is_degraded": self._degraded_mode,
            "frames_to_skip": self._skip_frames_count,
            "max_skip_frames": self._max_skip_frames,
            "buffer_health": self.buffer_health_percent,
            "dropped_frames": self._dropped_frames,
            "recovery_metrics": self._recovery_metrics.to_dict(),
        }

    def enter_degraded_mode(self, skip_frames: int = 5) -> None:
        """Manually enter degraded mode.

        Use this when external monitoring detects issues that require
        reducing load on the encoder.

        Args:
            skip_frames: Number of frames to skip (default: 5)
        """
        if not self._degraded_mode:
            self._degraded_mode = True
            self._skip_frames_count = min(skip_frames, self._max_skip_frames)
            logger.warning(
                f"[RECORDER] Entering degraded mode, skipping {self._skip_frames_count} frames"
            )

    def exit_degraded_mode(self) -> None:
        """Manually exit degraded mode.

        Use this when external monitoring confirms issues are resolved.
        """
        if self._degraded_mode:
            self._degraded_mode = False
            self._skip_frames_count = 0
            logger.info("[RECORDER] Exiting degraded mode")


def detect_hardware_acceleration() -> Dict[str, bool]:
    """Detect available hardware acceleration options.
    
    Returns:
        Dictionary with hardware acceleration availability
    """
    hw_accels = ["cuda", "videotoolbox", "qsv", "dxva2", "vaapi"]
    available = {}
    
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        for hw in hw_accels:
            available[hw] = hw in result.stdout
            
    except Exception as e:
        logger.warning(f"Failed to detect hardware acceleration: {e}")
        available = {hw: False for hw in hw_accels}
    
    return available


def get_recommended_config(
    quality: QualityProfile = QualityProfile.MEDIUM,
    resolution: Tuple[int, int] = (1280, 720),
) -> FFmpegConfig:
    """Get recommended configuration for common use cases.
    
    Args:
        quality: Quality profile to use
        resolution: Video resolution (width, height)
        
    Returns:
        Pre-configured FFmpegConfig
    """
    width, height = resolution
    
    # Auto-detect hardware acceleration
    hw_available = detect_hardware_acceleration()
    enable_hw = any(hw_available.values())
    
    return FFmpegConfig(
        codec=VideoCodec.H264,
        quality_profile=quality,
        width=width,
        height=height,
        enable_hw_accel=enable_hw,
    )
