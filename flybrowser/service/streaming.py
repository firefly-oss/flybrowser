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
Live Streaming Infrastructure for FlyBrowser.

This module provides real-time streaming capabilities for browser sessions:
- HLS (HTTP Live Streaming) with adaptive bitrate
- DASH (Dynamic Adaptive Streaming)
- RTMP relay for streaming platforms
- WebSocket-based live streaming
- Stream health monitoring and analytics
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from playwright.async_api import Page

from flybrowser.core.ffmpeg_recorder import (
    FFmpegConfig,
    FFmpegRecorder,
    QualityProfile,
    StreamingProtocol,
    VideoCodec,
)
from flybrowser.utils.logger import logger


class StreamState(str, Enum):
    """State of a streaming session."""
    
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class StreamHealth(str, Enum):
    """Health status of a stream."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class StreamMetrics:
    """Metrics for a streaming session."""
    
    frames_sent: int = 0
    bytes_sent: int = 0
    dropped_frames: int = 0
    current_bitrate: float = 0.0  # bps
    average_bitrate: float = 0.0  # bps
    current_fps: float = 0.0
    viewer_count: int = 0
    buffer_health: float = 100.0  # percentage
    last_update: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frames_sent": self.frames_sent,
            "bytes_sent": self.bytes_sent,
            "dropped_frames": self.dropped_frames,
            "current_bitrate": self.current_bitrate,
            "average_bitrate": self.average_bitrate,
            "current_fps": self.current_fps,
            "viewer_count": self.viewer_count,
            "buffer_health": self.buffer_health,
            "last_update": self.last_update,
        }


@dataclass
class StreamConfig:
    """Configuration for a streaming session.
    
    For localhost/LAN streaming, use LOCAL_HIGH or LOCAL_4K profiles
    with higher bitrates and PNG capture for best quality.
    
    Resolution presets:
    - 720p:  1280x720  (default)
    - 1080p: 1920x1080
    - 1440p: 2560x1440
    - 4K:    3840x2160
    """
    
    protocol: StreamingProtocol = StreamingProtocol.HLS
    quality_profile: QualityProfile = QualityProfile.HIGH  # Default to HIGH for better quality
    codec: VideoCodec = VideoCodec.H264
    width: int = 1920   # 1080p default for modern displays
    height: int = 1080
    frame_rate: int = 30
    enable_hw_accel: bool = True
    
    # HLS/DASH specific - shorter segments for lower latency
    segment_duration: int = 1  # 1 second for lower latency (was 2)
    playlist_size: int = 4     # smaller for lower latency (was 6)
    
    # RTMP specific
    rtmp_url: Optional[str] = None
    rtmp_key: Optional[str] = None
    
    # Stream limits
    max_viewers: int = 100
    max_duration_seconds: int = 3600  # 1 hour
    
    # Adaptive bitrate
    enable_abr: bool = False  # Adaptive BitRate
    abr_profiles: List[QualityProfile] = field(
        default_factory=lambda: [
            QualityProfile.LOW_BANDWIDTH,
            QualityProfile.MEDIUM,
            QualityProfile.HIGH,
        ]
    )
    
    # Logging control
    verbose_logging: bool = False  # Enable verbose logging for debugging


@dataclass
class StreamInfo:
    """Information about an active stream."""
    
    stream_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    node_id: str = ""
    state: StreamState = StreamState.INITIALIZING
    health: StreamHealth = StreamHealth.HEALTHY
    protocol: StreamingProtocol = StreamingProtocol.HLS
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    
    # Stream URLs
    hls_url: Optional[str] = None
    dash_url: Optional[str] = None
    rtmp_url: Optional[str] = None
    websocket_url: Optional[str] = None
    player_url: Optional[str] = None  # Embedded web player URL
    
    # Configuration
    config: StreamConfig = field(default_factory=StreamConfig)
    
    # Metrics
    metrics: StreamMetrics = field(default_factory=StreamMetrics)
    
    # Viewers
    viewer_ids: Set[str] = field(default_factory=set)

    # Recovery metrics from recorder
    recovery_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stream_id": self.stream_id,
            "session_id": self.session_id,
            "node_id": self.node_id,
            "state": self.state.value,
            "health": self.health.value,
            "protocol": self.protocol.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "hls_url": self.hls_url,
            "dash_url": self.dash_url,
            "rtmp_url": self.rtmp_url,
            "websocket_url": self.websocket_url,
            "player_url": self.player_url,
            "metrics": self.metrics.to_dict(),
            "viewer_count": len(self.viewer_ids),
            "uptime_seconds": time.time() - self.started_at if self.state == StreamState.ACTIVE else 0,
            "recovery_metrics": self.recovery_metrics,
        }


class StreamingSession:
    """Individual streaming session managing a single stream.
    
    Handles the lifecycle of a stream including:
    - FFmpeg process management
    - HLS/DASH playlist generation
    - RTMP relay
    - Metrics collection
    - Viewer management
    """
    
    def __init__(
        self,
        session_id: str,
        config: StreamConfig,
        output_dir: str,
        base_url: Optional[str] = None,
    ) -> None:
        """Initialize streaming session.
        
        Args:
            session_id: Browser session ID
            config: Stream configuration
            output_dir: Directory for stream output
            base_url: Base URL for accessing streams
        """
        self.session_id = session_id
        self.config = config
        self.output_dir = Path(output_dir)
        self.base_url = base_url or "http://localhost:8000"
        
        # Stream info (create first to get stream_id)
        self.info = StreamInfo(
            session_id=session_id,
            protocol=config.protocol,
            config=config,
        )
        
        # Create stream directory using stream_id (not session_id)
        self.stream_dir = self.output_dir / self.info.stream_id
        self.stream_dir.mkdir(parents=True, exist_ok=True)
        
        # FFmpeg recorder
        self._recorder: Optional[FFmpegRecorder] = None
        self._page: Optional[Page] = None

        # Monitoring
        self._metrics_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None

        # Viewer tracking
        self._viewer_connections: Dict[str, asyncio.Queue] = {}

        # Logging control
        self._verbose = config.verbose_logging

        # Recovery infrastructure
        self._stopping = False  # Flag to prevent recovery during shutdown
        self._recovery_attempts = 0
        self._max_recovery_attempts = 5
        self._last_recovery_time = 0.0
        self._recovery_backoff = 1.0  # Initial backoff in seconds
        self._max_recovery_backoff = 60.0  # Maximum backoff
        self._recovery_task: Optional[asyncio.Task] = None
        self._recovery_lock = asyncio.Lock()  # Prevent concurrent recovery attempts
        self._consecutive_failures = 0
        self._last_successful_frame_time = 0.0

        # Graceful degradation tracking
        self._degraded_since: float = 0.0  # Time when degraded state started
        self._total_degraded_time: float = 0.0  # Cumulative time in degraded state
        self._degradation_count: int = 0  # Number of degradation events

        if self._verbose:
            logger.info(f"StreamingSession created: {self.info.stream_id}")
    
    async def start(self, page: Page) -> StreamInfo:
        """Start the stream.
        
        Args:
            page: Playwright page to stream
            
        Returns:
            StreamInfo with stream details
        """
        self._page = page
        self.info.state = StreamState.INITIALIZING
        
        # Log stream initialization details
        logger.info(
            f"[STREAM] Initializing stream {self.info.stream_id}\n"
            f"  Session: {self.session_id}\n"
            f"  Protocol: {self.config.protocol.value}\n"
            f"  Quality: {self.config.quality_profile.value}\n"
            f"  Resolution: {self.config.width}x{self.config.height}\n"
            f"  Frame Rate: {self.config.frame_rate} fps\n"
            f"  Output Dir: {self.stream_dir}"
        )
        
        try:
            # Build FFmpeg configuration
            logger.info(f"[STREAM] Building FFmpeg configuration...")
            ffmpeg_config = self._build_ffmpeg_config()
            logger.info(
                f"[STREAM] FFmpeg config ready:\n"
                f"  Codec: {ffmpeg_config.codec.value}\n"
                f"  CRF: {ffmpeg_config.crf}\n"
                f"  Preset: {ffmpeg_config.preset}\n"
                f"  Output: {ffmpeg_config.output_path}"
            )
            
            # Create recorder
            logger.info(f"[STREAM] Creating FFmpegRecorder...")
            self._recorder = FFmpegRecorder(ffmpeg_config)
            
            # Start recording/streaming
            logger.info(f"[STREAM] Starting capture (page: {page})...")
            await self._recorder.start(page)
            logger.info(f"[STREAM] Capture started successfully")
            
            # Update stream info
            self._set_stream_urls()
            logger.info(
                f"[STREAM] URLs configured:\n"
                f"  HLS: {self.info.hls_url}\n"
                f"  Player: {self.info.player_url}"
            )
            
            # Wait for first HLS segment to be created before declaring stream active
            if self.config.protocol == StreamingProtocol.HLS:
                logger.info(f"[STREAM] Waiting for first HLS segment...")
                segment_ready = await self._wait_for_first_segment()
                if segment_ready:
                    logger.info(f"[STREAM] First HLS segment ready - stream is live!")
                else:
                    logger.warning(f"[STREAM] First segment timeout - stream may have issues")
            
            self.info.state = StreamState.ACTIVE
            
            # Start monitoring tasks
            self._metrics_task = asyncio.create_task(self._monitor_metrics())
            self._health_task = asyncio.create_task(self._monitor_health())
            
            logger.info(
                f"[STREAM] Stream ACTIVE: {self.info.stream_id}\n"
                f"  Play URL: {self.info.player_url or self.info.hls_url}"
            )
            return self.info
            
        except Exception as e:
            self.info.state = StreamState.ERROR
            self.info.health = StreamHealth.UNHEALTHY
            logger.error(f"[STREAM] Failed to start stream: {e}")
            import traceback
            logger.error(f"[STREAM] Traceback: {traceback.format_exc()}")
            raise
    
    async def _wait_for_first_segment(self, timeout: float = 10.0) -> bool:
        """Wait for the first HLS segment to be created.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if segment was found, False if timeout
        """
        playlist_path = self.stream_dir / "playlist.m3u8"
        start_time = time.time()
        check_interval = 0.1  # Check every 100ms
        
        while time.time() - start_time < timeout:
            # Check if playlist exists and has at least one segment reference
            if playlist_path.exists():
                try:
                    content = playlist_path.read_text()
                    # Look for .ts segment in playlist
                    if ".ts" in content and "#EXTINF" in content:
                        # Also verify at least one segment file exists
                        for line in content.split("\n"):
                            if line.endswith(".ts"):
                                segment_path = self.stream_dir / line.strip()
                                if segment_path.exists():
                                    if self._verbose:
                                        logger.debug(f"First segment ready: {line.strip()}")
                                    return True
                except Exception:
                    pass
            
            await asyncio.sleep(check_interval)
        
        logger.warning(f"Timeout waiting for first HLS segment after {timeout}s")
        return False
    
    async def stop(self) -> StreamInfo:
        """Stop the stream.

        Returns:
            Final StreamInfo with metrics
        """
        if self.info.state == StreamState.STOPPED:
            return self.info

        # Set stopping flag to prevent recovery attempts during shutdown
        self._stopping = True
        self.info.state = StreamState.STOPPED
        self.info.ended_at = time.time()

        # Cancel recovery task if running
        if self._recovery_task and not self._recovery_task.done():
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                pass

        # Stop monitoring tasks
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Stop recorder with robust error handling
        if self._recorder:
            try:
                await self._recorder.stop()
            except Exception as e:
                logger.error(f"Error stopping recorder: {e}")
                import traceback
                logger.debug(f"Recorder stop traceback: {traceback.format_exc()}")

        # Notify viewers
        try:
            await self._notify_viewers_stream_ended()
        except Exception as e:
            logger.debug(f"Error notifying viewers: {e}")

        # Log final stats with recovery info
        uptime = self.info.ended_at - self.info.started_at
        logger.info(
            f"Stream stopped: {self.info.stream_id} "
            f"(uptime: {uptime:.1f}s, frames: {self.info.metrics.frames_sent}, "
            f"viewers: {len(self.info.viewer_ids)}, "
            f"recovery_attempts: {self._recovery_attempts})"
        )
        return self.info

    async def _attempt_recovery(self) -> bool:
        """Attempt to recover the stream after a failure.

        Returns:
            True if recovery was successful, False otherwise
        """
        # Don't attempt recovery if we're stopping
        if self._stopping:
            logger.debug(f"Skipping recovery - stream is stopping")
            return False

        # Use lock to prevent concurrent recovery attempts
        async with self._recovery_lock:
            # Double-check after acquiring lock
            if self._stopping or self.info.state == StreamState.STOPPED:
                return False

            # Check if we've exceeded max recovery attempts
            if self._recovery_attempts >= self._max_recovery_attempts:
                logger.error(
                    f"[STREAM] Max recovery attempts ({self._max_recovery_attempts}) exceeded "
                    f"for stream {self.info.stream_id} - giving up"
                )
                self.info.state = StreamState.ERROR
                self.info.health = StreamHealth.UNHEALTHY
                return False

            # Apply exponential backoff
            current_time = time.time()
            time_since_last = current_time - self._last_recovery_time
            if time_since_last < self._recovery_backoff:
                wait_time = self._recovery_backoff - time_since_last
                logger.info(f"[STREAM] Waiting {wait_time:.1f}s before recovery attempt...")
                await asyncio.sleep(wait_time)

            self._recovery_attempts += 1
            self._last_recovery_time = time.time()

            logger.info(
                f"[STREAM] Recovery attempt {self._recovery_attempts}/{self._max_recovery_attempts} "
                f"for stream {self.info.stream_id}"
            )

            try:
                # Stop current recorder if it exists
                if self._recorder:
                    try:
                        await self._recorder.stop()
                    except Exception as e:
                        logger.debug(f"Error stopping recorder during recovery: {e}")

                # Check if we still have a valid page
                if not self._page:
                    logger.error("[STREAM] Cannot recover - no page reference")
                    return False

                # Check if page is still connected
                try:
                    # Try to access page to see if it's still valid
                    if self._page.is_closed():
                        logger.error("[STREAM] Cannot recover - page is closed")
                        return False
                except Exception as e:
                    logger.error(f"[STREAM] Cannot recover - page check failed: {e}")
                    return False

                # Rebuild FFmpeg config and create new recorder
                logger.info(f"[STREAM] Recreating FFmpeg recorder...")
                ffmpeg_config = self._build_ffmpeg_config()
                self._recorder = FFmpegRecorder(ffmpeg_config)

                # Start recording again
                await self._recorder.start(self._page)

                # Wait briefly to confirm it's working
                await asyncio.sleep(1.0)

                if self._recorder.is_recording:
                    logger.info(
                        f"[STREAM] Recovery successful for stream {self.info.stream_id} "
                        f"(attempt {self._recovery_attempts})"
                    )
                    self.info.state = StreamState.ACTIVE
                    self.info.health = StreamHealth.HEALTHY
                    self._consecutive_failures = 0
                    self._last_successful_frame_time = time.time()

                    # Increase backoff for next attempt (exponential)
                    self._recovery_backoff = min(
                        self._recovery_backoff * 2,
                        self._max_recovery_backoff
                    )
                    return True
                else:
                    logger.warning(f"[STREAM] Recovery failed - recorder not recording")
                    return False

            except Exception as e:
                logger.error(f"[STREAM] Recovery failed with error: {e}")
                import traceback
                logger.debug(f"Recovery traceback: {traceback.format_exc()}")

                # Increase backoff for next attempt
                self._recovery_backoff = min(
                    self._recovery_backoff * 2,
                    self._max_recovery_backoff
                )
                return False
    
    async def pause(self) -> bool:
        """Pause the stream."""
        if self.info.state == StreamState.ACTIVE:
            self.info.state = StreamState.PAUSED
            if self._verbose:
                logger.info(f"Stream paused: {self.info.stream_id}")
            return True
        return False
    
    async def resume(self) -> bool:
        """Resume the stream."""
        if self.info.state == StreamState.PAUSED:
            self.info.state = StreamState.ACTIVE
            if self._verbose:
                logger.info(f"Stream resumed: {self.info.stream_id}")
            return True
        return False
    
    async def add_viewer(self, viewer_id: str) -> bool:
        """Add a viewer to the stream.
        
        Args:
            viewer_id: Unique viewer identifier
            
        Returns:
            True if viewer was added
        """
        if len(self.info.viewer_ids) >= self.config.max_viewers:
            if self._verbose:
                logger.warning(f"Stream at max viewers: {self.info.stream_id}")
            return False
        
        self.info.viewer_ids.add(viewer_id)
        self.info.metrics.viewer_count = len(self.info.viewer_ids)
        
        # Create queue for viewer if WebSocket
        if self.config.protocol == StreamingProtocol.HLS:
            self._viewer_connections[viewer_id] = asyncio.Queue()
        
        if self._verbose:
            logger.debug(f"Viewer {viewer_id[:8]}... joined stream {self.info.stream_id[:8]}...")
        return True
    
    async def remove_viewer(self, viewer_id: str) -> bool:
        """Remove a viewer from the stream.
        
        Args:
            viewer_id: Viewer identifier
            
        Returns:
            True if viewer was removed
        """
        if viewer_id in self.info.viewer_ids:
            self.info.viewer_ids.remove(viewer_id)
            self.info.metrics.viewer_count = len(self.info.viewer_ids)
            
            if viewer_id in self._viewer_connections:
                del self._viewer_connections[viewer_id]
            
            if self._verbose:
                logger.debug(f"Viewer {viewer_id[:8]}... left stream {self.info.stream_id[:8]}...")
            return True
        return False
    
    def _build_ffmpeg_config(self) -> FFmpegConfig:
        """Build FFmpeg configuration for streaming."""
        output_path = None
        streaming_url = None
        
        if self.config.protocol == StreamingProtocol.HLS:
            output_path = str(self.stream_dir / "playlist.m3u8")
        elif self.config.protocol == StreamingProtocol.DASH:
            output_path = str(self.stream_dir / "manifest.mpd")
        elif self.config.protocol == StreamingProtocol.RTMP:
            if self.config.rtmp_url:
                streaming_url = self.config.rtmp_url
                if self.config.rtmp_key:
                    streaming_url = f"{streaming_url}/{self.config.rtmp_key}"
        
        return FFmpegConfig(
            codec=self.config.codec,
            quality_profile=self.config.quality_profile,
            output_path=output_path,
            streaming_protocol=self.config.protocol,
            streaming_url=streaming_url,
            width=self.config.width,
            height=self.config.height,
            frame_rate=self.config.frame_rate,
            enable_hw_accel=self.config.enable_hw_accel,
            verbose_logging=self.config.verbose_logging,
        )
    
    def _set_stream_urls(self) -> None:
        """Set stream URLs based on protocol."""
        base = self.base_url.rstrip("/")
        stream_path = f"/streams/{self.info.stream_id}"
        
        if self.config.protocol == StreamingProtocol.HLS:
            self.info.hls_url = f"{base}{stream_path}/playlist.m3u8"
        elif self.config.protocol == StreamingProtocol.DASH:
            self.info.dash_url = f"{base}{stream_path}/manifest.mpd"
        elif self.config.protocol == StreamingProtocol.RTMP:
            self.info.rtmp_url = self.config.rtmp_url
        
        # WebSocket URL (for live updates)
        self.info.websocket_url = f"ws://{base.replace('http://', '')}{stream_path}/ws"
        
        # Embedded web player URL (for HLS and DASH only)
        if self.config.protocol in [StreamingProtocol.HLS, StreamingProtocol.DASH]:
            self.info.player_url = f"{base}{stream_path}/player"
    
    async def _monitor_metrics(self) -> None:
        """Monitor stream metrics silently."""
        last_check = time.time()
        last_frames = 0
        
        try:
            while self.info.state == StreamState.ACTIVE:
                await asyncio.sleep(3)  # Update every 3 seconds (reduced frequency)
                
                if not self._recorder:
                    continue
                
                # Get current metrics from recorder
                metadata = self._recorder.metadata
                current_time = time.time()
                elapsed = current_time - last_check
                
                if elapsed > 0:
                    # Calculate rates
                    frame_delta = metadata.total_frames - last_frames
                    self.info.metrics.current_fps = frame_delta / elapsed

                    # Update metrics
                    self.info.metrics.frames_sent = metadata.total_frames
                    self.info.metrics.last_update = current_time

                    # Update dropped frames and buffer health from recorder
                    self.info.metrics.dropped_frames = self._recorder.dropped_frames
                    self.info.metrics.buffer_health = self._recorder.buffer_health_percent

                    # Calculate bytes_sent from segment files (HLS .ts files)
                    try:
                        total_bytes = sum(
                            f.stat().st_size
                            for f in self.stream_dir.glob("*.ts")
                            if f.is_file()
                        )
                        self.info.metrics.bytes_sent = total_bytes
                    except Exception:
                        pass  # Ignore errors in bytes calculation

                    # Update recovery metrics from recorder
                    self.info.recovery_metrics = self._recorder.get_recovery_metrics()

                    # Update averages
                    total_time = current_time - self.info.started_at
                    if total_time > 0:
                        self.info.metrics.average_bitrate = (
                            self.info.metrics.bytes_sent * 8 / total_time
                        )

                    last_check = current_time
                    last_frames = metadata.total_frames
                
        except asyncio.CancelledError:
            pass  # Clean exit
    
    async def _monitor_health(self) -> None:
        """Monitor stream health with automatic recovery on failures."""
        warmup_period = 15.0  # Longer warmup to avoid false positives
        start_time = time.time()
        last_health_log = 0.0
        last_stats_log = 0.0  # For comprehensive periodic stats logging
        stats_log_interval = 60.0  # Log comprehensive stats every 60 seconds
        recovery_in_progress = False

        # HLS segment stall detection
        last_segment_mtime: float = 0.0
        segment_stall_threshold = 30.0  # Seconds without new segment = stall

        try:
            while self.info.state == StreamState.ACTIVE:
                await asyncio.sleep(10)  # Check every 10 seconds (reduced frequency)

                # Don't perform health checks during shutdown
                if self._stopping:
                    break

                prev_health = self.info.health

                # Check if recovery task completed
                if self._recovery_task is not None and self._recovery_task.done():
                    try:
                        recovery_success = self._recovery_task.result()
                        recovery_in_progress = False
                        if recovery_success:
                            logger.info(f"Stream {self.info.stream_id}: Recovery completed successfully")
                            self._consecutive_failures = 0
                            self._last_successful_frame_time = time.time()
                            # Reset warmup after recovery
                            start_time = time.time()
                        else:
                            self._consecutive_failures += 1
                            logger.warning(
                                f"Stream {self.info.stream_id}: Recovery failed "
                                f"(consecutive failures: {self._consecutive_failures})"
                            )
                    except Exception as e:
                        recovery_in_progress = False
                        self._consecutive_failures += 1
                        logger.error(f"Stream {self.info.stream_id}: Recovery task error: {e}")
                    finally:
                        self._recovery_task = None

                # Check if recording is still active
                if self._recorder and not self._recorder.is_recording:
                    self.info.health = StreamHealth.UNHEALTHY

                    if prev_health != StreamHealth.UNHEALTHY:
                        logger.warning(f"Stream {self.info.stream_id}: Recorder stopped unexpectedly")

                    # Trigger recovery if not already in progress and not stopping
                    if not recovery_in_progress and not self._stopping:
                        if self._recovery_attempts < self._max_recovery_attempts:
                            logger.info(
                                f"Stream {self.info.stream_id}: Initiating automatic recovery "
                                f"(attempt {self._recovery_attempts + 1}/{self._max_recovery_attempts})"
                            )
                            recovery_in_progress = True
                            self._recovery_task = asyncio.create_task(self._attempt_recovery())
                        else:
                            logger.error(
                                f"Stream {self.info.stream_id}: Max recovery attempts "
                                f"({self._max_recovery_attempts}) exceeded, stopping stream"
                            )
                            # Stop the stream - max recovery attempts exceeded
                            self.info.state = StreamState.ERROR
                            break
                    continue

                # Update last successful frame time when recorder is active
                if self._recorder and self._recorder.is_recording:
                    self._last_successful_frame_time = time.time()

                # Skip checks during warmup period
                elapsed = time.time() - start_time
                if elapsed < warmup_period:
                    self.info.health = StreamHealth.HEALTHY
                    continue

                # Check FPS (only if we have enough frames)
                if self.info.metrics.frames_sent > 60:  # ~2 seconds of frames
                    min_fps = self.config.frame_rate * 0.2  # 20% threshold (more lenient)
                    if self.info.metrics.current_fps < min_fps:
                        self.info.health = StreamHealth.DEGRADED
                        # Enter degraded mode on recorder to skip frames
                        if self._recorder and not self._recorder.is_degraded:
                            self._recorder.enter_degraded_mode(skip_frames=3)
                            self._degraded_since = time.time()
                            self._degradation_count += 1
                        # Only log health changes, not every check
                        current_time = time.time()
                        if self._verbose and current_time - last_health_log > 30:
                            logger.debug(
                                f"Stream {self.info.stream_id}: Degraded FPS "
                                f"{self.info.metrics.current_fps:.1f} (min: {min_fps:.1f})"
                            )
                            last_health_log = current_time
                        continue

                # Check buffer health
                if self.info.metrics.buffer_health < 30:  # More lenient threshold
                    self.info.health = StreamHealth.DEGRADED
                    # Enter degraded mode on recorder with more aggressive frame skipping
                    if self._recorder and not self._recorder.is_degraded:
                        self._recorder.enter_degraded_mode(skip_frames=5)
                        self._degraded_since = time.time()
                        self._degradation_count += 1
                    continue

                # HLS segment stall detection
                if self.config.protocol == StreamingProtocol.HLS:
                    try:
                        # Find latest segment file modification time
                        segment_files = list(self.stream_dir.glob("*.ts"))
                        segment_count = len(segment_files)
                        if segment_files:
                            current_latest_mtime = max(
                                f.stat().st_mtime for f in segment_files
                            )
                            if last_segment_mtime == 0.0:
                                # First check, initialize
                                last_segment_mtime = current_latest_mtime
                                logger.debug(
                                    f"Stream {self.info.stream_id}: HLS initialized with "
                                    f"{segment_count} segments"
                                )
                            elif current_latest_mtime > last_segment_mtime:
                                # New segment created, update tracker
                                last_segment_mtime = current_latest_mtime
                                # Log segment progress periodically
                                if segment_count % 10 == 0:
                                    logger.debug(
                                        f"Stream {self.info.stream_id}: HLS segments: {segment_count}"
                                    )
                            else:
                                stall_duration = current_time - last_segment_mtime
                                if stall_duration > segment_stall_threshold:
                                    # No new segments for too long - stall detected
                                    logger.warning(
                                        f"Stream {self.info.stream_id}: HLS segment stall detected - "
                                        f"no new segments for {stall_duration:.1f}s, "
                                        f"last segment count: {segment_count}, "
                                        f"recorder recording: {self._recorder.is_recording if self._recorder else 'N/A'}"
                                    )
                                    self.info.health = StreamHealth.UNHEALTHY
                                    self._consecutive_failures += 1
                                    if self._consecutive_failures >= 2:
                                        logger.info(
                                            f"Stream {self.info.stream_id}: Triggering recovery "
                                            f"due to HLS segment stall"
                                        )
                                        asyncio.create_task(self._attempt_recovery())
                                    continue
                        else:
                            # No segments at all after warmup period
                            if elapsed > warmup_period:
                                logger.warning(
                                    f"Stream {self.info.stream_id}: No HLS segments found in {self.stream_dir}"
                                )
                    except Exception as e:
                        logger.debug(
                            f"Stream {self.info.stream_id}: Error checking HLS segments: {e}"
                        )

                # All checks passed
                self.info.health = StreamHealth.HEALTHY
                self._consecutive_failures = 0  # Reset on healthy state

                # Exit degraded mode if active
                if self._recorder and self._recorder.is_degraded:
                    self._recorder.exit_degraded_mode()
                    if self._degraded_since > 0:
                        self._total_degraded_time += time.time() - self._degraded_since
                        self._degraded_since = 0.0
                        logger.info(
                            f"Stream {self.info.stream_id}: Exited degraded mode, "
                            f"total degraded time: {self._total_degraded_time:.1f}s"
                        )

                # Reset recovery backoff after 5 minutes of sustained healthy operation
                if (self._recovery_attempts > 0 and
                    time.time() - self._last_recovery_time > 300):  # 5 minutes
                    logger.info(
                        f"Stream {self.info.stream_id}: Resetting recovery counters after "
                        f"5 minutes of stable operation"
                    )
                    self._recovery_attempts = 0
                    self._recovery_backoff = 1.0

                # Comprehensive periodic stats logging
                current_time = time.time()
                if current_time - last_stats_log >= stats_log_interval:
                    last_stats_log = current_time
                    uptime = current_time - self.info.started_at

                    # Get recorder metrics if available
                    recorder_metrics = {}
                    if self._recorder:
                        recorder_metrics = self._recorder.get_metrics()

                    # Calculate failure rate
                    total_recovery_attempts = self._recovery_attempts
                    recovery_success_rate = 0.0
                    if total_recovery_attempts > 0:
                        # Successful recoveries = attempts - consecutive failures still active
                        successful = max(0, total_recovery_attempts - self._consecutive_failures)
                        recovery_success_rate = (successful / total_recovery_attempts) * 100

                    logger.info(
                        f"Stream {self.info.stream_id} stats: "
                        f"uptime={uptime:.0f}s, "
                        f"health={self.info.health.name}, "
                        f"frames_sent={self.info.metrics.frames_sent}, "
                        f"dropped_frames={self.info.metrics.dropped_frames}, "
                        f"fps={self.info.metrics.current_fps:.1f}, "
                        f"buffer_health={self.info.metrics.buffer_health:.0f}%, "
                        f"recovery_attempts={self._recovery_attempts}, "
                        f"recovery_success_rate={recovery_success_rate:.0f}%, "
                        f"degradation_count={self._degradation_count}, "
                        f"total_degraded_time={self._total_degraded_time:.1f}s, "
                        f"ffmpeg_restarts={recorder_metrics.get('ffmpeg_restarts', 0)}"
                    )

        except asyncio.CancelledError:
            pass  # Clean exit
        except Exception as e:
            logger.error(f"Stream {self.info.stream_id}: Health monitor error: {e}")
        finally:
            # Clean up any pending recovery task
            if self._recovery_task is not None and not self._recovery_task.done():
                self._recovery_task.cancel()
                try:
                    await self._recovery_task
                except asyncio.CancelledError:
                    pass
    
    async def _notify_viewers_stream_ended(self) -> None:
        """Notify all viewers that stream has ended."""
        for viewer_queue in self._viewer_connections.values():
            try:
                await viewer_queue.put({"type": "stream_ended", "stream_id": self.info.stream_id})
            except Exception:
                pass  # Viewer may already be disconnected
    
    async def get_playlist_content(self) -> Optional[str]:
        """Get HLS playlist content.
        
        Returns:
            Playlist content if HLS stream
        """
        if self.config.protocol != StreamingProtocol.HLS:
            return None
        
        playlist_path = self.stream_dir / "playlist.m3u8"
        if not playlist_path.exists():
            return None
        
        try:
            return await asyncio.to_thread(playlist_path.read_text)
        except Exception:
            return None  # File may be being written
    
    async def get_segment(self, segment_name: str) -> Optional[bytes]:
        """Get HLS segment data.
        
        Args:
            segment_name: Segment file name
            
        Returns:
            Segment data if exists
        """
        segment_path = self.stream_dir / segment_name
        if not segment_path.exists():
            return None
        
        try:
            return await asyncio.to_thread(segment_path.read_bytes)
        except Exception:
            return None  # File may be being written

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics including uptime, failure rates, and degradation stats.

        Returns:
            Dictionary containing all streaming metrics for monitoring and debugging.
        """
        import time
        current_time = time.time()
        uptime = current_time - self.info.started_at if self.info.started_at > 0 else 0.0

        # Calculate recovery success rate
        total_recovery_attempts = self._recovery_attempts
        recovery_success_rate = 0.0
        if total_recovery_attempts > 0:
            successful = max(0, total_recovery_attempts - self._consecutive_failures)
            recovery_success_rate = (successful / total_recovery_attempts) * 100

        # Get recorder metrics if available
        recorder_metrics = {}
        if self._recorder:
            recorder_metrics = self._recorder.get_metrics()

        return {
            # Basic info
            "stream_id": self.info.stream_id,
            "state": self.info.state.name,
            "health": self.info.health.name,
            "protocol": self.info.protocol.value if self.info.protocol else None,

            # Uptime and timing
            "uptime_seconds": uptime,
            "started_at": self.info.started_at,

            # Frame metrics
            "frames_sent": self.info.metrics.frames_sent,
            "dropped_frames": self.info.metrics.dropped_frames,
            "bytes_sent": self.info.metrics.bytes_sent,
            "current_fps": self.info.metrics.current_fps,
            "buffer_health": self.info.metrics.buffer_health,

            # Recovery metrics
            "recovery_attempts": self._recovery_attempts,
            "consecutive_failures": self._consecutive_failures,
            "recovery_success_rate": recovery_success_rate,

            # Degradation metrics
            "is_degraded": self._degraded_since > 0,
            "degradation_count": self._degradation_count,
            "total_degraded_time_seconds": self._total_degraded_time,
            "current_degraded_duration": (
                current_time - self._degraded_since if self._degraded_since > 0 else 0.0
            ),

            # FFmpeg/recorder metrics
            "ffmpeg_restarts": recorder_metrics.get("ffmpeg_restarts", 0),
            "recorder_degraded": recorder_metrics.get("is_degraded", False),
            "recorder_metrics": recorder_metrics,
        }


class StreamingManager:
    """Manager for multiple streaming sessions.
    
    Coordinates streaming across the system:
    - Creates and manages streaming sessions
    - Tracks active streams
    - Handles viewer connections
    - Provides stream discovery
    - Enforces resource limits
    """
    
    def __init__(
        self,
        output_dir: str = "./streams",
        base_url: Optional[str] = None,
        max_concurrent_streams: int = 10,
    ) -> None:
        """Initialize streaming manager.
        
        Args:
            output_dir: Directory for stream output
            base_url: Base URL for accessing streams
            max_concurrent_streams: Maximum concurrent streams
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url
        self.max_concurrent_streams = max_concurrent_streams
        
        # Active streams
        self._streams: Dict[str, StreamingSession] = {}
        self._stream_lock = asyncio.Lock()
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(f"StreamingManager initialized (max streams: {max_concurrent_streams})")
    
    async def start(self) -> None:
        """Start the streaming manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("StreamingManager started")
    
    async def stop(self) -> None:
        """Stop the streaming manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Stop all active streams
        async with self._stream_lock:
            for stream in list(self._streams.values()):
                try:
                    await stream.stop()
                except Exception as e:
                    logger.error(f"Error stopping stream: {e}")
            self._streams.clear()
        
        logger.info("StreamingManager stopped")
    
    async def create_stream(
        self,
        session_id: str,
        page: Page,
        config: StreamConfig,
    ) -> StreamInfo:
        """Create and start a new stream.
        
        Args:
            session_id: Browser session ID
            page: Playwright page to stream
            config: Stream configuration
            
        Returns:
            StreamInfo for the new stream
            
        Raises:
            ValueError: If too many concurrent streams
        """
        logger.info(
            f"[STREAM_MGR] Creating stream for session {session_id}\n"
            f"  Protocol: {config.protocol.value}\n"
            f"  Quality: {config.quality_profile.value}\n"
            f"  Resolution: {config.width}x{config.height}@{config.frame_rate}fps\n"
            f"  Output Dir: {self.output_dir}\n"
            f"  Base URL: {self.base_url}"
        )
        
        async with self._stream_lock:
            if len(self._streams) >= self.max_concurrent_streams:
                logger.error(f"[STREAM_MGR] Max streams reached ({self.max_concurrent_streams})")
                raise ValueError(
                    f"Maximum concurrent streams reached ({self.max_concurrent_streams})"
                )
            
            # Check if session already has a stream
            for stream in self._streams.values():
                if stream.session_id == session_id:
                    logger.error(f"[STREAM_MGR] Session {session_id} already has active stream")
                    raise ValueError(f"Session {session_id} already has an active stream")
            
            # Create stream
            logger.info(f"[STREAM_MGR] Creating StreamingSession...")
            stream = StreamingSession(
                session_id=session_id,
                config=config,
                output_dir=str(self.output_dir),
                base_url=self.base_url,
            )
            
            # Start stream
            logger.info(f"[STREAM_MGR] Starting stream...")
            info = await stream.start(page)
            
            # Store stream
            self._streams[info.stream_id] = stream
            
            logger.info(
                f"[STREAM_MGR] Stream created successfully: {info.stream_id}\n"
                f"  HLS URL: {info.hls_url}\n"
                f"  Player URL: {info.player_url}"
            )
            return info
    
    async def stop_stream(self, stream_id: str) -> Optional[StreamInfo]:
        """Stop a stream.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            Final StreamInfo if stream existed
        """
        async with self._stream_lock:
            stream = self._streams.get(stream_id)
            if not stream:
                return None
            
            info = await stream.stop()
            del self._streams[stream_id]
            
            return info
    
    async def get_stream(self, stream_id: str) -> Optional[StreamInfo]:
        """Get stream information.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            StreamInfo if stream exists
        """
        stream = self._streams.get(stream_id)
        return stream.info if stream else None
    
    async def list_streams(
        self,
        session_id: Optional[str] = None,
    ) -> List[StreamInfo]:
        """List active streams.
        
        Args:
            session_id: Filter by session ID
            
        Returns:
            List of StreamInfo
        """
        streams = []
        async with self._stream_lock:
            for stream in self._streams.values():
                if session_id and stream.session_id != session_id:
                    continue
                streams.append(stream.info)
        
        return streams
    
    async def add_viewer(self, stream_id: str, viewer_id: str) -> bool:
        """Add a viewer to a stream.
        
        Args:
            stream_id: Stream identifier
            viewer_id: Viewer identifier
            
        Returns:
            True if viewer was added
        """
        stream = self._streams.get(stream_id)
        if not stream:
            return False
        
        return await stream.add_viewer(viewer_id)
    
    async def remove_viewer(self, stream_id: str, viewer_id: str) -> bool:
        """Remove a viewer from a stream.
        
        Args:
            stream_id: Stream identifier
            viewer_id: Viewer identifier
            
        Returns:
            True if viewer was removed
        """
        stream = self._streams.get(stream_id)
        if not stream:
            return False
        
        return await stream.remove_viewer(viewer_id)
    
    async def get_playlist(self, stream_id: str) -> Optional[str]:
        """Get HLS playlist for a stream.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            Playlist content
        """
        stream = self._streams.get(stream_id)
        if not stream:
            return None
        
        return await stream.get_playlist_content()
    
    async def get_segment(self, stream_id: str, segment_name: str) -> Optional[bytes]:
        """Get HLS segment data.
        
        Args:
            stream_id: Stream identifier
            segment_name: Segment file name
            
        Returns:
            Segment data
        """
        stream = self._streams.get(stream_id)
        if not stream:
            return None
        
        return await stream.get_segment(segment_name)
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of old streams."""
        try:
            while True:
                await asyncio.sleep(60)  # Run every minute
                
                async with self._stream_lock:
                    to_remove = []
                    
                    for stream_id, stream in self._streams.items():
                        # Remove stopped streams
                        if stream.info.state == StreamState.STOPPED:
                            to_remove.append(stream_id)
                            continue
                        
                        # Check max duration
                        uptime = time.time() - stream.info.started_at
                        if uptime > stream.config.max_duration_seconds:
                            logger.info(f"Stream exceeded max duration: {stream_id}")
                            await stream.stop()
                            to_remove.append(stream_id)
                            continue
                        
                        # Remove streams with no viewers for too long
                        # NOTE: Disabled viewer-based cleanup - streams should only be
                        # stopped explicitly or when max duration is exceeded
                        # if (len(stream.info.viewer_ids) == 0 and
                        #     uptime > 300):  # 5 minutes with no viewers
                        #     logger.info(f"Stream abandoned: {stream_id}")
                        #     await stream.stop()
                        #     to_remove.append(stream_id)
                    
                    for stream_id in to_remove:
                        del self._streams[stream_id]
                    
                    if to_remove:
                        logger.info(f"Cleaned up {len(to_remove)} streams")
                        
        except asyncio.CancelledError:
            logger.info("Cleanup loop stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics.
        
        Returns:
            Dictionary with streaming stats
        """
        active_streams = sum(
            1 for s in self._streams.values()
            if s.info.state == StreamState.ACTIVE
        )
        
        total_viewers = sum(
            len(s.info.viewer_ids)
            for s in self._streams.values()
        )
        
        total_bandwidth = sum(
            s.info.metrics.current_bitrate
            for s in self._streams.values()
        )
        
        return {
            "total_streams": len(self._streams),
            "active_streams": active_streams,
            "total_viewers": total_viewers,
            "total_bandwidth_bps": total_bandwidth,
            "max_concurrent_streams": self.max_concurrent_streams,
            "utilization": len(self._streams) / self.max_concurrent_streams,
        }
