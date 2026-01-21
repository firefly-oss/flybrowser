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
FFmpeg-based Video Recorder for FlyBrowser.

This module provides advanced video recording capabilities using ffmpeg:
- Modern codecs (H.264, H.265, VP9, VP8, AV1)
- Streaming protocols (RTMP, HLS, DASH)
- Hardware acceleration support
- Bandwidth-optimized encoding profiles
- Real-time frame capture from Playwright
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import Page

from flybrowser.exceptions import BrowserError
from flybrowser.utils.logger import logger


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
    
    LOW_BANDWIDTH = "low_bandwidth"      # 500kbps, optimized for slow connections
    MEDIUM = "medium"                     # 1.5Mbps, balanced quality/size
    HIGH = "high"                         # 3Mbps, high quality
    LOSSLESS = "lossless"                 # Lossless encoding


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
    """
    
    codec: VideoCodec = VideoCodec.H264
    quality_profile: QualityProfile = QualityProfile.MEDIUM
    output_path: Optional[str] = None
    streaming_protocol: StreamingProtocol = StreamingProtocol.FILE
    streaming_url: Optional[str] = None
    
    # Video settings
    width: int = 1280
    height: int = 720
    frame_rate: int = 25
    
    # Encoding settings (overridden by quality_profile)
    bitrate: Optional[str] = None
    crf: Optional[int] = None
    preset: str = "veryfast"
    
    # Advanced options
    enable_hw_accel: bool = True
    ffmpeg_path: Optional[str] = None
    pixel_format: str = "yuv420p"
    tune: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Apply quality profile settings."""
        self._apply_quality_profile()
        
        if not self.ffmpeg_path:
            self.ffmpeg_path = self._find_ffmpeg()
        
        if not self.output_path and self.streaming_protocol == StreamingProtocol.FILE:
            self.output_path = f"recording_{int(time.time())}.mp4"
    
    def _apply_quality_profile(self) -> None:
        """Apply pre-configured quality profile settings."""
        profiles = {
            QualityProfile.LOW_BANDWIDTH: {
                "bitrate": "500k",
                "crf": 28,
                "preset": "veryfast",
                "tune": "zerolatency",
            },
            QualityProfile.MEDIUM: {
                "bitrate": "1500k",
                "crf": 23,
                "preset": "veryfast",
                "tune": "zerolatency",
            },
            QualityProfile.HIGH: {
                "bitrate": "3000k",
                "crf": 20,
                "preset": "medium",
                "tune": None,
            },
            QualityProfile.LOSSLESS: {
                "bitrate": None,
                "crf": 0,
                "preset": "medium",
                "tune": None,
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
                self.tune = profile["tune"]
    
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


class FFmpegRecorder:
    """FFmpeg-based video recorder with advanced features.
    
    Records browser sessions using ffmpeg with support for:
    - Modern codecs (H.264, H.265, VP9)
    - Hardware acceleration
    - Live streaming (HLS, RTMP, DASH)
    - Bandwidth-optimized encoding
    
    Example:
        >>> config = FFmpegConfig(
        ...     codec=VideoCodec.H264,
        ...     quality_profile=QualityProfile.MEDIUM,
        ...     output_path="recording.mp4"
        ... )
        >>> recorder = FFmpegRecorder(config)
        >>> await recorder.start(page)
        >>> # ... browser actions ...
        >>> metadata = await recorder.stop()
    """
    
    def __init__(self, config: FFmpegConfig) -> None:
        """Initialize the recorder."""
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._recording = False
        self._capture_task: Optional[asyncio.Task] = None
        self._page: Optional[Page] = None
        self._metadata = RecordingMetadata(
            codec=config.codec.value,
            width=config.width,
            height=config.height,
            frame_rate=config.frame_rate,
        )
        self._frame_buffer: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._stop_event = asyncio.Event()
    
    async def start(self, page: Page) -> str:
        """Start recording the page.
        
        Args:
            page: Playwright page to record
            
        Returns:
            Recording ID
            
        Raises:
            BrowserError: If recording fails to start
        """
        if self._recording:
            raise BrowserError("Recording already in progress")
        
        self._page = page
        self._recording = True
        self._stop_event.clear()
        
        try:
            # Build ffmpeg command
            cmd = self._build_ffmpeg_command()
            logger.info(f"Starting ffmpeg: {' '.join(cmd)}")
            
            # Start ffmpeg process
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            # Start frame capture task
            self._capture_task = asyncio.create_task(self._capture_frames())
            
            # Start frame writing task
            asyncio.create_task(self._write_frames())
            
            logger.info(f"Recording started: {self._metadata.recording_id}")
            return self._metadata.recording_id
            
        except Exception as e:
            self._recording = False
            if self._process:
                self._process.kill()
            raise BrowserError(f"Failed to start recording: {e}") from e
    
    async def stop(self) -> RecordingMetadata:
        """Stop recording and return metadata.
        
        Returns:
            Recording metadata with file info
        """
        if not self._recording:
            logger.warning("No recording in progress")
            return self._metadata
        
        self._recording = False
        self._stop_event.set()
        
        # Wait for capture task to finish
        if self._capture_task:
            try:
                await asyncio.wait_for(self._capture_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Frame capture task timeout")
                self._capture_task.cancel()
        
        # Close ffmpeg stdin and wait for process to finish
        if self._process and self._process.stdin:
            try:
                self._process.stdin.close()
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        
        # Update metadata
        self._metadata.ended_at = time.time()
        self._metadata.duration_seconds = self._metadata.ended_at - self._metadata.started_at
        
        if self.config.output_path and os.path.exists(self.config.output_path):
            self._metadata.file_path = self.config.output_path
            self._metadata.file_size_bytes = os.path.getsize(self.config.output_path)
        
        if self.config.streaming_url:
            self._metadata.streaming_url = self.config.streaming_url
        
        logger.info(
            f"Recording stopped: {self._metadata.recording_id} "
            f"({self._metadata.total_frames} frames, {self._metadata.duration_seconds:.2f}s)"
        )
        
        return self._metadata
    
    def _build_ffmpeg_command(self) -> List[str]:
        """Build ffmpeg command based on configuration."""
        cmd = [self.config.ffmpeg_path, "-y"]
        
        # Hardware acceleration (input side)
        if self.config.enable_hw_accel:
            if self._is_hw_accel_available("cuda"):
                cmd.extend(["-hwaccel", "cuda"])
            elif self._is_hw_accel_available("videotoolbox"):
                cmd.extend(["-hwaccel", "videotoolbox"])
            elif self._is_hw_accel_available("qsv"):
                cmd.extend(["-hwaccel", "qsv"])
        
        # Input from stdin (raw video)
        cmd.extend([
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.config.width}x{self.config.height}",
            "-r", str(self.config.frame_rate),
            "-i", "pipe:0",
        ])
        
        # Video codec
        encoder = self.config.codec.get_encoder(self.config.enable_hw_accel)
        cmd.extend(["-c:v", encoder])
        
        # Encoding preset
        cmd.extend(["-preset", self.config.preset])
        
        # Tune
        if self.config.tune:
            cmd.extend(["-tune", self.config.tune])
        
        # Quality settings
        if self.config.crf is not None:
            cmd.extend(["-crf", str(self.config.crf)])
        
        if self.config.bitrate:
            cmd.extend([
                "-b:v", self.config.bitrate,
                "-maxrate", self.config.bitrate,
                "-bufsize", str(int(self.config.bitrate.replace("k", "").replace("M", "000")) * 2) + "k",
            ])
        
        # Pixel format
        cmd.extend(["-pix_fmt", self.config.pixel_format])
        
        # Protocol-specific settings
        if self.config.streaming_protocol == StreamingProtocol.HLS:
            cmd.extend([
                "-f", "hls",
                "-hls_time", "2",
                "-hls_list_size", "10",
                "-hls_flags", "delete_segments",
                "-start_number", "0",
            ])
            output = self.config.output_path or "playlist.m3u8"
            
        elif self.config.streaming_protocol == StreamingProtocol.DASH:
            cmd.extend([
                "-f", "dash",
                "-seg_duration", "2",
                "-window_size", "10",
            ])
            output = self.config.output_path or "manifest.mpd"
            
        elif self.config.streaming_protocol == StreamingProtocol.RTMP:
            cmd.extend(["-f", "flv"])
            output = self.config.streaming_url or "rtmp://localhost/live/stream"
            
        else:  # FILE
            cmd.extend(["-movflags", "+faststart"])
            output = self.config.output_path or "output.mp4"
        
        cmd.append(output)
        
        return cmd
    
    @staticmethod
    def _is_hw_accel_available(hw_accel: str) -> bool:
        """Check if hardware acceleration is available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-hwaccels"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return hw_accel in result.stdout
        except Exception:
            return False
    
    async def _capture_frames(self) -> None:
        """Capture frames from the page."""
        frame_interval = 1.0 / self.config.frame_rate
        
        try:
            while self._recording:
                start_time = time.time()
                
                try:
                    # Capture screenshot as PNG
                    screenshot_bytes = await self._page.screenshot(
                        type="png",
                        full_page=False,
                    )
                    
                    # Convert PNG to raw RGB24 (ffmpeg expects raw frames)
                    from PIL import Image
                    import io
                    
                    # Decode PNG to PIL Image
                    img = Image.open(io.BytesIO(screenshot_bytes))
                    
                    # Resize if needed
                    if img.size != (self.config.width, self.config.height):
                        img = img.resize((self.config.width, self.config.height), Image.Resampling.LANCZOS)
                    
                    # Convert to RGB (remove alpha channel if present)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Get raw RGB24 bytes
                    raw_frame = img.tobytes()
                    
                    # Add frame to buffer (non-blocking)
                    try:
                        self._frame_buffer.put_nowait(raw_frame)
                        self._metadata.total_frames += 1
                    except asyncio.QueueFull:
                        logger.warning("Frame buffer full, dropping frame")
                    
                except Exception as e:
                    logger.error(f"Frame capture error: {e}")
                
                # Maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=sleep_time
                    )
                    break
                except asyncio.TimeoutError:
                    pass
                    
        except asyncio.CancelledError:
            logger.info("Frame capture cancelled")
    
    async def _write_frames(self) -> None:
        """Write frames from buffer to ffmpeg stdin."""
        loop = asyncio.get_event_loop()
        
        try:
            while self._recording or not self._frame_buffer.empty():
                try:
                    # Get frame from buffer (with timeout)
                    frame_data = await asyncio.wait_for(
                        self._frame_buffer.get(),
                        timeout=1.0
                    )
                    
                    # Write to ffmpeg stdin in executor to avoid blocking
                    if self._process and self._process.stdin:
                        await loop.run_in_executor(
                            None,
                            self._process.stdin.write,
                            frame_data
                        )
                        
                except asyncio.TimeoutError:
                    if not self._recording:
                        break
                    continue
                except Exception as e:
                    logger.error(f"Frame write error: {e}")
                    break
                    
        except asyncio.CancelledError:
            logger.info("Frame writing cancelled")
    
    @property
    def is_recording(self) -> bool:
        """Check if recording is in progress."""
        return self._recording
    
    @property
    def metadata(self) -> RecordingMetadata:
        """Get current recording metadata."""
        return self._metadata


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
