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
FlyBrowser Observability Layer.

This package provides comprehensive observability capabilities for browser
automation sessions including:

- **Command Logging**: Structured logging of all SDK operations, LLM calls,
  and tool executions with queryable history and export capabilities.

- **Source Capture**: HTML snapshots, DOM tree capture, and HAR (HTTP Archive)
  network traffic logging for debugging and session replay.

- **Live View**: Real-time browser streaming via WebSocket with iFrame embedding
  and optional user control for interactive sessions.

Example:
    >>> from flybrowser.observability import (
    ...     CommandLogger,
    ...     SourceCaptureManager,
    ...     LiveViewServer,
    ...     ObservabilityConfig,
    ... )
    >>> 
    >>> # Configure observability
    >>> config = ObservabilityConfig(
    ...     enable_command_logging=True,
    ...     enable_source_capture=True,
    ...     enable_live_view=True,
    ...     live_view_port=8765,
    ... )
    >>> 
    >>> # Create managers
    >>> logger = CommandLogger(session_id="my-session")
    >>> capture = SourceCaptureManager(session_id="my-session")
    >>> live_view = LiveViewServer(port=config.live_view_port)
    >>> 
    >>> # Use with browser session
    >>> with logger.log_command("goto", {"url": "..."}):
    ...     pass
    >>> 
    >>> snapshot = await capture.capture_html(page)
    >>> 
    >>> # Export all data
    >>> logger.export("session_log.json")
    >>> capture.export_all("./captures/")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from flybrowser.observability.command_logger import (
    CommandLogger,
    CommandEntry,
    CommandType,
    LogLevel,
    LLMUsage,
    PageState,
    SessionHistory,
)

from flybrowser.observability.source_capture import (
    SourceCaptureManager,
    HTMLSnapshot,
    HARLog,
    HAREntry,
    CapturedResource,
    CaptureType,
    ResourceType,
)

from flybrowser.observability.live_view import (
    LiveViewServer,
    ViewerSession,
    StreamConfig,
    StreamQuality,
    ControlMode,
    MessageType,
)


@dataclass
class ObservabilityConfig:
    """
    Unified configuration for all observability features.
    
    This config can be passed to the SDK to enable/configure
    all observability capabilities at once.
    
    Example:
        >>> config = ObservabilityConfig(
        ...     enable_command_logging=True,
        ...     log_llm_prompts=True,
        ...     enable_source_capture=True,
        ...     capture_resources=True,
        ...     enable_live_view=True,
        ...     live_view_port=8080,
        ...     live_view_control_mode=ControlMode.INTERACT,
        ... )
    """
    # Command logging
    enable_command_logging: bool = True
    log_llm_prompts: bool = False
    log_llm_responses: bool = False
    max_log_entries: int = 10000
    auto_export_logs: bool = False
    log_export_path: Optional[str] = None
    
    # Source capture
    enable_source_capture: bool = False
    capture_resources: bool = False
    max_resource_size_bytes: int = 5 * 1024 * 1024  # 5MB
    max_snapshots: int = 100
    auto_capture_on_navigation: bool = False
    capture_har: bool = False
    
    # Live view
    enable_live_view: bool = False
    live_view_host: str = "0.0.0.0"
    live_view_port: int = 8765
    live_view_quality: StreamQuality = StreamQuality.MEDIUM
    live_view_control_mode: ControlMode = ControlMode.VIEW_ONLY
    live_view_require_auth: bool = False
    live_view_auth_token: Optional[str] = None
    live_view_max_viewers: int = 10
    
    # General
    session_id: Optional[str] = None
    output_dir: Optional[str] = None
    
    def create_command_logger(self) -> Optional[CommandLogger]:
        """Create a CommandLogger from this config."""
        if not self.enable_command_logging:
            return None
        
        return CommandLogger(
            session_id=self.session_id,
            enabled=True,
            max_entries=self.max_log_entries,
            log_llm_prompts=self.log_llm_prompts,
            log_llm_responses=self.log_llm_responses,
        )
    
    def create_source_capture_manager(self) -> Optional[SourceCaptureManager]:
        """Create a SourceCaptureManager from this config."""
        if not self.enable_source_capture:
            return None
        
        return SourceCaptureManager(
            session_id=self.session_id,
            enabled=True,
            capture_resources=self.capture_resources,
            max_resource_size_bytes=self.max_resource_size_bytes,
            max_snapshots=self.max_snapshots,
        )
    
    def create_live_view_server(self) -> Optional[LiveViewServer]:
        """Create a LiveViewServer from this config."""
        if not self.enable_live_view:
            return None
        
        return LiveViewServer(
            host=self.live_view_host,
            port=self.live_view_port,
            stream_config=StreamConfig.from_quality(self.live_view_quality),
            default_control_mode=self.live_view_control_mode,
            require_auth=self.live_view_require_auth,
            auth_token=self.live_view_auth_token,
            max_viewers=self.live_view_max_viewers,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "command_logging": {
                "enabled": self.enable_command_logging,
                "log_llm_prompts": self.log_llm_prompts,
                "log_llm_responses": self.log_llm_responses,
                "max_entries": self.max_log_entries,
            },
            "source_capture": {
                "enabled": self.enable_source_capture,
                "capture_resources": self.capture_resources,
                "max_snapshots": self.max_snapshots,
                "capture_har": self.capture_har,
            },
            "live_view": {
                "enabled": self.enable_live_view,
                "host": self.live_view_host,
                "port": self.live_view_port,
                "quality": self.live_view_quality.value,
                "control_mode": self.live_view_control_mode.value,
            },
            "session_id": self.session_id,
            "output_dir": self.output_dir,
        }


class ObservabilityManager:
    """
    Unified manager for all observability components.
    
    Provides a single interface to manage command logging,
    source capture, and live view together.
    
    Example:
        >>> config = ObservabilityConfig(
        ...     enable_command_logging=True,
        ...     enable_live_view=True,
        ... )
        >>> manager = ObservabilityManager(config)
        >>> 
        >>> # Attach to browser session
        >>> await manager.attach(page)
        >>> 
        >>> # Use context managers for operations
        >>> with manager.log_command("goto", {"url": "..."}):
        ...     pass
        >>> 
        >>> # Cleanup
        >>> await manager.detach()
        >>> manager.export_all("./output/")
    """
    
    def __init__(self, config: Optional[ObservabilityConfig] = None):
        """
        Initialize observability manager.
        
        Args:
            config: Observability configuration
        """
        self.config = config or ObservabilityConfig()
        
        self.command_logger = self.config.create_command_logger()
        self.source_capture = self.config.create_source_capture_manager()
        self.live_view = self.config.create_live_view_server()
        
        self._page = None
        self._attached = False
    
    async def attach(self, page: Any) -> None:
        """
        Attach to a browser page.
        
        Args:
            page: Playwright page object
        """
        self._page = page
        self._attached = True
        
        # Start live view if enabled
        if self.live_view:
            await self.live_view.start()
            await self.live_view.attach(page)
        
        # Start HAR capture if enabled
        if self.source_capture and self.config.capture_har:
            await self.source_capture.start_har_capture(page)
    
    async def detach(self) -> None:
        """Detach from current page."""
        # Stop HAR capture
        if self.source_capture and self.config.capture_har:
            await self.source_capture.stop_har_capture()
        
        # Stop live view
        if self.live_view:
            await self.live_view.stop()
        
        self._page = None
        self._attached = False
    
    def log_command(self, name: str, parameters: Dict[str, Any] = None, **kwargs):
        """Context manager for logging a command."""
        if self.command_logger:
            return self.command_logger.log_command(name, parameters, **kwargs)
        
        # Return no-op context manager
        from contextlib import nullcontext
        return nullcontext(CommandEntry())
    
    async def log_async(self, name: str, parameters: Dict[str, Any] = None, **kwargs):
        """Async context manager for logging a command."""
        if self.command_logger:
            return self.command_logger.log_async(name, parameters, **kwargs)
        
        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def noop():
            yield CommandEntry()
        return noop()
    
    async def capture_snapshot(self) -> Optional[HTMLSnapshot]:
        """Capture current page state."""
        if not self.source_capture or not self._page:
            return None
        return await self.source_capture.capture_html(self._page)
    
    def get_live_view_url(self) -> Optional[str]:
        """Get live view URL if enabled."""
        if self.live_view:
            return self.live_view.get_embed_url()
        return None
    
    def get_live_view_html(self) -> Optional[str]:
        """Get live view iFrame HTML if enabled."""
        if self.live_view:
            return self.live_view.get_embed_html()
        return None
    
    def export_all(self, output_dir: str) -> None:
        """
        Export all observability data to directory.
        
        Args:
            output_dir: Output directory path
        """
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export command logs
        if self.command_logger:
            self.command_logger.export(output_path / "commands.json")
        
        # Export source captures
        if self.source_capture:
            self.source_capture.export_all(output_path / "captures")
    
    @property
    def is_attached(self) -> bool:
        """Check if attached to a page."""
        return self._attached


# Public API
__all__ = [
    # Config
    "ObservabilityConfig",
    "ObservabilityManager",
    
    # Command logging
    "CommandLogger",
    "CommandEntry",
    "CommandType",
    "LogLevel",
    "LLMUsage",
    "PageState",
    "SessionHistory",
    
    # Source capture
    "SourceCaptureManager",
    "HTMLSnapshot",
    "HARLog",
    "HAREntry",
    "CapturedResource",
    "CaptureType",
    "ResourceType",
    
    # Live view
    "LiveViewServer",
    "ViewerSession",
    "StreamConfig",
    "StreamQuality",
    "ControlMode",
    "MessageType",
]
