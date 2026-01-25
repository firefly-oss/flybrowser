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
Live View iFrame Embedding for FlyBrowser.

This module provides real-time browser streaming capabilities including:

- WebSocket-based live view streaming
- iFrame embedding for external applications
- User control relay (mouse, keyboard events)
- Session takeover capabilities
- Low-latency screenshot streaming

Example:
    >>> from flybrowser.observability import LiveViewServer
    >>> 
    >>> # Start live view server
    >>> server = LiveViewServer(port=8765)
    >>> await server.start()
    >>> 
    >>> # Attach to browser session
    >>> await server.attach(page)
    >>> 
    >>> # Get embed URL for iFrame
    >>> embed_url = server.get_embed_url()
    >>> # Users can now view and control the browser at this URL
    >>> 
    >>> # Stop when done
    >>> await server.stop()
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
import weakref
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from flybrowser.utils.logger import logger

try:
    import aiohttp
    from aiohttp import web
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    web = None

try:
    from playwright.async_api import Page, CDPSession
except ImportError:
    Page = Any
    CDPSession = Any


class StreamQuality(str, Enum):
    """Stream quality presets."""
    LOW = "low"        # 10fps, 50% quality
    MEDIUM = "medium"  # 15fps, 70% quality
    HIGH = "high"      # 24fps, 85% quality
    ULTRA = "ultra"    # 30fps, 95% quality


class ControlMode(str, Enum):
    """User control modes."""
    VIEW_ONLY = "view_only"       # Can only watch
    INTERACT = "interact"          # Can interact (mouse, keyboard)
    FULL_CONTROL = "full_control"  # Full control including navigation


class MessageType(str, Enum):
    """WebSocket message types."""
    # Server -> Client
    FRAME = "frame"
    CURSOR = "cursor"
    SCROLL = "scroll"
    PAGE_INFO = "page_info"
    SESSION_INFO = "session_info"
    ERROR = "error"
    
    # Client -> Server
    MOUSE_MOVE = "mouse_move"
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    MOUSE_CLICK = "mouse_click"
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    KEY_PRESS = "key_press"
    SCROLL_EVENT = "scroll_event"
    NAVIGATE = "navigate"
    TAKE_CONTROL = "take_control"
    RELEASE_CONTROL = "release_control"


@dataclass
class StreamConfig:
    """Configuration for live view streaming."""
    quality: StreamQuality = StreamQuality.MEDIUM
    fps: int = 15
    jpeg_quality: int = 70
    max_width: int = 1920
    max_height: int = 1080
    compression: bool = True
    
    @classmethod
    def from_quality(cls, quality: StreamQuality) -> "StreamConfig":
        """Create config from quality preset."""
        presets = {
            StreamQuality.LOW: cls(
                quality=quality, fps=10, jpeg_quality=50,
                max_width=1280, max_height=720,
            ),
            StreamQuality.MEDIUM: cls(
                quality=quality, fps=15, jpeg_quality=70,
                max_width=1920, max_height=1080,
            ),
            StreamQuality.HIGH: cls(
                quality=quality, fps=24, jpeg_quality=85,
                max_width=1920, max_height=1080,
            ),
            StreamQuality.ULTRA: cls(
                quality=quality, fps=30, jpeg_quality=95,
                max_width=2560, max_height=1440,
            ),
        }
        return presets.get(quality, presets[StreamQuality.MEDIUM])


@dataclass
class ViewerSession:
    """A connected viewer session."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ws: Any = None  # WebSocket connection
    control_mode: ControlMode = ControlMode.VIEW_ONLY
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    has_control: bool = False
    user_agent: str = ""
    ip_address: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "control_mode": self.control_mode.value,
            "connected_at": self.connected_at,
            "has_control": self.has_control,
        }


class LiveViewServer:
    """
    Live View streaming server.
    
    Provides WebSocket-based real-time browser streaming with
    optional user control capabilities.
    
    Example:
        >>> server = LiveViewServer(port=8765)
        >>> await server.start()
        >>> 
        >>> # Attach browser page
        >>> await server.attach(page)
        >>> 
        >>> # Get embed code
        >>> iframe = server.get_embed_html()
        >>> # <iframe src="http://localhost:8765/view/..." ...>
        >>> 
        >>> # Control events are relayed to browser
        >>> await server.stop()
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        stream_config: Optional[StreamConfig] = None,
        default_control_mode: ControlMode = ControlMode.VIEW_ONLY,
        require_auth: bool = False,
        auth_token: Optional[str] = None,
        allowed_origins: Optional[List[str]] = None,
        max_viewers: int = 10,
    ):
        """
        Initialize live view server.
        
        Args:
            host: Server host
            port: Server port
            stream_config: Streaming configuration
            default_control_mode: Default control mode for viewers
            require_auth: Whether to require authentication
            auth_token: Auth token if required
            allowed_origins: CORS allowed origins
            max_viewers: Maximum concurrent viewers
        """
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required for LiveViewServer: pip install aiohttp")
        
        self.host = host
        self.port = port
        self.stream_config = stream_config or StreamConfig()
        self.default_control_mode = default_control_mode
        self.require_auth = require_auth
        self.auth_token = auth_token or str(uuid.uuid4())
        self.allowed_origins = allowed_origins or ["*"]
        self.max_viewers = max_viewers
        
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._page: Optional[Page] = None
        self._cdp_session: Optional[CDPSession] = None
        self._viewers: Dict[str, ViewerSession] = {}
        self._streaming = False
        self._stream_task: Optional[asyncio.Task] = None
        self._control_holder: Optional[str] = None
        self._session_id = str(uuid.uuid4())
        
        self._event_callbacks: Dict[str, List[Callable]] = {
            "viewer_connected": [],
            "viewer_disconnected": [],
            "control_requested": [],
            "control_released": [],
        }
        
        logger.info(f"[LIVE_VIEW] Initialized server on {host}:{port}")
    
    async def start(self) -> None:
        """Start the live view server."""
        self._app = web.Application()
        self._setup_routes()
        
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        
        logger.info(f"[LIVE_VIEW] Server started at http://{self.host}:{self.port}")
    
    async def stop(self) -> None:
        """Stop the live view server."""
        # Stop streaming
        await self.detach()
        
        # Disconnect all viewers
        for viewer in list(self._viewers.values()):
            try:
                await viewer.ws.close()
            except Exception:
                pass
        self._viewers.clear()
        
        # Stop server
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        
        self._app = None
        self._runner = None
        self._site = None
        
        logger.info("[LIVE_VIEW] Server stopped")
    
    async def attach(self, page: Page) -> None:
        """
        Attach to a browser page for streaming.
        
        Args:
            page: Playwright page to stream
        """
        self._page = page
        
        # Get CDP session for screenshots
        self._cdp_session = await page.context.new_cdp_session(page)
        
        # Start streaming
        self._streaming = True
        self._stream_task = asyncio.create_task(self._stream_loop())
        
        # Send initial page info to all viewers
        await self._broadcast_page_info()
        
        logger.info(f"[LIVE_VIEW] Attached to page: {page.url}")
    
    async def detach(self) -> None:
        """Detach from current page."""
        self._streaming = False
        
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None
        
        if self._cdp_session:
            try:
                await self._cdp_session.detach()
            except Exception:
                pass
            self._cdp_session = None
        
        self._page = None
        logger.info("[LIVE_VIEW] Detached from page")
    
    def _setup_routes(self) -> None:
        """Set up HTTP routes."""
        self._app.router.add_get("/", self._handle_index)
        self._app.router.add_get("/view/{session_id}", self._handle_view)
        self._app.router.add_get("/ws/{session_id}", self._handle_websocket)
        self._app.router.add_get("/api/session", self._handle_session_info)
        self._app.router.add_static("/static", Path(__file__).parent / "static", show_index=False)
    
    async def _handle_index(self, request: web.Request) -> web.Response:
        """Handle index page."""
        return web.Response(
            text=self._generate_viewer_html(),
            content_type="text/html",
        )
    
    async def _handle_view(self, request: web.Request) -> web.Response:
        """Handle view page with session."""
        session_id = request.match_info.get("session_id", "")
        
        # Validate session
        if session_id != self._session_id:
            return web.Response(status=404, text="Session not found")
        
        # Check auth if required
        if self.require_auth:
            token = request.query.get("token")
            if token != self.auth_token:
                return web.Response(status=401, text="Unauthorized")
        
        return web.Response(
            text=self._generate_viewer_html(),
            content_type="text/html",
        )
    
    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connection."""
        session_id = request.match_info.get("session_id", "")
        
        # Validate
        if session_id != self._session_id:
            return web.Response(status=404, text="Session not found")
        
        if self.require_auth:
            token = request.query.get("token")
            if token != self.auth_token:
                return web.Response(status=401, text="Unauthorized")
        
        # Check viewer limit
        if len(self._viewers) >= self.max_viewers:
            return web.Response(status=503, text="Maximum viewers reached")
        
        # Create WebSocket
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Create viewer session
        viewer = ViewerSession(
            ws=ws,
            control_mode=self.default_control_mode,
            user_agent=request.headers.get("User-Agent", ""),
            ip_address=request.remote or "",
        )
        self._viewers[viewer.id] = viewer
        
        logger.info(f"[LIVE_VIEW] Viewer connected: {viewer.id}")
        await self._emit_event("viewer_connected", viewer)
        
        # Send session info
        await self._send_to_viewer(viewer, {
            "type": MessageType.SESSION_INFO.value,
            "viewer_id": viewer.id,
            "control_mode": viewer.control_mode.value,
            "session_id": self._session_id,
        })
        
        # Send current page info
        if self._page:
            await self._send_page_info(viewer)
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_viewer_message(viewer, json.loads(msg.data))
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"[LIVE_VIEW] WebSocket error: {ws.exception()}")
        finally:
            # Cleanup
            del self._viewers[viewer.id]
            if self._control_holder == viewer.id:
                self._control_holder = None
            
            logger.info(f"[LIVE_VIEW] Viewer disconnected: {viewer.id}")
            await self._emit_event("viewer_disconnected", viewer)
        
        return ws
    
    async def _handle_session_info(self, request: web.Request) -> web.Response:
        """Handle session info API."""
        return web.json_response({
            "session_id": self._session_id,
            "streaming": self._streaming,
            "page_url": self._page.url if self._page else None,
            "viewers": len(self._viewers),
            "control_holder": self._control_holder,
        })
    
    async def _handle_viewer_message(self, viewer: ViewerSession, message: Dict[str, Any]) -> None:
        """Handle incoming message from viewer."""
        viewer.last_activity = time.time()
        msg_type = message.get("type")
        
        if not self._page:
            return
        
        # Control requests
        if msg_type == MessageType.TAKE_CONTROL.value:
            await self._handle_take_control(viewer)
            return
        
        if msg_type == MessageType.RELEASE_CONTROL.value:
            await self._handle_release_control(viewer)
            return
        
        # Input events - only process if viewer has control
        if not self._viewer_can_control(viewer):
            return
        
        try:
            if msg_type == MessageType.MOUSE_MOVE.value:
                await self._page.mouse.move(message["x"], message["y"])
            
            elif msg_type == MessageType.MOUSE_DOWN.value:
                await self._page.mouse.down(button=message.get("button", "left"))
            
            elif msg_type == MessageType.MOUSE_UP.value:
                await self._page.mouse.up(button=message.get("button", "left"))
            
            elif msg_type == MessageType.MOUSE_CLICK.value:
                await self._page.mouse.click(
                    message["x"], 
                    message["y"],
                    button=message.get("button", "left"),
                    click_count=message.get("click_count", 1),
                )
            
            elif msg_type == MessageType.KEY_DOWN.value:
                await self._page.keyboard.down(message["key"])
            
            elif msg_type == MessageType.KEY_UP.value:
                await self._page.keyboard.up(message["key"])
            
            elif msg_type == MessageType.KEY_PRESS.value:
                await self._page.keyboard.press(message["key"])
            
            elif msg_type == MessageType.SCROLL_EVENT.value:
                await self._page.mouse.wheel(
                    message.get("delta_x", 0),
                    message.get("delta_y", 0),
                )
            
            elif msg_type == MessageType.NAVIGATE.value:
                if viewer.control_mode == ControlMode.FULL_CONTROL:
                    await self._page.goto(message["url"])
        
        except Exception as e:
            logger.error(f"[LIVE_VIEW] Error handling input: {e}")
            await self._send_to_viewer(viewer, {
                "type": MessageType.ERROR.value,
                "message": str(e),
            })
    
    async def _handle_take_control(self, viewer: ViewerSession) -> None:
        """Handle control takeover request."""
        if viewer.control_mode == ControlMode.VIEW_ONLY:
            await self._send_to_viewer(viewer, {
                "type": MessageType.ERROR.value,
                "message": "View-only mode - control not allowed",
            })
            return
        
        # Release from current holder
        if self._control_holder and self._control_holder != viewer.id:
            old_holder = self._viewers.get(self._control_holder)
            if old_holder:
                old_holder.has_control = False
                await self._send_to_viewer(old_holder, {
                    "type": MessageType.SESSION_INFO.value,
                    "has_control": False,
                    "message": "Control taken by another viewer",
                })
        
        # Grant control
        self._control_holder = viewer.id
        viewer.has_control = True
        
        await self._send_to_viewer(viewer, {
            "type": MessageType.SESSION_INFO.value,
            "has_control": True,
        })
        
        logger.info(f"[LIVE_VIEW] Control granted to viewer: {viewer.id}")
        await self._emit_event("control_requested", viewer)
    
    async def _handle_release_control(self, viewer: ViewerSession) -> None:
        """Handle control release."""
        if self._control_holder == viewer.id:
            self._control_holder = None
            viewer.has_control = False
            
            await self._send_to_viewer(viewer, {
                "type": MessageType.SESSION_INFO.value,
                "has_control": False,
            })
            
            logger.info(f"[LIVE_VIEW] Control released by viewer: {viewer.id}")
            await self._emit_event("control_released", viewer)
    
    def _viewer_can_control(self, viewer: ViewerSession) -> bool:
        """Check if viewer can send control events."""
        if viewer.control_mode == ControlMode.VIEW_ONLY:
            return False
        
        if viewer.control_mode == ControlMode.FULL_CONTROL:
            return True
        
        # INTERACT mode - needs to have control
        return viewer.has_control or self._control_holder is None
    
    async def _stream_loop(self) -> None:
        """Main streaming loop."""
        frame_interval = 1.0 / self.stream_config.fps
        
        while self._streaming and self._page:
            try:
                start = time.time()
                
                # Capture screenshot via CDP
                if self._cdp_session:
                    result = await self._cdp_session.send(
                        "Page.captureScreenshot",
                        {
                            "format": "jpeg",
                            "quality": self.stream_config.jpeg_quality,
                        }
                    )
                    
                    frame_data = result.get("data", "")
                    
                    # Broadcast to all viewers
                    await self._broadcast({
                        "type": MessageType.FRAME.value,
                        "data": frame_data,
                        "timestamp": time.time(),
                    })
                
                # Maintain frame rate
                elapsed = time.time() - start
                if elapsed < frame_interval:
                    await asyncio.sleep(frame_interval - elapsed)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[LIVE_VIEW] Stream error: {e}")
                await asyncio.sleep(0.1)
    
    async def _broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all viewers."""
        data = json.dumps(message)
        dead_viewers = []
        
        for viewer_id, viewer in self._viewers.items():
            try:
                await viewer.ws.send_str(data)
            except Exception:
                dead_viewers.append(viewer_id)
        
        # Cleanup dead connections
        for viewer_id in dead_viewers:
            if viewer_id in self._viewers:
                del self._viewers[viewer_id]
    
    async def _send_to_viewer(self, viewer: ViewerSession, message: Dict[str, Any]) -> None:
        """Send message to specific viewer."""
        try:
            await viewer.ws.send_str(json.dumps(message))
        except Exception as e:
            logger.error(f"[LIVE_VIEW] Failed to send to viewer {viewer.id}: {e}")
    
    async def _broadcast_page_info(self) -> None:
        """Broadcast current page info to all viewers."""
        for viewer in self._viewers.values():
            await self._send_page_info(viewer)
    
    async def _send_page_info(self, viewer: ViewerSession) -> None:
        """Send page info to viewer."""
        if not self._page:
            return
        
        try:
            await self._send_to_viewer(viewer, {
                "type": MessageType.PAGE_INFO.value,
                "url": self._page.url,
                "title": await self._page.title(),
                "viewport": await self._page.viewport_size(),
            })
        except Exception as e:
            logger.error(f"[LIVE_VIEW] Failed to send page info: {e}")
    
    async def _emit_event(self, event: str, *args) -> None:
        """Emit event to callbacks."""
        for callback in self._event_callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args)
                else:
                    callback(*args)
            except Exception as e:
                logger.error(f"[LIVE_VIEW] Event callback error: {e}")
    
    def on(self, event: str, callback: Callable) -> None:
        """Register event callback."""
        if event not in self._event_callbacks:
            self._event_callbacks[event] = []
        self._event_callbacks[event].append(callback)
    
    def get_url(self) -> str:
        """Get the view URL."""
        return f"http://{self.host}:{self.port}/view/{self._session_id}"
    
    def get_embed_url(self, with_token: bool = True) -> str:
        """Get URL for iFrame embedding."""
        url = self.get_url()
        if self.require_auth and with_token:
            url += f"?token={self.auth_token}"
        return url
    
    def get_embed_html(
        self, 
        width: str = "100%", 
        height: str = "600px",
        style: str = "",
    ) -> str:
        """
        Get HTML for iFrame embedding.
        
        Args:
            width: iFrame width
            height: iFrame height
            style: Additional CSS styles
            
        Returns:
            HTML string for embedding
        """
        url = self.get_embed_url()
        return f'''<iframe 
    src="{url}" 
    width="{width}" 
    height="{height}" 
    style="border: 1px solid #ccc; {style}"
    allow="clipboard-read; clipboard-write"
    sandbox="allow-scripts allow-same-origin allow-forms"
></iframe>'''
    
    def _generate_viewer_html(self) -> str:
        """Generate the viewer HTML page."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlyBrowser Live View</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #1a1a2e;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #eee;
            overflow: hidden;
        }}
        #container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}
        #toolbar {{
            background: #16213e;
            padding: 8px 16px;
            display: flex;
            align-items: center;
            gap: 16px;
            border-bottom: 1px solid #0f3460;
        }}
        #toolbar .status {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        #toolbar .status .dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #e94560;
        }}
        #toolbar .status .dot.connected {{ background: #4ecca3; }}
        #toolbar .url {{
            flex: 1;
            padding: 6px 12px;
            background: #0f3460;
            border: none;
            border-radius: 4px;
            color: #eee;
            font-size: 14px;
        }}
        #toolbar button {{
            padding: 6px 16px;
            background: #e94560;
            border: none;
            border-radius: 4px;
            color: white;
            cursor: pointer;
            font-size: 14px;
        }}
        #toolbar button:hover {{ background: #ff6b6b; }}
        #toolbar button.active {{ background: #4ecca3; }}
        #viewport {{
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            background: #0f0f23;
            overflow: hidden;
        }}
        #canvas {{
            max-width: 100%;
            max-height: 100%;
            cursor: crosshair;
        }}
        #canvas.controlling {{ cursor: default; }}
        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
        }}
        #loading .spinner {{
            width: 40px;
            height: 40px;
            border: 3px solid #0f3460;
            border-top: 3px solid #e94560;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 16px;
        }}
        @keyframes spin {{ 100% {{ transform: rotate(360deg); }} }}
        #info {{
            position: absolute;
            bottom: 16px;
            right: 16px;
            background: rgba(22, 33, 62, 0.9);
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="toolbar">
            <div class="status">
                <div class="dot" id="statusDot"></div>
                <span id="statusText">Connecting...</span>
            </div>
            <input type="text" class="url" id="urlBar" readonly>
            <button id="controlBtn">Take Control</button>
        </div>
        <div id="viewport">
            <div id="loading">
                <div class="spinner"></div>
                <div>Connecting to session...</div>
            </div>
            <canvas id="canvas" style="display: none;"></canvas>
        </div>
        <div id="info">
            <span id="fps">0 FPS</span> | 
            <span id="latency">0ms</span>
        </div>
    </div>
    
    <script>
        const sessionId = '{self._session_id}';
        const wsUrl = `ws://${{window.location.host}}/ws/${{sessionId}}`;
        
        let ws;
        let canvas, ctx;
        let hasControl = false;
        let frameCount = 0;
        let lastFpsUpdate = Date.now();
        
        function connect() {{
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {{
                document.getElementById('statusDot').classList.add('connected');
                document.getElementById('statusText').textContent = 'Connected';
            }};
            
            ws.onclose = () => {{
                document.getElementById('statusDot').classList.remove('connected');
                document.getElementById('statusText').textContent = 'Disconnected';
                setTimeout(connect, 2000);
            }};
            
            ws.onmessage = (e) => {{
                const msg = JSON.parse(e.data);
                handleMessage(msg);
            }};
        }}
        
        function handleMessage(msg) {{
            switch (msg.type) {{
                case 'frame':
                    renderFrame(msg.data, msg.timestamp);
                    break;
                case 'page_info':
                    document.getElementById('urlBar').value = msg.url;
                    document.title = msg.title || 'FlyBrowser Live View';
                    break;
                case 'session_info':
                    hasControl = msg.has_control;
                    updateControlButton();
                    break;
            }}
        }}
        
        function renderFrame(data, timestamp) {{
            if (!canvas) {{
                canvas = document.getElementById('canvas');
                ctx = canvas.getContext('2d');
                document.getElementById('loading').style.display = 'none';
                canvas.style.display = 'block';
            }}
            
            const img = new Image();
            img.onload = () => {{
                if (canvas.width !== img.width || canvas.height !== img.height) {{
                    canvas.width = img.width;
                    canvas.height = img.height;
                }}
                ctx.drawImage(img, 0, 0);
                
                // Update FPS
                frameCount++;
                const now = Date.now();
                if (now - lastFpsUpdate >= 1000) {{
                    document.getElementById('fps').textContent = frameCount + ' FPS';
                    frameCount = 0;
                    lastFpsUpdate = now;
                }}
                
                // Update latency
                const latency = now - timestamp * 1000;
                document.getElementById('latency').textContent = Math.round(latency) + 'ms';
            }};
            img.src = 'data:image/jpeg;base64,' + data;
        }}
        
        function updateControlButton() {{
            const btn = document.getElementById('controlBtn');
            if (hasControl) {{
                btn.textContent = 'Release Control';
                btn.classList.add('active');
                canvas?.classList.add('controlling');
            }} else {{
                btn.textContent = 'Take Control';
                btn.classList.remove('active');
                canvas?.classList.remove('controlling');
            }}
        }}
        
        function sendInput(type, data) {{
            if (ws?.readyState === WebSocket.OPEN) {{
                ws.send(JSON.stringify({{ type, ...data }}));
            }}
        }}
        
        // Input handlers
        document.addEventListener('DOMContentLoaded', () => {{
            connect();
            
            document.getElementById('controlBtn').onclick = () => {{
                sendInput(hasControl ? 'release_control' : 'take_control', {{}});
            }};
            
            const canvas = document.getElementById('canvas');
            
            canvas.onmousemove = (e) => {{
                const rect = canvas.getBoundingClientRect();
                const scaleX = canvas.width / rect.width;
                const scaleY = canvas.height / rect.height;
                sendInput('mouse_move', {{
                    x: Math.round((e.clientX - rect.left) * scaleX),
                    y: Math.round((e.clientY - rect.top) * scaleY)
                }});
            }};
            
            canvas.onmousedown = (e) => {{
                e.preventDefault();
                const rect = canvas.getBoundingClientRect();
                const scaleX = canvas.width / rect.width;
                const scaleY = canvas.height / rect.height;
                sendInput('mouse_click', {{
                    x: Math.round((e.clientX - rect.left) * scaleX),
                    y: Math.round((e.clientY - rect.top) * scaleY),
                    button: ['left', 'middle', 'right'][e.button]
                }});
            }};
            
            canvas.oncontextmenu = (e) => e.preventDefault();
            
            canvas.onwheel = (e) => {{
                e.preventDefault();
                sendInput('scroll_event', {{
                    delta_x: e.deltaX,
                    delta_y: e.deltaY
                }});
            }};
            
            document.onkeydown = (e) => {{
                if (hasControl && !e.target.matches('input')) {{
                    e.preventDefault();
                    sendInput('key_down', {{ key: e.key }});
                }}
            }};
            
            document.onkeyup = (e) => {{
                if (hasControl && !e.target.matches('input')) {{
                    sendInput('key_up', {{ key: e.key }});
                }}
            }};
        }});
    </script>
</body>
</html>'''
    
    @property
    def viewers(self) -> List[ViewerSession]:
        """Get list of connected viewers."""
        return list(self._viewers.values())
    
    @property
    def viewer_count(self) -> int:
        """Get number of connected viewers."""
        return len(self._viewers)
    
    @property
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._streaming
    
    def set_viewer_control_mode(self, viewer_id: str, mode: ControlMode) -> bool:
        """Set control mode for a specific viewer."""
        if viewer_id in self._viewers:
            self._viewers[viewer_id].control_mode = mode
            return True
        return False
    
    async def revoke_control(self) -> None:
        """Revoke control from current holder."""
        if self._control_holder:
            viewer = self._viewers.get(self._control_holder)
            if viewer:
                viewer.has_control = False
                await self._send_to_viewer(viewer, {
                    "type": MessageType.SESSION_INFO.value,
                    "has_control": False,
                    "message": "Control revoked",
                })
            self._control_holder = None
