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
Source Code Capture for FlyBrowser.

This module provides comprehensive source capture capabilities including:

- HTML snapshots with inline resources
- DOM state capture at specific points
- Network traffic logging (HAR format)
- Resource archiving (CSS, JS, images)
- Diff tracking between captures

Example:
    >>> from flybrowser.observability import SourceCaptureManager
    >>> 
    >>> manager = SourceCaptureManager(session_id="my-session")
    >>> 
    >>> # Capture page source
    >>> snapshot = await manager.capture_html(page)
    >>> 
    >>> # Enable HAR logging
    >>> await manager.start_har_capture(page)
    >>> # ... navigate and interact ...
    >>> har = await manager.stop_har_capture()
    >>> har.export("traffic.har")
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse

from flybrowser.utils.logger import logger

try:
    from playwright.async_api import Page, Request, Response, Route
except ImportError:
    Page = Any
    Request = Any
    Response = Any
    Route = Any

try:
    import difflib
except ImportError:
    difflib = None


class CaptureType(str, Enum):
    """Types of source captures."""
    HTML = "html"
    DOM = "dom"
    MHTML = "mhtml"
    SCREENSHOT = "screenshot"
    FULL_PAGE = "full_page"


class ResourceType(str, Enum):
    """Types of captured resources."""
    DOCUMENT = "document"
    STYLESHEET = "stylesheet"
    SCRIPT = "script"
    IMAGE = "image"
    FONT = "font"
    XHR = "xhr"
    FETCH = "fetch"
    WEBSOCKET = "websocket"
    OTHER = "other"


@dataclass
class CapturedResource:
    """A captured network resource."""
    url: str
    resource_type: ResourceType
    mime_type: str = ""
    content: bytes = b""
    content_hash: str = ""
    size_bytes: int = 0
    encoded: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "resource_type": self.resource_type.value,
            "mime_type": self.mime_type,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "encoded": self.encoded,
        }
    
    def get_content_base64(self) -> str:
        """Get content as base64 string."""
        return base64.b64encode(self.content).decode("utf-8")


@dataclass
class HTMLSnapshot:
    """
    A captured HTML snapshot with metadata.
    
    Contains the full HTML source, optional inline resources,
    and capture metadata.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    
    # Page info
    url: str = ""
    title: str = ""
    
    # Content
    html: str = ""
    dom_tree: Optional[Dict[str, Any]] = None
    
    # Resources (if captured)
    resources: List[CapturedResource] = field(default_factory=list)
    
    # Metadata
    captured_at: float = field(default_factory=time.time)
    capture_type: CaptureType = CaptureType.HTML
    viewport: Dict[str, int] = field(default_factory=dict)
    scroll_position: Dict[str, int] = field(default_factory=dict)
    
    # Metrics
    html_size_bytes: int = 0
    total_resources_size_bytes: int = 0
    
    def __post_init__(self):
        if self.html and not self.html_size_bytes:
            self.html_size_bytes = len(self.html.encode("utf-8"))
        if self.resources and not self.total_resources_size_bytes:
            self.total_resources_size_bytes = sum(r.size_bytes for r in self.resources)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "url": self.url,
            "title": self.title,
            "captured_at": self.captured_at,
            "capture_type": self.capture_type.value,
            "viewport": self.viewport,
            "scroll_position": self.scroll_position,
            "html_size_bytes": self.html_size_bytes,
            "total_resources_size_bytes": self.total_resources_size_bytes,
            "resources_count": len(self.resources),
        }
    
    def export_html(self, path: Union[str, Path]) -> None:
        """Export HTML to file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.html)
        logger.info(f"[SOURCE_CAPTURE] Exported HTML to {path}")
    
    def export_mhtml(self, path: Union[str, Path]) -> None:
        """Export as MHTML archive with embedded resources."""
        path = Path(path)
        
        boundary = f"----=_Part_{uuid.uuid4().hex}"
        parts = []
        
        # Main HTML part
        parts.append(f"""Content-Type: text/html; charset="utf-8"
Content-Transfer-Encoding: quoted-printable
Content-Location: {self.url}

{self.html}""")
        
        # Resource parts
        for resource in self.resources:
            content_b64 = resource.get_content_base64()
            parts.append(f"""Content-Type: {resource.mime_type}
Content-Transfer-Encoding: base64
Content-Location: {resource.url}

{content_b64}""")
        
        # Assemble MHTML
        mhtml = f"""From: <Saved by FlyBrowser>
Subject: {self.title}
Date: {datetime.fromtimestamp(self.captured_at).isoformat()}
MIME-Version: 1.0
Content-Type: multipart/related; boundary="{boundary}"

"""
        
        for part in parts:
            mhtml += f"--{boundary}\n{part}\n\n"
        
        mhtml += f"--{boundary}--"
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(mhtml)
        
        logger.info(f"[SOURCE_CAPTURE] Exported MHTML to {path}")
    
    def diff_from(self, other: "HTMLSnapshot") -> Optional[str]:
        """Get unified diff from another snapshot."""
        if difflib is None:
            return None
        
        diff = difflib.unified_diff(
            other.html.splitlines(keepends=True),
            self.html.splitlines(keepends=True),
            fromfile=f"snapshot_{other.id[:8]}",
            tofile=f"snapshot_{self.id[:8]}",
        )
        return "".join(diff)


@dataclass
class HAREntry:
    """A single HAR (HTTP Archive) entry."""
    started_at: float
    request_url: str
    request_method: str
    request_headers: List[Dict[str, str]]
    request_body: Optional[str] = None
    
    response_status: int = 0
    response_status_text: str = ""
    response_headers: List[Dict[str, str]] = field(default_factory=list)
    response_body: Optional[bytes] = None
    response_body_size: int = 0
    response_mime_type: str = ""
    
    ended_at: Optional[float] = None
    time_ms: float = 0.0
    
    # Timing breakdown
    timings: Dict[str, float] = field(default_factory=dict)
    
    def complete(
        self,
        status: int,
        status_text: str,
        headers: List[Dict[str, str]],
        body: Optional[bytes] = None,
        mime_type: str = "",
    ) -> None:
        """Complete the entry with response data."""
        self.ended_at = time.time()
        self.time_ms = (self.ended_at - self.started_at) * 1000
        self.response_status = status
        self.response_status_text = status_text
        self.response_headers = headers
        self.response_body = body
        self.response_body_size = len(body) if body else 0
        self.response_mime_type = mime_type
    
    def to_har_dict(self) -> Dict[str, Any]:
        """Convert to HAR format dict."""
        return {
            "startedDateTime": datetime.fromtimestamp(self.started_at).isoformat() + "Z",
            "time": self.time_ms,
            "request": {
                "method": self.request_method,
                "url": self.request_url,
                "httpVersion": "HTTP/1.1",
                "headers": self.request_headers,
                "queryString": [],
                "cookies": [],
                "headersSize": -1,
                "bodySize": len(self.request_body.encode()) if self.request_body else 0,
                "postData": {
                    "mimeType": "application/x-www-form-urlencoded",
                    "text": self.request_body,
                } if self.request_body else None,
            },
            "response": {
                "status": self.response_status,
                "statusText": self.response_status_text,
                "httpVersion": "HTTP/1.1",
                "headers": self.response_headers,
                "cookies": [],
                "content": {
                    "size": self.response_body_size,
                    "mimeType": self.response_mime_type,
                    "text": base64.b64encode(self.response_body).decode() if self.response_body else "",
                    "encoding": "base64" if self.response_body else None,
                },
                "redirectURL": "",
                "headersSize": -1,
                "bodySize": self.response_body_size,
            },
            "cache": {},
            "timings": self.timings or {
                "send": 0,
                "wait": self.time_ms,
                "receive": 0,
            },
        }


@dataclass
class HARLog:
    """
    HTTP Archive (HAR) log for network traffic.
    
    Standard format for capturing and exporting network requests/responses.
    """
    session_id: str = ""
    entries: List[HAREntry] = field(default_factory=list)
    
    # Capture window
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    
    # Page info
    pages: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_page(self, url: str, title: str = "") -> str:
        """Add a page reference and return its ID."""
        page_id = f"page_{len(self.pages) + 1}"
        self.pages.append({
            "startedDateTime": datetime.now().isoformat() + "Z",
            "id": page_id,
            "title": title or url,
            "pageTimings": {
                "onContentLoad": -1,
                "onLoad": -1,
            },
        })
        return page_id
    
    def add_entry(self, entry: HAREntry) -> None:
        """Add a HAR entry."""
        self.entries.append(entry)
    
    def to_har_dict(self) -> Dict[str, Any]:
        """Convert to full HAR format."""
        return {
            "log": {
                "version": "1.2",
                "creator": {
                    "name": "FlyBrowser",
                    "version": "1.0.0",
                },
                "browser": {
                    "name": "Chromium",
                    "version": "latest",
                },
                "pages": self.pages,
                "entries": [e.to_har_dict() for e in self.entries],
            }
        }
    
    def export(self, path: Union[str, Path]) -> None:
        """Export HAR to file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_har_dict(), f, indent=2)
        logger.info(f"[SOURCE_CAPTURE] Exported HAR with {len(self.entries)} entries to {path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_size = sum(e.response_body_size for e in self.entries)
        total_time = sum(e.time_ms for e in self.entries)
        
        by_type: Dict[str, int] = {}
        for entry in self.entries:
            mime = entry.response_mime_type.split("/")[0] if entry.response_mime_type else "unknown"
            by_type[mime] = by_type.get(mime, 0) + 1
        
        return {
            "total_requests": len(self.entries),
            "total_size_bytes": total_size,
            "total_time_ms": total_time,
            "by_type": by_type,
            "errors": len([e for e in self.entries if e.response_status >= 400]),
        }


class SourceCaptureManager:
    """
    Main source capture manager.
    
    Handles HTML snapshots, DOM capture, and HAR logging for
    comprehensive session debugging.
    
    Example:
        >>> manager = SourceCaptureManager(session_id="my-session")
        >>> 
        >>> # Capture current page state
        >>> snapshot = await manager.capture_html(page)
        >>> 
        >>> # Start network capture
        >>> await manager.start_har_capture(page)
        >>> # ... page interactions ...
        >>> har = await manager.stop_har_capture()
        >>> 
        >>> # Export all captures
        >>> manager.export_all("./captures/")
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        enabled: bool = True,
        capture_resources: bool = False,
        max_resource_size_bytes: int = 5 * 1024 * 1024,  # 5MB
        max_snapshots: int = 100,
    ):
        """
        Initialize source capture manager.
        
        Args:
            session_id: Session identifier
            enabled: Whether capture is enabled
            capture_resources: Whether to capture inline resources
            max_resource_size_bytes: Max size for individual resources
            max_snapshots: Maximum snapshots to keep
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.enabled = enabled
        self.capture_resources = capture_resources
        self.max_resource_size_bytes = max_resource_size_bytes
        self.max_snapshots = max_snapshots
        
        self._snapshots: List[HTMLSnapshot] = []
        self._har_log: Optional[HARLog] = None
        self._har_capturing = False
        self._pending_requests: Dict[str, HAREntry] = {}
        self._lock = asyncio.Lock()
        
        logger.info(f"[SOURCE_CAPTURE] Initialized for session {self.session_id}")
    
    async def capture_html(
        self,
        page: Page,
        include_resources: Optional[bool] = None,
        capture_type: CaptureType = CaptureType.HTML,
    ) -> HTMLSnapshot:
        """
        Capture current page HTML.
        
        Args:
            page: Playwright page
            include_resources: Whether to inline resources (overrides default)
            capture_type: Type of capture
            
        Returns:
            HTMLSnapshot with captured content
        """
        if not self.enabled:
            return HTMLSnapshot()
        
        include_res = include_resources if include_resources is not None else self.capture_resources
        
        # Get basic page info
        url = page.url
        title = await page.title()
        html = await page.content()
        
        # Get viewport and scroll position
        viewport = await page.evaluate("""() => ({
            width: window.innerWidth,
            height: window.innerHeight
        })""")
        
        scroll_position = await page.evaluate("""() => ({
            x: window.scrollX,
            y: window.scrollY
        })""")
        
        # Capture resources if requested
        resources = []
        if include_res:
            resources = await self._capture_resources(page, html, url)
        
        snapshot = HTMLSnapshot(
            session_id=self.session_id,
            url=url,
            title=title,
            html=html,
            capture_type=capture_type,
            viewport=viewport,
            scroll_position=scroll_position,
            resources=resources,
        )
        
        # Add to history
        async with self._lock:
            self._snapshots.append(snapshot)
            if len(self._snapshots) > self.max_snapshots:
                self._snapshots = self._snapshots[-self.max_snapshots:]
        
        logger.info(f"[SOURCE_CAPTURE] Captured HTML snapshot: {url} ({snapshot.html_size_bytes} bytes)")
        return snapshot
    
    async def capture_dom(self, page: Page) -> HTMLSnapshot:
        """
        Capture DOM tree structure.
        
        Returns simplified DOM tree without full HTML.
        """
        if not self.enabled:
            return HTMLSnapshot()
        
        dom_tree = await page.evaluate("""() => {
            function captureNode(node, depth = 0) {
                if (depth > 20) return null;  // Max depth
                
                const result = {
                    tag: node.tagName?.toLowerCase() || '#text',
                    children: []
                };
                
                if (node.id) result.id = node.id;
                if (node.className && typeof node.className === 'string') {
                    result.class = node.className.split(' ').filter(c => c);
                }
                
                // Capture text for leaf nodes
                if (node.nodeType === Node.TEXT_NODE) {
                    const text = node.textContent?.trim();
                    if (text) result.text = text.substring(0, 100);
                    return result;
                }
                
                // Capture key attributes
                if (node.href) result.href = node.href;
                if (node.src) result.src = node.src;
                if (node.type) result.type = node.type;
                if (node.name) result.name = node.name;
                
                // Capture children
                for (const child of node.childNodes || []) {
                    const captured = captureNode(child, depth + 1);
                    if (captured && (captured.tag !== '#text' || captured.text)) {
                        result.children.push(captured);
                    }
                }
                
                return result;
            }
            
            return captureNode(document.documentElement);
        }""")
        
        snapshot = HTMLSnapshot(
            session_id=self.session_id,
            url=page.url,
            title=await page.title(),
            dom_tree=dom_tree,
            capture_type=CaptureType.DOM,
        )
        
        async with self._lock:
            self._snapshots.append(snapshot)
        
        logger.info(f"[SOURCE_CAPTURE] Captured DOM tree: {page.url}")
        return snapshot
    
    async def capture_full_page(self, page: Page, output_dir: Union[str, Path]) -> HTMLSnapshot:
        """
        Capture full page with screenshot and resources.
        
        Creates a complete archive of the page state.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Capture HTML with resources
        snapshot = await self.capture_html(page, include_resources=True, capture_type=CaptureType.FULL_PAGE)
        
        # Take screenshot
        screenshot_path = output_dir / f"screenshot_{snapshot.id[:8]}.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        
        # Save HTML
        html_path = output_dir / f"page_{snapshot.id[:8]}.html"
        snapshot.export_html(html_path)
        
        # Save MHTML archive
        mhtml_path = output_dir / f"archive_{snapshot.id[:8]}.mhtml"
        snapshot.export_mhtml(mhtml_path)
        
        logger.info(f"[SOURCE_CAPTURE] Full page capture saved to {output_dir}")
        return snapshot
    
    async def _capture_resources(
        self, 
        page: Page, 
        html: str, 
        base_url: str,
    ) -> List[CapturedResource]:
        """Capture page resources (CSS, JS, images)."""
        resources = []
        
        # Extract resource URLs from HTML
        css_urls = re.findall(r'<link[^>]+href=["\']([^"\']+)["\'][^>]*>', html)
        js_urls = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html)
        img_urls = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html)
        
        all_urls = [
            (url, ResourceType.STYLESHEET) for url in css_urls
        ] + [
            (url, ResourceType.SCRIPT) for url in js_urls
        ] + [
            (url, ResourceType.IMAGE) for url in img_urls
        ]
        
        for url, res_type in all_urls[:50]:  # Limit to 50 resources
            try:
                # Resolve relative URLs
                full_url = urljoin(base_url, url)
                
                # Fetch resource
                response = await page.request.get(full_url)
                if response.ok:
                    content = await response.body()
                    
                    if len(content) <= self.max_resource_size_bytes:
                        resource = CapturedResource(
                            url=full_url,
                            resource_type=res_type,
                            mime_type=response.headers.get("content-type", ""),
                            content=content,
                            content_hash=hashlib.md5(content).hexdigest(),
                            size_bytes=len(content),
                        )
                        resources.append(resource)
            except Exception as e:
                logger.debug(f"[SOURCE_CAPTURE] Failed to capture resource {url}: {e}")
        
        return resources
    
    async def start_har_capture(self, page: Page) -> None:
        """
        Start capturing network traffic as HAR.
        
        Must call stop_har_capture() to get results.
        """
        if not self.enabled or self._har_capturing:
            return
        
        self._har_log = HARLog(session_id=self.session_id)
        self._har_log.started_at = time.time()
        self._har_log.add_page(page.url, await page.title())
        self._pending_requests = {}
        self._har_capturing = True
        
        # Set up request interception
        async def on_request(request: Request) -> None:
            if not self._har_capturing:
                return
            
            entry = HAREntry(
                started_at=time.time(),
                request_url=request.url,
                request_method=request.method,
                request_headers=[{"name": k, "value": v} for k, v in request.headers.items()],
                request_body=request.post_data,
            )
            
            self._pending_requests[request.url + str(id(request))] = entry
        
        async def on_response(response: Response) -> None:
            if not self._har_capturing:
                return
            
            request = response.request
            key = request.url + str(id(request))
            
            if key in self._pending_requests:
                entry = self._pending_requests.pop(key)
                
                # Get response body if small enough
                body = None
                try:
                    body = await response.body()
                    if len(body) > self.max_resource_size_bytes:
                        body = None  # Too large
                except Exception:
                    pass
                
                entry.complete(
                    status=response.status,
                    status_text=response.status_text,
                    headers=[{"name": k, "value": v} for k, v in response.headers.items()],
                    body=body,
                    mime_type=response.headers.get("content-type", ""),
                )
                
                self._har_log.add_entry(entry)
        
        page.on("request", on_request)
        page.on("response", on_response)
        
        # Store handlers for cleanup
        self._request_handler = on_request
        self._response_handler = on_response
        self._har_page = page
        
        logger.info(f"[SOURCE_CAPTURE] Started HAR capture for {page.url}")
    
    async def stop_har_capture(self) -> Optional[HARLog]:
        """
        Stop HAR capture and return the log.
        
        Returns:
            HARLog with captured traffic, or None if not capturing
        """
        if not self._har_capturing or not self._har_log:
            return None
        
        self._har_capturing = False
        self._har_log.ended_at = time.time()
        
        # Remove handlers
        if hasattr(self, "_har_page") and self._har_page:
            try:
                self._har_page.remove_listener("request", self._request_handler)
                self._har_page.remove_listener("response", self._response_handler)
            except Exception:
                pass
        
        har = self._har_log
        self._har_log = None
        self._pending_requests = {}
        
        summary = har.get_summary()
        logger.info(
            f"[SOURCE_CAPTURE] Stopped HAR capture: "
            f"{summary['total_requests']} requests, "
            f"{summary['total_size_bytes']} bytes"
        )
        
        return har
    
    def get_snapshots(self) -> List[HTMLSnapshot]:
        """Get all captured snapshots."""
        return list(self._snapshots)
    
    def get_latest_snapshot(self) -> Optional[HTMLSnapshot]:
        """Get the most recent snapshot."""
        return self._snapshots[-1] if self._snapshots else None
    
    def diff_snapshots(
        self, 
        index1: int = -2, 
        index2: int = -1,
    ) -> Optional[str]:
        """
        Get diff between two snapshots.
        
        Args:
            index1: Index of first snapshot (default: second to last)
            index2: Index of second snapshot (default: last)
            
        Returns:
            Unified diff string, or None if not enough snapshots
        """
        if len(self._snapshots) < 2:
            return None
        
        try:
            snapshot1 = self._snapshots[index1]
            snapshot2 = self._snapshots[index2]
            return snapshot2.diff_from(snapshot1)
        except (IndexError, TypeError):
            return None
    
    def clear(self) -> None:
        """Clear all captured data."""
        self._snapshots = []
        self._har_log = None
        self._har_capturing = False
        self._pending_requests = {}
    
    def export_all(self, output_dir: Union[str, Path]) -> None:
        """
        Export all captures to directory.
        
        Creates:
        - snapshots/ - HTML and MHTML files
        - har/ - HAR file if captured
        - index.json - Index of all captures
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export snapshots
        snapshots_dir = output_dir / "snapshots"
        snapshots_dir.mkdir(exist_ok=True)
        
        snapshot_index = []
        for snapshot in self._snapshots:
            filename = f"snapshot_{snapshot.id[:8]}"
            
            # Export HTML
            snapshot.export_html(snapshots_dir / f"{filename}.html")
            
            # Export MHTML if has resources
            if snapshot.resources:
                snapshot.export_mhtml(snapshots_dir / f"{filename}.mhtml")
            
            snapshot_index.append({
                **snapshot.to_dict(),
                "files": {
                    "html": f"snapshots/{filename}.html",
                    "mhtml": f"snapshots/{filename}.mhtml" if snapshot.resources else None,
                },
            })
        
        # Export HAR if available
        har_info = None
        if self._har_log and self._har_log.entries:
            har_dir = output_dir / "har"
            har_dir.mkdir(exist_ok=True)
            har_path = har_dir / "traffic.har"
            self._har_log.export(har_path)
            har_info = {
                "path": "har/traffic.har",
                "summary": self._har_log.get_summary(),
            }
        
        # Write index
        index = {
            "session_id": self.session_id,
            "exported_at": time.time(),
            "snapshots": snapshot_index,
            "har": har_info,
        }
        
        with open(output_dir / "index.json", "w") as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"[SOURCE_CAPTURE] Exported all captures to {output_dir}")
