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
Raft RPC Transport Layer for FlyBrowser HA Cluster.

This module provides the network transport for Raft consensus messages.
It uses HTTP/JSON for simplicity and compatibility, with support for:
- Async request/response
- Connection pooling
- Timeout handling
- Retry logic
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

import aiohttp
from aiohttp import web

from flybrowser.service.cluster.raft.messages import (
    AppendEntriesRequest,
    AppendEntriesResponse,
    RaftMessage,
    RequestVoteRequest,
    RequestVoteResponse,
    parse_message,
)
from flybrowser.utils.logger import logger

if TYPE_CHECKING:
    from flybrowser.service.cluster.raft.node import RaftNode


class RaftTransport:
    """HTTP-based transport for Raft RPC messages.
    
    Provides both server (receiving RPCs) and client (sending RPCs) functionality.
    
    Example:
        >>> transport = RaftTransport("0.0.0.0", 4321)
        >>> transport.set_handler(raft_node.handle_rpc)
        >>> await transport.start()
        >>> response = await transport.send_request_vote("peer:4321", request)
        >>> await transport.stop()
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 4321,
        timeout_ms: int = 200,
    ) -> None:
        """Initialize the transport.
        
        Args:
            host: Host to bind the RPC server
            port: Port to bind the RPC server
            timeout_ms: Timeout for RPC calls in milliseconds
        """
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._session: Optional[aiohttp.ClientSession] = None
        
        # RPC handler (set by RaftNode)
        self._handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
        
        self._running = False
    
    def set_handler(self, handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """Set the RPC handler function.
        
        Args:
            handler: Function that takes a message dict and returns a response dict
        """
        self._handler = handler
    
    async def start(self) -> None:
        """Start the transport server."""
        if self._running:
            return
        
        # Create HTTP client session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=100,  # Max connections
            limit_per_host=10,  # Max per host
            keepalive_timeout=30,
        )
        self._session = aiohttp.ClientSession(connector=connector)
        
        # Create and start HTTP server
        self._app = web.Application()
        self._app.router.add_post("/raft/rpc", self._handle_rpc)
        self._app.router.add_get("/raft/health", self._handle_health)
        
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        
        self._running = True
        logger.info(f"Raft transport started on {self.host}:{self.port}")
    
    async def stop(self) -> None:
        """Stop the transport server."""
        if not self._running:
            return
        
        self._running = False
        
        if self._session:
            await self._session.close()
            self._session = None
        
        if self._site:
            await self._site.stop()
        
        if self._runner:
            await self._runner.cleanup()
        
        logger.info("Raft transport stopped")
    
    async def _handle_rpc(self, request: web.Request) -> web.Response:
        """Handle incoming RPC request."""
        try:
            data = await request.json()
            
            if self._handler:
                response = self._handler(data)
                return web.json_response(response)
            else:
                return web.json_response({"error": "No handler"}, status=500)
        
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            logger.error(f"RPC handler error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle health check request."""
        return web.json_response({"status": "ok", "running": self._running})

    # ==================== Client Methods ====================

    async def _send_rpc(
        self,
        peer_address: str,
        message: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Send an RPC to a peer and return the response.

        Args:
            peer_address: Peer address in host:port format
            message: Message to send

        Returns:
            Response dict or None on failure
        """
        if not self._session:
            return None

        url = f"http://{peer_address}/raft/rpc"
        timeout = aiohttp.ClientTimeout(total=self.timeout_ms / 1000)

        try:
            async with self._session.post(url, json=message, timeout=timeout) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.warning(f"RPC to {peer_address} failed: {resp.status}")
                    return None
        except asyncio.TimeoutError:
            logger.debug(f"RPC to {peer_address} timed out")
            return None
        except aiohttp.ClientError as e:
            logger.debug(f"RPC to {peer_address} failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected RPC error to {peer_address}: {e}")
            return None

    async def send_request_vote(
        self,
        peer_address: str,
        request: RequestVoteRequest,
    ) -> Optional[RequestVoteResponse]:
        """Send a RequestVote RPC to a peer.

        Args:
            peer_address: Peer address in host:port format
            request: The RequestVote request

        Returns:
            RequestVoteResponse or None on failure
        """
        response = await self._send_rpc(peer_address, request.to_dict())
        if response:
            return RequestVoteResponse.from_dict(response)
        return None

    async def send_append_entries(
        self,
        peer_address: str,
        request: AppendEntriesRequest,
    ) -> Optional[AppendEntriesResponse]:
        """Send an AppendEntries RPC to a peer.

        Args:
            peer_address: Peer address in host:port format
            request: The AppendEntries request

        Returns:
            AppendEntriesResponse or None on failure
        """
        response = await self._send_rpc(peer_address, request.to_dict())
        if response:
            return AppendEntriesResponse.from_dict(response)
        return None

    async def send_append_entries_batch(
        self,
        peers: Dict[str, AppendEntriesRequest],
    ) -> Dict[str, Optional[AppendEntriesResponse]]:
        """Send AppendEntries RPCs to multiple peers concurrently.

        Args:
            peers: Dict mapping peer_address to request

        Returns:
            Dict mapping peer_address to response (or None on failure)
        """
        tasks = {
            peer: self.send_append_entries(peer, request)
            for peer, request in peers.items()
        }

        results = {}
        responses = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for peer, response in zip(tasks.keys(), responses):
            if isinstance(response, Exception):
                results[peer] = None
            else:
                results[peer] = response

        return results

    async def send_request_vote_batch(
        self,
        peers: Dict[str, RequestVoteRequest],
    ) -> Dict[str, Optional[RequestVoteResponse]]:
        """Send RequestVote RPCs to multiple peers concurrently.

        Args:
            peers: Dict mapping peer_address to request

        Returns:
            Dict mapping peer_address to response (or None on failure)
        """
        tasks = {
            peer: self.send_request_vote(peer, request)
            for peer, request in peers.items()
        }

        results = {}
        responses = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for peer, response in zip(tasks.keys(), responses):
            if isinstance(response, Exception):
                results[peer] = None
            else:
                results[peer] = response

        return results

    async def check_peer_health(self, peer_address: str) -> bool:
        """Check if a peer is healthy.

        Args:
            peer_address: Peer address in host:port format

        Returns:
            True if peer is healthy
        """
        if not self._session:
            return False

        url = f"http://{peer_address}/raft/health"
        timeout = aiohttp.ClientTimeout(total=1.0)

        try:
            async with self._session.get(url, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

