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
Worker Node for FlyBrowser Cluster.

A worker node handles browser instances and executes automation tasks
as directed by the cluster coordinator.

Example:
    >>> worker = WorkerNode(
    ...     coordinator_url="http://coordinator:8001",
    ...     max_browsers=10
    ... )
    >>> await worker.start()
    >>> # Worker is now registered and accepting tasks
    >>> await worker.stop()
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

import aiohttp

from flybrowser.core.browser_pool import BrowserPool, PoolConfig
from flybrowser.service.cluster.protocol import (
    MessageType,
    NodeInfo,
    NodeMessage,
    NodeStatus,
)
from flybrowser.service.session_manager import SessionManager
from flybrowser.utils.logger import logger


class WorkerNode:
    """Worker node in a FlyBrowser cluster.
    
    The worker node:
    - Registers with the coordinator on startup
    - Sends periodic heartbeats with status updates
    - Manages local browser pool and sessions
    - Executes tasks assigned by the coordinator
    
    Attributes:
        coordinator_url: URL of the cluster coordinator
        host: Host address for this worker's API
        port: Port for this worker's API
        max_browsers: Maximum browsers this worker can handle
        heartbeat_interval: Interval between heartbeats
    """
    
    def __init__(
        self,
        coordinator_url: str = "http://localhost:8001",
        host: str = "0.0.0.0",
        port: int = 8000,
        max_browsers: int = 10,
        heartbeat_interval: float = 5.0,
    ) -> None:
        """Initialize the worker node."""
        self.coordinator_url = coordinator_url.rstrip("/")
        self.host = host
        self.port = port
        self.max_browsers = max_browsers
        self.heartbeat_interval = heartbeat_interval
        
        self._node_info = NodeInfo(
            host=host,
            port=port,
            role="worker",
            status=NodeStatus.STARTING,
            max_browsers=max_browsers,
        )
        
        self._session_manager: Optional[SessionManager] = None
        self._browser_pool: Optional[BrowserPool] = None
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._http_session: Optional[aiohttp.ClientSession] = None
    
    @property
    def node_id(self) -> str:
        """Get the node ID."""
        return self._node_info.node_id
    
    async def start(self) -> None:
        """Start the worker node."""
        if self._running:
            logger.warning("Worker node already running")
            return
        
        logger.info(f"Starting worker node {self.node_id}")
        
        # Initialize HTTP session
        self._http_session = aiohttp.ClientSession()
        
        # Initialize browser pool
        pool_config = PoolConfig(
            min_size=1,
            max_size=self.max_browsers,
            headless=True,
        )
        self._browser_pool = BrowserPool(pool_config)
        await self._browser_pool.start()
        
        # Initialize session manager
        self._session_manager = SessionManager(max_sessions=self.max_browsers * 2)
        
        self._running = True
        self._node_info.status = NodeStatus.READY
        
        # Register with coordinator
        await self._register()
        
        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        logger.info(f"Worker node {self.node_id} started")
    
    async def stop(self) -> None:
        """Stop the worker node."""
        if not self._running:
            return
        
        logger.info(f"Stopping worker node {self.node_id}")
        self._running = False
        self._node_info.status = NodeStatus.DRAINING
        
        # Unregister from coordinator
        await self._unregister()
        
        # Stop heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup sessions
        if self._session_manager:
            await self._session_manager.cleanup_all()
        
        # Stop browser pool
        if self._browser_pool:
            await self._browser_pool.stop()
        
        # Close HTTP session
        if self._http_session:
            await self._http_session.close()

        self._node_info.status = NodeStatus.OFFLINE
        logger.info(f"Worker node {self.node_id} stopped")

    def _update_node_info(self) -> None:
        """Update node info with current stats."""
        if self._session_manager:
            self._node_info.active_sessions = self._session_manager.get_active_session_count()
        if self._browser_pool:
            stats = self._browser_pool.get_stats()
            self._node_info.active_browsers = stats.get("busy_sessions", 0)

    async def _register(self) -> bool:
        """Register with the coordinator."""
        try:
            message = NodeMessage.create_register(self.node_id, self._node_info)

            async with self._http_session.post(
                f"{self.coordinator_url}/cluster/register",
                json=message.to_dict(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    logger.info(f"Registered with coordinator at {self.coordinator_url}")
                    return True
                else:
                    logger.error(f"Failed to register: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error registering with coordinator: {e}")
            return False

    async def _unregister(self) -> bool:
        """Unregister from the coordinator."""
        try:
            message = NodeMessage(
                message_type=MessageType.UNREGISTER,
                sender_id=self.node_id,
                payload={"node_id": self.node_id},
            )

            async with self._http_session.post(
                f"{self.coordinator_url}/cluster/unregister",
                json=message.to_dict(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Error unregistering from coordinator: {e}")
            return False

    async def _send_heartbeat(self) -> Optional[Dict[str, Any]]:
        """Send heartbeat to coordinator."""
        try:
            self._update_node_info()
            message = NodeMessage.create_heartbeat(self.node_id, self._node_info)

            async with self._http_session.post(
                f"{self.coordinator_url}/cluster/heartbeat",
                json=message.to_dict(),
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("payload", {}).get("cluster_status")
                return None
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
            return None

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat loop."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                cluster_status = await self._send_heartbeat()

                if cluster_status:
                    logger.debug(
                        f"Heartbeat OK - Cluster: {cluster_status.get('node_count', 0)} nodes, "
                        f"{cluster_status.get('available_capacity', 0)} capacity"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current worker status."""
        self._update_node_info()
        return {
            "node_info": self._node_info.to_dict(),
            "running": self._running,
            "coordinator_url": self.coordinator_url,
        }

    async def __aenter__(self) -> "WorkerNode":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

