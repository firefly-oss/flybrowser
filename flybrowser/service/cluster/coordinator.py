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
Cluster Coordinator for FlyBrowser.

The coordinator manages worker nodes in a cluster deployment, handling:
- Worker registration and discovery
- Health monitoring via heartbeats
- Load balancing and task distribution
- Cluster-wide session management

Example:
    >>> coordinator = ClusterCoordinator(host="0.0.0.0", port=8001)
    >>> await coordinator.start()
    >>> # Workers can now register and receive tasks
    >>> await coordinator.stop()
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from flybrowser.service.cluster.protocol import (
    MessageType,
    NodeInfo,
    NodeMessage,
    NodeStatus,
)
from flybrowser.utils.logger import logger


class ClusterCoordinator:
    """Manages worker nodes in a FlyBrowser cluster.
    
    The coordinator is responsible for:
    - Accepting worker registrations
    - Monitoring worker health via heartbeats
    - Distributing sessions across workers based on load
    - Handling worker failures and rebalancing
    
    Attributes:
        host: Host address to bind to
        port: Port for the coordinator API
        node_timeout: Seconds before a node is considered dead
        heartbeat_interval: Expected heartbeat interval from workers
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8001,
        node_timeout: float = 30.0,
        heartbeat_interval: float = 5.0,
    ) -> None:
        """Initialize the cluster coordinator."""
        self.host = host
        self.port = port
        self.node_timeout = node_timeout
        self.heartbeat_interval = heartbeat_interval
        
        self._nodes: Dict[str, NodeInfo] = {}
        self._session_assignments: Dict[str, str] = {}  # session_id -> node_id
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Coordinator's own node info
        self._node_info = NodeInfo(
            host=host,
            port=port,
            role="coordinator",
            status=NodeStatus.READY,
        )
    
    async def start(self) -> None:
        """Start the coordinator."""
        if self._running:
            logger.warning("Coordinator already running")
            return
        
        logger.info(f"Starting cluster coordinator on {self.host}:{self.port}")
        self._running = True
        self._node_info.status = NodeStatus.READY
        
        # Start health monitoring
        self._monitor_task = asyncio.create_task(self._monitor_nodes())
        
        logger.info("Cluster coordinator started")
    
    async def stop(self) -> None:
        """Stop the coordinator."""
        if not self._running:
            return
        
        logger.info("Stopping cluster coordinator")
        self._running = False
        self._node_info.status = NodeStatus.OFFLINE
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Cluster coordinator stopped")
    
    async def register_node(self, node_info: NodeInfo) -> bool:
        """Register a worker node with the cluster."""
        async with self._lock:
            if node_info.node_id in self._nodes:
                logger.warning(f"Node {node_info.node_id} already registered, updating")
            
            node_info.status = NodeStatus.READY
            node_info.last_heartbeat = time.time()
            self._nodes[node_info.node_id] = node_info
            
            logger.info(
                f"Node registered: {node_info.node_id} "
                f"({node_info.host}:{node_info.port}, capacity={node_info.max_browsers})"
            )
            return True
    
    async def unregister_node(self, node_id: str) -> bool:
        """Unregister a worker node from the cluster."""
        async with self._lock:
            if node_id not in self._nodes:
                return False
            
            del self._nodes[node_id]
            
            # Reassign sessions from this node
            orphaned = [sid for sid, nid in self._session_assignments.items() if nid == node_id]
            for session_id in orphaned:
                del self._session_assignments[session_id]
            
            logger.info(f"Node unregistered: {node_id} ({len(orphaned)} sessions orphaned)")
            return True
    
    async def handle_heartbeat(self, message: NodeMessage) -> NodeMessage:
        """Handle a heartbeat from a worker node."""
        node_info_data = message.payload.get("node_info", {})
        node_info = NodeInfo.from_dict(node_info_data)

        async with self._lock:
            if node_info.node_id in self._nodes:
                # Update existing node
                existing = self._nodes[node_info.node_id]
                existing.active_browsers = node_info.active_browsers
                existing.active_sessions = node_info.active_sessions
                existing.status = node_info.status
                existing.last_heartbeat = time.time()
            else:
                # Auto-register new node
                await self.register_node(node_info)

        return NodeMessage(
            message_type=MessageType.HEARTBEAT_ACK,
            sender_id=self._node_info.node_id,
            recipient_id=message.sender_id,
            correlation_id=message.message_id,
            payload={"cluster_status": self.get_cluster_status()},
        )

    def select_node_for_session(self) -> Optional[NodeInfo]:
        """Select the best node for a new session based on load."""
        available_nodes = [
            node for node in self._nodes.values()
            if node.is_available
        ]

        if not available_nodes:
            return None

        # Select node with most available capacity
        return max(available_nodes, key=lambda n: n.available_capacity)

    async def assign_session(self, session_id: str) -> Optional[str]:
        """Assign a session to a worker node."""
        async with self._lock:
            node = self.select_node_for_session()
            if not node:
                logger.warning("No available nodes for session assignment")
                return None

            self._session_assignments[session_id] = node.node_id
            logger.info(f"Session {session_id} assigned to node {node.node_id}")
            return node.node_id

    def get_node_for_session(self, session_id: str) -> Optional[NodeInfo]:
        """Get the node handling a specific session."""
        node_id = self._session_assignments.get(session_id)
        if node_id:
            return self._nodes.get(node_id)
        return None

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status."""
        total_capacity = sum(n.max_browsers for n in self._nodes.values())
        total_active = sum(n.active_browsers for n in self._nodes.values())

        return {
            "coordinator_id": self._node_info.node_id,
            "node_count": len(self._nodes),
            "total_capacity": total_capacity,
            "total_active": total_active,
            "available_capacity": total_capacity - total_active,
            "session_count": len(self._session_assignments),
            "nodes": [n.to_dict() for n in self._nodes.values()],
        }

    def get_nodes(self) -> List[NodeInfo]:
        """Get all registered nodes."""
        return list(self._nodes.values())

    async def _monitor_nodes(self) -> None:
        """Monitor node health and remove dead nodes."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                now = time.time()
                dead_nodes = []

                async with self._lock:
                    for node_id, node in self._nodes.items():
                        if now - node.last_heartbeat > self.node_timeout:
                            dead_nodes.append(node_id)
                            logger.warning(f"Node {node_id} timed out")

                for node_id in dead_nodes:
                    await self.unregister_node(node_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in node monitor: {e}")

    async def __aenter__(self) -> "ClusterCoordinator":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

