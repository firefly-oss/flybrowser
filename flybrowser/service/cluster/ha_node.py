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
High-Availability Cluster Node for FlyBrowser.

This module provides the HAClusterNode class which integrates:
- Raft consensus for leader election and state replication
- Load balancing for session distribution
- Automatic failover and session migration
- Health monitoring and metrics

The HAClusterNode is the main entry point for running FlyBrowser in cluster mode.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import psutil

from flybrowser.core.browser_pool import BrowserPool, PoolConfig
from flybrowser.service.cluster.discovery import (
    ClusterDiscovery,
    ClusterMember,
    DiscoveryConfig,
    DiscoveryMethod,
)
from flybrowser.service.cluster.load_balancer import LoadBalancer, LoadBalancingStrategy
from flybrowser.service.cluster.raft import RaftConfig, RaftNode, NodeRole, StateMachine
from flybrowser.service.cluster.raft.state_machine import (
    CommandType,
    NodeHealth,
    NodeState,
    SessionState,
    SessionStatus,
)
from flybrowser.service.cluster.exceptions import (
    NotLeaderError,
    SessionNotFoundError,
    SessionMigrationError,
    NodeCapacityError,
)
from flybrowser.service.session_manager import SessionManager
from flybrowser.utils.logger import logger


@dataclass
class HANodeConfig:
    """Configuration for an HA cluster node.

    Attributes:
        node_id: Unique node identifier
        api_host: Host for HTTP API
        api_port: Port for HTTP API
        raft_host: Host for Raft RPC
        raft_port: Port for Raft RPC
        peers: List of peer nodes (host:raft_port:api_port format)
        data_dir: Directory for persistent data
        max_sessions: Maximum browser sessions this node can handle
        lb_strategy: Load balancing strategy
        discovery: Discovery configuration for auto-configuration
    """
    node_id: str = ""
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    raft_host: str = "0.0.0.0"
    raft_port: int = 4321
    peers: List[str] = field(default_factory=list)
    data_dir: str = "./data"
    max_sessions: int = 10
    lb_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOAD
    discovery: Optional[DiscoveryConfig] = None

    def __post_init__(self) -> None:
        if not self.node_id:
            self.node_id = str(uuid.uuid4())[:8]

    @property
    def api_address(self) -> str:
        """Get full API address."""
        return f"{self.api_host}:{self.api_port}"

    @property
    def raft_address(self) -> str:
        """Get full Raft address."""
        return f"{self.raft_host}:{self.raft_port}"

    def to_raft_config(self) -> RaftConfig:
        """Convert to RaftConfig."""
        return RaftConfig(
            node_id=self.node_id,
            bind_host=self.raft_host,
            bind_port=self.raft_port,
            api_host=self.api_host,
            api_port=self.api_port,
            cluster_nodes=self.peers,
            data_dir=f"{self.data_dir}/raft",
        )


class HAClusterNode:
    """High-Availability cluster node for FlyBrowser.
    
    Integrates Raft consensus, load balancing, and browser pool management
    for production-grade distributed deployment.
    
    Example:
        >>> config = HANodeConfig(
        ...     node_id="node1",
        ...     api_port=8000,
        ...     raft_port=4321,
        ...     peers=["node2:4322:8001", "node3:4323:8002"],
        ... )
        >>> node = HAClusterNode(config)
        >>> await node.start()
        >>> 
        >>> # Create a session (routed to best node)
        >>> session_id = await node.create_session()
        >>> 
        >>> await node.stop()
    """
    
    def __init__(self, config: HANodeConfig) -> None:
        """Initialize the HA cluster node."""
        self.config = config
        self.node_id = config.node_id

        # Raft consensus
        self._raft = RaftNode(config.to_raft_config())

        # Load balancer
        self._load_balancer = LoadBalancer(strategy=config.lb_strategy)

        # Local browser pool
        pool_config = PoolConfig(min_size=1, max_size=config.max_sessions)
        self._browser_pool = BrowserPool(pool_config)

        # Session manager
        self._session_manager = SessionManager(max_sessions=config.max_sessions * 2)

        # Discovery (for auto-configuration)
        self._discovery: Optional[ClusterDiscovery] = None
        if config.discovery:
            self._discovery = ClusterDiscovery(
                node_id=config.node_id,
                host=config.api_host if config.api_host != "0.0.0.0" else self._get_local_ip(),
                api_port=config.api_port,
                raft_port=config.raft_port,
                config=config.discovery,
                on_member_join=self._handle_member_join,
                on_member_leave=self._handle_member_leave,
                on_member_update=self._handle_member_update,
            )

        # Background tasks
        self._running = False
        self._health_task: Optional[asyncio.Task] = None
        self._sync_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_leader_change: Optional[Callable[[Optional[str]], None]] = None

    def _get_local_ip(self) -> str:
        """Get local IP address for discovery."""
        import socket
        try:
            # Connect to a public DNS to get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _handle_member_join(self, member: ClusterMember) -> None:
        """Handle new member joining the cluster."""
        logger.info(f"New cluster member discovered: {member.node_id} ({member.api_address})")
        # Update Raft peers dynamically
        if self._discovery:
            new_peers = self._discovery.get_peers()
            self._raft.update_peers(new_peers)

    def _handle_member_leave(self, member: ClusterMember) -> None:
        """Handle member leaving the cluster."""
        logger.info(f"Cluster member left: {member.node_id}")
        # Update Raft peers dynamically
        if self._discovery:
            new_peers = self._discovery.get_peers()
            self._raft.update_peers(new_peers)

    def _handle_member_update(self, member: ClusterMember) -> None:
        """Handle member update."""
        logger.debug(f"Cluster member updated: {member.node_id}")
    
    # ==================== Properties ====================
    
    @property
    def is_leader(self) -> bool:
        """Check if this node is the cluster leader."""
        return self._raft.is_leader
    
    @property
    def leader_id(self) -> Optional[str]:
        """Get the current leader's ID."""
        return self._raft.leader_id
    
    @property
    def role(self) -> NodeRole:
        """Get current Raft role."""
        return self._raft.role
    
    @property
    def state_machine(self) -> StateMachine:
        """Get the replicated state machine."""
        return self._raft.state_machine

    # ==================== Lifecycle ====================

    async def start(self) -> None:
        """Start the HA cluster node."""
        if self._running:
            return

        logger.info(f"Starting HA cluster node {self.node_id}")

        # Start discovery if configured (before Raft to discover peers)
        if self._discovery:
            logger.info("Starting cluster discovery...")
            await self._discovery.start()

            # Wait a bit for peer discovery
            await asyncio.sleep(1.0)

            # Update Raft peers from discovery
            discovered_peers = self._discovery.get_peers()
            if discovered_peers:
                logger.info(f"Discovered {len(discovered_peers)} peers via {self.config.discovery.method.value}")
                self._raft.update_peers(discovered_peers)

        # Set up Raft callbacks
        self._raft.set_callbacks(
            on_leader_change=self._handle_leader_change,
            on_role_change=self._handle_role_change,
        )

        # Start Raft consensus
        await self._raft.start()

        # Start browser pool
        await self._browser_pool.start()

        self._running = True

        # Start background tasks
        self._health_task = asyncio.create_task(self._health_report_loop())
        self._sync_task = asyncio.create_task(self._state_sync_loop())

        # Register self with cluster
        await self._register_self()

        logger.info(f"HA cluster node {self.node_id} started as {self.role.value}")

    async def stop(self) -> None:
        """Stop the HA cluster node."""
        if not self._running:
            return

        logger.info(f"Stopping HA cluster node {self.node_id}")
        self._running = False

        # Unregister from cluster
        await self._unregister_self()

        # Cancel background tasks
        for task in [self._health_task, self._sync_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop discovery
        if self._discovery:
            await self._discovery.stop()

        # Stop browser pool
        await self._browser_pool.stop()

        # Stop Raft
        await self._raft.stop()

        logger.info(f"HA cluster node {self.node_id} stopped")

    # ==================== Cluster Registration ====================

    async def _register_self(self) -> None:
        """Register this node with the cluster."""
        node_state = self._get_node_state()

        if self.is_leader:
            # Leader can directly apply
            await self._raft.submit_command({
                "type": CommandType.REGISTER_NODE.value,
                "node": node_state.to_dict(),
            })
        else:
            # Wait for leader to be elected, then it will sync
            pass

    async def _unregister_self(self) -> None:
        """Unregister this node from the cluster."""
        if self.is_leader:
            try:
                await self._raft.submit_command({
                    "type": CommandType.UNREGISTER_NODE.value,
                    "node_id": self.node_id,
                }, timeout=2.0)
            except Exception as e:
                logger.warning(f"Failed to unregister: {e}")

    def _get_node_state(self) -> NodeState:
        """Get current node state for reporting."""
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        return NodeState(
            node_id=self.node_id,
            api_address=self.config.api_address,
            raft_address=self.config.raft_address,
            health=NodeHealth.HEALTHY,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            active_sessions=self._session_manager.get_active_session_count(),
            max_sessions=self.config.max_sessions,
        )

    # ==================== Background Tasks ====================

    async def _health_report_loop(self) -> None:
        """Periodically report health to cluster."""
        while self._running:
            try:
                await asyncio.sleep(5.0)  # Report every 5 seconds

                if self.is_leader:
                    # Update our own state
                    node_state = self._get_node_state()
                    await self._raft.submit_command({
                        "type": CommandType.UPDATE_NODE.value,
                        "node_id": self.node_id,
                        "updates": {
                            "cpu_percent": node_state.cpu_percent,
                            "memory_percent": node_state.memory_percent,
                            "active_sessions": node_state.active_sessions,
                            "health": node_state.health.value,
                        },
                    }, timeout=2.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Health report error: {e}")

    async def _state_sync_loop(self) -> None:
        """Sync state machine to load balancer."""
        while self._running:
            try:
                await asyncio.sleep(1.0)  # Sync every second

                # Update load balancer with current node states
                nodes = self.state_machine.get_all_nodes()
                self._load_balancer.update_nodes(nodes)

                # Update session affinity
                for session in self.state_machine.get_all_sessions():
                    self._load_balancer.register_session(session.session_id, session.node_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"State sync error: {e}")

    # ==================== Callbacks ====================

    def _handle_leader_change(self, leader_id: Optional[str]) -> None:
        """Handle leader change event."""
        logger.info(f"Leader changed to: {leader_id}")
        if self._on_leader_change:
            self._on_leader_change(leader_id)

    def _handle_role_change(self, role: NodeRole) -> None:
        """Handle role change event."""
        logger.info(f"Role changed to: {role.value}")

    # ==================== Session Management ====================

    async def create_session(
        self,
        client_id: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new browser session.

        If this node is the leader, it selects the best node and creates
        the session. If not leader, raises an error (client should redirect).

        Args:
            client_id: Optional client identifier for session affinity
            llm_config: LLM configuration dictionary containing:
                - llm_provider: LLM provider name (openai, anthropic, ollama)
                - llm_model: LLM model name (optional)
                - api_key: API key for the LLM provider
                - headless: Run browser in headless mode (default: true)
                - browser_type: Browser type (chromium, firefox, webkit)

        Returns:
            Session ID

        Raises:
            NotLeaderError: If not leader
            NodeCapacityError: If no capacity available
        """
        if not self.is_leader:
            raise NotLeaderError(
                leader_id=self.leader_id,
                leader_address=self.get_leader_api_address(),
            )

        # Default LLM config
        if llm_config is None:
            llm_config = {}

        # Select best node for session
        target_node = self._load_balancer.select_node_for_session()
        if not target_node:
            # No available nodes, use self if we have capacity
            self_state = self._get_node_state()
            if self_state.available_capacity > 0:
                target_node = self_state
            else:
                raise NodeCapacityError(
                    total_capacity=self.config.max_sessions,
                    used_capacity=self._session_manager.get_active_session_count(),
                )

        session_id = str(uuid.uuid4())

        # Store LLM config in session metadata for use when creating the browser
        # Mask API key before storing in replicated state
        session_metadata = {
            "llm_provider": llm_config.get("llm_provider", "openai"),
            "llm_model": llm_config.get("llm_model"),
            "api_key_masked": "***" if llm_config.get("api_key") else None,
            "headless": llm_config.get("headless", True),
            "browser_type": llm_config.get("browser_type", "chromium"),
        }
        
        # Keep actual API key only for local use
        local_llm_config = {
            **session_metadata,
            "api_key": llm_config.get("api_key"),
        }

        # If session is on this node, create local browser FIRST
        # This prevents orphaned state machine entries if browser creation fails
        if target_node.node_id == self.node_id:
            try:
                await self._create_local_session(session_id, local_llm_config)
            except Exception as e:
                logger.error(f"Failed to create local session {session_id}: {e}")
                raise

        # Now commit to state machine (browser is already running)
        session_state = SessionState(
            session_id=session_id,
            node_id=target_node.node_id,
            status=SessionStatus.ACTIVE if target_node.node_id == self.node_id else SessionStatus.PENDING,
            client_id=client_id,
            metadata=session_metadata,
        )

        await self._raft.submit_command({
            "type": CommandType.CREATE_SESSION.value,
            "session": session_state.to_dict(),
        })

        return session_id

    async def _create_local_session(
        self,
        session_id: str,
        llm_config: Dict[str, Any],
    ) -> None:
        """Create a browser session locally with LLM configuration.

        Args:
            session_id: Session ID
            llm_config: LLM configuration dictionary
        """
        await self._session_manager.create_session(
            llm_provider=llm_config.get("llm_provider", "openai"),
            llm_model=llm_config.get("llm_model"),
            api_key=llm_config.get("api_key"),
            headless=llm_config.get("headless", True),
            browser_type=llm_config.get("browser_type", "chromium"),
            session_id=session_id,
        )

    async def delete_session(self, session_id: str) -> bool:
        """Delete a browser session.

        Args:
            session_id: Session to delete

        Returns:
            True if deleted
            
        Raises:
            NotLeaderError: If not leader
            SessionNotFoundError: If session not found
        """
        if not self.is_leader:
            raise NotLeaderError(
                leader_id=self.leader_id,
                leader_address=self.get_leader_api_address(),
            )

        # Check if session exists
        session = self.state_machine.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        # If session was on this node, delete locally first
        if session.node_id == self.node_id:
            try:
                await self._session_manager.delete_session(session_id)
            except KeyError:
                # Session may not exist locally
                pass

        # Delete from state machine
        await self._raft.submit_command({
            "type": CommandType.DELETE_SESSION.value,
            "session_id": session_id,
        })

        return True

    async def migrate_session(
        self,
        session_id: str,
        target_node_id: str,
    ) -> bool:
        """Migrate a session to a different node.
        
        This exports session state from the source node and imports it
        on the target node.
        
        Args:
            session_id: Session to migrate
            target_node_id: Target node ID
            
        Returns:
            True if migration succeeded
            
        Raises:
            NotLeaderError: If not leader
            SessionNotFoundError: If session not found
            SessionMigrationError: If migration fails
        """
        if not self.is_leader:
            raise NotLeaderError(
                leader_id=self.leader_id,
                leader_address=self.get_leader_api_address(),
            )
        
        session = self.state_machine.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)
        
        source_node_id = session.node_id
        if source_node_id == target_node_id:
            logger.info(f"Session {session_id} already on target node {target_node_id}")
            return True
        
        logger.info(f"Migrating session {session_id} from {source_node_id} to {target_node_id}")
        
        try:
            # Export session state from source (if it's us)
            session_state_data = None
            if source_node_id == self.node_id:
                session_state_data = await self._export_session_state(session_id)
            # TODO: For remote nodes, we'd need an RPC to export state
            
            # Update state machine first (mark as migrating)
            await self._raft.submit_command({
                "type": CommandType.MIGRATE_SESSION.value,
                "session_id": session_id,
                "target_node_id": target_node_id,
            })
            
            # Import on target (if it's us)
            if target_node_id == self.node_id and session_state_data:
                await self._import_session_state(session_id, session_state_data)
            
            # Update session status to active
            await self._raft.submit_command({
                "type": CommandType.UPDATE_SESSION.value,
                "session_id": session_id,
                "updates": {"status": SessionStatus.ACTIVE.value},
            })
            
            logger.info(f"Session {session_id} migrated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Session migration failed: {e}")
            # Mark session as error
            await self._raft.submit_command({
                "type": CommandType.UPDATE_SESSION.value,
                "session_id": session_id,
                "updates": {"status": SessionStatus.ERROR.value},
            })
            raise SessionMigrationError(
                session_id=session_id,
                source_node=source_node_id,
                target_node=target_node_id,
                message=str(e),
            )

    async def _export_session_state(self, session_id: str) -> Dict[str, Any]:
        """Export session state for migration.
        
        This captures browser state including cookies, localStorage, and navigation history.
        
        Args:
            session_id: Session to export
            
        Returns:
            Dictionary with session state data
        """
        try:
            browser = self._session_manager.get_session(session_id)
            
            # Get current URL and title
            current_url = await browser.page_controller.get_url()
            title = await browser.page_controller.get_title()
            
            # Get cookies
            cookies = await browser.page_controller.page.context.cookies()
            
            # Get localStorage (if accessible)
            local_storage = {}
            try:
                local_storage = await browser.page_controller.page.evaluate(
                    "() => Object.entries(localStorage)"
                )
            except Exception:
                pass
            
            # Get session storage
            session_storage = {}
            try:
                session_storage = await browser.page_controller.page.evaluate(
                    "() => Object.entries(sessionStorage)"
                )
            except Exception:
                pass
            
            return {
                "session_id": session_id,
                "url": current_url,
                "title": title,
                "cookies": cookies,
                "local_storage": dict(local_storage) if local_storage else {},
                "session_storage": dict(session_storage) if session_storage else {},
                "exported_at": time.time(),
            }
        except KeyError:
            logger.warning(f"Session {session_id} not found locally for export")
            return {}

    async def _import_session_state(
        self,
        session_id: str,
        state_data: Dict[str, Any],
    ) -> None:
        """Import session state after migration.
        
        Args:
            session_id: Session ID
            state_data: State data from export
        """
        if not state_data:
            return
        
        # Get session metadata from state machine
        session = self.state_machine.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)
        
        # Create new browser session with same config
        llm_config = session.metadata or {}
        await self._create_local_session(session_id, llm_config)
        
        try:
            browser = self._session_manager.get_session(session_id)
            
            # Restore cookies
            if state_data.get("cookies"):
                await browser.page_controller.page.context.add_cookies(state_data["cookies"])
            
            # Navigate to the URL
            if state_data.get("url"):
                await browser.goto(state_data["url"])
            
            # Restore localStorage
            if state_data.get("local_storage"):
                for key, value in state_data["local_storage"].items():
                    await browser.page_controller.page.evaluate(
                        f"localStorage.setItem({repr(key)}, {repr(value)})"
                    )
            
            # Restore sessionStorage
            if state_data.get("session_storage"):
                for key, value in state_data["session_storage"].items():
                    await browser.page_controller.page.evaluate(
                        f"sessionStorage.setItem({repr(key)}, {repr(value)})"
                    )
            
            logger.info(f"Session {session_id} state imported successfully")
            
        except Exception as e:
            logger.error(f"Failed to import session state: {e}")
            raise

    def get_node_for_session(self, session_id: str) -> Optional[str]:
        """Get the API address of the node handling a session.

        Args:
            session_id: Session ID

        Returns:
            API address or None if session not found
        """
        return self._load_balancer.get_node_address_for_session(session_id)

    # ==================== Status ====================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive node status."""
        return {
            "node_id": self.node_id,
            "role": self.role.value,
            "is_leader": self.is_leader,
            "leader_id": self.leader_id,
            "api_address": self.config.api_address,
            "raft_address": self.config.raft_address,
            "raft": self._raft.get_status(),
            "cluster": self.state_machine.get_cluster_stats(),
            "load_balancer": self._load_balancer.get_stats(),
            "browser_pool": self._browser_pool.get_stats(),
        }

    def get_leader_api_address(self) -> Optional[str]:
        """Get the leader's API address for redirects."""
        if self.is_leader:
            return self.config.api_address

        leader_id = self.leader_id
        if leader_id:
            leader_node = self.state_machine.get_node(leader_id)
            if leader_node:
                return leader_node.api_address

        return None

    async def __aenter__(self) -> "HAClusterNode":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

