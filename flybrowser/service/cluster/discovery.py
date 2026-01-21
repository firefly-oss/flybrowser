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
Cluster Discovery and Auto-Configuration for FlyBrowser.

This module provides intelligent peer discovery and automatic cluster
configuration using multiple discovery methods:

1. Static: Manual peer list configuration
2. DNS: DNS-based service discovery (SRV records)
3. Kubernetes: K8s API-based discovery
4. Gossip: SWIM-based gossip protocol for peer discovery

Features:
- Automatic peer discovery without manual configuration
- Graceful handling of node joins and leaves
- Network partition detection and recovery
- Dynamic peer list updates across all nodes
- Health-based membership management
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import socket
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from flybrowser.utils.logger import logger


class DiscoveryMethod(str, Enum):
    """Discovery method for finding cluster peers."""
    STATIC = "static"
    DNS = "dns"
    KUBERNETES = "kubernetes"
    GOSSIP = "gossip"
    MULTICAST = "multicast"


class MemberStatus(str, Enum):
    """Status of a cluster member."""
    ALIVE = "alive"
    SUSPECT = "suspect"
    DEAD = "dead"
    LEFT = "left"


@dataclass
class ClusterMember:
    """Represents a member of the cluster."""
    node_id: str
    host: str
    api_port: int
    raft_port: int
    status: MemberStatus = MemberStatus.ALIVE
    incarnation: int = 0
    last_seen: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def api_address(self) -> str:
        """Get API address."""
        return f"{self.host}:{self.api_port}"

    @property
    def raft_address(self) -> str:
        """Get Raft address."""
        return f"{self.host}:{self.raft_port}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "api_port": self.api_port,
            "raft_port": self.raft_port,
            "status": self.status.value,
            "incarnation": self.incarnation,
            "last_seen": self.last_seen,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterMember":
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            host=data["host"],
            api_port=data["api_port"],
            raft_port=data["raft_port"],
            status=MemberStatus(data.get("status", "alive")),
            incarnation=data.get("incarnation", 0),
            last_seen=data.get("last_seen", time.time()),
            metadata=data.get("metadata", {}),
        )


class GossipMessageType(str, Enum):
    """Types of gossip messages."""
    PING = "ping"
    PING_REQ = "ping_req"
    ACK = "ack"
    SYNC = "sync"
    JOIN = "join"
    LEAVE = "leave"
    SUSPECT = "suspect"
    ALIVE = "alive"
    DEAD = "dead"


@dataclass
class GossipMessage:
    """A gossip protocol message."""
    msg_type: GossipMessageType
    sender_id: str
    target_id: Optional[str] = None
    incarnation: int = 0
    members: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        data = {
            "type": self.msg_type.value,
            "sender": self.sender_id,
            "target": self.target_id,
            "incarnation": self.incarnation,
            "members": self.members,
            "ts": self.timestamp,
        }
        return json.dumps(data).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "GossipMessage":
        """Deserialize from bytes."""
        obj = json.loads(data.decode("utf-8"))
        return cls(
            msg_type=GossipMessageType(obj["type"]),
            sender_id=obj["sender"],
            target_id=obj.get("target"),
            incarnation=obj.get("incarnation", 0),
            members=obj.get("members", []),
            timestamp=obj.get("ts", time.time()),
        )


@dataclass
class DiscoveryConfig:
    """Configuration for cluster discovery."""
    method: DiscoveryMethod = DiscoveryMethod.GOSSIP
    # Static discovery
    static_peers: List[str] = field(default_factory=list)
    # DNS discovery
    dns_name: str = ""
    dns_port: int = 8000
    # Kubernetes discovery
    k8s_namespace: str = "default"
    k8s_service: str = "flybrowser"
    k8s_label_selector: str = "app=flybrowser"
    # Gossip settings
    gossip_port: int = 7946
    gossip_interval_ms: int = 200
    gossip_fanout: int = 3
    suspect_timeout_ms: int = 5000
    dead_timeout_ms: int = 30000
    # Multicast discovery
    multicast_group: str = "239.255.255.250"
    multicast_port: int = 7947


class ClusterDiscovery:
    """Manages cluster peer discovery and membership.

    Implements the SWIM (Scalable Weakly-consistent Infection-style
    Process Group Membership) protocol for efficient peer discovery
    and failure detection.

    Example:
        >>> config = DiscoveryConfig(method=DiscoveryMethod.GOSSIP)
        >>> discovery = ClusterDiscovery(
        ...     node_id="node1",
        ...     host="192.168.1.10",
        ...     api_port=8000,
        ...     raft_port=4321,
        ...     config=config,
        ... )
        >>> await discovery.start()
        >>>
        >>> # Get current cluster members
        >>> members = discovery.get_members()
        >>>
        >>> await discovery.stop()
    """

    def __init__(
        self,
        node_id: str,
        host: str,
        api_port: int,
        raft_port: int,
        config: Optional[DiscoveryConfig] = None,
        on_member_join: Optional[Callable[[ClusterMember], None]] = None,
        on_member_leave: Optional[Callable[[ClusterMember], None]] = None,
        on_member_update: Optional[Callable[[ClusterMember], None]] = None,
    ) -> None:
        """Initialize cluster discovery.

        Args:
            node_id: Unique identifier for this node
            host: Host address for this node
            api_port: API port for this node
            raft_port: Raft port for this node
            config: Discovery configuration
            on_member_join: Callback when a member joins
            on_member_leave: Callback when a member leaves
            on_member_update: Callback when a member is updated
        """
        self.node_id = node_id
        self.host = host
        self.api_port = api_port
        self.raft_port = raft_port
        self.config = config or DiscoveryConfig()

        # Callbacks
        self._on_member_join = on_member_join
        self._on_member_leave = on_member_leave
        self._on_member_update = on_member_update

        # Self member
        self._self = ClusterMember(
            node_id=node_id,
            host=host,
            api_port=api_port,
            raft_port=raft_port,
            status=MemberStatus.ALIVE,
            incarnation=int(time.time()),
        )

        # Membership state
        self._members: Dict[str, ClusterMember] = {node_id: self._self}
        self._lock = asyncio.Lock()

        # Gossip state
        self._running = False
        self._gossip_task: Optional[asyncio.Task] = None
        self._probe_task: Optional[asyncio.Task] = None
        self._udp_transport: Optional[asyncio.DatagramTransport] = None
        self._udp_protocol: Optional["GossipProtocol"] = None

        # Probe state (for failure detection)
        self._probe_index = 0
        self._pending_acks: Dict[str, asyncio.Future] = {}

    async def start(self) -> None:
        """Start the discovery service."""
        if self._running:
            return

        self._running = True
        logger.info(f"Starting cluster discovery (method={self.config.method.value})")

        # Start based on discovery method
        if self.config.method == DiscoveryMethod.GOSSIP:
            await self._start_gossip()
        elif self.config.method == DiscoveryMethod.STATIC:
            await self._discover_static()
        elif self.config.method == DiscoveryMethod.DNS:
            await self._discover_dns()
        elif self.config.method == DiscoveryMethod.KUBERNETES:
            await self._discover_kubernetes()
        elif self.config.method == DiscoveryMethod.MULTICAST:
            await self._start_multicast()

        logger.info(f"Cluster discovery started with {len(self._members)} members")

    async def stop(self) -> None:
        """Stop the discovery service."""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping cluster discovery")

        # Announce leave
        await self._announce_leave()

        # Stop tasks
        if self._gossip_task:
            self._gossip_task.cancel()
            try:
                await self._gossip_task
            except asyncio.CancelledError:
                pass

        if self._probe_task:
            self._probe_task.cancel()
            try:
                await self._probe_task
            except asyncio.CancelledError:
                pass

        # Close UDP transport
        if self._udp_transport:
            self._udp_transport.close()

        logger.info("Cluster discovery stopped")

    def get_members(self, alive_only: bool = True) -> List[ClusterMember]:
        """Get current cluster members.

        Args:
            alive_only: Only return alive members

        Returns:
            List of cluster members
        """
        members = list(self._members.values())
        if alive_only:
            members = [m for m in members if m.status == MemberStatus.ALIVE]
        return members

    def get_peers(self) -> List[str]:
        """Get peer addresses in host:raft_port:api_port format.

        Returns:
            List of peer addresses for Raft configuration
        """
        peers = []
        for member in self._members.values():
            if member.node_id != self.node_id and member.status == MemberStatus.ALIVE:
                peers.append(f"{member.host}:{member.raft_port}:{member.api_port}")
        return peers

    async def join(
        self,
        seed_nodes: List[str],
        max_retries: int = 5,
        initial_delay: float = 0.5,
    ) -> bool:
        """Join a cluster using seed nodes with exponential backoff.

        Args:
            seed_nodes: List of seed node addresses (host:port)
            max_retries: Maximum number of retry rounds
            initial_delay: Initial delay between attempts in seconds

        Returns:
            True if successfully joined
        """
        logger.info(f"Joining cluster via seed nodes: {seed_nodes}")

        for attempt in range(max_retries):
            # Shuffle seeds to distribute load
            shuffled_seeds = list(seed_nodes)
            random.shuffle(shuffled_seeds)
            
            for seed in shuffled_seeds:
                try:
                    host, port = seed.rsplit(":", 1)
                    port = int(port)

                    # Send join message
                    msg = GossipMessage(
                        msg_type=GossipMessageType.JOIN,
                        sender_id=self.node_id,
                        members=[self._self.to_dict()],
                    )

                    await self._send_udp(host, port, msg.to_bytes())

                    # Wait for sync response with increasing timeout
                    wait_time = initial_delay * (1 + attempt * 0.5)
                    await asyncio.sleep(wait_time)

                    if len(self._members) > 1:
                        logger.info(
                            f"Successfully joined cluster with {len(self._members)} members "
                            f"after {attempt + 1} attempt(s)"
                        )
                        return True

                except Exception as e:
                    logger.debug(f"Failed to join via {seed}: {e}")

            # Exponential backoff between rounds
            if attempt < max_retries - 1:
                backoff = min(initial_delay * (2 ** attempt), 30.0)  # Cap at 30s
                logger.info(
                    f"Join attempt {attempt + 1}/{max_retries} failed, "
                    f"retrying in {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)

        logger.warning(
            f"Failed to join cluster after {max_retries} attempts. "
            f"Will continue trying via gossip."
        )
        return False

    # ==================== Internal Methods ====================

    async def _start_gossip(self) -> None:
        """Start gossip-based discovery."""
        # Create UDP server for gossip
        loop = asyncio.get_event_loop()

        self._udp_protocol = GossipProtocol(self)
        transport, _ = await loop.create_datagram_endpoint(
            lambda: self._udp_protocol,
            local_addr=(self.host, self.config.gossip_port),
        )
        self._udp_transport = transport

        # Start gossip and probe tasks
        self._gossip_task = asyncio.create_task(self._gossip_loop())
        self._probe_task = asyncio.create_task(self._probe_loop())

        # Join via static peers if configured
        if self.config.static_peers:
            await self.join([
                f"{p.split(':')[0]}:{self.config.gossip_port}"
                for p in self.config.static_peers
            ])

    async def _gossip_loop(self) -> None:
        """Periodic gossip to random peers."""
        interval = self.config.gossip_interval_ms / 1000.0

        while self._running:
            try:
                await asyncio.sleep(interval)
                await self._gossip_to_random_peers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in gossip loop: {e}")

    async def _gossip_to_random_peers(self) -> None:
        """Send gossip to random subset of peers."""
        alive_members = [
            m for m in self._members.values()
            if m.node_id != self.node_id and m.status == MemberStatus.ALIVE
        ]

        if not alive_members:
            return

        # Select random peers (fanout)
        targets = random.sample(
            alive_members,
            min(self.config.gossip_fanout, len(alive_members))
        )

        # Build sync message with current membership
        msg = GossipMessage(
            msg_type=GossipMessageType.SYNC,
            sender_id=self.node_id,
            members=[m.to_dict() for m in self._members.values()],
        )

        for target in targets:
            try:
                await self._send_udp(
                    target.host,
                    self.config.gossip_port,
                    msg.to_bytes()
                )
            except Exception as e:
                logger.debug(f"Failed to gossip to {target.node_id}: {e}")

    async def _probe_loop(self) -> None:
        """Periodic probing for failure detection."""
        interval = self.config.gossip_interval_ms / 1000.0

        while self._running:
            try:
                await asyncio.sleep(interval)
                await self._probe_random_member()
                await self._check_suspect_timeouts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in probe loop: {e}")

    async def _probe_random_member(self) -> None:
        """Probe a random member for failure detection."""
        members = [
            m for m in self._members.values()
            if m.node_id != self.node_id and m.status in (MemberStatus.ALIVE, MemberStatus.SUSPECT)
        ]

        if not members:
            return

        # Round-robin through members
        self._probe_index = (self._probe_index + 1) % len(members)
        target = members[self._probe_index]

        # Send ping
        msg = GossipMessage(
            msg_type=GossipMessageType.PING,
            sender_id=self.node_id,
            target_id=target.node_id,
        )

        try:
            # Create future for ack
            ack_future = asyncio.get_event_loop().create_future()
            self._pending_acks[target.node_id] = ack_future

            await self._send_udp(
                target.host,
                self.config.gossip_port,
                msg.to_bytes()
            )

            # Wait for ack with timeout
            try:
                await asyncio.wait_for(
                    ack_future,
                    timeout=self.config.gossip_interval_ms / 1000.0
                )
                # Got ack - member is alive
                async with self._lock:
                    if target.node_id in self._members:
                        self._members[target.node_id].last_seen = time.time()
                        if self._members[target.node_id].status == MemberStatus.SUSPECT:
                            self._members[target.node_id].status = MemberStatus.ALIVE
                            logger.info(f"Member {target.node_id} recovered from suspect")
            except asyncio.TimeoutError:
                # No ack - try indirect ping
                await self._indirect_ping(target)
        finally:
            self._pending_acks.pop(target.node_id, None)

    async def _indirect_ping(self, target: ClusterMember) -> None:
        """Try indirect ping through other members."""
        others = [
            m for m in self._members.values()
            if m.node_id not in (self.node_id, target.node_id)
            and m.status == MemberStatus.ALIVE
        ]

        if not others:
            # No other members - mark as suspect
            await self._mark_suspect(target)
            return

        # Ask random members to ping target
        helpers = random.sample(others, min(3, len(others)))

        for helper in helpers:
            msg = GossipMessage(
                msg_type=GossipMessageType.PING_REQ,
                sender_id=self.node_id,
                target_id=target.node_id,
            )
            await self._send_udp(
                helper.host,
                self.config.gossip_port,
                msg.to_bytes()
            )

        # Wait a bit for indirect ack
        await asyncio.sleep(self.config.gossip_interval_ms / 1000.0)

        # Check if we got an ack
        if target.node_id in self._pending_acks:
            # Still no ack - mark as suspect
            await self._mark_suspect(target)


    async def _mark_suspect(self, member: ClusterMember) -> None:
        """Mark a member as suspect."""
        async with self._lock:
            if member.node_id in self._members:
                if self._members[member.node_id].status == MemberStatus.ALIVE:
                    self._members[member.node_id].status = MemberStatus.SUSPECT
                    logger.warning(f"Member {member.node_id} marked as suspect")

    async def _check_suspect_timeouts(self) -> None:
        """Check for suspect members that should be marked dead."""
        now = time.time()
        suspect_timeout = self.config.suspect_timeout_ms / 1000.0
        dead_timeout = self.config.dead_timeout_ms / 1000.0

        async with self._lock:
            for member in list(self._members.values()):
                if member.node_id == self.node_id:
                    continue

                age = now - member.last_seen

                if member.status == MemberStatus.SUSPECT and age > suspect_timeout:
                    member.status = MemberStatus.DEAD
                    logger.warning(f"Member {member.node_id} marked as dead")
                    if self._on_member_leave:
                        self._on_member_leave(member)

                elif member.status == MemberStatus.DEAD and age > dead_timeout:
                    # Remove from membership
                    del self._members[member.node_id]
                    logger.info(f"Member {member.node_id} removed from cluster")

    async def _announce_leave(self) -> None:
        """Announce that we're leaving the cluster."""
        msg = GossipMessage(
            msg_type=GossipMessageType.LEAVE,
            sender_id=self.node_id,
            members=[self._self.to_dict()],
        )

        # Send to all known members
        for member in self._members.values():
            if member.node_id != self.node_id:
                try:
                    await self._send_udp(
                        member.host,
                        self.config.gossip_port,
                        msg.to_bytes()
                    )
                except Exception:
                    pass

    async def _send_udp(self, host: str, port: int, data: bytes) -> None:
        """Send UDP message."""
        if self._udp_transport:
            self._udp_transport.sendto(data, (host, port))

    async def _handle_message(self, msg: GossipMessage, addr: Tuple[str, int]) -> None:
        """Handle incoming gossip message."""
        if msg.msg_type == GossipMessageType.PING:
            await self._handle_ping(msg, addr)
        elif msg.msg_type == GossipMessageType.PING_REQ:
            await self._handle_ping_req(msg, addr)
        elif msg.msg_type == GossipMessageType.ACK:
            await self._handle_ack(msg)
        elif msg.msg_type == GossipMessageType.SYNC:
            await self._handle_sync(msg)
        elif msg.msg_type == GossipMessageType.JOIN:
            await self._handle_join(msg, addr)
        elif msg.msg_type == GossipMessageType.LEAVE:
            await self._handle_leave(msg)
        elif msg.msg_type == GossipMessageType.SUSPECT:
            await self._handle_suspect(msg)
        elif msg.msg_type == GossipMessageType.ALIVE:
            await self._handle_alive(msg)
        elif msg.msg_type == GossipMessageType.DEAD:
            await self._handle_dead(msg)

    async def _handle_ping(self, msg: GossipMessage, addr: Tuple[str, int]) -> None:
        """Handle ping message."""
        ack = GossipMessage(
            msg_type=GossipMessageType.ACK,
            sender_id=self.node_id,
            target_id=msg.sender_id,
        )
        await self._send_udp(addr[0], addr[1], ack.to_bytes())

    async def _handle_ping_req(self, msg: GossipMessage, addr: Tuple[str, int]) -> None:
        """Handle indirect ping request."""
        if msg.target_id and msg.target_id in self._members:
            target = self._members[msg.target_id]
            # Ping the target
            ping = GossipMessage(
                msg_type=GossipMessageType.PING,
                sender_id=self.node_id,
                target_id=msg.target_id,
            )
            await self._send_udp(target.host, self.config.gossip_port, ping.to_bytes())

    async def _handle_ack(self, msg: GossipMessage) -> None:
        """Handle ack message."""
        if msg.sender_id in self._pending_acks:
            self._pending_acks[msg.sender_id].set_result(True)

    async def _handle_sync(self, msg: GossipMessage) -> None:
        """Handle sync message - merge membership."""
        for member_data in msg.members:
            await self._merge_member(ClusterMember.from_dict(member_data))

    async def _handle_join(self, msg: GossipMessage, addr: Tuple[str, int]) -> None:
        """Handle join message."""
        for member_data in msg.members:
            member = ClusterMember.from_dict(member_data)
            await self._merge_member(member)

        # Send full membership back
        sync = GossipMessage(
            msg_type=GossipMessageType.SYNC,
            sender_id=self.node_id,
            members=[m.to_dict() for m in self._members.values()],
        )
        await self._send_udp(addr[0], addr[1], sync.to_bytes())

    async def _handle_leave(self, msg: GossipMessage) -> None:
        """Handle leave message."""
        async with self._lock:
            if msg.sender_id in self._members:
                self._members[msg.sender_id].status = MemberStatus.LEFT
                logger.info(f"Member {msg.sender_id} left the cluster")
                if self._on_member_leave:
                    self._on_member_leave(self._members[msg.sender_id])

    async def _handle_suspect(self, msg: GossipMessage) -> None:
        """Handle suspect message."""
        if msg.target_id == self.node_id:
            # We're being suspected - refute with higher incarnation
            self._self.incarnation += 1
            alive = GossipMessage(
                msg_type=GossipMessageType.ALIVE,
                sender_id=self.node_id,
                incarnation=self._self.incarnation,
                members=[self._self.to_dict()],
            )
            await self._gossip_to_random_peers()
        elif msg.target_id and msg.target_id in self._members:
            member = self._members[msg.target_id]
            if msg.incarnation >= member.incarnation:
                await self._mark_suspect(member)

    async def _handle_alive(self, msg: GossipMessage) -> None:
        """Handle alive message."""
        for member_data in msg.members:
            member = ClusterMember.from_dict(member_data)
            if member.node_id in self._members:
                existing = self._members[member.node_id]
                if member.incarnation > existing.incarnation:
                    existing.incarnation = member.incarnation
                    existing.status = MemberStatus.ALIVE
                    existing.last_seen = time.time()

    async def _handle_dead(self, msg: GossipMessage) -> None:
        """Handle dead message."""
        if msg.target_id and msg.target_id in self._members:
            if msg.target_id != self.node_id:
                self._members[msg.target_id].status = MemberStatus.DEAD

    async def _merge_member(self, member: ClusterMember) -> None:
        """Merge a member into our membership list."""
        async with self._lock:
            if member.node_id == self.node_id:
                return

            if member.node_id in self._members:
                existing = self._members[member.node_id]
                # Update if newer incarnation or better status
                if member.incarnation > existing.incarnation:
                    self._members[member.node_id] = member
                    member.last_seen = time.time()
                    if self._on_member_update:
                        self._on_member_update(member)
            else:
                # New member
                member.last_seen = time.time()
                self._members[member.node_id] = member
                logger.info(f"New member discovered: {member.node_id} ({member.api_address})")
                if self._on_member_join:
                    self._on_member_join(member)

    # ==================== Discovery Methods ====================

    async def _discover_static(self) -> None:
        """Discover peers from static configuration."""
        for peer in self.config.static_peers:
            parts = peer.split(":")
            if len(parts) >= 3:
                host, raft_port, api_port = parts[0], int(parts[1]), int(parts[2])
            elif len(parts) == 2:
                host, api_port = parts[0], int(parts[1])
                raft_port = 4321
            else:
                continue

            # Generate node_id from address
            node_id = hashlib.md5(f"{host}:{api_port}".encode()).hexdigest()[:12]

            member = ClusterMember(
                node_id=node_id,
                host=host,
                api_port=api_port,
                raft_port=raft_port,
            )
            await self._merge_member(member)

    async def _discover_dns(self) -> None:
        """Discover peers via DNS SRV records."""
        try:
            import dns.resolver

            answers = dns.resolver.resolve(self.config.dns_name, "SRV")
            for rdata in answers:
                host = str(rdata.target).rstrip(".")
                port = rdata.port

                node_id = hashlib.md5(f"{host}:{port}".encode()).hexdigest()[:12]
                member = ClusterMember(
                    node_id=node_id,
                    host=host,
                    api_port=port,
                    raft_port=4321,
                )
                await self._merge_member(member)
        except ImportError:
            logger.warning("dnspython not installed, DNS discovery unavailable")
        except Exception as e:
            logger.error(f"DNS discovery failed: {e}")

    async def _discover_kubernetes(self) -> None:
        """Discover peers via Kubernetes API."""
        try:
            from kubernetes import client, config as k8s_config

            try:
                k8s_config.load_incluster_config()
            except Exception:
                k8s_config.load_kube_config()

            v1 = client.CoreV1Api()

            # Get endpoints for the service
            endpoints = v1.read_namespaced_endpoints(
                self.config.k8s_service,
                self.config.k8s_namespace,
            )

            for subset in endpoints.subsets or []:
                for address in subset.addresses or []:
                    host = address.ip
                    port = subset.ports[0].port if subset.ports else 8000

                    node_id = address.target_ref.name if address.target_ref else \
                        hashlib.md5(f"{host}:{port}".encode()).hexdigest()[:12]

                    member = ClusterMember(
                        node_id=node_id,
                        host=host,
                        api_port=port,
                        raft_port=4321,
                    )
                    await self._merge_member(member)
        except ImportError:
            logger.warning("kubernetes client not installed, K8s discovery unavailable")
        except Exception as e:
            logger.error(f"Kubernetes discovery failed: {e}")

    async def _start_multicast(self) -> None:
        """Start multicast-based discovery."""
        # Create multicast socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", self.config.multicast_port))

        # Join multicast group
        mreq = struct.pack(
            "4sl",
            socket.inet_aton(self.config.multicast_group),
            socket.INADDR_ANY,
        )
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        # Announce ourselves
        announce = GossipMessage(
            msg_type=GossipMessageType.JOIN,
            sender_id=self.node_id,
            members=[self._self.to_dict()],
        )
        sock.sendto(
            announce.to_bytes(),
            (self.config.multicast_group, self.config.multicast_port),
        )

        # Start listening task
        self._gossip_task = asyncio.create_task(
            self._multicast_listen_loop(sock)
        )

    async def _multicast_listen_loop(self, sock: socket.socket) -> None:
        """Listen for multicast messages."""
        loop = asyncio.get_event_loop()

        while self._running:
            try:
                data, addr = await loop.run_in_executor(
                    None, lambda: sock.recvfrom(65535)
                )
                msg = GossipMessage.from_bytes(data)
                await self._handle_message(msg, addr)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in multicast loop: {e}")


class GossipProtocol(asyncio.DatagramProtocol):
    """UDP protocol for gossip messages."""

    def __init__(self, discovery: ClusterDiscovery) -> None:
        self.discovery = discovery

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        """Handle received datagram."""
        try:
            msg = GossipMessage.from_bytes(data)
            asyncio.create_task(self.discovery._handle_message(msg, addr))
        except Exception as e:
            logger.debug(f"Failed to parse gossip message: {e}")
