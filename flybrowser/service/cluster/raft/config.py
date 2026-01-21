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
Raft Consensus Configuration for FlyBrowser HA Cluster.

This module provides configuration classes for the custom Raft consensus
implementation, including timing parameters, network settings, and cluster topology.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class RaftConfig:
    """Configuration for Raft consensus.

    Timing parameters follow Raft paper recommendations:
    - heartbeat_interval << election_timeout_min
    - election_timeout randomized between min and max
    - Typical: heartbeat=150ms, election=300-500ms for LAN
    - For WAN: heartbeat=300ms, election=1000-2000ms

    Attributes:
        node_id: Unique identifier for this node
        bind_host: Host to bind the Raft RPC server
        bind_port: Port to bind the Raft RPC server
        api_host: Host for the HTTP API server
        api_port: Port for the HTTP API server
        cluster_nodes: List of other node addresses (host:raft_port:api_port)
        data_dir: Directory for persistent Raft log storage
        election_timeout_min_ms: Minimum election timeout in milliseconds
        election_timeout_max_ms: Maximum election timeout in milliseconds
        heartbeat_interval_ms: Leader heartbeat interval in milliseconds
        rpc_timeout_ms: Timeout for RPC calls in milliseconds
        max_entries_per_request: Maximum log entries per AppendEntries RPC
        snapshot_threshold: Number of log entries before taking snapshot
        max_log_size_bytes: Maximum log size before compaction
        batch_interval_ms: Interval for batching log entries
    """

    node_id: str = ""
    bind_host: str = "0.0.0.0"
    bind_port: int = 4321
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cluster_nodes: List[str] = field(default_factory=list)
    data_dir: str = "./raft_data"

    # Timing parameters (tuned for LAN, adjust for WAN)
    election_timeout_min_ms: int = 300   # 300ms minimum
    election_timeout_max_ms: int = 500   # 500ms maximum
    heartbeat_interval_ms: int = 100     # 100ms heartbeat
    rpc_timeout_ms: int = 200            # 200ms RPC timeout

    # Log management
    max_entries_per_request: int = 100   # Max entries per AppendEntries
    snapshot_threshold: int = 10000      # Snapshot every 10k entries
    max_log_size_bytes: int = 100_000_000  # 100MB max log size
    batch_interval_ms: int = 10          # Batch entries every 10ms

    # Callbacks (set at runtime, not serialized)
    on_leader_change: Optional[Callable[[Optional[str]], None]] = field(
        default=None, repr=False, compare=False
    )
    on_state_change: Optional[Callable[[str], None]] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Generate node_id if not provided."""
        if not self.node_id:
            self.node_id = str(uuid.uuid4())[:8]

    @property
    def bind_address(self) -> str:
        """Get the full bind address for Raft RPC."""
        return f"{self.bind_host}:{self.bind_port}"

    @property
    def api_address(self) -> str:
        """Get the full API address."""
        return f"{self.api_host}:{self.api_port}"

    @property
    def is_single_node(self) -> bool:
        """Check if this is a single-node cluster."""
        return len(self.cluster_nodes) == 0

    @property
    def cluster_size(self) -> int:
        """Get total cluster size including self."""
        return len(self.cluster_nodes) + 1

    @property
    def quorum_size(self) -> int:
        """Calculate quorum size for the cluster (majority)."""
        return (self.cluster_size // 2) + 1

    @classmethod
    def from_env(cls) -> "RaftConfig":
        """Create RaftConfig from environment variables.

        Environment variables:
            FLYBROWSER_NODE_ID: Unique node identifier
            FLYBROWSER_RAFT_HOST: Raft bind host
            FLYBROWSER_RAFT_PORT: Raft bind port
            FLYBROWSER_API_HOST: API bind host
            FLYBROWSER_API_PORT: API bind port
            FLYBROWSER_CLUSTER_PEERS: Comma-separated list of cluster peers (preferred)
            FLYBROWSER_CLUSTER_NODES: Alias for CLUSTER_PEERS (deprecated)
            FLYBROWSER_DATA_DIR: Directory for Raft data
            FLYBROWSER_ELECTION_TIMEOUT_MIN: Min election timeout (ms)
            FLYBROWSER_ELECTION_TIMEOUT_MAX: Max election timeout (ms)
            FLYBROWSER_HEARTBEAT_INTERVAL: Heartbeat interval (ms)

        Returns:
            RaftConfig with values from environment
        """
        # Support both CLUSTER_PEERS (preferred) and CLUSTER_NODES (deprecated)
        cluster_nodes_str = os.environ.get(
            "FLYBROWSER_CLUSTER_PEERS",
            os.environ.get("FLYBROWSER_CLUSTER_NODES", "")
        )
        cluster_nodes = [n.strip() for n in cluster_nodes_str.split(",") if n.strip()]

        return cls(
            node_id=os.environ.get("FLYBROWSER_NODE_ID", ""),
            bind_host=os.environ.get("FLYBROWSER_RAFT_HOST", "0.0.0.0"),
            bind_port=int(os.environ.get("FLYBROWSER_RAFT_PORT", "4321")),
            api_host=os.environ.get("FLYBROWSER_API_HOST", "0.0.0.0"),
            api_port=int(os.environ.get("FLYBROWSER_API_PORT", "8000")),
            cluster_nodes=cluster_nodes,
            data_dir=os.environ.get("FLYBROWSER_DATA_DIR", "./raft_data"),
            election_timeout_min_ms=int(
                os.environ.get("FLYBROWSER_ELECTION_TIMEOUT_MIN", "300")
            ),
            election_timeout_max_ms=int(
                os.environ.get("FLYBROWSER_ELECTION_TIMEOUT_MAX", "500")
            ),
            heartbeat_interval_ms=int(
                os.environ.get("FLYBROWSER_HEARTBEAT_INTERVAL", "100")
            ),
        )

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.node_id:
            errors.append("node_id is required")

        if self.bind_port < 1024 or self.bind_port > 65535:
            errors.append("bind_port must be between 1024 and 65535")

        if self.api_port < 1024 or self.api_port > 65535:
            errors.append("api_port must be between 1024 and 65535")

        if self.election_timeout_min_ms >= self.election_timeout_max_ms:
            errors.append("election_timeout_min_ms must be less than election_timeout_max_ms")

        if self.heartbeat_interval_ms >= self.election_timeout_min_ms:
            errors.append("heartbeat_interval_ms should be less than election_timeout_min_ms")

        # For HA, we need at least 3 nodes (can tolerate 1 failure)
        if self.cluster_size > 1 and self.cluster_size < 3:
            errors.append("Cluster mode requires at least 3 nodes for fault tolerance")

        return errors

    def get_peer_addresses(self) -> List[tuple]:
        """Parse cluster_nodes into (host, raft_port, api_port) tuples."""
        peers = []
        for node in self.cluster_nodes:
            parts = node.split(":")
            if len(parts) == 2:
                # host:raft_port format, assume api_port = raft_port - 321
                peers.append((parts[0], int(parts[1]), int(parts[1]) - 321))
            elif len(parts) == 3:
                # host:raft_port:api_port format
                peers.append((parts[0], int(parts[1]), int(parts[2])))
            else:
                # Just host, use defaults
                peers.append((node, 4321, 8000))
        return peers

