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
High-Availability Cluster Support for FlyBrowser.

This module provides production-grade distributed deployment capabilities
for FlyBrowser with Raft consensus for leader election and state replication.

Architecture:
- Raft Consensus: Leader election and log replication for HA
- StateMachine: Replicated state for sessions, nodes, and configuration
- LoadBalancer: Intelligent request routing with session affinity
- Automatic failover and session migration

Deployment Modes:
1. Standalone: Single node, no clustering
2. Cluster: 3+ nodes with Raft consensus for fault tolerance

Example (Standalone):
    >>> from flybrowser.service.app import create_app
    >>> app = create_app()
    >>> # Run with uvicorn

Example (Cluster - 3 nodes):
    Node 1:
    >>> from flybrowser.service.cluster import HAClusterNode
    >>> node = HAClusterNode(
    ...     node_id="node1",
    ...     raft_port=4321,
    ...     api_port=8000,
    ...     peers=["node2:4322:8001", "node3:4323:8002"],
    ... )
    >>> await node.start()

    Node 2 & 3: Similar configuration with different ports
"""

from flybrowser.service.cluster.coordinator import ClusterCoordinator
from flybrowser.service.cluster.discovery import (
    ClusterDiscovery,
    ClusterMember,
    DiscoveryConfig,
    DiscoveryMethod,
    MemberStatus,
)
from flybrowser.service.cluster.ha_node import HAClusterNode, HANodeConfig
from flybrowser.service.cluster.load_balancer import LoadBalancer, LoadBalancingStrategy
from flybrowser.service.cluster.protocol import (
    MessageType,
    NodeInfo,
    NodeMessage,
    NodeStatus,
)
from flybrowser.service.cluster.raft import (
    NodeRole,
    RaftConfig,
    RaftNode,
    StateMachine,
)
from flybrowser.service.cluster.worker import WorkerNode

__all__ = [
    # Main HA entry point
    "HAClusterNode",
    "HANodeConfig",
    # Discovery
    "ClusterDiscovery",
    "ClusterMember",
    "DiscoveryConfig",
    "DiscoveryMethod",
    "MemberStatus",
    # Legacy (for backward compatibility)
    "ClusterCoordinator",
    "WorkerNode",
    "NodeMessage",
    "MessageType",
    "NodeInfo",
    "NodeStatus",
    # HA components
    "LoadBalancer",
    "LoadBalancingStrategy",
    "NodeRole",
    "RaftConfig",
    "RaftNode",
    "StateMachine",
]
