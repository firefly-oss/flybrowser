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
Load Balancer for FlyBrowser HA Cluster.

This module provides intelligent load balancing for distributing browser
sessions across cluster nodes. Features include:
- Capacity-aware routing (CPU, memory, active sessions)
- Session affinity (requests for same session go to same node)
- Automatic failover when nodes become unavailable
- Weighted round-robin with health awareness
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from flybrowser.service.cluster.raft.state_machine import NodeHealth, NodeState, SessionState
from flybrowser.utils.logger import logger


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_LOAD = "least_load"
    RANDOM = "random"
    WEIGHTED = "weighted"


@dataclass
class NodeScore:
    """Scoring for a node used in load balancing decisions."""
    node_id: str
    api_address: str
    score: float  # Lower is better
    available_capacity: int
    health: NodeHealth
    
    @property
    def is_available(self) -> bool:
        """Check if node is available for new sessions."""
        return self.health == NodeHealth.HEALTHY and self.available_capacity > 0


class LoadBalancer:
    """Intelligent load balancer for cluster session distribution.
    
    Selects the best node for new sessions based on:
    - Node health status
    - Available capacity (max_sessions - active_sessions)
    - Resource usage (CPU, memory)
    - Current load score
    
    Also handles session affinity to ensure requests for the same
    session are routed to the correct node.
    
    Example:
        >>> lb = LoadBalancer()
        >>> lb.update_nodes(nodes)
        >>> node = lb.select_node_for_session()
        >>> lb.register_session("session-123", node.node_id)
        >>> target = lb.get_node_for_session("session-123")
    """
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOAD,
        health_check_interval: float = 5.0,
    ) -> None:
        """Initialize the load balancer.
        
        Args:
            strategy: Load balancing strategy to use
            health_check_interval: Interval for health checks in seconds
        """
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        
        self._lock = threading.RLock()
        
        # Node state
        self._nodes: Dict[str, NodeState] = {}
        self._node_scores: Dict[str, NodeScore] = {}
        
        # Session affinity
        self._session_to_node: Dict[str, str] = {}
        
        # Round-robin state
        self._rr_index = 0
        self._rr_order: List[str] = []
        
        # Health tracking
        self._last_health_check: Dict[str, float] = {}
        self._consecutive_failures: Dict[str, int] = {}
    
    def update_nodes(self, nodes: List[NodeState]) -> None:
        """Update the list of available nodes.
        
        Args:
            nodes: List of current node states
        """
        with self._lock:
            self._nodes = {n.node_id: n for n in nodes}
            self._update_scores()
            self._update_rr_order()
    
    def update_node(self, node: NodeState) -> None:
        """Update a single node's state.
        
        Args:
            node: Updated node state
        """
        with self._lock:
            self._nodes[node.node_id] = node
            self._update_node_score(node)
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the load balancer.
        
        Args:
            node_id: ID of node to remove
        """
        with self._lock:
            self._nodes.pop(node_id, None)
            self._node_scores.pop(node_id, None)
            if node_id in self._rr_order:
                self._rr_order.remove(node_id)
    
    def _update_scores(self) -> None:
        """Update scores for all nodes."""
        for node in self._nodes.values():
            self._update_node_score(node)
    
    def _update_node_score(self, node: NodeState) -> None:
        """Update score for a single node."""
        # Calculate score (lower is better)
        # Factors: load_score (0-1), health penalty, capacity bonus
        
        base_score = node.load_score
        
        # Health penalty
        if node.health == NodeHealth.DEGRADED:
            base_score += 0.3
        elif node.health == NodeHealth.UNHEALTHY:
            base_score += 1.0
        elif node.health == NodeHealth.UNKNOWN:
            base_score += 0.5
        
        # Capacity bonus (prefer nodes with more capacity)
        capacity_ratio = node.available_capacity / max(1, node.max_sessions)
        base_score -= capacity_ratio * 0.2
        
        self._node_scores[node.node_id] = NodeScore(
            node_id=node.node_id,
            api_address=node.api_address,
            score=max(0, base_score),
            available_capacity=node.available_capacity,
            health=node.health,
        )

    def _update_rr_order(self) -> None:
        """Update round-robin order based on available nodes."""
        available = [
            node_id for node_id, score in self._node_scores.items()
            if score.is_available
        ]
        self._rr_order = sorted(available)
        if self._rr_index >= len(self._rr_order):
            self._rr_index = 0

    # ==================== Node Selection ====================

    def select_node_for_session(self) -> Optional[NodeState]:
        """Select the best node for a new session.

        Returns:
            Selected NodeState or None if no nodes available
        """
        with self._lock:
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._select_round_robin()
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._select_least_connections()
            elif self.strategy == LoadBalancingStrategy.LEAST_LOAD:
                return self._select_least_load()
            elif self.strategy == LoadBalancingStrategy.RANDOM:
                return self._select_random()
            elif self.strategy == LoadBalancingStrategy.WEIGHTED:
                return self._select_weighted()
            else:
                return self._select_least_load()

    def _select_round_robin(self) -> Optional[NodeState]:
        """Select using round-robin strategy."""
        if not self._rr_order:
            return None

        # Find next available node
        for _ in range(len(self._rr_order)):
            node_id = self._rr_order[self._rr_index]
            self._rr_index = (self._rr_index + 1) % len(self._rr_order)

            score = self._node_scores.get(node_id)
            if score and score.is_available:
                return self._nodes.get(node_id)

        return None

    def _select_least_connections(self) -> Optional[NodeState]:
        """Select node with fewest active sessions."""
        available = [
            (node_id, self._nodes[node_id].active_sessions)
            for node_id, score in self._node_scores.items()
            if score.is_available and node_id in self._nodes
        ]

        if not available:
            return None

        # Sort by active sessions (ascending)
        available.sort(key=lambda x: x[1])
        return self._nodes.get(available[0][0])

    def _select_least_load(self) -> Optional[NodeState]:
        """Select node with lowest load score."""
        available = [
            (node_id, score.score)
            for node_id, score in self._node_scores.items()
            if score.is_available
        ]

        if not available:
            return None

        # Sort by score (ascending - lower is better)
        available.sort(key=lambda x: x[1])
        return self._nodes.get(available[0][0])

    def _select_random(self) -> Optional[NodeState]:
        """Select a random available node."""
        available = [
            node_id for node_id, score in self._node_scores.items()
            if score.is_available
        ]

        if not available:
            return None

        return self._nodes.get(random.choice(available))

    def _select_weighted(self) -> Optional[NodeState]:
        """Select using weighted random based on available capacity."""
        available = [
            (node_id, score.available_capacity)
            for node_id, score in self._node_scores.items()
            if score.is_available
        ]

        if not available:
            return None

        # Weighted random selection
        total_capacity = sum(cap for _, cap in available)
        if total_capacity == 0:
            return self._nodes.get(available[0][0])

        r = random.uniform(0, total_capacity)
        cumulative = 0
        for node_id, capacity in available:
            cumulative += capacity
            if r <= cumulative:
                return self._nodes.get(node_id)

        return self._nodes.get(available[-1][0])

    # ==================== Session Affinity ====================

    def register_session(self, session_id: str, node_id: str) -> None:
        """Register a session's node assignment.

        Args:
            session_id: Session identifier
            node_id: Node handling the session
        """
        with self._lock:
            self._session_to_node[session_id] = node_id

    def unregister_session(self, session_id: str) -> None:
        """Unregister a session.

        Args:
            session_id: Session identifier
        """
        with self._lock:
            self._session_to_node.pop(session_id, None)

    def get_node_for_session(self, session_id: str) -> Optional[NodeState]:
        """Get the node handling a specific session.

        Args:
            session_id: Session identifier

        Returns:
            NodeState or None if session not found
        """
        with self._lock:
            node_id = self._session_to_node.get(session_id)
            if node_id:
                return self._nodes.get(node_id)
            return None

    def get_node_address_for_session(self, session_id: str) -> Optional[str]:
        """Get the API address of the node handling a session.

        Args:
            session_id: Session identifier

        Returns:
            API address or None if session not found
        """
        node = self.get_node_for_session(session_id)
        if node:
            return node.api_address
        return None

    # ==================== Failover ====================

    def handle_node_failure(self, node_id: str) -> List[str]:
        """Handle a node failure and return orphaned sessions.

        Args:
            node_id: ID of failed node

        Returns:
            List of session IDs that need to be migrated
        """
        with self._lock:
            # Find sessions on failed node
            orphaned = [
                session_id for session_id, nid in self._session_to_node.items()
                if nid == node_id
            ]

            # Remove node
            self.remove_node(node_id)

            logger.warning(f"Node {node_id} failed, {len(orphaned)} sessions orphaned")
            return orphaned

    def migrate_session(self, session_id: str, target_node_id: str) -> bool:
        """Migrate a session to a new node.

        Args:
            session_id: Session to migrate
            target_node_id: Target node ID

        Returns:
            True if migration registered successfully
        """
        with self._lock:
            if target_node_id not in self._nodes:
                return False

            self._session_to_node[session_id] = target_node_id
            logger.info(f"Session {session_id} migrated to node {target_node_id}")
            return True

    # ==================== Status ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            available_nodes = sum(1 for s in self._node_scores.values() if s.is_available)
            total_capacity = sum(n.max_sessions for n in self._nodes.values())
            used_capacity = sum(n.active_sessions for n in self._nodes.values())

            return {
                "strategy": self.strategy.value,
                "total_nodes": len(self._nodes),
                "available_nodes": available_nodes,
                "total_capacity": total_capacity,
                "used_capacity": used_capacity,
                "available_capacity": total_capacity - used_capacity,
                "active_sessions": len(self._session_to_node),
                "nodes": [
                    {
                        "node_id": s.node_id,
                        "score": s.score,
                        "available_capacity": s.available_capacity,
                        "health": s.health.value,
                        "is_available": s.is_available,
                    }
                    for s in self._node_scores.values()
                ],
            }

