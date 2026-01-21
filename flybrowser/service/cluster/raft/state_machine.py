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
Replicated State Machine for FlyBrowser HA Cluster.

This module provides the state machine that is replicated across all cluster
nodes via Raft consensus. It tracks cluster-wide state including:
- Node registry (which nodes are in the cluster)
- Session assignments (which session is on which node)
- Job queue state
- Cluster configuration

Commands are applied to the state machine in log order, ensuring all nodes
have identical state.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


class CommandType(str, Enum):
    """Types of commands that can be applied to the state machine."""
    # Node management
    REGISTER_NODE = "register_node"
    UNREGISTER_NODE = "unregister_node"
    UPDATE_NODE = "update_node"
    
    # Session management
    CREATE_SESSION = "create_session"
    UPDATE_SESSION = "update_session"
    DELETE_SESSION = "delete_session"
    MIGRATE_SESSION = "migrate_session"
    
    # Configuration
    SET_CONFIG = "set_config"
    DELETE_CONFIG = "delete_config"
    
    # No-op for leader election
    NOOP = "noop"


class NodeHealth(str, Enum):
    """Health status of a cluster node."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class SessionStatus(str, Enum):
    """Status of a browser session."""
    PENDING = "pending"
    ACTIVE = "active"
    IDLE = "idle"
    MIGRATING = "migrating"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class NodeState:
    """State of a node in the cluster."""
    node_id: str
    api_address: str
    raft_address: str
    health: NodeHealth = NodeHealth.UNKNOWN
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    active_sessions: int = 0
    max_sessions: int = 10
    last_heartbeat: float = field(default_factory=time.time)
    joined_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def available_capacity(self) -> int:
        """Get available session capacity."""
        return max(0, self.max_sessions - self.active_sessions)
    
    @property
    def load_score(self) -> float:
        """Calculate load score (0-1, lower is better)."""
        session_load = self.active_sessions / max(1, self.max_sessions)
        resource_load = (self.cpu_percent + self.memory_percent) / 200
        return (session_load * 0.6) + (resource_load * 0.4)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "api_address": self.api_address,
            "raft_address": self.raft_address,
            "health": self.health.value,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "active_sessions": self.active_sessions,
            "max_sessions": self.max_sessions,
            "available_capacity": self.available_capacity,
            "load_score": self.load_score,
            "last_heartbeat": self.last_heartbeat,
            "joined_at": self.joined_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeState":
        return cls(
            node_id=data["node_id"],
            api_address=data["api_address"],
            raft_address=data["raft_address"],
            health=NodeHealth(data.get("health", "unknown")),
            cpu_percent=data.get("cpu_percent", 0.0),
            memory_percent=data.get("memory_percent", 0.0),
            active_sessions=data.get("active_sessions", 0),
            max_sessions=data.get("max_sessions", 10),
            last_heartbeat=data.get("last_heartbeat", time.time()),
            joined_at=data.get("joined_at", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SessionState:
    """State of a browser session."""
    session_id: str
    node_id: str
    status: SessionStatus = SessionStatus.PENDING
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    client_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "node_id": self.node_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "client_id": self.client_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        return cls(
            session_id=data["session_id"],
            node_id=data["node_id"],
            status=SessionStatus(data.get("status", "pending")),
            created_at=data.get("created_at", time.time()),
            last_activity=data.get("last_activity", time.time()),
            client_id=data.get("client_id"),
            metadata=data.get("metadata", {}),
        )


class StateMachine:
    """Replicated state machine for cluster-wide data.

    This class maintains the cluster state that is replicated across all nodes.
    Commands are applied in log order to ensure consistency.

    Thread-safe for concurrent access.

    Example:
        >>> sm = StateMachine()
        >>> sm.apply({"type": "register_node", "node": {...}})
        >>> nodes = sm.get_all_nodes()
    """

    def __init__(self) -> None:
        """Initialize the state machine."""
        self._lock = threading.RLock()

        # State
        self._nodes: Dict[str, NodeState] = {}
        self._sessions: Dict[str, SessionState] = {}
        self._config: Dict[str, Any] = {}

        # Last applied index for idempotency
        self._last_applied_index: int = 0

        # Callbacks for state changes
        self._on_node_change: Optional[Callable[[str, Optional[NodeState]], None]] = None
        self._on_session_change: Optional[Callable[[str, Optional[SessionState]], None]] = None

    def set_callbacks(
        self,
        on_node_change: Optional[Callable[[str, Optional[NodeState]], None]] = None,
        on_session_change: Optional[Callable[[str, Optional[SessionState]], None]] = None,
    ) -> None:
        """Set callbacks for state changes."""
        self._on_node_change = on_node_change
        self._on_session_change = on_session_change

    def apply(self, command: Dict[str, Any], index: int = 0) -> Any:
        """Apply a command to the state machine.

        Args:
            command: The command to apply
            index: The log index of this command (for idempotency)

        Returns:
            Result of the command (command-specific)
        """
        with self._lock:
            # Skip if already applied (idempotency)
            if index > 0 and index <= self._last_applied_index:
                return None

            cmd_type = CommandType(command.get("type", "noop"))
            result = None

            if cmd_type == CommandType.REGISTER_NODE:
                result = self._apply_register_node(command)
            elif cmd_type == CommandType.UNREGISTER_NODE:
                result = self._apply_unregister_node(command)
            elif cmd_type == CommandType.UPDATE_NODE:
                result = self._apply_update_node(command)
            elif cmd_type == CommandType.CREATE_SESSION:
                result = self._apply_create_session(command)
            elif cmd_type == CommandType.UPDATE_SESSION:
                result = self._apply_update_session(command)
            elif cmd_type == CommandType.DELETE_SESSION:
                result = self._apply_delete_session(command)
            elif cmd_type == CommandType.MIGRATE_SESSION:
                result = self._apply_migrate_session(command)
            elif cmd_type == CommandType.SET_CONFIG:
                result = self._apply_set_config(command)
            elif cmd_type == CommandType.DELETE_CONFIG:
                result = self._apply_delete_config(command)
            elif cmd_type == CommandType.NOOP:
                result = True

            if index > 0:
                self._last_applied_index = index

            return result

    # ==================== Node Commands ====================

    def _apply_register_node(self, command: Dict[str, Any]) -> bool:
        """Register a new node."""
        node_data = command.get("node", {})
        node = NodeState.from_dict(node_data)
        self._nodes[node.node_id] = node

        if self._on_node_change:
            self._on_node_change(node.node_id, node)

        return True

    def _apply_unregister_node(self, command: Dict[str, Any]) -> bool:
        """Unregister a node."""
        node_id = command.get("node_id", "")
        if node_id in self._nodes:
            del self._nodes[node_id]

            # Mark sessions on this node as error
            for session in self._sessions.values():
                if session.node_id == node_id:
                    session.status = SessionStatus.ERROR
                    session.node_id = ""

            if self._on_node_change:
                self._on_node_change(node_id, None)

            return True
        return False

    def _apply_update_node(self, command: Dict[str, Any]) -> bool:
        """Update a node's state."""
        node_id = command.get("node_id", "")
        updates = command.get("updates", {})

        if node_id in self._nodes:
            node = self._nodes[node_id]
            for key, value in updates.items():
                if hasattr(node, key):
                    if key == "health":
                        value = NodeHealth(value)
                    setattr(node, key, value)
            node.last_heartbeat = time.time()

            if self._on_node_change:
                self._on_node_change(node_id, node)

            return True
        return False

    # ==================== Session Commands ====================

    def _apply_create_session(self, command: Dict[str, Any]) -> bool:
        """Create a new session."""
        session_data = command.get("session", {})
        session = SessionState.from_dict(session_data)
        self._sessions[session.session_id] = session

        # Update node's active session count
        if session.node_id in self._nodes:
            self._nodes[session.node_id].active_sessions += 1

        if self._on_session_change:
            self._on_session_change(session.session_id, session)

        return True

    def _apply_update_session(self, command: Dict[str, Any]) -> bool:
        """Update a session's state."""
        session_id = command.get("session_id", "")
        updates = command.get("updates", {})

        if session_id in self._sessions:
            session = self._sessions[session_id]
            for key, value in updates.items():
                if hasattr(session, key):
                    if key == "status":
                        value = SessionStatus(value)
                    setattr(session, key, value)
            session.last_activity = time.time()

            if self._on_session_change:
                self._on_session_change(session_id, session)

            return True
        return False

    def _apply_delete_session(self, command: Dict[str, Any]) -> bool:
        """Delete a session."""
        session_id = command.get("session_id", "")

        if session_id in self._sessions:
            session = self._sessions[session_id]

            # Update node's active session count
            if session.node_id in self._nodes:
                self._nodes[session.node_id].active_sessions = max(
                    0, self._nodes[session.node_id].active_sessions - 1
                )

            del self._sessions[session_id]

            if self._on_session_change:
                self._on_session_change(session_id, None)

            return True
        return False

    def _apply_migrate_session(self, command: Dict[str, Any]) -> bool:
        """Migrate a session to a different node."""
        session_id = command.get("session_id", "")
        target_node_id = command.get("target_node_id", "")

        if session_id in self._sessions and target_node_id in self._nodes:
            session = self._sessions[session_id]
            old_node_id = session.node_id

            # Update old node's count
            if old_node_id in self._nodes:
                self._nodes[old_node_id].active_sessions = max(
                    0, self._nodes[old_node_id].active_sessions - 1
                )

            # Update session
            session.node_id = target_node_id
            session.status = SessionStatus.MIGRATING
            session.last_activity = time.time()

            # Update new node's count
            self._nodes[target_node_id].active_sessions += 1

            if self._on_session_change:
                self._on_session_change(session_id, session)

            return True
        return False

    # ==================== Config Commands ====================

    def _apply_set_config(self, command: Dict[str, Any]) -> bool:
        """Set a configuration value."""
        key = command.get("key", "")
        value = command.get("value")
        if key:
            self._config[key] = value
            return True
        return False

    def _apply_delete_config(self, command: Dict[str, Any]) -> bool:
        """Delete a configuration value."""
        key = command.get("key", "")
        if key in self._config:
            del self._config[key]
            return True
        return False

    # ==================== Query Methods (Read-only) ====================

    def get_node(self, node_id: str) -> Optional[NodeState]:
        """Get a node by ID."""
        with self._lock:
            return self._nodes.get(node_id)

    def get_all_nodes(self) -> List[NodeState]:
        """Get all nodes."""
        with self._lock:
            return list(self._nodes.values())

    def get_healthy_nodes(self) -> List[NodeState]:
        """Get all healthy nodes with available capacity."""
        with self._lock:
            return [
                n for n in self._nodes.values()
                if n.health == NodeHealth.HEALTHY and n.available_capacity > 0
            ]

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get a session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_all_sessions(self) -> List[SessionState]:
        """Get all sessions."""
        with self._lock:
            return list(self._sessions.values())

    def get_sessions_for_node(self, node_id: str) -> List[SessionState]:
        """Get all sessions on a specific node."""
        with self._lock:
            return [s for s in self._sessions.values() if s.node_id == node_id]

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        with self._lock:
            return self._config.get(key, default)

    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration."""
        with self._lock:
            return dict(self._config)

    # ==================== Snapshot Methods ====================

    def serialize(self) -> bytes:
        """Serialize the state machine to bytes for snapshotting."""
        with self._lock:
            state = {
                "nodes": {k: v.to_dict() for k, v in self._nodes.items()},
                "sessions": {k: v.to_dict() for k, v in self._sessions.items()},
                "config": self._config,
                "last_applied_index": self._last_applied_index,
            }
            return json.dumps(state).encode("utf-8")

    def deserialize(self, data: bytes) -> None:
        """Restore state machine from snapshot bytes."""
        with self._lock:
            state = json.loads(data.decode("utf-8"))

            self._nodes = {
                k: NodeState.from_dict(v) for k, v in state.get("nodes", {}).items()
            }
            self._sessions = {
                k: SessionState.from_dict(v) for k, v in state.get("sessions", {}).items()
            }
            self._config = state.get("config", {})
            self._last_applied_index = state.get("last_applied_index", 0)

    @property
    def last_applied_index(self) -> int:
        """Get the last applied log index."""
        with self._lock:
            return self._last_applied_index

    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics."""
        with self._lock:
            total_capacity = sum(n.max_sessions for n in self._nodes.values())
            total_active = sum(n.active_sessions for n in self._nodes.values())
            healthy_nodes = sum(1 for n in self._nodes.values() if n.health == NodeHealth.HEALTHY)

            return {
                "node_count": len(self._nodes),
                "healthy_nodes": healthy_nodes,
                "total_capacity": total_capacity,
                "total_active_sessions": total_active,
                "available_capacity": total_capacity - total_active,
                "session_count": len(self._sessions),
            }
