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
Inter-node communication protocol for FlyBrowser cluster mode.

This module defines the message types and data structures used for
communication between coordinator and worker nodes in a cluster.

Protocol Overview:
- Uses HTTP/JSON for simplicity and compatibility
- Supports heartbeat, task assignment, and status reporting
- Designed for reliability with acknowledgments and retries
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageType(str, Enum):
    """Types of messages in the cluster protocol."""
    
    # Node lifecycle
    REGISTER = "register"  # Worker registering with coordinator
    UNREGISTER = "unregister"  # Worker leaving the cluster
    HEARTBEAT = "heartbeat"  # Periodic health check
    HEARTBEAT_ACK = "heartbeat_ack"  # Heartbeat acknowledgment
    
    # Task management
    TASK_ASSIGN = "task_assign"  # Coordinator assigning task to worker
    TASK_ACCEPT = "task_accept"  # Worker accepting task
    TASK_REJECT = "task_reject"  # Worker rejecting task (busy/error)
    TASK_COMPLETE = "task_complete"  # Worker reporting task completion
    TASK_FAILED = "task_failed"  # Worker reporting task failure
    
    # Cluster management
    NODE_STATUS = "node_status"  # Node status update
    CLUSTER_STATUS = "cluster_status"  # Full cluster status
    SHUTDOWN = "shutdown"  # Graceful shutdown request


class NodeStatus(str, Enum):
    """Status of a node in the cluster."""
    
    STARTING = "starting"  # Node is starting up
    READY = "ready"  # Node is ready to accept tasks
    BUSY = "busy"  # Node is at capacity
    DRAINING = "draining"  # Node is finishing tasks before shutdown
    OFFLINE = "offline"  # Node is not responding


@dataclass
class NodeInfo:
    """Information about a node in the cluster.
    
    Attributes:
        node_id: Unique identifier for the node
        host: Host address of the node
        port: Port for the node's API
        role: Node role (coordinator or worker)
        status: Current node status
        max_browsers: Maximum browsers this node can handle
        active_browsers: Current number of active browsers
        active_sessions: Current number of active sessions
        last_heartbeat: Timestamp of last heartbeat
        metadata: Additional node metadata
    """
    
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    host: str = "localhost"
    port: int = 8000
    role: str = "worker"
    status: NodeStatus = NodeStatus.STARTING
    max_browsers: int = 10
    active_browsers: int = 0
    active_sessions: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def available_capacity(self) -> int:
        """Get available browser capacity."""
        return max(0, self.max_browsers - self.active_browsers)
    
    @property
    def is_available(self) -> bool:
        """Check if node can accept new tasks."""
        return self.status == NodeStatus.READY and self.available_capacity > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "role": self.role,
            "status": self.status.value,
            "max_browsers": self.max_browsers,
            "active_browsers": self.active_browsers,
            "active_sessions": self.active_sessions,
            "last_heartbeat": self.last_heartbeat,
            "available_capacity": self.available_capacity,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeInfo":
        """Create from dictionary."""
        return cls(
            node_id=data.get("node_id", str(uuid.uuid4())),
            host=data.get("host", "localhost"),
            port=data.get("port", 8000),
            role=data.get("role", "worker"),
            status=NodeStatus(data.get("status", "starting")),
            max_browsers=data.get("max_browsers", 10),
            active_browsers=data.get("active_browsers", 0),
            active_sessions=data.get("active_sessions", 0),
            last_heartbeat=data.get("last_heartbeat", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class NodeMessage:
    """Message exchanged between nodes in the cluster.
    
    Attributes:
        message_id: Unique message identifier
        message_type: Type of message
        sender_id: ID of the sending node
        recipient_id: ID of the recipient node (optional for broadcasts)
        timestamp: Message creation timestamp
        payload: Message payload data
        correlation_id: ID linking related messages (e.g., request/response)
    """
    
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.HEARTBEAT
    sender_id: str = ""
    recipient_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeMessage":
        """Create from dictionary."""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            message_type=MessageType(data.get("message_type", "heartbeat")),
            sender_id=data.get("sender_id", ""),
            recipient_id=data.get("recipient_id"),
            timestamp=data.get("timestamp", time.time()),
            payload=data.get("payload", {}),
            correlation_id=data.get("correlation_id"),
        )

    @classmethod
    def create_heartbeat(cls, sender_id: str, node_info: NodeInfo) -> "NodeMessage":
        """Create a heartbeat message."""
        return cls(
            message_type=MessageType.HEARTBEAT,
            sender_id=sender_id,
            payload={"node_info": node_info.to_dict()},
        )

    @classmethod
    def create_register(cls, sender_id: str, node_info: NodeInfo) -> "NodeMessage":
        """Create a registration message."""
        return cls(
            message_type=MessageType.REGISTER,
            sender_id=sender_id,
            payload={"node_info": node_info.to_dict()},
        )

    @classmethod
    def create_task_assign(
        cls, sender_id: str, recipient_id: str, task_id: str, task_data: Dict[str, Any]
    ) -> "NodeMessage":
        """Create a task assignment message."""
        return cls(
            message_type=MessageType.TASK_ASSIGN,
            sender_id=sender_id,
            recipient_id=recipient_id,
            payload={"task_id": task_id, "task_data": task_data},
        )

