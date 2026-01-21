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
Custom exceptions for FlyBrowser cluster operations.

This module provides cluster-specific exception types for better
error handling and client communication.
"""

from __future__ import annotations

from typing import Optional


class ClusterError(Exception):
    """Base exception for cluster-related errors."""

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        """Initialize the cluster error.

        Args:
            message: Error message
            retry_after: Optional seconds to wait before retrying
        """
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after


class NotLeaderError(ClusterError):
    """Raised when an operation requires leader but current node is not leader."""

    def __init__(
        self,
        message: str = "Not the cluster leader",
        leader_id: Optional[str] = None,
        leader_address: Optional[str] = None,
    ) -> None:
        """Initialize the not leader error.

        Args:
            message: Error message
            leader_id: ID of the current leader (if known)
            leader_address: API address of the current leader (if known)
        """
        super().__init__(message, retry_after=0.1)
        self.leader_id = leader_id
        self.leader_address = leader_address

    def __str__(self) -> str:
        if self.leader_address:
            return f"{self.message}. Redirect to: {self.leader_address}"
        return self.message


class NoLeaderError(ClusterError):
    """Raised when no leader is available in the cluster."""

    def __init__(
        self,
        message: str = "No leader available",
        retry_after: float = 1.0,
    ) -> None:
        """Initialize the no leader error.

        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
        """
        super().__init__(message, retry_after=retry_after)


class ClusterConfigurationError(ClusterError):
    """Raised when cluster configuration is invalid or changing."""

    def __init__(
        self,
        message: str = "Cluster configuration error",
        in_transition: bool = False,
    ) -> None:
        """Initialize the configuration error.

        Args:
            message: Error message
            in_transition: Whether cluster is in configuration transition
        """
        super().__init__(message, retry_after=1.0 if in_transition else None)
        self.in_transition = in_transition


class SessionNotFoundError(ClusterError):
    """Raised when a session is not found in the cluster."""

    def __init__(
        self,
        session_id: str,
        message: Optional[str] = None,
    ) -> None:
        """Initialize the session not found error.

        Args:
            session_id: The session ID that was not found
            message: Optional custom error message
        """
        super().__init__(message or f"Session not found: {session_id}")
        self.session_id = session_id


class SessionMigrationError(ClusterError):
    """Raised when session migration fails."""

    def __init__(
        self,
        session_id: str,
        source_node: Optional[str] = None,
        target_node: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        """Initialize the session migration error.

        Args:
            session_id: The session ID being migrated
            source_node: Source node ID
            target_node: Target node ID
            message: Optional custom error message
        """
        super().__init__(
            message or f"Failed to migrate session {session_id}",
            retry_after=5.0,
        )
        self.session_id = session_id
        self.source_node = source_node
        self.target_node = target_node


class NodeCapacityError(ClusterError):
    """Raised when cluster has no capacity for new sessions."""

    def __init__(
        self,
        message: str = "No capacity available in cluster",
        total_capacity: int = 0,
        used_capacity: int = 0,
    ) -> None:
        """Initialize the capacity error.

        Args:
            message: Error message
            total_capacity: Total cluster capacity
            used_capacity: Currently used capacity
        """
        super().__init__(message, retry_after=5.0)
        self.total_capacity = total_capacity
        self.used_capacity = used_capacity


class ConsistencyError(ClusterError):
    """Raised when consistency requirements cannot be met."""

    def __init__(
        self,
        message: str = "Cannot satisfy consistency requirement",
        requested_consistency: str = "strong",
    ) -> None:
        """Initialize the consistency error.

        Args:
            message: Error message
            requested_consistency: The consistency level that was requested
        """
        super().__init__(message, retry_after=0.5)
        self.requested_consistency = requested_consistency


class RaftError(ClusterError):
    """Raised for Raft consensus-related errors."""

    def __init__(
        self,
        message: str = "Raft consensus error",
        term: Optional[int] = None,
    ) -> None:
        """Initialize the Raft error.

        Args:
            message: Error message
            term: Current Raft term (if relevant)
        """
        super().__init__(message)
        self.term = term


class CommandTimeoutError(ClusterError):
    """Raised when a command times out waiting for commit."""

    def __init__(
        self,
        message: str = "Command timed out",
        index: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Initialize the command timeout error.

        Args:
            message: Error message
            index: Log index of the command
            timeout: Timeout value in seconds
        """
        super().__init__(message, retry_after=1.0)
        self.index = index
        self.timeout = timeout
