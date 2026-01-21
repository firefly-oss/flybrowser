# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for cluster exceptions."""

import pytest

from flybrowser.service.cluster.exceptions import (
    ClusterConfigurationError,
    ClusterError,
    CommandTimeoutError,
    ConsistencyError,
    NodeCapacityError,
    NoLeaderError,
    NotLeaderError,
    RaftError,
    SessionMigrationError,
    SessionNotFoundError,
)


class TestClusterError:
    """Tests for base ClusterError."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = ClusterError("Something went wrong")
        
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.retry_after is None

    def test_with_retry_after(self):
        """Test error with retry_after."""
        error = ClusterError("Retry later", retry_after=5.0)
        
        assert error.retry_after == 5.0


class TestNotLeaderError:
    """Tests for NotLeaderError."""

    def test_default_message(self):
        """Test default error message."""
        error = NotLeaderError()
        
        assert "Not the cluster leader" in str(error)

    def test_with_leader_info(self):
        """Test with leader information."""
        error = NotLeaderError(
            leader_id="node-1",
            leader_address="localhost:8000"
        )
        
        assert error.leader_id == "node-1"
        assert error.leader_address == "localhost:8000"
        assert "localhost:8000" in str(error)

    def test_has_retry_after(self):
        """Test NotLeaderError has retry_after."""
        error = NotLeaderError()
        
        assert error.retry_after is not None
        assert error.retry_after == 0.1


class TestNoLeaderError:
    """Tests for NoLeaderError."""

    def test_default_message(self):
        """Test default error message."""
        error = NoLeaderError()
        
        assert "No leader available" in str(error)

    def test_default_retry_after(self):
        """Test default retry_after value."""
        error = NoLeaderError()
        
        assert error.retry_after == 1.0


class TestClusterConfigurationError:
    """Tests for ClusterConfigurationError."""

    def test_default_message(self):
        """Test default error message."""
        error = ClusterConfigurationError()
        
        assert "Cluster configuration error" in str(error)

    def test_in_transition(self):
        """Test error during configuration transition."""
        error = ClusterConfigurationError(in_transition=True)
        
        assert error.in_transition is True
        assert error.retry_after == 1.0

    def test_not_in_transition(self):
        """Test error not during transition."""
        error = ClusterConfigurationError(in_transition=False)
        
        assert error.in_transition is False
        assert error.retry_after is None


class TestSessionNotFoundError:
    """Tests for SessionNotFoundError."""

    def test_with_session_id(self):
        """Test error with session ID."""
        error = SessionNotFoundError("sess-123")
        
        assert error.session_id == "sess-123"
        assert "sess-123" in str(error)

    def test_custom_message(self):
        """Test with custom message."""
        error = SessionNotFoundError("sess-123", message="Custom message")
        
        assert str(error) == "Custom message"


class TestSessionMigrationError:
    """Tests for SessionMigrationError."""

    def test_with_details(self):
        """Test error with migration details."""
        error = SessionMigrationError(
            session_id="sess-123",
            source_node="node-1",
            target_node="node-2"
        )
        
        assert error.session_id == "sess-123"
        assert error.source_node == "node-1"
        assert error.target_node == "node-2"
        assert error.retry_after == 5.0


class TestNodeCapacityError:
    """Tests for NodeCapacityError."""

    def test_default_message(self):
        """Test default error message."""
        error = NodeCapacityError()
        
        assert "No capacity available" in str(error)

    def test_with_capacity_info(self):
        """Test with capacity information."""
        error = NodeCapacityError(
            total_capacity=100,
            used_capacity=100
        )
        
        assert error.total_capacity == 100
        assert error.used_capacity == 100
        assert error.retry_after == 5.0


class TestConsistencyError:
    """Tests for ConsistencyError."""

    def test_default_message(self):
        """Test default error message."""
        error = ConsistencyError()
        
        assert "consistency" in str(error).lower()

    def test_with_consistency_level(self):
        """Test with consistency level."""
        error = ConsistencyError(requested_consistency="strong")
        
        assert error.requested_consistency == "strong"


class TestRaftError:
    """Tests for RaftError."""

    def test_default_message(self):
        """Test default error message."""
        error = RaftError()
        
        assert "Raft consensus error" in str(error)

    def test_with_term(self):
        """Test with Raft term."""
        error = RaftError(term=5)
        
        assert error.term == 5


class TestCommandTimeoutError:
    """Tests for CommandTimeoutError."""

    def test_default_message(self):
        """Test default error message."""
        error = CommandTimeoutError()
        
        assert "timed out" in str(error).lower()

    def test_with_details(self):
        """Test with timeout details."""
        error = CommandTimeoutError(
            index=42,
            timeout=30.0
        )
        
        assert error.index == 42
        assert error.timeout == 30.0
        assert error.retry_after == 1.0
