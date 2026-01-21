# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Raft state machine."""

import time
from unittest.mock import MagicMock

import pytest

from flybrowser.service.cluster.raft.state_machine import (
    CommandType,
    NodeHealth,
    NodeState,
    SessionState,
    SessionStatus,
    StateMachine,
)


class TestCommandType:
    """Tests for CommandType enum."""

    def test_all_commands(self):
        """Test all command types exist."""
        assert CommandType.REGISTER_NODE == "register_node"
        assert CommandType.UNREGISTER_NODE == "unregister_node"
        assert CommandType.CREATE_SESSION == "create_session"
        assert CommandType.DELETE_SESSION == "delete_session"
        assert CommandType.NOOP == "noop"


class TestNodeHealth:
    """Tests for NodeHealth enum."""

    def test_all_health_states(self):
        """Test all health states exist."""
        assert NodeHealth.HEALTHY == "healthy"
        assert NodeHealth.DEGRADED == "degraded"
        assert NodeHealth.UNHEALTHY == "unhealthy"
        assert NodeHealth.UNKNOWN == "unknown"


class TestSessionStatus:
    """Tests for SessionStatus enum."""

    def test_all_statuses(self):
        """Test all session statuses exist."""
        assert SessionStatus.PENDING == "pending"
        assert SessionStatus.ACTIVE == "active"
        assert SessionStatus.MIGRATING == "migrating"
        assert SessionStatus.CLOSED == "closed"


class TestNodeState:
    """Tests for NodeState."""

    def test_creation(self):
        """Test NodeState creation."""
        node = NodeState(
            node_id="node-1",
            api_address="localhost:8000",
            raft_address="localhost:4321",
        )
        
        assert node.node_id == "node-1"
        assert node.api_address == "localhost:8000"
        assert node.health == NodeHealth.UNKNOWN

    def test_available_capacity(self):
        """Test available_capacity property."""
        node = NodeState(
            node_id="node-1",
            api_address="localhost:8000",
            raft_address="localhost:4321",
            active_sessions=3,
            max_sessions=10,
        )
        
        assert node.available_capacity == 7

    def test_load_score(self):
        """Test load_score calculation."""
        node = NodeState(
            node_id="node-1",
            api_address="localhost:8000",
            raft_address="localhost:4321",
            active_sessions=5,
            max_sessions=10,
            cpu_percent=50.0,
            memory_percent=30.0,
        )
        
        score = node.load_score
        assert 0 <= score <= 1

    def test_to_dict(self):
        """Test serialization."""
        node = NodeState(
            node_id="node-1",
            api_address="localhost:8000",
            raft_address="localhost:4321",
        )
        
        d = node.to_dict()
        
        assert d["node_id"] == "node-1"
        assert d["api_address"] == "localhost:8000"

    def test_from_dict(self):
        """Test deserialization."""
        d = {
            "node_id": "node-1",
            "api_address": "localhost:8000",
            "raft_address": "localhost:4321",
            "health": "healthy",
        }
        
        node = NodeState.from_dict(d)
        
        assert node.node_id == "node-1"
        assert node.health == NodeHealth.HEALTHY


class TestSessionState:
    """Tests for SessionState."""

    def test_creation(self):
        """Test SessionState creation."""
        session = SessionState(
            session_id="sess-123",
            node_id="node-1",
        )
        
        assert session.session_id == "sess-123"
        assert session.node_id == "node-1"
        assert session.status == SessionStatus.PENDING

    def test_to_dict(self):
        """Test serialization."""
        session = SessionState(
            session_id="sess-123",
            node_id="node-1",
            status=SessionStatus.ACTIVE,
        )
        
        d = session.to_dict()
        
        assert d["session_id"] == "sess-123"
        assert d["status"] == "active"

    def test_from_dict(self):
        """Test deserialization."""
        d = {
            "session_id": "sess-123",
            "node_id": "node-1",
            "status": "active",
        }
        
        session = SessionState.from_dict(d)
        
        assert session.session_id == "sess-123"
        assert session.status == SessionStatus.ACTIVE


class TestStateMachine:
    """Tests for StateMachine."""

    def test_init(self):
        """Test StateMachine initialization."""
        sm = StateMachine()
        
        assert len(sm._nodes) == 0
        assert len(sm._sessions) == 0
        assert sm._last_applied_index == 0

    def test_apply_register_node(self):
        """Test applying register_node command."""
        sm = StateMachine()
        
        command = {
            "type": "register_node",
            "node": {
                "node_id": "node-1",
                "api_address": "localhost:8000",
                "raft_address": "localhost:4321",
            }
        }
        
        sm.apply(command, index=1)
        
        assert "node-1" in sm._nodes
        assert sm._nodes["node-1"].api_address == "localhost:8000"

    def test_apply_unregister_node(self):
        """Test applying unregister_node command."""
        sm = StateMachine()
        
        # First register
        sm.apply({
            "type": "register_node",
            "node": {
                "node_id": "node-1",
                "api_address": "localhost:8000",
                "raft_address": "localhost:4321",
            }
        }, index=1)
        
        # Then unregister
        sm.apply({
            "type": "unregister_node",
            "node_id": "node-1",
        }, index=2)
        
        assert "node-1" not in sm._nodes

    def test_apply_create_session(self):
        """Test applying create_session command."""
        sm = StateMachine()
        
        # Register node first
        sm.apply({
            "type": "register_node",
            "node": {
                "node_id": "node-1",
                "api_address": "localhost:8000",
                "raft_address": "localhost:4321",
            }
        }, index=1)
        
        # Create session
        sm.apply({
            "type": "create_session",
            "session": {
                "session_id": "sess-123",
                "node_id": "node-1",
            }
        }, index=2)
        
        assert "sess-123" in sm._sessions
        assert sm._sessions["sess-123"].node_id == "node-1"

    def test_apply_delete_session(self):
        """Test applying delete_session command."""
        sm = StateMachine()
        
        # Create session
        sm.apply({
            "type": "create_session",
            "session": {
                "session_id": "sess-123",
                "node_id": "node-1",
            }
        }, index=1)
        
        # Delete session
        sm.apply({
            "type": "delete_session",
            "session_id": "sess-123",
        }, index=2)
        
        assert "sess-123" not in sm._sessions

    def test_apply_idempotency(self):
        """Test that applying same index twice is idempotent."""
        sm = StateMachine()
        
        command = {
            "type": "register_node",
            "node": {
                "node_id": "node-1",
                "api_address": "localhost:8000",
                "raft_address": "localhost:4321",
            }
        }
        
        sm.apply(command, index=1)
        sm.apply(command, index=1)  # Same index
        
        assert len(sm._nodes) == 1

    def test_apply_noop(self):
        """Test applying noop command."""
        sm = StateMachine()
        
        result = sm.apply({"type": "noop"}, index=1)
        
        # NOOP should not change state
        assert len(sm._nodes) == 0
        assert len(sm._sessions) == 0

    def test_get_all_nodes(self):
        """Test getting all nodes."""
        sm = StateMachine()
        
        sm.apply({
            "type": "register_node",
            "node": {"node_id": "node-1", "api_address": "a", "raft_address": "b"}
        }, index=1)
        sm.apply({
            "type": "register_node",
            "node": {"node_id": "node-2", "api_address": "c", "raft_address": "d"}
        }, index=2)
        
        nodes = sm.get_all_nodes()
        
        assert len(nodes) == 2

    def test_get_session(self):
        """Test getting a session."""
        sm = StateMachine()
        
        sm.apply({
            "type": "create_session",
            "session": {"session_id": "sess-1", "node_id": "node-1"}
        }, index=1)
        
        session = sm.get_session("sess-1")
        
        assert session is not None
        assert session.session_id == "sess-1"

    def test_callback_on_node_change(self):
        """Test callback is called on node change."""
        sm = StateMachine()
        callback = MagicMock()
        sm.set_callbacks(on_node_change=callback)
        
        sm.apply({
            "type": "register_node",
            "node": {"node_id": "node-1", "api_address": "a", "raft_address": "b"}
        }, index=1)
        
        callback.assert_called()


class TestStateMachineSnapshot:
    """Tests for StateMachine snapshot operations."""

    def test_export_state(self):
        """Test exporting state for snapshot."""
        sm = StateMachine()
        
        sm.apply({
            "type": "register_node",
            "node": {"node_id": "node-1", "api_address": "a", "raft_address": "b"}
        }, index=1)
        sm.apply({
            "type": "create_session",
            "session": {"session_id": "sess-1", "node_id": "node-1"}
        }, index=2)
        
        state_bytes = sm.serialize()
        
        assert len(state_bytes) > 0

    def test_import_state(self):
        """Test importing state from snapshot."""
        sm1 = StateMachine()
        
        sm1.apply({
            "type": "register_node",
            "node": {"node_id": "node-1", "api_address": "a", "raft_address": "b"}
        }, index=1)
        
        state_bytes = sm1.serialize()
        
        # Import into new state machine
        sm2 = StateMachine()
        sm2.deserialize(state_bytes)
        
        assert "node-1" in sm2._nodes
