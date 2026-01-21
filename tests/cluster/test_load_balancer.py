# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for cluster load balancer."""

import pytest

from flybrowser.service.cluster.load_balancer import (
    LoadBalancer,
    LoadBalancingStrategy,
    NodeScore,
)
from flybrowser.service.cluster.raft.state_machine import NodeHealth, NodeState


class TestLoadBalancingStrategy:
    """Tests for LoadBalancingStrategy enum."""

    def test_all_strategies(self):
        """Test all strategies exist."""
        assert LoadBalancingStrategy.ROUND_ROBIN == "round_robin"
        assert LoadBalancingStrategy.LEAST_CONNECTIONS == "least_connections"
        assert LoadBalancingStrategy.LEAST_LOAD == "least_load"
        assert LoadBalancingStrategy.RANDOM == "random"


class TestNodeScore:
    """Tests for NodeScore."""

    def test_node_score(self):
        """Test NodeScore creation."""
        score = NodeScore(
            node_id="node-1",
            api_address="localhost:8000",
            score=0.5,
            available_capacity=5,
            health=NodeHealth.HEALTHY,
        )
        
        assert score.node_id == "node-1"
        assert score.is_available is True

    def test_is_available_unhealthy(self):
        """Test is_available when unhealthy."""
        score = NodeScore(
            node_id="node-1",
            api_address="localhost:8000",
            score=0.5,
            available_capacity=5,
            health=NodeHealth.UNHEALTHY,
        )
        
        assert score.is_available is False

    def test_is_available_no_capacity(self):
        """Test is_available when no capacity."""
        score = NodeScore(
            node_id="node-1",
            api_address="localhost:8000",
            score=0.5,
            available_capacity=0,
            health=NodeHealth.HEALTHY,
        )
        
        assert score.is_available is False


class TestLoadBalancer:
    """Tests for LoadBalancer."""

    def test_init(self):
        """Test LoadBalancer initialization."""
        lb = LoadBalancer()
        
        assert lb.strategy == LoadBalancingStrategy.LEAST_LOAD
        assert len(lb._nodes) == 0

    def test_update_nodes(self):
        """Test updating nodes."""
        lb = LoadBalancer()
        
        nodes = [
            NodeState(
                node_id="node-1",
                api_address="localhost:8000",
                raft_address="localhost:4321",
                health=NodeHealth.HEALTHY,
                max_sessions=10,
            ),
            NodeState(
                node_id="node-2",
                api_address="localhost:8001",
                raft_address="localhost:4322",
                health=NodeHealth.HEALTHY,
                max_sessions=10,
            ),
        ]
        
        lb.update_nodes(nodes)
        
        assert len(lb._nodes) == 2
        assert len(lb._node_scores) == 2

    def test_update_single_node(self):
        """Test updating a single node."""
        lb = LoadBalancer()
        
        node = NodeState(
            node_id="node-1",
            api_address="localhost:8000",
            raft_address="localhost:4321",
            health=NodeHealth.HEALTHY,
        )
        
        lb.update_node(node)
        
        assert "node-1" in lb._nodes

    def test_remove_node(self):
        """Test removing a node."""
        lb = LoadBalancer()
        
        node = NodeState(
            node_id="node-1",
            api_address="localhost:8000",
            raft_address="localhost:4321",
        )
        lb.update_node(node)
        
        lb.remove_node("node-1")
        
        assert "node-1" not in lb._nodes

    def test_select_node_for_session_empty(self):
        """Test selecting node when no nodes available."""
        lb = LoadBalancer()
        
        result = lb.select_node_for_session()
        
        assert result is None

    def test_select_node_for_session_least_load(self):
        """Test selecting node with least load strategy."""
        lb = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_LOAD)
        
        nodes = [
            NodeState(
                node_id="node-1",
                api_address="localhost:8000",
                raft_address="localhost:4321",
                health=NodeHealth.HEALTHY,
                active_sessions=5,
                max_sessions=10,
            ),
            NodeState(
                node_id="node-2",
                api_address="localhost:8001",
                raft_address="localhost:4322",
                health=NodeHealth.HEALTHY,
                active_sessions=2,
                max_sessions=10,
            ),
        ]
        lb.update_nodes(nodes)
        
        selected = lb.select_node_for_session()
        
        assert selected is not None
        # node-2 has lower load
        assert selected.node_id == "node-2"

    def test_select_node_for_session_round_robin(self):
        """Test selecting node with round robin strategy."""
        lb = LoadBalancer(strategy=LoadBalancingStrategy.ROUND_ROBIN)
        
        nodes = [
            NodeState(
                node_id="node-1",
                api_address="localhost:8000",
                raft_address="localhost:4321",
                health=NodeHealth.HEALTHY,
                max_sessions=10,
            ),
            NodeState(
                node_id="node-2",
                api_address="localhost:8001",
                raft_address="localhost:4322",
                health=NodeHealth.HEALTHY,
                max_sessions=10,
            ),
        ]
        lb.update_nodes(nodes)
        
        # Get first selection
        first = lb.select_node_for_session()
        # Get second selection (should be different)
        second = lb.select_node_for_session()
        
        assert first is not None
        assert second is not None
        # Round robin should alternate
        assert first.node_id != second.node_id

    def test_register_session(self):
        """Test registering a session."""
        lb = LoadBalancer()
        
        lb.register_session("sess-123", "node-1")
        
        assert lb._session_to_node["sess-123"] == "node-1"

    def test_unregister_session(self):
        """Test unregistering a session."""
        lb = LoadBalancer()
        
        lb.register_session("sess-123", "node-1")
        lb.unregister_session("sess-123")
        
        assert "sess-123" not in lb._session_to_node

    def test_get_node_for_session(self):
        """Test getting node for an existing session."""
        lb = LoadBalancer()
        
        node = NodeState(
            node_id="node-1",
            api_address="localhost:8000",
            raft_address="localhost:4321",
        )
        lb.update_node(node)
        lb.register_session("sess-123", "node-1")
        
        result = lb.get_node_for_session("sess-123")
        
        assert result is not None
        assert result.node_id == "node-1"

    def test_get_node_for_session_not_found(self):
        """Test getting node for non-existent session."""
        lb = LoadBalancer()
        
        result = lb.get_node_for_session("unknown")
        
        assert result is None


class TestLoadBalancerHealthPenalty:
    """Tests for health-based scoring."""

    def test_degraded_health_penalty(self):
        """Test degraded nodes get lower priority."""
        lb = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_LOAD)
        
        nodes = [
            NodeState(
                node_id="node-1",
                api_address="localhost:8000",
                raft_address="localhost:4321",
                health=NodeHealth.DEGRADED,
                active_sessions=1,
                max_sessions=10,
            ),
            NodeState(
                node_id="node-2",
                api_address="localhost:8001",
                raft_address="localhost:4322",
                health=NodeHealth.HEALTHY,
                active_sessions=3,
                max_sessions=10,
            ),
        ]
        lb.update_nodes(nodes)
        
        selected = lb.select_node_for_session()
        
        # Healthy node should be preferred despite higher load
        assert selected is not None
        assert selected.node_id == "node-2"

    def test_unhealthy_node_excluded(self):
        """Test unhealthy nodes are excluded."""
        lb = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_LOAD)
        
        nodes = [
            NodeState(
                node_id="node-1",
                api_address="localhost:8000",
                raft_address="localhost:4321",
                health=NodeHealth.UNHEALTHY,
                max_sessions=10,
            ),
            NodeState(
                node_id="node-2",
                api_address="localhost:8001",
                raft_address="localhost:4322",
                health=NodeHealth.HEALTHY,
                max_sessions=10,
            ),
        ]
        lb.update_nodes(nodes)
        
        selected = lb.select_node_for_session()
        
        assert selected is not None
        assert selected.node_id == "node-2"
