# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MonitoringAgent."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from flybrowser.agents.monitoring_agent import (
    ChangeEvent,
    ChangeType,
    ComparisonOperator,
    MonitoringAgent,
    MonitoringCondition,
    MonitoringError,
    MonitoringSession,
)


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_all_change_types(self):
        """Test all change types exist."""
        assert ChangeType.CONTENT == "content"
        assert ChangeType.ELEMENT == "element"
        assert ChangeType.VALUE == "value"
        assert ChangeType.PRESENCE == "presence"
        assert ChangeType.ABSENCE == "absence"


class TestComparisonOperator:
    """Tests for ComparisonOperator enum."""

    def test_all_operators(self):
        """Test all comparison operators exist."""
        assert ComparisonOperator.EQUALS == "equals"
        assert ComparisonOperator.NOT_EQUALS == "not_equals"
        assert ComparisonOperator.GREATER_THAN == "greater_than"
        assert ComparisonOperator.LESS_THAN == "less_than"
        assert ComparisonOperator.GREATER_OR_EQUAL == "greater_or_equal"
        assert ComparisonOperator.LESS_OR_EQUAL == "less_or_equal"
        assert ComparisonOperator.CONTAINS == "contains"
        assert ComparisonOperator.NOT_CONTAINS == "not_contains"


class TestMonitoringCondition:
    """Tests for MonitoringCondition dataclass."""

    def test_default_values(self):
        """Test default MonitoringCondition values."""
        condition = MonitoringCondition(description="Monitor price")
        
        assert condition.description == "Monitor price"
        assert condition.change_type == ChangeType.CONTENT
        assert condition.operator is None
        assert condition.threshold is None
        assert condition.element_selector is None

    def test_with_threshold(self):
        """Test MonitoringCondition with threshold."""
        condition = MonitoringCondition(
            description="Price drops below $50",
            change_type=ChangeType.VALUE,
            operator=ComparisonOperator.LESS_THAN,
            threshold=50.0,
            element_selector="#price"
        )
        
        assert condition.change_type == ChangeType.VALUE
        assert condition.operator == ComparisonOperator.LESS_THAN
        assert condition.threshold == 50.0


class TestChangeEvent:
    """Tests for ChangeEvent dataclass."""

    def test_change_event_creation(self):
        """Test ChangeEvent creation."""
        event = ChangeEvent(
            timestamp=datetime.now(),
            change_type=ChangeType.VALUE,
            description="Price changed",
            old_value="$100",
            new_value="$90"
        )
        
        assert event.change_type == ChangeType.VALUE
        assert event.old_value == "$100"
        assert event.new_value == "$90"


class TestMonitoringSession:
    """Tests for MonitoringSession dataclass."""

    def test_session_creation(self):
        """Test MonitoringSession creation."""
        conditions = [
            MonitoringCondition(description="Monitor content")
        ]
        
        session = MonitoringSession(
            session_id="sess-123",
            conditions=conditions,
            start_time=datetime.now(),
            poll_interval=10.0,
            max_duration=300.0
        )
        
        assert session.session_id == "sess-123"
        assert len(session.conditions) == 1
        assert session.poll_interval == 10.0
        assert session.is_active is True
        assert len(session.changes_detected) == 0


class TestMonitoringAgentInit:
    """Tests for MonitoringAgent initialization."""

    def test_init_default_values(self):
        """Test default initialization values."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = MonitoringAgent(mock_page, mock_detector, mock_llm)
        
        assert agent.page is mock_page
        assert agent.detector is mock_detector
        assert agent.llm is mock_llm
        assert agent.default_poll_interval == 5.0
        assert agent.default_max_duration == 3600.0
        assert len(agent._active_sessions) == 0

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = MonitoringAgent(
            mock_page,
            mock_detector,
            mock_llm,
            default_poll_interval=10.0,
            default_max_duration=1800.0
        )
        
        assert agent.default_poll_interval == 10.0
        assert agent.default_max_duration == 1800.0


class TestMonitoringAgentCheckThreshold:
    """Tests for MonitoringAgent._check_threshold()."""

    def test_less_than_numeric(self):
        """Test less than comparison with numeric value."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = MonitoringAgent(mock_page, mock_detector, mock_llm)
        
        condition = MonitoringCondition(
            description="Price below 50",
            change_type=ChangeType.VALUE,
            operator=ComparisonOperator.LESS_THAN,
            threshold=50
        )
        
        assert agent._check_threshold("$45.00", condition) is True
        assert agent._check_threshold("$55.00", condition) is False

    def test_greater_than_numeric(self):
        """Test greater than comparison."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = MonitoringAgent(mock_page, mock_detector, mock_llm)
        
        condition = MonitoringCondition(
            description="Value above 100",
            change_type=ChangeType.VALUE,
            operator=ComparisonOperator.GREATER_THAN,
            threshold=100
        )
        
        assert agent._check_threshold("150", condition) is True
        assert agent._check_threshold("50", condition) is False

    def test_contains_string(self):
        """Test contains comparison for strings."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = MonitoringAgent(mock_page, mock_detector, mock_llm)
        
        condition = MonitoringCondition(
            description="Text contains 'success'",
            change_type=ChangeType.VALUE,
            operator=ComparisonOperator.CONTAINS,
            threshold="success"
        )
        
        assert agent._check_threshold("Operation success!", condition) is True
        assert agent._check_threshold("Operation failed", condition) is False

    def test_no_operator_returns_true(self):
        """Test returns True when no operator specified."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = MonitoringAgent(mock_page, mock_detector, mock_llm)
        
        condition = MonitoringCondition(
            description="Any change",
            change_type=ChangeType.VALUE
        )
        
        assert agent._check_threshold("any value", condition) is True


class TestMonitoringAgentChangeToDict:
    """Tests for MonitoringAgent._change_to_dict()."""

    def test_change_to_dict(self):
        """Test converting ChangeEvent to dictionary."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = MonitoringAgent(mock_page, mock_detector, mock_llm)
        
        event = ChangeEvent(
            timestamp=datetime(2026, 1, 15, 10, 30, 0),
            change_type=ChangeType.VALUE,
            description="Price changed",
            old_value="$100",
            new_value="$90"
        )
        
        result = agent._change_to_dict(event)
        
        assert result["change_type"] == "value"
        assert result["description"] == "Price changed"
        assert result["old_value"] == "$100"
        assert result["new_value"] == "$90"
        assert "timestamp" in result


class TestMonitoringAgentStopMonitoring:
    """Tests for MonitoringAgent.stop_monitoring()."""

    @pytest.mark.asyncio
    async def test_stop_monitoring_active_session(self):
        """Test stopping an active monitoring session."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = MonitoringAgent(mock_page, mock_detector, mock_llm)
        
        # Create a mock session
        session = MonitoringSession(
            session_id="sess-123",
            conditions=[],
            start_time=datetime.now(),
            is_active=True
        )
        agent._active_sessions["sess-123"] = session
        
        result = await agent.stop_monitoring("sess-123")
        
        assert result is True
        assert session.is_active is False

    @pytest.mark.asyncio
    async def test_stop_monitoring_nonexistent_session(self):
        """Test stopping non-existent session returns False."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = MonitoringAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.stop_monitoring("nonexistent")
        
        assert result is False


class TestMonitoringAgentGetActiveSessions:
    """Tests for MonitoringAgent.get_active_sessions()."""

    def test_get_active_sessions(self):
        """Test getting list of active session IDs."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = MonitoringAgent(mock_page, mock_detector, mock_llm)
        
        # Add sessions
        active_session = MonitoringSession(
            session_id="active-1",
            conditions=[],
            start_time=datetime.now(),
            is_active=True
        )
        inactive_session = MonitoringSession(
            session_id="inactive-1",
            conditions=[],
            start_time=datetime.now(),
            is_active=False
        )
        
        agent._active_sessions["active-1"] = active_session
        agent._active_sessions["inactive-1"] = inactive_session
        
        result = agent.get_active_sessions()
        
        assert "active-1" in result
        assert "inactive-1" not in result
        assert len(result) == 1


class TestMonitoringAgentGetContentHash:
    """Tests for MonitoringAgent._get_content_hash()."""

    @pytest.mark.asyncio
    async def test_get_content_hash(self):
        """Test content hash generation."""
        mock_page = MagicMock()
        mock_page.get_html = AsyncMock(return_value="<html><body>Test</body></html>")
        
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = MonitoringAgent(mock_page, mock_detector, mock_llm)
        
        hash1 = await agent._get_content_hash()
        hash2 = await agent._get_content_hash()
        
        # Same content should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length

    @pytest.mark.asyncio
    async def test_different_content_different_hash(self):
        """Test different content produces different hash."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = MonitoringAgent(mock_page, mock_detector, mock_llm)
        
        mock_page.get_html = AsyncMock(return_value="<html><body>Content 1</body></html>")
        hash1 = await agent._get_content_hash()
        
        mock_page.get_html = AsyncMock(return_value="<html><body>Content 2</body></html>")
        hash2 = await agent._get_content_hash()
        
        assert hash1 != hash2


class TestMonitoringAgentExecute:
    """Tests for MonitoringAgent.execute()."""

    @pytest.mark.asyncio
    async def test_execute_raises_on_error(self):
        """Test execute raises MonitoringError on failure."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(side_effect=Exception("Page error"))
        
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = MonitoringAgent(mock_page, mock_detector, mock_llm)
        
        with pytest.raises(MonitoringError, match="Failed to execute monitoring"):
            await agent.execute("Monitor changes", max_duration=0.1)
