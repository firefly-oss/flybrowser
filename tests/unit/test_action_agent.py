# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ActionAgent."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flybrowser.agents.action_agent import (
    ActionAgent,
    ActionResult,
    ActionStep,
    ActionType,
)
from flybrowser.exceptions import ActionError


class TestActionType:
    """Tests for ActionType enum."""

    def test_all_action_types(self):
        """Test all action types exist."""
        assert ActionType.CLICK == "click"
        assert ActionType.TYPE == "type"
        assert ActionType.FILL == "fill"
        assert ActionType.SELECT == "select"
        assert ActionType.HOVER == "hover"
        assert ActionType.SCROLL == "scroll"
        assert ActionType.WAIT == "wait"
        assert ActionType.PRESS_KEY == "press_key"
        assert ActionType.CLEAR == "clear"
        assert ActionType.CHECK == "check"
        assert ActionType.UNCHECK == "uncheck"
        assert ActionType.SCREENSHOT == "screenshot"


class TestActionStep:
    """Tests for ActionStep dataclass."""

    def test_default_values(self):
        """Test default ActionStep values."""
        step = ActionStep(action_type=ActionType.CLICK)
        
        assert step.action_type == ActionType.CLICK
        assert step.target is None
        assert step.value is None
        assert step.options == {}

    def test_with_values(self):
        """Test ActionStep with values."""
        step = ActionStep(
            action_type=ActionType.TYPE,
            target="#input",
            value="hello",
            options={"delay": 100}
        )
        
        assert step.action_type == ActionType.TYPE
        assert step.target == "#input"
        assert step.value == "hello"
        assert step.options == {"delay": 100}


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_successful_result(self):
        """Test successful action result."""
        result = ActionResult(
            success=True,
            steps_completed=3,
            total_steps=3
        )
        
        assert result.success is True
        assert result.steps_completed == 3
        assert result.total_steps == 3
        assert result.error is None

    def test_failed_result(self):
        """Test failed action result."""
        result = ActionResult(
            success=False,
            steps_completed=1,
            total_steps=3,
            error="Step 2 failed"
        )
        
        assert result.success is False
        assert result.error == "Step 2 failed"


class TestActionAgentInit:
    """Tests for ActionAgent initialization."""

    def test_init_default_values(self):
        """Test default initialization values from performance config."""
        from flybrowser.core.performance import get_performance_config
        perf = get_performance_config()
        
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = ActionAgent(mock_page, mock_detector, mock_llm)
        
        assert agent.page is mock_page
        assert agent.detector is mock_detector
        assert agent.llm is mock_llm
        assert agent.max_retries == perf.max_retries
        assert agent.retry_delay == perf.retry_delay_seconds
        assert agent.action_timeout == perf.action_timeout_ms
        assert agent.pii_handler is None

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        mock_pii = MagicMock()
        
        agent = ActionAgent(
            mock_page,
            mock_detector,
            mock_llm,
            max_retries=5,
            retry_delay=2.0,
            action_timeout=60000,
            pii_handler=mock_pii
        )
        
        assert agent.max_retries == 5
        assert agent.retry_delay == 2.0
        assert agent.action_timeout == 60000
        assert agent.pii_handler is mock_pii


class TestActionAgentExecute:
    """Tests for ActionAgent.execute()."""

    @pytest.mark.asyncio
    async def test_execute_dry_run(self):
        """Test execute with dry_run=True returns plan without executing."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(return_value="https://example.com")
        mock_page.get_title = AsyncMock(return_value="Example")
        mock_page.get_html = AsyncMock(return_value="<html></html>")
        mock_page.screenshot = AsyncMock(return_value=b"screenshot")
        mock_page.evaluate = AsyncMock(return_value=[])
        
        mock_detector = MagicMock()
        
        mock_llm = MagicMock()
        mock_llm.generate_with_vision = AsyncMock(
            return_value=MagicMock(content=json.dumps({
                "actions": [
                    {"action_type": "click", "target": "button"}
                ]
            }))
        )
        
        agent = ActionAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.execute("Click the button", dry_run=True)
        
        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["steps_completed"] == 0
        assert len(result["plan"]) == 1

    @pytest.mark.asyncio
    async def test_execute_returns_error_on_failure(self):
        """Test execute returns error dict on failure instead of raising."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(side_effect=Exception("Page error"))
        
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = ActionAgent(mock_page, mock_detector, mock_llm)
        
        # ActionAgent now returns error dict instead of raising
        result = await agent.execute("Click the button")
        
        assert result["success"] is False
        assert "error" in result
        assert "Page error" in result["error"]


class TestActionAgentFillForm:
    """Tests for ActionAgent.fill_form()."""

    @pytest.mark.asyncio
    async def test_fill_form_success(self):
        """Test fill_form successfully fills fields."""
        mock_page = MagicMock()
        mock_page.page = MagicMock()
        mock_page.page.locator = MagicMock(return_value=MagicMock(
            clear=AsyncMock()
        ))
        
        mock_detector = MagicMock()
        mock_detector.find_element = AsyncMock(return_value={
            "selector": "#email",
            "selector_type": "css"
        })
        mock_detector.type_text = AsyncMock()
        
        mock_llm = MagicMock()
        
        agent = ActionAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.fill_form({
            "email": "test@example.com"
        })
        
        assert result["fields_filled"] == 1
        assert result["total_fields"] == 1

    @pytest.mark.asyncio
    async def test_fill_form_handles_errors(self):
        """Test fill_form handles field errors."""
        mock_page = MagicMock()
        mock_page.page = MagicMock()
        
        mock_detector = MagicMock()
        mock_detector.find_element = AsyncMock(
            side_effect=Exception("Element not found")
        )
        
        mock_llm = MagicMock()
        
        agent = ActionAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.fill_form({
            "email": "test@example.com"
        })
        
        assert result["success"] is False
        assert result["fields_filled"] == 0
        assert len(result["errors"]) == 1


class TestActionAgentClick:
    """Tests for ActionAgent.click()."""

    @pytest.mark.asyncio
    async def test_click_executes_action(self):
        """Test click calls execute with click instruction."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(return_value="https://example.com")
        mock_page.get_title = AsyncMock(return_value="Example")
        mock_page.get_html = AsyncMock(return_value="<html></html>")
        mock_page.screenshot = AsyncMock(return_value=b"screenshot")
        mock_page.evaluate = AsyncMock(return_value=[])
        
        mock_detector = MagicMock()
        mock_detector.find_element = AsyncMock(return_value={
            "selector": "button",
            "selector_type": "css"
        })
        mock_detector.click = AsyncMock()
        
        mock_llm = MagicMock()
        mock_llm.generate_with_vision = AsyncMock(
            return_value=MagicMock(content=json.dumps({
                "actions": [{"action_type": "click", "target": "button"}]
            }))
        )
        
        agent = ActionAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.click("the button")
        
        assert result["success"] is True


class TestActionAgentTypeText:
    """Tests for ActionAgent.type_text()."""

    @pytest.mark.asyncio
    async def test_type_text_executes_action(self):
        """Test type_text calls execute with type instruction."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(return_value="https://example.com")
        mock_page.get_title = AsyncMock(return_value="Example")
        mock_page.get_html = AsyncMock(return_value="<html></html>")
        mock_page.screenshot = AsyncMock(return_value=b"screenshot")
        mock_page.evaluate = AsyncMock(return_value=[])
        
        mock_detector = MagicMock()
        mock_detector.find_element = AsyncMock(return_value={
            "selector": "input",
            "selector_type": "css"
        })
        mock_detector.type_text = AsyncMock()
        
        mock_llm = MagicMock()
        mock_llm.generate_with_vision = AsyncMock(
            return_value=MagicMock(content=json.dumps({
                "actions": [{"action_type": "type", "target": "input", "value": "hello"}]
            }))
        )
        
        agent = ActionAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.type_text("search box", "hello")
        
        assert result["success"] is True


class TestActionAgentSecureFillForm:
    """Tests for ActionAgent.secure_fill_form()."""

    @pytest.mark.asyncio
    async def test_secure_fill_form_requires_pii_handler(self):
        """Test secure_fill_form raises without PIIHandler."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = ActionAgent(mock_page, mock_detector, mock_llm)
        
        with pytest.raises(ValueError, match="PIIHandler is required"):
            await agent.secure_fill_form({"email": "cred-123"})

    @pytest.mark.asyncio
    async def test_secure_fill_form_with_pii_handler(self):
        """Test secure_fill_form works with PIIHandler."""
        mock_page = MagicMock()
        mock_page.page = MagicMock()
        mock_page.page.locator = MagicMock(return_value=MagicMock(
            clear=AsyncMock()
        ))
        
        mock_detector = MagicMock()
        mock_detector.find_element = AsyncMock(return_value={
            "selector": "#email",
            "selector_type": "css"
        })
        
        mock_llm = MagicMock()
        
        mock_pii = MagicMock()
        mock_pii.secure_fill = AsyncMock(return_value=True)
        
        agent = ActionAgent(
            mock_page, mock_detector, mock_llm,
            pii_handler=mock_pii
        )
        
        result = await agent.secure_fill_form({"email": "cred-123"})
        
        assert result["fields_filled"] == 1


class TestActionAgentHelpers:
    """Tests for ActionAgent helper methods."""

    def test_step_to_dict(self):
        """Test _step_to_dict converts ActionStep to dict."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = ActionAgent(mock_page, mock_detector, mock_llm)
        
        step = ActionStep(
            action_type=ActionType.CLICK,
            target="button",
            value=None,
            options={"force": True}
        )
        
        result = agent._step_to_dict(step)
        
        assert result["action_type"] == "click"
        assert result["target"] == "button"
        assert result["value"] is None
        assert result["options"] == {"force": True}
