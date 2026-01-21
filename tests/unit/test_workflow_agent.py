# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for WorkflowAgent."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from flybrowser.agents.workflow_agent import (
    StepStatus,
    StepType,
    Workflow,
    WorkflowAgent,
    WorkflowError,
    WorkflowResult,
    WorkflowStep,
)


class TestStepType:
    """Tests for StepType enum."""

    def test_all_step_types(self):
        """Test all step types exist."""
        assert StepType.NAVIGATE == "navigate"
        assert StepType.ACTION == "action"
        assert StepType.EXTRACT == "extract"
        assert StepType.WAIT == "wait"
        assert StepType.CONDITION == "condition"
        assert StepType.LOOP == "loop"
        assert StepType.ASSERT == "assert"
        assert StepType.STORE == "store"


class TestStepStatus:
    """Tests for StepStatus enum."""

    def test_all_step_statuses(self):
        """Test all step statuses exist."""
        assert StepStatus.PENDING == "pending"
        assert StepStatus.RUNNING == "running"
        assert StepStatus.COMPLETED == "completed"
        assert StepStatus.FAILED == "failed"
        assert StepStatus.SKIPPED == "skipped"


class TestWorkflowStep:
    """Tests for WorkflowStep dataclass."""

    def test_default_values(self):
        """Test default WorkflowStep values."""
        step = WorkflowStep(
            step_id="step1",
            step_type=StepType.ACTION,
            instruction="Click button"
        )
        
        assert step.step_id == "step1"
        assert step.step_type == StepType.ACTION
        assert step.instruction == "Click button"
        assert step.condition is None
        assert step.loop_count is None
        assert step.on_failure == "stop"
        assert step.max_retries == 3
        assert step.timeout == 30.0
        assert step.status == StepStatus.PENDING

    def test_with_custom_values(self):
        """Test WorkflowStep with custom values."""
        step = WorkflowStep(
            step_id="step1",
            step_type=StepType.NAVIGATE,
            instruction="Go to page",
            on_failure="continue",
            max_retries=5,
            store_as="result"
        )
        
        assert step.on_failure == "continue"
        assert step.max_retries == 5
        assert step.store_as == "result"


class TestWorkflowResult:
    """Tests for WorkflowResult dataclass."""

    def test_successful_result(self):
        """Test successful workflow result."""
        result = WorkflowResult(
            success=True,
            steps_completed=3,
            total_steps=3,
            duration=5.5
        )
        
        assert result.success is True
        assert result.steps_completed == 3
        assert result.total_steps == 3
        assert result.error is None

    def test_failed_result(self):
        """Test failed workflow result."""
        result = WorkflowResult(
            success=False,
            steps_completed=1,
            total_steps=3,
            error="Step 2 failed"
        )
        
        assert result.success is False
        assert result.error == "Step 2 failed"


class TestWorkflow:
    """Tests for Workflow dataclass."""

    def test_workflow_creation(self):
        """Test workflow creation."""
        steps = [
            WorkflowStep(step_id="s1", step_type=StepType.NAVIGATE, instruction="Go to page"),
            WorkflowStep(step_id="s2", step_type=StepType.ACTION, instruction="Click button")
        ]
        
        workflow = Workflow(
            workflow_id="wf1",
            name="Test Workflow",
            description="A test workflow",
            steps=steps
        )
        
        assert workflow.workflow_id == "wf1"
        assert workflow.name == "Test Workflow"
        assert len(workflow.steps) == 2
        assert isinstance(workflow.created_at, datetime)


class TestWorkflowAgentInit:
    """Tests for WorkflowAgent initialization."""

    def test_init_default_values(self):
        """Test default initialization values."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = WorkflowAgent(mock_page, mock_detector, mock_llm)
        
        assert agent.page is mock_page
        assert agent.detector is mock_detector
        assert agent.llm is mock_llm
        assert agent._action_agent is None
        assert agent._navigation_agent is None
        assert agent._extraction_agent is None

    def test_init_with_sub_agents(self):
        """Test initialization with sub-agents."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        mock_action_agent = MagicMock()
        mock_nav_agent = MagicMock()
        mock_extract_agent = MagicMock()
        
        agent = WorkflowAgent(
            mock_page,
            mock_detector,
            mock_llm,
            action_agent=mock_action_agent,
            navigation_agent=mock_nav_agent,
            extraction_agent=mock_extract_agent
        )
        
        assert agent._action_agent is mock_action_agent
        assert agent._navigation_agent is mock_nav_agent
        assert agent._extraction_agent is mock_extract_agent


class TestWorkflowAgentRunWorkflow:
    """Tests for WorkflowAgent.run_workflow()."""

    @pytest.mark.asyncio
    async def test_run_workflow_successful(self):
        """Test successful workflow execution."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(return_value="https://example.com")
        mock_page.get_title = AsyncMock(return_value="Example")
        mock_page.get_html = AsyncMock(return_value="<html></html>")
        mock_page.goto = AsyncMock()
        
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = WorkflowAgent(mock_page, mock_detector, mock_llm)
        
        workflow = Workflow(
            workflow_id="wf1",
            name="Test",
            description="Test workflow",
            steps=[
                WorkflowStep(
                    step_id="s1",
                    step_type=StepType.NAVIGATE,
                    instruction="https://example.com"
                )
            ]
        )
        
        result = await agent.run_workflow(workflow)
        
        assert result.steps_completed == 1
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_workflow_step_failure_stops(self):
        """Test workflow stops on step failure with on_failure='stop'."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(side_effect=Exception("Error"))
        
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = WorkflowAgent(mock_page, mock_detector, mock_llm)
        
        workflow = Workflow(
            workflow_id="wf1",
            name="Test",
            description="Test workflow",
            steps=[
                WorkflowStep(
                    step_id="s1",
                    step_type=StepType.NAVIGATE,
                    instruction="Go to page",
                    on_failure="stop"
                ),
                WorkflowStep(
                    step_id="s2",
                    step_type=StepType.ACTION,
                    instruction="Click button"
                )
            ]
        )
        
        result = await agent.run_workflow(workflow)
        
        assert result.success is False
        assert result.steps_completed == 0
        assert result.error is not None


class TestWorkflowAgentExecute:
    """Tests for WorkflowAgent.execute()."""

    @pytest.mark.asyncio
    async def test_execute_plans_and_runs_workflow(self):
        """Test execute plans and runs workflow."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(return_value="https://example.com")
        mock_page.get_title = AsyncMock(return_value="Example")
        mock_page.get_html = AsyncMock(return_value="<html></html>")
        mock_page.goto = AsyncMock()
        
        mock_detector = MagicMock()
        
        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock(return_value={
            "name": "Test Workflow",
            "description": "Test",
            "steps": [
                {"step_type": "navigate", "instruction": "https://example.com"}
            ]
        })
        
        agent = WorkflowAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.execute("Navigate to example.com")
        
        assert result["success"] is True
        assert result["total_steps"] == 1

    @pytest.mark.asyncio
    async def test_execute_raises_workflow_error(self):
        """Test execute raises WorkflowError on failure."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(side_effect=Exception("Error"))
        
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = WorkflowAgent(mock_page, mock_detector, mock_llm)
        
        with pytest.raises(WorkflowError, match="Failed to execute workflow"):
            await agent.execute("Do something")


class TestWorkflowAgentSubstituteVariables:
    """Tests for WorkflowAgent._substitute_variables()."""

    def test_substitute_variables(self):
        """Test variable substitution."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = WorkflowAgent(mock_page, mock_detector, mock_llm)
        
        text = "Hello {{name}}, your email is {{email}}"
        variables = {"name": "John", "email": "john@example.com"}
        
        result = agent._substitute_variables(text, variables)
        
        assert result == "Hello John, your email is john@example.com"

    def test_substitute_variables_no_match(self):
        """Test variable substitution with no matches."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = WorkflowAgent(mock_page, mock_detector, mock_llm)
        
        text = "Hello world"
        variables = {"name": "John"}
        
        result = agent._substitute_variables(text, variables)
        
        assert result == "Hello world"


class TestWorkflowAgentTemplates:
    """Tests for WorkflowAgent template management."""

    def test_register_and_get_template(self):
        """Test registering and retrieving templates."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = WorkflowAgent(mock_page, mock_detector, mock_llm)
        
        workflow = Workflow(
            workflow_id="login-flow",
            name="Login Flow",
            description="Standard login workflow",
            steps=[]
        )
        
        agent.register_template(workflow)
        
        retrieved = agent.get_template("login-flow")
        
        assert retrieved is workflow
        assert retrieved.name == "Login Flow"

    def test_get_nonexistent_template(self):
        """Test getting non-existent template returns None."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = WorkflowAgent(mock_page, mock_detector, mock_llm)
        
        result = agent.get_template("nonexistent")
        
        assert result is None


class TestWorkflowAgentStepExecution:
    """Tests for WorkflowAgent step execution methods."""

    @pytest.mark.asyncio
    async def test_execute_wait_step(self):
        """Test wait step execution."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = WorkflowAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent._execute_wait("Wait 1 second")
        
        assert result["waited"] == 1

    @pytest.mark.asyncio
    async def test_execute_wait_step_with_seconds(self):
        """Test wait step with specific seconds."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = WorkflowAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent._execute_wait("Wait for 2 seconds")
        
        assert result["waited"] == 2

    @pytest.mark.asyncio
    async def test_execute_navigate_with_url(self):
        """Test navigate step with URL."""
        mock_page = MagicMock()
        mock_page.goto = AsyncMock()
        
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = WorkflowAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent._execute_navigate("https://example.com")
        
        assert result["navigated"] is True
        mock_page.goto.assert_awaited_once_with("https://example.com")
