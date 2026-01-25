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
Agent Orchestrator with safety mechanisms and approval workflows.

This module provides the AgentOrchestrator which coordinates ReAct agents
with comprehensive safety features:
- Multi-layered circuit breaker for preventing runaway execution
- Approval workflows for dangerous/sensitive actions
- Multiple execution modes (AUTONOMOUS, SUPERVISED, INTERACTIVE)
- Progress tracking and stagnation detection

Example:
    >>> orchestrator = AgentOrchestrator(
    ...     react_agent=agent,
    ...     execution_mode=ExecutionMode.SUPERVISED,
    ... )
    >>> result = await orchestrator.execute("Search for Python tutorials")
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from flybrowser.agents.types import (
    Action,
    ExecutionState,
    Observation,
    ReActStep,
    SafetyLevel,
    ToolResult,
)
from flybrowser.utils.logger import logger


class ExecutionMode(str, Enum):
    """Mode for orchestrator execution."""

    AUTONOMOUS = "autonomous"      # Full autonomous execution
    SUPERVISED = "supervised"      # Pause for dangerous actions
    INTERACTIVE = "interactive"    # Pause for all actions


class StopReason(str, Enum):
    """Reasons for stopping execution."""

    COMPLETED = "completed"            # Task completed successfully
    MAX_ITERATIONS = "max_iterations"  # Hit iteration limit
    MAX_TIME = "max_time"              # Hit time limit
    MAX_LLM_CALLS = "max_llm_calls"    # Hit LLM call limit
    STAGNATION = "stagnation"          # No progress detected
    USER_CANCELLED = "user_cancelled"  # User cancelled execution
    CIRCUIT_BREAKER = "circuit_breaker"  # Circuit breaker tripped
    APPROVAL_DENIED = "approval_denied"  # Human denied action
    ERROR = "error"                    # Unrecoverable error


@dataclass
class SafetyConfig:
    """Configuration for safety mechanisms."""

    max_iterations: int = 50
    max_time_seconds: float = 300.0
    max_llm_calls: int = 100
    max_consecutive_failures: int = 5
    stagnation_window: int = 5        # Steps to check for stagnation
    min_progress_rate: float = 0.1    # Minimum progress per window
    require_approval_for: List[SafetyLevel] = field(
        default_factory=lambda: [SafetyLevel.DANGEROUS]
    )
    approval_timeout_seconds: float = 300.0


@dataclass
class ProgressTracker:
    """Tracks execution progress for stagnation detection."""

    recent_actions: List[str] = field(default_factory=list)
    action_hashes: Set[str] = field(default_factory=set)
    successful_steps: int = 0
    failed_steps: int = 0
    repeated_actions: int = 0
    last_meaningful_progress: float = field(default_factory=time.time)

    def record_action(self, action: Action, success: bool) -> None:
        """Record an action for progress tracking."""
        import hashlib
        action_hash = hashlib.md5(
            f"{action.tool_name}:{action.parameters}".encode()
        ).hexdigest()[:8]

        if action_hash in self.action_hashes:
            self.repeated_actions += 1
        else:
            self.action_hashes.add(action_hash)
            self.last_meaningful_progress = time.time()

        self.recent_actions.append(action_hash)
        if len(self.recent_actions) > 20:
            self.recent_actions.pop(0)

        if success:
            self.successful_steps += 1
        else:
            self.failed_steps += 1

    def is_stagnating(self, window: int = 5) -> bool:
        """Check if execution is stagnating."""
        if len(self.recent_actions) < window:
            return False
        recent = self.recent_actions[-window:]
        unique = len(set(recent))
        return unique < window * 0.5  # Less than 50% unique actions


@dataclass
class CircuitBreakerState:
    """State tracking for the circuit breaker."""

    iterations: int = 0
    llm_calls: int = 0
    consecutive_failures: int = 0
    start_time: float = field(default_factory=time.time)
    stop_reason: Optional[StopReason] = None
    is_open: bool = False  # True = tripped, blocking execution


# ApprovalCallback type - async function that returns (approved, reason)
ApprovalCallback = Callable[[Action, str], Awaitable[tuple[bool, str]]]


class CircuitBreaker:
    """
    Multi-layered safety mechanism for autonomous execution.

    Implements protection layers:
    - Layer 1: Hard limits (iterations, time, LLM calls)
    - Layer 2: Progress detection (stagnation, repeated actions)
    - Layer 3: Consecutive failure tracking
    """


    def __init__(self, config: SafetyConfig) -> None:
        """Initialize circuit breaker with safety configuration."""
        self.config = config
        self.state = CircuitBreakerState()
        self.progress = ProgressTracker()

    def start(self) -> None:
        """Start/reset the circuit breaker for a new execution."""
        self.state = CircuitBreakerState()
        self.progress = ProgressTracker()

    def should_stop(self) -> bool:
        """Check if execution should stop based on safety limits."""
        if self.state.is_open:
            return True

        # Layer 1: Hard limits
        if self.state.iterations >= self.config.max_iterations:
            self._trip(StopReason.MAX_ITERATIONS)
            return True

        elapsed = time.time() - self.state.start_time
        if elapsed >= self.config.max_time_seconds:
            self._trip(StopReason.MAX_TIME)
            return True

        if self.state.llm_calls >= self.config.max_llm_calls:
            self._trip(StopReason.MAX_LLM_CALLS)
            return True

        # Layer 2: Progress detection
        if self.progress.is_stagnating(self.config.stagnation_window):
            self._trip(StopReason.STAGNATION)
            return True

        # Layer 3: Consecutive failures
        if self.state.consecutive_failures >= self.config.max_consecutive_failures:
            self._trip(StopReason.CIRCUIT_BREAKER)
            return True

        return False

    def _trip(self, reason: StopReason) -> None:
        """Trip the circuit breaker."""
        self.state.is_open = True
        self.state.stop_reason = reason
        logger.warning(f"Circuit breaker tripped: {reason.value}")

    def record_action(self, action: Action, success: bool) -> None:
        """Record an action execution."""
        self.state.iterations += 1
        self.progress.record_action(action, success)

        if success:
            self.state.consecutive_failures = 0
        else:
            self.state.consecutive_failures += 1

    def record_llm_call(self, tokens: int = 0) -> None:
        """Record an LLM call."""
        self.state.llm_calls += 1

    def get_stop_reason(self) -> Optional[StopReason]:
        """Get the reason for stopping, if any."""
        return self.state.stop_reason

    def get_report(self) -> Dict[str, Any]:
        """Get a report of the circuit breaker state."""
        elapsed = time.time() - self.state.start_time
        return {
            "iterations": self.state.iterations,
            "llm_calls": self.state.llm_calls,
            "elapsed_seconds": round(elapsed, 2),
            "consecutive_failures": self.state.consecutive_failures,
            "successful_steps": self.progress.successful_steps,
            "failed_steps": self.progress.failed_steps,
            "repeated_actions": self.progress.repeated_actions,
            "is_open": self.state.is_open,
            "stop_reason": self.state.stop_reason.value if self.state.stop_reason else None,
        }



class AgentOrchestrator:
    """
    Orchestrates ReAct agent execution with safety mechanisms.

    Provides:
    - Multiple execution modes (AUTONOMOUS, SUPERVISED, INTERACTIVE)
    - Approval workflows for dangerous actions
    - Circuit breaker for preventing runaway execution
    - Progress tracking and reporting

    Example:
        >>> orchestrator = AgentOrchestrator(
        ...     react_agent=agent,
        ...     execution_mode=ExecutionMode.SUPERVISED,
        ... )
        >>> result = await orchestrator.execute("Navigate to example.com")
    """

    def __init__(
        self,
        react_agent: Any,  # ReActAgent - avoid circular import
        execution_mode: ExecutionMode = ExecutionMode.AUTONOMOUS,
        safety_config: Optional[SafetyConfig] = None,
        approval_callback: Optional[ApprovalCallback] = None,
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            react_agent: The ReActAgent to orchestrate
            execution_mode: How to handle human-in-the-loop
            safety_config: Configuration for safety mechanisms
            approval_callback: Async callback for approval requests
        """
        self.react_agent = react_agent
        self.execution_mode = execution_mode
        self.safety_config = safety_config or SafetyConfig()
        self.approval_callback = approval_callback

        self.circuit_breaker = CircuitBreaker(self.safety_config)
        self._state = ExecutionState.IDLE
        self._current_task: Optional[str] = None
        self._steps: List[ReActStep] = []
        self._cancelled = False

    @property
    def state(self) -> ExecutionState:
        """Get current execution state."""
        return self._state

    async def execute(self, task: str) -> Dict[str, Any]:
        """
        Execute a task with safety mechanisms.

        Args:
            task: The task to execute

        Returns:
            Execution result with steps, status, and circuit breaker report
        """
        self._current_task = task
        self._steps = []
        self._cancelled = False
        self._state = ExecutionState.THINKING

        # Start circuit breaker
        self.circuit_breaker.start()

        logger.info(f"Starting orchestrated execution: {task[:100]}...")

        try:
            # Execute using the ReAct agent with step callback
            result = await self.react_agent.execute(
                task=task,
                step_callback=self._handle_step,
            )

            # Check final state
            if self._cancelled:
                self._state = ExecutionState.CANCELLED
                stop_reason = StopReason.USER_CANCELLED
            elif result.get("success", False):
                self._state = ExecutionState.COMPLETED
                stop_reason = StopReason.COMPLETED
            else:
                self._state = ExecutionState.FAILED
                stop_reason = self.circuit_breaker.get_stop_reason() or StopReason.ERROR

            return {
                "success": result.get("success", False),
                "result": result.get("result"),
                "steps": self._steps,
                "stop_reason": stop_reason.value,
                "circuit_breaker_report": self.circuit_breaker.get_report(),
                "execution_mode": self.execution_mode.value,
            }

        except Exception as e:
            logger.error(f"Orchestrator execution failed: {e}")
            self._state = ExecutionState.FAILED
            return {
                "success": False,
                "error": str(e),
                "steps": self._steps,
                "stop_reason": StopReason.ERROR.value,
                "circuit_breaker_report": self.circuit_breaker.get_report(),
                "execution_mode": self.execution_mode.value,
            }

    async def _handle_step(self, step: ReActStep) -> Optional[bool]:
        """
        Handle a step from the ReAct agent.

        Args:
            step: The ReAct step to process

        Returns:
            None to continue, False to stop execution
        """
        self._steps.append(step)

        # Record LLM call
        self.circuit_breaker.record_llm_call()

        # Check circuit breaker
        if self.circuit_breaker.should_stop():
            logger.warning(f"Circuit breaker stopping execution")
            return False

        # Check if cancelled
        if self._cancelled:
            return False

        # Process action if present
        if step.action:
            action = step.action

            # Check if approval is needed
            if self._should_require_approval(action):
                self._state = ExecutionState.WAITING_APPROVAL
                approved, reason = await self._request_approval(action)

                if not approved:
                    logger.info(f"Action denied: {reason}")
                    self.circuit_breaker._trip(StopReason.APPROVAL_DENIED)
                    return False

                self._state = ExecutionState.ACTING

            # Record the action (success determined by observation)
            success = step.observation and step.observation.success
            self.circuit_breaker.record_action(action, success)

        return None  # Continue execution

    def _should_require_approval(self, action: Action) -> bool:
        """
        Determine if an action requires human approval.

        Args:
            action: The action to check

        Returns:
            True if approval is required
        """
        # Interactive mode: always require approval
        if self.execution_mode == ExecutionMode.INTERACTIVE:
            return True

        # Autonomous mode: never require approval
        if self.execution_mode == ExecutionMode.AUTONOMOUS:
            return False

        # Supervised mode: check safety level
        return action.safety_level in self.safety_config.require_approval_for

    async def _request_approval(self, action: Action) -> tuple[bool, str]:
        """
        Request approval for an action.

        Args:
            action: The action requiring approval

        Returns:
            Tuple of (approved, reason)
        """
        if not self.approval_callback:
            # No callback = auto-approve in supervised mode, deny in interactive
            if self.execution_mode == ExecutionMode.SUPERVISED:
                logger.warning(
                    f"No approval callback, auto-approving: {action.tool_name}"
                )
                return True, "auto-approved (no callback)"
            else:
                return False, "no approval callback configured"

        try:
            # Create description for approval request
            description = (
                f"Tool: {action.tool_name}\n"
                f"Safety Level: {action.safety_level.value}\n"
                f"Parameters: {action.parameters}\n"
                f"Reason: {action.reasoning or 'Not provided'}"
            )

            # Request approval with timeout
            approved, reason = await asyncio.wait_for(
                self.approval_callback(action, description),
                timeout=self.safety_config.approval_timeout_seconds,
            )

            return approved, reason

        except asyncio.TimeoutError:
            logger.warning("Approval request timed out")
            return False, "approval timeout"
        except Exception as e:
            logger.error(f"Approval request failed: {e}")
            return False, f"approval error: {e}"

    def cancel(self) -> None:
        """Cancel the current execution."""
        self._cancelled = True
        logger.info("Execution cancelled by user")

    def pause(self) -> None:
        """Pause execution (agent will pause at next step)."""
        self._state = ExecutionState.PAUSED

    def resume(self) -> None:
        """Resume paused execution."""
        if self._state == ExecutionState.PAUSED:
            self._state = ExecutionState.THINKING
