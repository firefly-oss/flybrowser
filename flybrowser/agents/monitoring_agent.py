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
Monitoring agent for tracking page changes and data updates.

This module provides the MonitoringAgent class which uses LLMs to monitor
web pages for changes, track specific data points, and trigger callbacks
when conditions are met.

The agent supports:
- Page content change detection
- Specific element monitoring
- Data value tracking with thresholds
- Periodic polling with configurable intervals
- Callback notifications on changes

Use Cases:
- Price monitoring on e-commerce sites
- Stock availability tracking
- Content update detection
- Form submission status monitoring
- API response monitoring

Example:
    >>> agent = MonitoringAgent(page_controller, element_detector, llm_provider)
    >>> result = await agent.execute("Monitor the price and alert if it drops below $50")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from flybrowser.agents.base_agent import BaseAgent
from flybrowser.exceptions import FlyBrowserError
from flybrowser.utils.logger import logger


class MonitoringError(FlyBrowserError):
    """Exception raised when monitoring fails."""
    pass


class ChangeType(str, Enum):
    """Types of changes that can be detected."""

    CONTENT = "content"
    ELEMENT = "element"
    VALUE = "value"
    PRESENCE = "presence"
    ABSENCE = "absence"


class ComparisonOperator(str, Enum):
    """Operators for value comparisons."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_OR_EQUAL = "greater_or_equal"
    LESS_OR_EQUAL = "less_or_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"


@dataclass
class MonitoringCondition:
    """
    Defines a condition to monitor.

    Attributes:
        description: Natural language description of what to monitor
        change_type: Type of change to detect
        operator: Comparison operator for value monitoring
        threshold: Threshold value for comparisons
        element_selector: Optional CSS selector for element monitoring
    """

    description: str
    change_type: ChangeType = ChangeType.CONTENT
    operator: Optional[ComparisonOperator] = None
    threshold: Optional[Union[str, int, float]] = None
    element_selector: Optional[str] = None


@dataclass
class ChangeEvent:
    """
    Represents a detected change.

    Attributes:
        timestamp: When the change was detected
        change_type: Type of change detected
        description: Description of the change
        old_value: Previous value (if applicable)
        new_value: New value (if applicable)
        details: Additional details about the change
    """

    timestamp: datetime
    change_type: ChangeType
    description: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringSession:
    """
    Represents an active monitoring session.

    Attributes:
        session_id: Unique identifier for the session
        conditions: List of conditions being monitored
        start_time: When monitoring started
        poll_interval: Interval between checks in seconds
        max_duration: Maximum monitoring duration in seconds
        changes_detected: List of detected changes
        is_active: Whether the session is still active
    """

    session_id: str
    conditions: List[MonitoringCondition]
    start_time: datetime
    poll_interval: float = 5.0
    max_duration: float = 3600.0
    changes_detected: List[ChangeEvent] = field(default_factory=list)
    is_active: bool = True
    _baseline: Dict[str, Any] = field(default_factory=dict)


class MonitoringAgent(BaseAgent):
    """
    Agent specialized in monitoring web pages for changes.

    This agent uses LLMs to understand monitoring requirements and track
    changes on web pages. It supports various monitoring strategies including
    content hashing, element tracking, and value comparisons.

    **Rationale for this agent:**
    Monitoring is a critical capability for many automation use cases:
    - E-commerce: Track price changes, stock availability
    - News/Content: Detect new articles or updates
    - Business: Monitor dashboards, reports, KPIs
    - Testing: Verify page state after actions
    - Compliance: Track changes to terms, policies

    The agent inherits from BaseAgent and has access to:
    - page_controller: For page operations
    - element_detector: For element location
    - llm: For intelligent change analysis

    Example:
        >>> agent = MonitoringAgent(page_controller, element_detector, llm)
        >>>
        >>> # Monitor for any content changes
        >>> result = await agent.execute("Monitor this page for any changes")
        >>>
        >>> # Monitor specific element
        >>> result = await agent.monitor_element(
        ...     "the product price",
        ...     condition=MonitoringCondition(
        ...         description="Price drops below $100",
        ...         change_type=ChangeType.VALUE,
        ...         operator=ComparisonOperator.LESS_THAN,
        ...         threshold=100
        ...     )
        ... )
    """

    def __init__(
        self,
        page_controller,
        element_detector,
        llm_provider,
        default_poll_interval: float = 5.0,
        default_max_duration: float = 3600.0,
        pii_handler=None,
    ) -> None:
        """
        Initialize the monitoring agent.

        Args:
            page_controller: PageController instance for page operations
            element_detector: ElementDetector instance for element location
            llm_provider: BaseLLMProvider instance for LLM operations
            default_poll_interval: Default polling interval in seconds (default: 5.0)
            default_max_duration: Default max monitoring duration in seconds (default: 3600)
            pii_handler: Optional PIIHandler for secure handling of sensitive data
        """
        super().__init__(page_controller, element_detector, llm_provider, pii_handler=pii_handler)
        self.default_poll_interval = default_poll_interval
        self.default_max_duration = default_max_duration
        self._active_sessions: Dict[str, MonitoringSession] = {}

    async def execute(
        self,
        instruction: str,
        poll_interval: Optional[float] = None,
        max_duration: Optional[float] = None,
        callback: Optional[Callable[[ChangeEvent], None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a natural language monitoring instruction.

        This method uses an LLM to understand the monitoring requirement
        and sets up appropriate monitoring.

        Args:
            instruction: Natural language monitoring instruction.
                Examples:
                - "Monitor this page for any changes"
                - "Watch the price and alert if it drops below $50"
                - "Track when the 'Add to Cart' button becomes available"
            poll_interval: Interval between checks in seconds
            max_duration: Maximum monitoring duration in seconds
            callback: Optional callback function for change notifications

        Returns:
            Dictionary containing:
            - session_id: Unique identifier for the monitoring session
            - changes_detected: List of detected changes
            - monitoring_duration: How long monitoring ran
            - success: Whether monitoring completed successfully

        Example:
            >>> result = await agent.execute(
            ...     "Monitor the stock price and alert if it changes by more than 5%",
            ...     poll_interval=10,
            ...     max_duration=300
            ... )
        """
        try:
            # Mask instruction for logging to avoid exposing PII
            logger.info(f"Starting monitoring: {self.mask_for_log(instruction)}")

            poll_interval = poll_interval or self.default_poll_interval
            max_duration = max_duration or self.default_max_duration

            # Parse the instruction to determine monitoring conditions
            conditions = await self._parse_monitoring_instruction(instruction)

            # Create monitoring session
            session = await self._create_session(
                conditions, poll_interval, max_duration
            )

            # Run monitoring loop
            await self._run_monitoring_loop(session, callback)

            return {
                "session_id": session.session_id,
                "changes_detected": [
                    self._change_to_dict(c) for c in session.changes_detected
                ],
                "monitoring_duration": (
                    datetime.now() - session.start_time
                ).total_seconds(),
                "success": True,
            }

        except Exception as e:
            logger.error(f"Monitoring failed: {self.mask_for_log(str(e))}")
            raise MonitoringError(f"Failed to execute monitoring: {e}") from e

    async def _parse_monitoring_instruction(
        self, instruction: str
    ) -> List[MonitoringCondition]:
        """Parse natural language instruction into monitoring conditions."""
        context = await self.get_page_context()

        # Mask instruction for LLM to avoid exposing PII
        safe_instruction = self.mask_for_llm(instruction)

        prompt = f"""Analyze this monitoring instruction and extract the conditions to monitor.

Instruction: {safe_instruction}

Current page URL: {context['url']}
Page title: {context['title']}

Return a JSON object with:
- conditions: Array of monitoring conditions, each with:
  - description: What to monitor
  - change_type: One of "content", "element", "value", "presence", "absence"
  - operator: For value monitoring, one of "equals", "not_equals", "greater_than", "less_than", "contains"
  - threshold: The threshold value for comparisons
  - element_description: Natural language description of the element to monitor
"""

        schema = {
            "type": "object",
            "properties": {
                "conditions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "change_type": {"type": "string"},
                            "operator": {"type": "string"},
                            "threshold": {},
                            "element_description": {"type": "string"},
                        },
                        "required": ["description", "change_type"],
                    },
                },
            },
            "required": ["conditions"],
        }

        result = await self.llm.generate_structured(
            prompt=prompt,
            schema=schema,
            temperature=0.3,
        )

        conditions = []
        for cond_data in result.get("conditions", []):
            try:
                change_type = ChangeType(cond_data.get("change_type", "content"))
            except ValueError:
                change_type = ChangeType.CONTENT

            operator = None
            if cond_data.get("operator"):
                try:
                    operator = ComparisonOperator(cond_data["operator"])
                except ValueError:
                    pass

            conditions.append(
                MonitoringCondition(
                    description=cond_data.get("description", instruction),
                    change_type=change_type,
                    operator=operator,
                    threshold=cond_data.get("threshold"),
                    element_selector=cond_data.get("element_description"),
                )
            )

        if not conditions:
            # Default to content monitoring
            conditions.append(
                MonitoringCondition(
                    description=instruction,
                    change_type=ChangeType.CONTENT,
                )
            )

        return conditions

    async def _create_session(
        self,
        conditions: List[MonitoringCondition],
        poll_interval: float,
        max_duration: float,
    ) -> MonitoringSession:
        """Create a new monitoring session."""
        import uuid

        session_id = str(uuid.uuid4())[:8]

        session = MonitoringSession(
            session_id=session_id,
            conditions=conditions,
            start_time=datetime.now(),
            poll_interval=poll_interval,
            max_duration=max_duration,
            changes_detected=[],
            is_active=True,
            _baseline={},
        )

        # Capture baseline state
        session._baseline = await self._capture_state(conditions)

        self._active_sessions[session_id] = session
        logger.info(f"Created monitoring session: {session_id}")

        return session

    async def _capture_state(
        self, conditions: List[MonitoringCondition]
    ) -> Dict[str, Any]:
        """Capture current state for all monitoring conditions."""
        state = {
            "content_hash": await self._get_content_hash(),
            "timestamp": datetime.now().isoformat(),
            "elements": {},
        }

        for i, condition in enumerate(conditions):
            if condition.element_selector:
                try:
                    element_info = await self.detector.find_element(
                        condition.element_selector, use_vision=False
                    )
                    selector = element_info["selector"]
                    text = await self.detector.get_text(selector)
                    state["elements"][f"condition_{i}"] = {
                        "selector": selector,
                        "text": text,
                        "exists": True,
                    }
                except Exception:
                    state["elements"][f"condition_{i}"] = {"exists": False}

        return state

    async def _get_content_hash(self) -> str:
        """Get hash of page content for change detection."""
        html = await self.page.get_html()
        return hashlib.md5(html.encode()).hexdigest()

    async def _run_monitoring_loop(
        self,
        session: MonitoringSession,
        callback: Optional[Callable[[ChangeEvent], None]] = None,
    ) -> None:
        """Run the monitoring loop."""
        start_time = datetime.now()

        while session.is_active:
            # Check if max duration exceeded
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= session.max_duration:
                logger.info(f"Monitoring session {session.session_id} reached max duration")
                session.is_active = False
                break

            # Check for changes
            current_state = await self._capture_state(session.conditions)
            changes = await self._detect_changes(
                session.conditions, session._baseline, current_state
            )

            for change in changes:
                session.changes_detected.append(change)
                logger.info(f"Change detected: {change.description}")

                if callback:
                    try:
                        callback(change)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

            # Update baseline
            session._baseline = current_state

            # Wait for next poll
            await asyncio.sleep(session.poll_interval)

    async def _detect_changes(
        self,
        conditions: List[MonitoringCondition],
        baseline: Dict[str, Any],
        current: Dict[str, Any],
    ) -> List[ChangeEvent]:
        """Detect changes between baseline and current state."""
        changes = []

        # Check content hash
        if baseline.get("content_hash") != current.get("content_hash"):
            for condition in conditions:
                if condition.change_type == ChangeType.CONTENT:
                    changes.append(
                        ChangeEvent(
                            timestamp=datetime.now(),
                            change_type=ChangeType.CONTENT,
                            description=f"Page content changed: {condition.description}",
                            old_value=baseline.get("content_hash"),
                            new_value=current.get("content_hash"),
                        )
                    )

        # Check element-specific conditions
        for i, condition in enumerate(conditions):
            key = f"condition_{i}"
            old_elem = baseline.get("elements", {}).get(key, {})
            new_elem = current.get("elements", {}).get(key, {})

            if condition.change_type == ChangeType.PRESENCE:
                if not old_elem.get("exists") and new_elem.get("exists"):
                    changes.append(
                        ChangeEvent(
                            timestamp=datetime.now(),
                            change_type=ChangeType.PRESENCE,
                            description=f"Element appeared: {condition.description}",
                        )
                    )

            elif condition.change_type == ChangeType.ABSENCE:
                if old_elem.get("exists") and not new_elem.get("exists"):
                    changes.append(
                        ChangeEvent(
                            timestamp=datetime.now(),
                            change_type=ChangeType.ABSENCE,
                            description=f"Element disappeared: {condition.description}",
                        )
                    )

            elif condition.change_type == ChangeType.VALUE:
                old_text = old_elem.get("text", "")
                new_text = new_elem.get("text", "")

                if old_text != new_text:
                    if self._check_threshold(new_text, condition):
                        changes.append(
                            ChangeEvent(
                                timestamp=datetime.now(),
                                change_type=ChangeType.VALUE,
                                description=f"Value changed: {condition.description}",
                                old_value=old_text,
                                new_value=new_text,
                            )
                        )

        return changes

    def _check_threshold(
        self, value: str, condition: MonitoringCondition
    ) -> bool:
        """Check if value meets threshold condition."""
        if not condition.operator or condition.threshold is None:
            return True

        try:
            # Try to parse as number
            num_value = float(value.replace("$", "").replace(",", "").strip())
            threshold = float(condition.threshold)

            if condition.operator == ComparisonOperator.LESS_THAN:
                return num_value < threshold
            elif condition.operator == ComparisonOperator.GREATER_THAN:
                return num_value > threshold
            elif condition.operator == ComparisonOperator.EQUALS:
                return num_value == threshold
            elif condition.operator == ComparisonOperator.LESS_OR_EQUAL:
                return num_value <= threshold
            elif condition.operator == ComparisonOperator.GREATER_OR_EQUAL:
                return num_value >= threshold
        except ValueError:
            # String comparison
            if condition.operator == ComparisonOperator.CONTAINS:
                return str(condition.threshold) in value
            elif condition.operator == ComparisonOperator.NOT_CONTAINS:
                return str(condition.threshold) not in value
            elif condition.operator == ComparisonOperator.EQUALS:
                return value == str(condition.threshold)

        return True

    def _change_to_dict(self, change: ChangeEvent) -> Dict[str, Any]:
        """Convert ChangeEvent to dictionary."""
        return {
            "timestamp": change.timestamp.isoformat(),
            "change_type": change.change_type.value,
            "description": change.description,
            "old_value": change.old_value,
            "new_value": change.new_value,
            "details": change.details,
        }

    async def stop_monitoring(self, session_id: str) -> bool:
        """
        Stop an active monitoring session.

        Args:
            session_id: ID of the session to stop

        Returns:
            True if session was stopped, False if not found
        """
        if session_id in self._active_sessions:
            self._active_sessions[session_id].is_active = False
            logger.info(f"Stopped monitoring session: {session_id}")
            return True
        return False

    def get_active_sessions(self) -> List[str]:
        """Get list of active monitoring session IDs."""
        return [
            sid for sid, session in self._active_sessions.items()
            if session.is_active
        ]

