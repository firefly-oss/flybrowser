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
Action agent for executing browser actions using natural language.

This module provides the ActionAgent class which uses LLMs to understand
natural language instructions and execute corresponding browser actions
like clicking, typing, form filling, and complex multi-step workflows.

The agent supports:
- Natural language action commands
- Multi-step action planning
- Error handling with retry logic
- Form filling and submission
- Complex interaction sequences

Example:
    >>> agent = ActionAgent(page_controller, element_detector, llm_provider)
    >>> result = await agent.execute("Click the login button and enter my email")
    >>> print(result)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from flybrowser.agents.base_agent import BaseAgent
from flybrowser.agents.validation_agent import ResponseValidator
from flybrowser.exceptions import ActionError, ElementNotFoundError
from flybrowser.llm.prompts import ACTION_PLANNING_PROMPT, ACTION_PLANNING_SYSTEM
from flybrowser.utils.logger import logger
from flybrowser.utils.timing import StepTimer


class ActionType(str, Enum):
    """Types of browser actions that can be executed."""

    CLICK = "click"
    TYPE = "type"
    FILL = "fill"
    SELECT = "select"
    HOVER = "hover"
    SCROLL = "scroll"
    WAIT = "wait"
    PRESS_KEY = "press_key"
    CLEAR = "clear"
    CHECK = "check"
    UNCHECK = "uncheck"
    SCREENSHOT = "screenshot"


@dataclass
class ActionStep:
    """
    Represents a single action step in an action plan.

    Attributes:
        action_type: Type of action to perform
        target: Element description or selector
        value: Value for the action (e.g., text to type)
        options: Additional options for the action
    """

    action_type: ActionType
    target: Optional[str] = None
    value: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    """
    Result of executing an action or action plan.

    Attributes:
        success: Whether the action(s) succeeded
        steps_completed: Number of steps completed
        total_steps: Total number of steps in the plan
        error: Error message if failed
        details: Additional details about the execution
    """

    success: bool
    steps_completed: int = 0
    total_steps: int = 0
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class ActionAgent(BaseAgent):
    """
    Agent specialized in executing browser actions using natural language.

    This agent uses LLMs to understand natural language instructions and
    translate them into browser actions. It supports complex multi-step
    workflows with planning, error handling, and retry logic.

    The agent inherits from BaseAgent and has access to:
    - page_controller: For page operations
    - element_detector: For element location
    - llm: For intelligent action planning

    Attributes:
        max_retries: Maximum number of retries for failed actions
        retry_delay: Delay between retries in seconds
        action_timeout: Timeout for individual actions in milliseconds

    Example:
        >>> agent = ActionAgent(page_controller, element_detector, llm)
        >>>
        >>> # Simple action
        >>> result = await agent.execute("Click the submit button")
        >>>
        >>> # Complex multi-step action
        >>> result = await agent.execute(
        ...     "Fill in the login form with email test@example.com and password secret123"
        ... )
        >>>
        >>> # Form filling
        >>> result = await agent.fill_form({
        ...     "email": "test@example.com",
        ...     "password": "secret123"
        ... })
    """

    def __init__(
        self,
        page_controller,
        element_detector,
        llm_provider,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        action_timeout: int = 30000,
        pii_handler=None,
    ) -> None:
        """
        Initialize the action agent.

        Args:
            page_controller: PageController instance for page operations
            element_detector: ElementDetector instance for element location
            llm_provider: BaseLLMProvider instance for LLM operations
            max_retries: Maximum retries for failed actions (default: 3)
            retry_delay: Delay between retries in seconds (default: 1.0)
            action_timeout: Timeout for actions in milliseconds (default: 30000)
            pii_handler: Optional PIIHandler for secure handling of sensitive data
        """
        super().__init__(page_controller, element_detector, llm_provider, pii_handler=pii_handler)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.action_timeout = action_timeout
        self.validator = ResponseValidator(llm_provider)

    async def execute(
        self,
        instruction: str,
        use_vision: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a natural language action instruction.

        This method uses an LLM to understand the instruction, plan the
        necessary steps, and execute them in sequence with error handling.

        Args:
            instruction: Natural language instruction describing the action.
                Examples:
                - "Click the login button"
                - "Type 'hello world' in the search box"
                - "Fill the form with name John and email john@example.com"
                - "Select 'United States' from the country dropdown"
            use_vision: Whether to use vision for element detection (default: True)
            dry_run: If True, plan actions but don't execute (default: False)

        Returns:
            Dictionary containing:
            - success: Whether all actions succeeded
            - steps_completed: Number of steps completed
            - total_steps: Total number of planned steps
            - plan: The action plan that was executed
            - error: Error message if failed

        Raises:
            ActionError: If action planning or execution fails

        Example:
            >>> result = await agent.execute("Click the submit button")
            >>> print(result["success"])
            True

            >>> result = await agent.execute(
            ...     "Fill login form with email test@example.com",
            ...     dry_run=True
            ... )
            >>> print(result["plan"])  # See planned actions without executing
        """
        # Start timing
        timer = StepTimer()
        timer.start()
        
        try:
            # Mask instruction for logging to avoid exposing PII
            logger.info(f"Executing action: {self.mask_for_log(instruction)}")

            # Plan the actions (instruction is masked for LLM in _plan_actions)
            timer.start_step("planning")
            plan = await self._plan_actions(instruction, use_vision)
            timer.end_step("planning")
            
            # Log planned steps before execution
            plan_summary = ", ".join([
                f"{i+1}. [{step.action_type.value}] {step.target or '(no target)'}"
                for i, step in enumerate(plan)
            ])
            logger.info(f"Planned {len(plan)} action steps: {plan_summary}")

            if dry_run:
                return {
                    "success": True,
                    "steps_completed": 0,
                    "total_steps": len(plan),
                    "plan": [self._step_to_dict(step) for step in plan],
                    "dry_run": True,
                    "timing": timer.get_timings().to_dict(),
                }

            # Execute the plan
            timer.start_step("execution")
            result = await self._execute_plan(plan, use_vision)
            timer.end_step("execution")

            logger.info(
                f"Action completed: {result.steps_completed}/{result.total_steps} steps"
            )

            return {
                "success": result.success,
                "steps_completed": result.steps_completed,
                "total_steps": result.total_steps,
                "plan": [self._step_to_dict(step) for step in plan],
                "error": result.error,
                "details": result.details,
                "timing": timer.get_timings().to_dict(),
            }

        except Exception as e:
            logger.error(f"Action execution failed: {self.mask_for_log(str(e))}")
            # Return error dict instead of raising, for better error handling
            return {
                "success": False,
                "steps_completed": 0,
                "total_steps": 0,
                "plan": [],
                "error": str(e),
                "details": {"exception_type": type(e).__name__},
                "timing": timer.get_timings().to_dict(),
            }

    async def _plan_actions(
        self, instruction: str, use_vision: bool = True
    ) -> List[ActionStep]:
        """
        Plan actions based on natural language instruction.

        Args:
            instruction: Natural language instruction
            use_vision: Whether to use vision for context

        Returns:
            List of ActionStep objects representing the plan
        """
        context = await self.get_page_context()

        # Get visible interactive elements for context
        elements_info = await self._get_interactive_elements()

        # Mask instruction for LLM to avoid exposing PII
        safe_instruction = self.mask_for_llm(instruction)

        prompt = ACTION_PLANNING_PROMPT.format(
            instruction=safe_instruction,
            url=context["url"],
            title=context["title"],
            elements=json.dumps(elements_info[:20], indent=2),  # Limit elements
        )

        # Define the expected response schema
        schema = {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action_type": {
                                "type": "string",
                                "enum": [t.value for t in ActionType],
                            },
                            "target": {"type": "string"},
                            "value": {"type": "string"},
                            "options": {"type": "object"},
                        },
                        "required": ["action_type"],
                    },
                },
                "reasoning": {"type": "string"},
            },
            "required": ["actions"],
        }

        if use_vision:
            screenshot = await self.page.screenshot()
            # Use structured generation with vision context
            response = await self.llm.generate_with_vision(
                prompt=prompt,
                image_data=screenshot,
                system_prompt=ACTION_PLANNING_SYSTEM,
                temperature=0.3,
            )
            # Validate and fix response
            result = await self.validator.validate_and_fix(
                response.content,
                schema,
                context=f"Planning actions for: {safe_instruction}"
            )
        else:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=ACTION_PLANNING_SYSTEM,
                temperature=0.3,
            )
            # Validate and fix response
            result = await self.validator.validate_and_fix(
                response.content,
                schema,
                context=f"Planning actions for: {safe_instruction}"
            )

        # Convert to ActionStep objects
        steps = []
        for action_data in result.get("actions", []):
            try:
                action_type = ActionType(action_data.get("action_type", "click"))
            except ValueError:
                action_type = ActionType.CLICK

            steps.append(
                ActionStep(
                    action_type=action_type,
                    target=action_data.get("target"),
                    value=action_data.get("value"),
                    options=action_data.get("options", {}),
                )
            )

        logger.info(f"Planned {len(steps)} action steps")
        return steps


    async def _execute_plan(
        self, plan: List[ActionStep], use_vision: bool = True
    ) -> ActionResult:
        """
        Execute a planned sequence of actions.

        Args:
            plan: List of ActionStep objects to execute
            use_vision: Whether to use vision for element detection

        Returns:
            ActionResult with execution details
        """
        total_steps = len(plan)
        completed = 0
        details: Dict[str, Any] = {"step_results": []}

        for i, step in enumerate(plan):
            step_result = {"step": i + 1, "action": step.action_type.value}

            for attempt in range(self.max_retries):
                try:
                    await self._execute_step(step, use_vision)
                    step_result["success"] = True
                    step_result["attempts"] = attempt + 1
                    completed += 1
                    break
                except Exception as e:
                    step_result["error"] = str(e)
                    if attempt < self.max_retries - 1:
                        logger.warning(
                            f"Step {i + 1} failed (attempt {attempt + 1}), retrying..."
                        )
                        await asyncio.sleep(self.retry_delay)
                    else:
                        step_result["success"] = False
                        details["step_results"].append(step_result)
                        return ActionResult(
                            success=False,
                            steps_completed=completed,
                            total_steps=total_steps,
                            error=f"Step {i + 1} failed after {self.max_retries} attempts: {e}",
                            details=details,
                        )

            details["step_results"].append(step_result)

            # Small delay between steps for stability
            if i < total_steps - 1:
                await asyncio.sleep(0.2)

        return ActionResult(
            success=True,
            steps_completed=completed,
            total_steps=total_steps,
            details=details,
        )

    async def _execute_step(self, step: ActionStep, use_vision: bool = True) -> None:
        """
        Execute a single action step.

        Args:
            step: ActionStep to execute
            use_vision: Whether to use vision for element detection

        Raises:
            ActionError: If the step fails to execute
        """
        logger.debug(f"Executing step: {step.action_type.value} on {step.target}")

        # Find element if target is specified
        element_info = None
        if step.target:
            try:
                element_info = await self.detector.find_element(
                    step.target, use_vision=use_vision
                )
            except ElementNotFoundError as e:
                raise ActionError(f"Could not find element: {step.target}") from e

        selector = element_info["selector"] if element_info else None
        selector_type = element_info.get("selector_type", "css") if element_info else "css"

        # Resolve any credential placeholders in the value before execution
        # This is the key step: LLM sees {{CREDENTIAL:password}}, browser gets real value
        resolved_value = step.value
        if step.value and self.has_placeholders(step.value):
            resolved_value = self.resolve_placeholders(step.value)
            logger.debug(f"Resolved credential placeholders for action")

        # Execute the action based on type
        if step.action_type == ActionType.CLICK:
            if not selector:
                raise ActionError("Click action requires a target element")
            await self.detector.click(selector, selector_type)

        elif step.action_type == ActionType.TYPE:
            if not selector:
                raise ActionError("Type action requires a target element")
            if not resolved_value:
                raise ActionError("Type action requires a value")
            await self.detector.type_text(selector, resolved_value, selector_type)

        elif step.action_type == ActionType.FILL:
            if not selector:
                raise ActionError("Fill action requires a target element")
            await self.detector.type_text(selector, resolved_value or "", selector_type)

        elif step.action_type == ActionType.SELECT:
            if not selector:
                raise ActionError("Select action requires a target element")
            await self._select_option(selector, resolved_value or "", selector_type)

        elif step.action_type == ActionType.HOVER:
            if not selector:
                raise ActionError("Hover action requires a target element")
            await self._hover_element(selector, selector_type)

        elif step.action_type == ActionType.SCROLL:
            direction = resolved_value or "down"
            amount = step.options.get("amount", 300)
            await self._scroll(direction, amount)

        elif step.action_type == ActionType.WAIT:
            wait_time = float(resolved_value or "1")
            await asyncio.sleep(wait_time)

        elif step.action_type == ActionType.PRESS_KEY:
            if not resolved_value:
                raise ActionError("Press key action requires a key value")
            await self._press_key(resolved_value)

        elif step.action_type == ActionType.CLEAR:
            if not selector:
                raise ActionError("Clear action requires a target element")
            await self._clear_field(selector, selector_type)

        elif step.action_type == ActionType.CHECK:
            if not selector:
                raise ActionError("Check action requires a target element")
            await self._set_checkbox(selector, True, selector_type)

        elif step.action_type == ActionType.UNCHECK:
            if not selector:
                raise ActionError("Uncheck action requires a target element")
            await self._set_checkbox(selector, False, selector_type)

        elif step.action_type == ActionType.SCREENSHOT:
            await self.page.screenshot()

        else:
            raise ActionError(f"Unknown action type: {step.action_type}")

    async def _get_interactive_elements(self) -> List[Dict[str, Any]]:
        """Get list of interactive elements on the page."""
        script = """
        () => {
            const elements = [];
            const interactiveSelectors = 'a, button, input, select, textarea, [role="button"], [onclick]';
            document.querySelectorAll(interactiveSelectors).forEach((el, idx) => {
                if (idx < 50) {  // Limit to 50 elements
                    const rect = el.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        elements.push({
                            tag: el.tagName.toLowerCase(),
                            type: el.type || null,
                            text: (el.textContent || '').trim().substring(0, 50),
                            placeholder: el.placeholder || null,
                            name: el.name || null,
                            id: el.id || null,
                            class: el.className || null,
                        });
                    }
                }
            });
            return elements;
        }
        """
        try:
            return await self.page.evaluate(script)
        except Exception:
            return []

    async def _select_option(
        self, selector: str, value: str, selector_type: str = "css"
    ) -> None:
        """Select an option from a dropdown."""
        try:
            if selector_type == "xpath":
                locator = self.page.page.locator(f"xpath={selector}")
            else:
                locator = self.page.page.locator(selector)
            await locator.select_option(value)
        except Exception as e:
            raise ActionError(f"Failed to select option: {e}") from e

    async def _hover_element(self, selector: str, selector_type: str = "css") -> None:
        """Hover over an element."""
        try:
            if selector_type == "xpath":
                locator = self.page.page.locator(f"xpath={selector}")
            else:
                locator = self.page.page.locator(selector)
            await locator.hover()
        except Exception as e:
            raise ActionError(f"Failed to hover: {e}") from e

    async def _scroll(self, direction: str, amount: int = 300) -> None:
        """Scroll the page."""
        try:
            if direction == "down":
                await self.page.evaluate(f"window.scrollBy(0, {amount})")
            elif direction == "up":
                await self.page.evaluate(f"window.scrollBy(0, -{amount})")
            elif direction == "left":
                await self.page.evaluate(f"window.scrollBy(-{amount}, 0)")
            elif direction == "right":
                await self.page.evaluate(f"window.scrollBy({amount}, 0)")
            elif direction == "top":
                await self.page.evaluate("window.scrollTo(0, 0)")
            elif direction == "bottom":
                await self.page.evaluate(
                    "window.scrollTo(0, document.body.scrollHeight)"
                )
        except Exception as e:
            raise ActionError(f"Failed to scroll: {e}") from e

    async def _press_key(self, key: str) -> None:
        """Press a keyboard key."""
        try:
            await self.page.page.keyboard.press(key)
        except Exception as e:
            raise ActionError(f"Failed to press key: {e}") from e

    async def _clear_field(self, selector: str, selector_type: str = "css") -> None:
        """Clear an input field."""
        try:
            if selector_type == "xpath":
                locator = self.page.page.locator(f"xpath={selector}")
            else:
                locator = self.page.page.locator(selector)
            await locator.clear()
        except Exception as e:
            raise ActionError(f"Failed to clear field: {e}") from e

    async def _set_checkbox(
        self, selector: str, checked: bool, selector_type: str = "css"
    ) -> None:
        """Set checkbox state."""
        try:
            if selector_type == "xpath":
                locator = self.page.page.locator(f"xpath={selector}")
            else:
                locator = self.page.page.locator(selector)
            await locator.set_checked(checked)
        except Exception as e:
            raise ActionError(f"Failed to set checkbox: {e}") from e

    def _step_to_dict(self, step: ActionStep) -> Dict[str, Any]:
        """Convert ActionStep to dictionary."""
        return {
            "action_type": step.action_type.value,
            "target": step.target,
            "value": step.value,
            "options": step.options,
        }

    async def fill_form(
        self,
        field_values: Dict[str, str],
        submit: bool = False,
        use_vision: bool = True,
    ) -> Dict[str, Any]:
        """
        Fill a form with the provided field values.

        This method intelligently locates form fields and fills them with
        the provided values. It can optionally submit the form after filling.

        Args:
            field_values: Dictionary mapping field descriptions to values.
                Keys can be field names, labels, or natural language descriptions.
                Examples:
                - {"email": "test@example.com", "password": "secret123"}
                - {"First Name": "John", "Last Name": "Doe"}
            submit: Whether to submit the form after filling (default: False)
            use_vision: Whether to use vision for field detection (default: True)

        Returns:
            Dictionary containing:
            - success: Whether form filling succeeded
            - fields_filled: Number of fields successfully filled
            - total_fields: Total number of fields to fill
            - errors: List of any errors encountered

        Example:
            >>> result = await agent.fill_form({
            ...     "email": "test@example.com",
            ...     "password": "secret123"
            ... }, submit=True)
        """
        errors = []
        filled = 0
        total = len(field_values)

        # Mask field values for logging
        masked_values = self.mask_dict(field_values)

        for field_desc, value in field_values.items():
            try:
                # Find the field
                element_info = await self.detector.find_element(
                    f"the {field_desc} input field", use_vision=use_vision
                )
                selector = element_info["selector"]
                selector_type = element_info.get("selector_type", "css")

                # Clear and fill
                await self._clear_field(selector, selector_type)
                await self.detector.type_text(selector, value, selector_type)
                filled += 1
                # Log with masked value to avoid exposing PII
                logger.info(f"Filled field '{field_desc}' with value '{masked_values.get(field_desc, '***')}'")

            except Exception as e:
                errors.append({"field": field_desc, "error": str(e)})
                logger.warning(f"Failed to fill field '{field_desc}': {self.mask_for_log(str(e))}")

        # Submit if requested
        if submit and filled > 0:
            try:
                result = await self.execute("Click the submit button", use_vision)
                if not result["success"]:
                    errors.append({"field": "submit", "error": result.get("error")})
            except Exception as e:
                errors.append({"field": "submit", "error": str(e)})

        return {
            "success": len(errors) == 0,
            "fields_filled": filled,
            "total_fields": total,
            "errors": errors,
        }

    async def click(self, description: str, use_vision: bool = True) -> Dict[str, Any]:
        """
        Click an element described in natural language.

        Args:
            description: Natural language description of the element to click
            use_vision: Whether to use vision for element detection

        Returns:
            Dictionary with success status and details

        Example:
            >>> await agent.click("the login button")
            >>> await agent.click("the first product in the list")
        """
        return await self.execute(f"Click {description}", use_vision=use_vision)

    async def type_text(
        self, description: str, text: str, use_vision: bool = True
    ) -> Dict[str, Any]:
        """
        Type text into an element described in natural language.

        Args:
            description: Natural language description of the element
            text: Text to type
            use_vision: Whether to use vision for element detection

        Returns:
            Dictionary with success status and details

        Example:
            >>> await agent.type_text("the search box", "python tutorials")
        """
        return await self.execute(
            f"Type '{text}' into {description}", use_vision=use_vision
        )

    async def secure_fill_form(
        self,
        field_credentials: Dict[str, str],
        submit: bool = False,
        use_vision: bool = True,
    ) -> Dict[str, Any]:
        """
        Securely fill a form using stored credentials from PIIHandler.

        This method fills form fields using credential IDs from the PIIHandler,
        ensuring that sensitive values are never exposed to the LLM or logged.

        Args:
            field_credentials: Dictionary mapping field descriptions to credential IDs.
                Keys are field names/labels/descriptions, values are credential IDs
                from PIIHandler.store_credential().
                Example:
                - {"email": email_cred_id, "password": password_cred_id}
            submit: Whether to submit the form after filling (default: False)
            use_vision: Whether to use vision for field detection (default: True)

        Returns:
            Dictionary containing:
            - success: Whether form filling succeeded
            - fields_filled: Number of fields successfully filled
            - total_fields: Total number of fields to fill
            - errors: List of any errors encountered

        Raises:
            ValueError: If no PIIHandler is configured

        Example:
            >>> # Store credentials first
            >>> email_id = pii_handler.store_credential("email", "user@example.com")
            >>> pwd_id = pii_handler.store_credential("password", "secret123")
            >>>
            >>> # Then use secure_fill_form
            >>> result = await agent.secure_fill_form({
            ...     "email": email_id,
            ...     "password": pwd_id
            ... }, submit=True)
        """
        if not self.pii_handler:
            raise ValueError("PIIHandler is required for secure_fill_form. Initialize agent with pii_handler parameter.")

        errors = []
        filled = 0
        total = len(field_credentials)

        for field_desc, credential_id in field_credentials.items():
            try:
                # Find the field
                element_info = await self.detector.find_element(
                    f"the {field_desc} input field", use_vision=use_vision
                )
                selector = element_info["selector"]

                # Clear the field first
                try:
                    await self._clear_field(selector, element_info.get("selector_type", "css"))
                except Exception:
                    pass  # Field might not need clearing

                # Securely fill using PIIHandler
                success = await self.secure_fill(selector, credential_id)
                if success:
                    filled += 1
                    logger.info(f"Securely filled field '{field_desc}' (credential: {credential_id})")
                else:
                    errors.append({"field": field_desc, "error": "Failed to fill field"})

            except Exception as e:
                errors.append({"field": field_desc, "error": str(e)})
                logger.warning(f"Failed to securely fill field '{field_desc}': {self.mask_for_log(str(e))}")

        # Submit if requested
        if submit and filled > 0:
            try:
                result = await self.execute("Click the submit button", use_vision)
                if not result["success"]:
                    errors.append({"field": "submit", "error": result.get("error")})
            except Exception as e:
                errors.append({"field": "submit", "error": str(e)})

        return {
            "success": len(errors) == 0,
            "fields_filled": filled,
            "total_fields": total,
            "errors": errors,
        }

