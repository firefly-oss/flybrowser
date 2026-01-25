# Copyright 2026 Firefly Software Solutions Inc.
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
Core ReAct Agent implementation.

Implements the Reasoning and Acting (ReAct) paradigm for intelligent
browser automation with explicit thought-action-observation cycles.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .types import (
    Action,
    ExecutionOutcome,
    ExecutionState,
    MemoryPriority,
    Observation,
    OperationMode,
    ReActStep,
    ReasoningStrategy,
    SafetyLevel,
    Thought,
    ToolResult,
    REACT_RESPONSE_SCHEMA,
    parse_react_response,
    validate_react_response,
    build_repair_prompt,
)
from .config import AgentConfig
from .parser import ReActParser, ParseResult, ParseFormat
from .memory import AgentMemory
from .tools.registry import ToolRegistry
from .planner import TaskPlanner, ExecutionPlan, PhaseStatus, GoalStatus
from .goal_interpreter import GoalInterpreter
from .scope_validator import get_scope_validator
from .sitemap_graph import (
    SitemapGraph, SitemapLimits, LinkType,
    analyze_exploration_intent_async, is_site_exploration_task_async,
    filter_navigation_links_async, reset_link_filter_memory, ExplorationIntentAnalyzer,
    ExplorationDAG
)
from .parallel_explorer import ParallelPageExplorer, ParallelExplorationStats
from flybrowser.prompts import PromptManager
from flybrowser.llm.base import ModelCapability

if TYPE_CHECKING:
    from flybrowser.core.page import PageController
    from flybrowser.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result of agent task execution."""

    success: bool
    result: Any = None
    error: Optional[str] = None
    steps: List[ReActStep] = field(default_factory=list)
    total_iterations: int = 0
    execution_time_ms: float = 0.0
    final_state: ExecutionState = ExecutionState.COMPLETED
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method for compatibility."""
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "steps": self.steps,
            "total_iterations": self.total_iterations,
            "execution_time_ms": self.execution_time_ms,
            "final_state": self.final_state.value if isinstance(self.final_state, ExecutionState) else self.final_state,
            "metadata": self.metadata,
        }

    @classmethod
    def success_result(cls, result: Any, steps: List[ReActStep], **kwargs) -> "AgentResult":
        """Create a successful agent result."""
        return cls(success=True, result=result, steps=steps, **kwargs)

    @classmethod
    def failure_result(cls, error: str, steps: List[ReActStep], **kwargs) -> "AgentResult":
        """Create a failed agent result."""
        return cls(
            success=False,
            error=error,
            steps=steps,
            final_state=ExecutionState.FAILED,
            **kwargs
        )


class ReActAgent:
    """
    Core ReAct agent implementing thought-action-observation cycles.

    The agent receives a task, reasons about how to accomplish it,
    executes browser actions through tools, observes results, and
    iterates until the task is complete or fails.

    Attributes:
        page: Browser page controller for executing actions
        llm: LLM provider for reasoning
        tool_registry: Registry of available tools
        memory: Agent memory system for context management
        config: Agent configuration

    Example:
        >>> agent = ReActAgent(page, llm, registry, memory)
        >>> result = await agent.execute("Search for 'python tutorials'")
        >>> print(result.success, result.result)
    """

    def __init__(
        self,
        page_controller: "PageController",
        llm_provider: "BaseLLMProvider",
        tool_registry: ToolRegistry,
        memory: Optional[AgentMemory] = None,
        config: Optional[AgentConfig] = None,
        approval_callback: Optional[Callable[[Action], bool]] = None,
        enable_autonomous_planning: bool = True,
        element_detector: Optional[Any] = None,
    ) -> None:
        """
        Initialize the ReAct agent.

        Args:
            page_controller: Browser page controller
            llm_provider: LLM for reasoning
            tool_registry: Available tools registry
            memory: Memory system (created if not provided)
            config: Agent configuration
            approval_callback: Callback for dangerous action approval
            enable_autonomous_planning: Enable automatic planning for complex tasks
            element_detector: Optional element detector for AI-based element finding
        """
        self.page = page_controller
        self.llm = llm_provider
        self.memory = memory or AgentMemory()
        self.config = config or AgentConfig()
        self.element_detector = element_detector
        self.approval_callback = approval_callback
        self.enable_autonomous_planning = enable_autonomous_planning
        
        # Get model capabilities
        self.model_info = llm_provider.get_model_info()
        
        # Filter tool registry based on model capabilities
        self.tool_registry = tool_registry.get_filtered_registry(
            self.model_info.capabilities,
            warn_suboptimal=True,
        )
        
        # Log capability awareness
        vision_status = "vision-enabled" if ModelCapability.VISION in self.model_info.capabilities else "text-only"
        logger.info(
            f"Agent initialized with {llm_provider.model} ({vision_status})"
        )
        logger.debug(f"Available tools: {len(self.tool_registry.list_tools())}")

        # Initialize parser with filtered tool registry
        self.parser = ReActParser(tool_registry=self.tool_registry)

        # Initialize prompt manager for template-based prompts
        self.prompt_manager = PromptManager()

        # Initialize task planner with model capabilities and config
        self.planner = TaskPlanner(
            llm_provider,
            self.prompt_manager,
            model_capabilities=self.model_info.capabilities,
            config=self.config,
        )
        
        # Initialize goal interpreter for fast-path execution
        self.goal_interpreter = GoalInterpreter()
        
        # Initialize page exploration components (optional)
        self._page_explorer = None
        self._page_analyzer = None
        self._parallel_explorer: Optional[ParallelPageExplorer] = None
        if self.config.page_exploration.enabled and ModelCapability.VISION in self.model_info.capabilities:
            try:
                from .tools.page_explorer import PageExplorerTool
                from .page_analyzer import PageAnalyzer
                self._page_explorer = PageExplorerTool(self.page, self.config.page_exploration)
                self._page_analyzer = PageAnalyzer(self.llm, self.prompt_manager, self.config.page_exploration)
                logger.info("Page exploration enabled (vision-based)")
                
                # Initialize parallel explorer if configured
                if self.config.parallel_exploration.enable_parallel:
                    self._parallel_explorer = ParallelPageExplorer(
                        page_controller=self.page,
                        llm_provider=self.llm,
                        page_explorer=self._page_explorer,
                        page_analyzer=self._page_analyzer,
                        memory=self.memory,
                        config=self.config.parallel_exploration,
                    )
                    logger.info(
                        f"Parallel exploration enabled "
                        f"(max_parallel={self.config.parallel_exploration.max_parallel_pages})"
                    )
            except ImportError as e:
                logger.warning(f"Page exploration disabled: {e}")

        # Execution state
        self._state = ExecutionState.IDLE
        self._current_step: Optional[ReActStep] = None
        self._steps: List[ReActStep] = []
        self._consecutive_failures = 0
        self._task_completed = False
        self._task_failed = False
        self._final_result: Any = None
        self._error_message: Optional[str] = None
        self._current_plan: Optional[ExecutionPlan] = None
        self._progress_callback: Optional[Callable[[str], None]] = None
        
        # Operation mode (detected per-task)
        self.operation_mode: OperationMode = OperationMode.AUTO

    @property
    def state(self) -> ExecutionState:
        """Current execution state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Whether agent is currently executing."""
        return self._state not in (
            ExecutionState.IDLE,
            ExecutionState.COMPLETED,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
        )

    @property
    def current_step(self) -> Optional[ReActStep]:
        """Current step being executed."""
        return self._current_step

    def set_progress_callback(self, callback: Callable[[str], None]) -> None:
        """Set a callback for progress reporting."""
        self._progress_callback = callback

    def _report_progress(self, message: str) -> None:
        """Report progress if callback is set."""
        if self._progress_callback:
            self._progress_callback(message)
        logger.info(f"[PROGRESS] {message}")

    async def execute(
        self,
        task: str,
        step_callback: Optional[Callable[[ReActStep], Optional[bool]]] = None,
        operation_mode: Optional[OperationMode] = None,
        vision_override: Optional[bool] = None,
    ) -> AgentResult:
        """
        Execute a task using the ReAct loop with optional autonomous planning.

        Args:
            task: Natural language task description
            step_callback: Optional callback called after each step (returns False to stop)
            operation_mode: Explicitly set operation mode (should be set by SDK method)
            vision_override: Override vision behavior for this execution.
                - None: Use model capabilities and operation mode to decide (default)
                - True: Force vision on (if model supports it)
                - False: Force vision off (text-only mode)

        Returns:
            AgentResult with success/failure and execution details
        """
        start_time = time.time()
        iteration = 0

        # Reset state for new execution
        self._reset_state()
        
        # Store vision override for this execution
        self._vision_override = vision_override
        
        # Set operation mode if provided (from SDK method)
        if operation_mode is not None:
            self.operation_mode = operation_mode
        
        self.memory.start_task(task)
        
        # Phase 1: Quick preliminary check if this might be a site exploration task
        # Actual limits are determined AFTER first page exploration (Phase 2)
        await self._preliminary_sitemap_check_async(task)

        logger.info(f"Starting task execution: {task[:100]}")
        
        # Operation mode should be set by SDK method call, not detected from keywords
        # If not set (legacy usage), default to AUTO
        if self.operation_mode == OperationMode.AUTO:
            logger.debug(f"Operation mode: {self.operation_mode.value} (set by SDK)")
        else:
            logger.info(f"Operation mode: {self.operation_mode.value}")
        
        # Store operation mode in memory for context
        self.memory.set_operation_mode(self.operation_mode)

        # Check if autonomous planning should be used
        if self.enable_autonomous_planning and self.planner.should_create_plan(task):
            self._report_progress("Analyzing task complexity...")
            return await self._execute_with_planning(task, step_callback, start_time)
        else:
            return await self._execute_direct(task, step_callback, start_time)

    async def _execute_direct(self, task: str, step_callback: Optional[Callable], start_time: float) -> AgentResult:
        """
        Execute task directly without planning (for simple tasks).

        Args:
            task: Task description
            step_callback: Optional step callback
            start_time: Execution start time

        Returns:
            AgentResult
        """
        iteration = 0
        logger.info("Executing in DIRECT mode (no planning)")

        try:
            while not self._should_stop(iteration):
                iteration += 1
                self._state = ExecutionState.THINKING
                
                # Build prompts with memory context using PromptManager
                prompts = self._build_prompt(task)

                # Get LLM reasoning (with vision if enabled)
                response = await self._generate_with_optional_vision(
                    prompts=prompts,
                    iteration=iteration,
                )

                # Parse response
                parse_result = self.parser.parse(response.content)

                if not parse_result.success:
                    self._consecutive_failures += 1
                    self._consecutive_parse_failures += 1
                    logger.warning(f"Parse failed ({self._consecutive_parse_failures}x): {parse_result.error}")
                    continue
                
                # Reset parse failure counter on successful parse
                self._consecutive_parse_failures = 0

                # Create step record
                step = ReActStep(
                    step_number=iteration,
                    thought=parse_result.thought,
                    action=parse_result.action,
                )

                # Handle action execution
                if parse_result.action:
                    # Record action for loop detection BEFORE execution
                    self._record_action_for_loop_detection(parse_result.action)
                    
                    observation = await self._execute_action(parse_result.action, step)
                    step.observation = observation
                    
                    # TRIGGER EXPLORATION after EVERY successful navigation (vision models only)
                    if (observation.success and 
                        parse_result.action.tool_name in ("navigate", "goto") and 
                        self._should_explore_page(task)):
                        self._report_progress("Exploring page structure...")
                        exploration_success = await self._explore_current_page()
                        if exploration_success:
                            self._report_progress("Page exploration complete")
                        else:
                            logger.warning("Page exploration failed - continuing without it")

                    # Record in memory
                    self.memory.record_cycle(
                        thought=parse_result.thought,
                        action=parse_result.action,
                        observation=observation,
                        outcome=ExecutionOutcome.SUCCESS if observation.success else ExecutionOutcome.FAILURE,
                    )

                self._steps.append(step)
                self._current_step = step

                # Call step callback if provided
                if step_callback:
                    result = await step_callback(step)
                    if result is False:
                        logger.info("Execution stopped by step callback")
                        break

                # Check for completion signals
                if self._check_completion(parse_result.action):
                    break

        except asyncio.CancelledError:
            self._state = ExecutionState.CANCELLED
            logger.info("Task execution cancelled")
        except Exception as e:
            self._state = ExecutionState.FAILED
            self._error_message = str(e)
            logger.exception(f"Task execution failed: {e}")

        execution_time = (time.time() - start_time) * 1000
        return self._create_result(iteration, execution_time)

    async def _execute_with_planning(self, task: str, step_callback: Optional[Callable], start_time: float) -> AgentResult:
        """
        Execute task with autonomous planning (for complex tasks).

        Creates a structured plan, executes phase-by-phase, and adapts on failures.

        Args:
            task: Task description
            step_callback: Optional step callback
            start_time: Execution start time

        Returns:
            AgentResult with plan metadata
        """
        iteration = 0
        logger.info("Executing in PLANNING mode")
        
        # Create execution plan (exploration happens AFTER first navigation)
        plan = await self.planner.create_plan(task, context={}, memory=self.memory)
        self._current_plan = plan
        self.planner.store_plan_in_memory(plan, self.memory)

        self._report_progress(
            f"Plan created: {len(plan.phases)} phases, "
            f"{sum(len(p.goals) for p in plan.phases)} goals"
        )

        # Execute each phase
        try:
            for phase_idx in range(len(plan.phases)):
                phase = plan.phases[phase_idx]
                phase.mark_in_progress()
                self.planner.update_plan_in_memory(plan, self.memory)

                self._report_progress(
                    f"Phase {phase_idx + 1}/{len(plan.phases)}: {phase.name}"
                )

                # Execute each goal in the phase
                for goal in phase.goals:
                    goal.mark_in_progress()
                    self.planner.update_plan_in_memory(plan, self.memory)

                    self._report_progress(f"  > Goal: {goal.description}")

                    # Execute ReAct loop for this goal
                    # Agent determines completion via 'complete' tool - no artificial per-goal limits
                    # Reset consecutive failures for this new goal (per-goal tracking in planning mode)
                    goal_start_failures = self._consecutive_failures
                    self._consecutive_failures = 0
                    fast_path_used = False  # Track if we've used fast-path for this goal

                    while not self._should_stop(iteration):
                        iteration += 1
                        self._state = ExecutionState.THINKING

                        # Try fast-path: check if goal can be directly mapped to action
                        # ONLY on first iteration - after that, use LLM reasoning
                        # Respect config flag for fast-path optimization
                        direct_action = None
                        if not fast_path_used and self.config.enable_fast_path_optimization:
                            direct_action = self.goal_interpreter.parse_goal(
                                task, current_goal_desc=goal.description
                            )
                        
                        if direct_action:
                            # Fast-path success! Skip LLM reasoning entirely
                            fast_path_used = True  # Mark as used so we don't try again
                            logger.info(f"[Fast-path] Goal '{goal.description[:50]}...' -> {direct_action.tool_name}({direct_action.parameters})")
                            parse_result = ParseResult(
                                success=True,
                                thought=Thought(content=f"Direct action for goal: {goal.description}"),
                                action=direct_action,
                                format_detected=ParseFormat.JSON,
                                confidence=1.0,
                                raw_content=f"Fast-path: {direct_action.tool_name}",
                            )
                        else:
                            # No fast-path - use LLM reasoning
                            # Build prompt with plan context
                            prompts = self._build_prompt(task)

                            # Get LLM reasoning (with vision if enabled)
                            response = await self._generate_with_optional_vision(
                                prompts=prompts,
                                iteration=iteration,
                            )

                            # Parse response
                            parse_result = self.parser.parse(response.content)

                        if not parse_result.success:
                            self._consecutive_failures += 1
                            self._consecutive_parse_failures += 1
                            logger.warning(f"Parse failed ({self._consecutive_parse_failures}x): {parse_result.error}")
                            continue
                        
                        # Reset parse failure counter on successful parse
                        self._consecutive_parse_failures = 0

                        # Create step record
                        step = ReActStep(
                            step_number=iteration,
                            thought=parse_result.thought,
                            action=parse_result.action,
                        )

                        # Handle action execution
                        if parse_result.action:
                            # Record action for loop detection BEFORE execution
                            self._record_action_for_loop_detection(parse_result.action)
                            
                            observation = await self._execute_action(parse_result.action, step)
                            step.observation = observation
                            
                            # TRIGGER EXPLORATION after EVERY successful navigation (vision models only)
                            if (observation.success and 
                                parse_result.action.tool_name in ("navigate", "goto") and 
                                self._should_explore_page(task)):
                                self._report_progress("Exploring page structure...")
                                exploration_success = await self._explore_current_page()
                                if exploration_success:
                                    self._report_progress("Page exploration complete")
                                else:
                                    logger.warning("Page exploration failed - continuing without it")

                            # Record in memory
                            self.memory.record_cycle(
                                thought=parse_result.thought,
                                action=parse_result.action,
                                observation=observation,
                                outcome=ExecutionOutcome.SUCCESS if observation.success else ExecutionOutcome.FAILURE,
                            )

                        self._steps.append(step)
                        self._current_step = step

                        # Call step callback if provided
                        if step_callback:
                            result = await step_callback(step)
                            if result is False:
                                logger.info("Execution stopped by step callback (circuit breaker)")
                                
                                # Try replanning instead of immediately failing
                                if self.config.enable_self_reflection and self._current_plan:
                                    self._report_progress(" Circuit breaker triggered - replanning...")
                                    
                                    try:
                                        # Adapt plan with failure context
                                        adapted_plan = await self.planner.adapt_plan(
                                            self._current_plan,
                                            {"reason": "Circuit breaker: stagnation detected", 
                                             "phase": phase.name,
                                             "goal": goal.description}
                                        )
                                        self._current_plan = adapted_plan
                                        self.planner.update_plan_in_memory(adapted_plan, self.memory)
                                        
                                        # Reset consecutive failures and continue with new plan
                                        self._consecutive_failures = 0
                                        self._report_progress(" Replan complete - continuing with adapted strategy")
                                        
                                        # Mark current goal as replanning and continue
                                        goal.status = GoalStatus.REPLANNING
                                        break  # Break inner loop to apply new plan
                                    except Exception as e:
                                        logger.error(f"Replanning failed: {e}")
                                        self._task_failed = True
                                        break
                                else:
                                    # Replanning not enabled or no plan - mark as failed
                                    self._task_failed = True
                                    break

                        # Check if goal is complete
                        if parse_result.action and parse_result.action.tool_name == "complete":
                            goal.mark_completed()
                            self.planner.update_plan_in_memory(plan, self.memory)
                            self._report_progress(f"  [ok] Goal completed: {goal.description}")
                            break

                        # Check if goal failed
                        if parse_result.action and parse_result.action.tool_name == "fail":
                            error_msg = parse_result.action.parameters.get("reason", "Goal failed")
                            goal.mark_failed(error_msg)
                            self.planner.update_plan_in_memory(plan, self.memory)
                            break
                        
                        # Auto-complete for successful fast-path actions
                        # Fast-path bypasses LLM so agent can't call 'complete' naturally
                        # If fast-path action succeeded, consider goal complete
                        if direct_action and observation and observation.success:
                            # Fast-path action succeeded - mark goal complete
                            goal.mark_completed()
                            self.planner.update_plan_in_memory(plan, self.memory)
                            self._report_progress(f"  [ok] Goal completed (fast-path): {goal.description}")
                            break

                    # If goal completed, continue to next goal
                    if goal.status == GoalStatus.COMPLETED:
                        # Restore global failure count (don't let previous goal failures affect next goal)
                        self._consecutive_failures = goal_start_failures
                        continue
                    
                    # If goal still in progress, it failed to complete properly
                    # (LLM must explicitly call 'complete' tool or goal remains incomplete)
                    if goal.status == GoalStatus.IN_PROGRESS:
                        logger.warning(f"Goal not completed: {goal.description}")
                        goal.mark_failed("Goal did not complete - no 'complete' tool call")
                        self._report_progress(f"  [fail] Goal incomplete: {goal.description}")
                    
                    # Restore global failure count before continuing to next goal
                    self._consecutive_failures = goal_start_failures

                # Check if phase is complete
                if phase.are_all_goals_completed():
                    phase.mark_completed()
                    self.planner.update_plan_in_memory(plan, self.memory)
                    self._report_progress(f"[ok] Phase {phase_idx + 1} completed: {phase.name}")
                    # Advance to next phase only on success
                    plan.advance_to_next_phase()
                elif phase.has_failed_goals():
                    phase.mark_failed("Some goals failed")
                    self.planner.update_plan_in_memory(plan, self.memory)

                    # Try to adapt plan
                    if self.config.enable_self_reflection:
                        self._report_progress(" Adapting plan due to failures...")
                        plan = await self.planner.adapt_plan(
                            plan,
                            {"phase": phase.name, "reason": "Goals failed"}
                        )
                        self._current_plan = plan
                        self.planner.update_plan_in_memory(plan, self.memory)
                        
                        # CRITICAL: Reset task_failed flag after successful adaptation
                        # Otherwise _should_stop() will immediately exit on next goal
                        self._task_failed = False
                        self._error_message = None
                        
                        # Skip remaining failed phases and continue with adapted plan
                        # The adapted plan should have already advanced to appropriate phase
                    
                    # Advance to next phase (adapted plan determines what's next)
                    plan.advance_to_next_phase()

            # All phases complete - extract final result
            self._task_completed = True
            self._state = ExecutionState.COMPLETED
            
            # Extract result from execution history
            self._final_result = self._extract_final_result()
            
            self._report_progress(" All phases completed successfully!")

        except asyncio.CancelledError:
            self._state = ExecutionState.CANCELLED
            logger.info("Task execution cancelled")
        except Exception as e:
            self._state = ExecutionState.FAILED
            self._error_message = str(e)
            logger.exception(f"Planned execution failed: {e}")

        execution_time = (time.time() - start_time) * 1000
        result = self._create_result(iteration, execution_time)
        
        # Add plan metadata to result
        result.metadata["execution_plan"] = plan.to_dict()
        result.metadata["planning_enabled"] = True
        
        return result

    def _reset_state(self) -> None:
        """Reset internal state for a new execution."""
        self._state = ExecutionState.IDLE
        self._current_step = None
        self._steps = []
        self._consecutive_failures = 0
        self._consecutive_parse_failures = 0  # Track parse failures separately
        self._task_completed = False
        self._task_failed = False
        self._final_result = None
        self._error_message = None
        self._current_plan = None
        self._vision_override = None  # Reset vision override for new execution
        
        # Loop detection state
        self._action_history: List[str] = []  # Recent action signatures for loop detection
        self._state_hashes: List[str] = []  # Page state hashes for stagnation detection
        
        # Site exploration tracking
        self._is_site_exploration = False
        self._current_task = ""
        self._sitemap_limits: Optional[SitemapLimits] = None
        
        # Parallel exploration tracking
        self._parallel_exploration_done = False
    
    async def _preliminary_sitemap_check_async(self, task: str) -> None:
        """
        Quick preliminary check if task likely involves site exploration.
        
        This is Phase 1 - just detects if we should pay attention to exploration.
        The actual limits are determined AFTER first page is explored (Phase 2).
        
        Args:
            task: User's task description
        """
        self._current_task = task
        
        # Quick LLM check without page context
        self._report_progress(" Preliminary task analysis...")
        
        is_exploration = await is_site_exploration_task_async(task, self.llm)
        
        if not is_exploration:
            self._is_site_exploration = False
            logger.info("[SitemapGraph] Preliminary: NOT a site exploration task")
            return
        
        # Mark as potential exploration - limits will be set after first page
        self._is_site_exploration = True
        self._sitemap_limits = None  # Will be set after first page with context
        
        # Clear link filter memory for fresh analysis in this session
        reset_link_filter_memory()
        
        self._report_progress(" Site exploration detected - will analyze depth after homepage")
        logger.info("[SitemapGraph] Preliminary: Site exploration likely - awaiting page context")
    
    async def _analyze_and_init_sitemap_with_context_async(
        self, 
        url: str, 
        page_map: Optional[Any] = None
    ) -> None:
        """
        Phase 2: Analyze exploration intent WITH page context and initialize sitemap.
        
        Called AFTER first page exploration to make an INFORMED decision about
        exploration depth based on actual site structure.
        
        Args:
            url: The homepage URL that was navigated to
            page_map: PageMap with page analysis results
        """
        if not self._is_site_exploration:
            return
        
        # Already initialized - skip
        if self.memory.has_sitemap_graph():
            return
        
        # Build page context from PageMap for informed analysis
        page_context = {
            'url': url,
            'title': '',
            'summary': '',
            'nav_links': [],
            'sections': []
        }
        
        if page_map:
            try:
                page_context['title'] = getattr(page_map, 'title', '') or ''
                page_context['summary'] = getattr(page_map, 'summary', '') or ''
                
                # Extract sections (PageSection dataclass objects)
                sections = getattr(page_map, 'sections', [])
                if sections and isinstance(sections, list):
                    extracted_sections = []
                    for s in sections:
                        try:
                            # PageSection has .name and .type attributes
                            name = getattr(s, 'name', '')
                            if not name:
                                section_type = getattr(s, 'type', None)
                                if section_type and hasattr(section_type, 'value'):
                                    name = section_type.value
                                else:
                                    name = str(section_type) if section_type else 'Unknown'
                            extracted_sections.append({'name': name})
                        except Exception:
                            continue
                    page_context['sections'] = extracted_sections
                
                # Extract navigation links from PageMap's DOM data
                dom_links = getattr(page_map, 'dom_navigation_links', {}) or {}
                if dom_links and isinstance(dom_links, dict):
                    raw_nav_links = []
                    all_links = dom_links.get('all_links', []) or []
                    for link in all_links[:30]:  # Top 30 links (before filtering)
                        if isinstance(link, dict):
                            href = link.get('href', '')
                            text = link.get('text', '')
                            if href and text and not href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                                raw_nav_links.append({'text': text, 'href': href, 'url': href})
                    
                    # Use LLM to filter out language variants and duplicates
                    if raw_nav_links:
                        self._report_progress(" Filtering duplicate/language variant links...")
                        filtered_links = await filter_navigation_links_async(
                            links=raw_nav_links,
                            llm_provider=self.llm,
                            current_url=url,
                            task=self._current_task
                        )
                        page_context['nav_links'] = filtered_links
                    else:
                        page_context['nav_links'] = []
            except Exception as e:
                logger.warning(f"[SitemapGraph] Error extracting page context: {e}")
                # Continue with partial context
        
        # Now analyze with INFORMED context
        self._report_progress(" Analyzing exploration scope with page context...")
        
        limits = await analyze_exploration_intent_async(
            task=self._current_task,
            llm_provider=self.llm,
            page_context=page_context
        )
        self._sitemap_limits = limits
        
        # Initialize the sitemap graph with informed limits
        self.memory.init_sitemap_graph(url, limits)
        
        self._report_progress(
            f" Sitemap initialized (depth={limits.max_depth}, "
            f"L1={limits.max_level1_pages}, L2={limits.max_level2_pages})"
        )
        logger.info(f"[SitemapGraph] Initialized with homepage: {url} (with informed limits)")
    
    async def _update_sitemap_after_navigation_async(self, url: str, page_map: Optional[Any] = None) -> None:
        """
        Update SitemapGraph after a successful navigation.
        
        Called after page exploration completes to:
        1. Mark visited pages
        2. Add discovered links from PageMap (filtered for duplicates/language variants)
        
        Note: Sitemap initialization happens separately via _analyze_and_init_sitemap_with_context_async
        
        Args:
            url: The URL that was navigated to
            page_map: Optional PageMap with page analysis results
        """
        if not self._is_site_exploration:
            return
        
        graph = self.memory.get_sitemap_graph()
        if not graph:
            # Graph not initialized yet - will be done by _analyze_and_init_sitemap_with_context_async
            return
        
        # Extract info from PageMap if available
        title = ""
        summary = ""
        section_count = 0
        raw_links = []
        
        if page_map:
            title = getattr(page_map, 'title', '')
            summary = getattr(page_map, 'summary', '')
            sections = getattr(page_map, 'sections', [])
            section_count = len(sections) if isinstance(sections, list) else 0
            
            # Extract navigation links from PageMap's DOM data
            dom_links = getattr(page_map, 'dom_navigation_links', {})
            if dom_links:
                # Prioritize main navigation links
                for link in dom_links.get('all_links', []):
                    href = link.get('href', '')
                    text = link.get('text', '')
                    if href and not href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                        raw_links.append({'url': href, 'text': text, 'href': href})
        
        # Mark page as visited
        graph.mark_visited(
            url=url,
            title=title,
            summary=summary,
            section_count=section_count,
            page_map_stored=self.memory.has_page_map(url)
        )
        
        # Filter discovered links to remove language variants and duplicates
        # This uses the link filter's memory to avoid redundant LLM calls
        if raw_links:
            filtered_links = await filter_navigation_links_async(
                links=raw_links,
                llm_provider=self.llm,
                current_url=url,
                task=self._current_task
            )
            
            # Add filtered links to sitemap (graph handles depth limits)
            if filtered_links:
                added = graph.add_discovered_links(url, filtered_links, LinkType.MAIN_NAV)
                if added > 0:
                    logger.info(f"[SitemapGraph] Added {added} new pages to explore (after filtering)")
        
        # Log exploration status
        logger.info(
            f"[SitemapGraph] Status: {graph.visited_count}/{graph.total_count} visited, "
            f"{graph.pending_count} pending"
        )

    def _should_stop(self, iteration: int) -> bool:
        """
        Check if execution should stop.
        
        Includes comprehensive loop detection to prevent infinite loops:
        - Max iterations check
        - Consecutive failures check
        - Repeated action detection
        - Same tool loop detection
        - Parse failure limits

        Args:
            iteration: Current iteration count

        Returns:
            True if execution should stop
        """
        # Max iterations reached
        if iteration >= self.config.max_iterations:
            logger.warning(f"Max iterations ({self.config.max_iterations}) reached")
            return True

        # Task completed or failed
        if self._task_completed or self._task_failed:
            return True

        # Too many consecutive failures
        if self._consecutive_failures >= self.config.safety.max_consecutive_failures:
            logger.warning(f"Max consecutive failures ({self.config.safety.max_consecutive_failures}) reached")
            self._task_failed = True
            self._error_message = "Too many consecutive failures"
            return True
        
        # Too many consecutive parse failures (LLM output loop)
        if self._consecutive_parse_failures >= self.config.safety.max_consecutive_parse_failures:
            logger.warning(f"Max consecutive parse failures ({self.config.safety.max_consecutive_parse_failures}) reached - LLM may be in output loop")
            self._task_failed = True
            self._error_message = "Too many consecutive parse failures - LLM output loop detected"
            return True
        
        # Loop detection (if enabled)
        if self.config.safety.enable_loop_detection:
            loop_detected, loop_reason = self._detect_action_loop()
            if loop_detected:
                logger.warning(f"Loop detected: {loop_reason}")
                self._task_failed = True
                self._error_message = f"Action loop detected: {loop_reason}"
                return True

        return False
    
    def _detect_action_loop(self) -> tuple[bool, str]:
        """
        Detect if the agent is stuck in an action loop.
        
        Checks for:
        - Same action (tool + params) repeated N times
        - Same tool called N times in a row
        - State not changing over window
        
        Returns:
            Tuple of (loop_detected, reason)
        """
        if not self._steps:
            return False, ""
        
        safety = self.config.safety
        history_size = safety.action_history_size
        recent_steps = self._steps[-history_size:] if len(self._steps) >= history_size else self._steps
        
        # Check for repeated identical actions
        if len(recent_steps) >= safety.max_repeated_actions:
            recent_actions = []
            for step in recent_steps:
                if step.action:
                    # Create action signature (tool + key params)
                    sig = self._get_action_signature(step.action)
                    recent_actions.append(sig)
            
            if recent_actions:
                # Check if last N actions are identical
                last_n = recent_actions[-safety.max_repeated_actions:]
                if len(set(last_n)) == 1:  # All same
                    return True, f"Same action '{last_n[0]}' repeated {safety.max_repeated_actions} times"
        
        # Check for same tool called repeatedly
        if len(recent_steps) >= safety.max_same_tool_calls:
            recent_tools = []
            for step in recent_steps:
                if step.action:
                    recent_tools.append(step.action.tool_name)
            
            if recent_tools:
                last_n = recent_tools[-safety.max_same_tool_calls:]
                if len(set(last_n)) == 1:  # All same tool
                    tool_name = last_n[0]
                    # Allow certain tools to repeat:
                    # - extract_text: for different selectors
                    # - screenshot, wait: utility operations
                    # - navigate: for site exploration (visiting multiple pages)
                    # - scroll: for page exploration
                    # - complete: for completing different goals/phases in planning mode
                    allowed_repeat_tools = ('extract_text', 'screenshot', 'wait', 'navigate', 'scroll', 'complete')
                    if tool_name not in allowed_repeat_tools:
                        return True, f"Tool '{tool_name}' called {safety.max_same_tool_calls} times in a row"
        
        return False, ""
    
    def _get_action_signature(self, action: Action) -> str:
        """
        Create a signature for an action for comparison.
        
        Args:
            action: The action to create signature for
            
        Returns:
            String signature like "click:selector=#btn"
        """
        import hashlib
        import json
        
        # Include tool name and sorted params
        params_str = json.dumps(action.parameters, sort_keys=True, default=str)
        sig = f"{action.tool_name}:{params_str}"
        
        # Hash if too long
        if len(sig) > 100:
            sig_hash = hashlib.md5(sig.encode()).hexdigest()[:16]
            return f"{action.tool_name}:{sig_hash}"
        
        return sig
    
    def _record_action_for_loop_detection(self, action: Action) -> None:
        """
        Record an action for loop detection tracking.
        
        Args:
            action: The action being executed
        """
        if not self.config.safety.enable_loop_detection:
            return
        
        sig = self._get_action_signature(action)
        self._action_history.append(sig)
        
        # Trim history to configured size
        max_size = self.config.safety.action_history_size
        if len(self._action_history) > max_size:
            self._action_history = self._action_history[-max_size:]

    def _build_prompt(self, task: str) -> Dict[str, str]:
        """
        Build the prompts for LLM with task and context using PromptManager.
        
        Selects the appropriate prompt template based on the configured
        reasoning strategy (standard ReAct, Chain of Thought, or Tree of Thoughts).
        Includes execution plan context if planning is active.

        Args:
            task: The task description

        Returns:
            Dictionary with 'system' and 'user' prompt strings
        """
        # Get memory context
        memory_context = self.memory.format_for_prompt()

        # Get available tools description
        tools_prompt = self.tool_registry.generate_tools_prompt()
        
        # Get plan context if available
        plan_context = ""
        if self._current_plan:
            plan_context = self._current_plan.format_for_prompt()
        
        # Select prompt template based on reasoning strategy
        prompt_name = self._select_prompt_for_strategy()
        
        # Get sitemap exploration status if active
        sitemap_status = ""
        if self._is_site_exploration and self.memory.has_sitemap_graph():
            sitemap_status = self.memory.get_sitemap_status()
        
        # Build prompt variables
        prompt_vars = {
            "task": task,
            "available_tools": tools_prompt,
            "memory_context": memory_context or "",
            "plan_context": plan_context,
            "sitemap_status": sitemap_status,
            "vision_enabled": ModelCapability.VISION in self.model_info.capabilities,
        }
        
        # Add strategy-specific variables
        if self.config.reasoning_strategy == ReasoningStrategy.TREE_OF_THOUGHT:
            # For ToT, include previous failures to avoid repeating them
            if self.memory:
                failures = self.memory.short_term.get_by_outcome(ExecutionOutcome.FAILURE)
                if failures:
                    # Use configurable failure history size
                    history_size = self.config.tot_failure_history_size
                    failure_text = "\n".join([
                        f"- {f.action.tool_name}: {f.error_message or 'Failed'}"
                        for f in failures[-history_size:]
                    ])
                    prompt_vars["previous_failures"] = failure_text
        
        # Use PromptManager to get templated prompts
        try:
            prompts = self.prompt_manager.get_prompt(
                prompt_name,
                **prompt_vars
            )
        except Exception as e:
            logger.warning(
                f"Failed to load prompt '{prompt_name}', falling back to react_agent: {e}"
            )
            # Fallback to standard react_agent prompt
            prompts = self.prompt_manager.get_prompt(
                "react_agent",
                task=task,
                available_tools=tools_prompt,
                memory_context=memory_context or "",
            )
        
        # DEBUG: Log memory context to verify it contains results
        if memory_context:
            logger.debug(f"[PROMPT DEBUG] Memory context ({len(memory_context)} chars):\n{memory_context[:1000]}...")
        else:
            logger.warning("[PROMPT DEBUG] Memory context is EMPTY!")

        return prompts
    
    def _select_prompt_for_strategy(self) -> str:
        """
        Select the appropriate prompt template based on reasoning strategy.
        
        Returns:
            Prompt template name
        """
        strategy_to_prompt = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: "react_chain_of_thought",
            ReasoningStrategy.TREE_OF_THOUGHT: "react_tree_of_thoughts",
            ReasoningStrategy.REACT_STANDARD: "react_agent",
            ReasoningStrategy.SELF_REFLECTION: "react_agent",  # Use standard with memory
            ReasoningStrategy.PLAN_AND_SOLVE: "react_agent",   # Use standard for now
        }
        
        prompt_name = strategy_to_prompt.get(
            self.config.reasoning_strategy,
            "react_agent"  # Default fallback
        )
        
        logger.debug(
            f"Selected prompt '{prompt_name}' for strategy {self.config.reasoning_strategy.value}"
        )
        
        return prompt_name
    
    async def _generate_with_optional_vision(
        self,
        prompts: Dict[str, str],
        iteration: int,
    ) -> Any:
        """
        Generate LLM response with optional vision integration using STRUCTURED OUTPUT.
        
        ALWAYS uses structured output (JSON mode) for deterministic response format.
        This eliminates parsing errors and ensures consistent action/thought extraction.
        
        Includes automatic repair mechanism: if the LLM returns malformed JSON that
        doesn't match the schema, we ask the LLM to fix it using the original context.
        
        For vision-enabled models, conditionally captures and includes screenshots.
        For text-only models, uses standard structured text generation.
        
        Args:
            prompts: System and user prompts
            iteration: Current iteration number
            
        Returns:
            LLM response object with JSON content
        """
        import json
        from dataclasses import dataclass as dc
        
        @dc
        class StructuredResponse:
            """Response wrapper for structured output."""
            content: str
            finish_reason: str = "stop"
        
        max_repair_attempts = self.config.llm.max_repair_attempts
        
        # Check if vision should be used
        if self._should_use_vision(iteration):
            try:
                # Capture screenshot
                screenshot_bytes = await self.page.screenshot(full_page=False)
                logger.info(f"[VISION] Captured screenshot ({len(screenshot_bytes) // 1024}KB) for iteration {iteration}")
                
                # Calculate max_tokens dynamically based on image size and prompts
                if self.config.llm.enable_dynamic_tokens:
                    from flybrowser.agents.config import calculate_max_tokens_for_vision_response
                    max_tokens = calculate_max_tokens_for_vision_response(
                        system_prompt=prompts["system"],
                        user_prompt=prompts["user"],
                        image_size_bytes=len(screenshot_bytes),
                        context_tokens=0,
                        safety_margin=self.config.llm.token_safety_margin
                    )
                    logger.info(f"[VISION] Dynamically calculated max_tokens={max_tokens} "
                               f"(image: {len(screenshot_bytes)//1024}KB, margin: {self.config.llm.token_safety_margin})")
                else:
                    max_tokens = getattr(self.config.llm, 'reasoning_vision_max_tokens', None)
                    if max_tokens is None:
                        max_tokens = max(2048, self.config.llm.reasoning_max_tokens * 2)
                logger.debug(f"[VISION] Using max_tokens={max_tokens}")
                
                # Further increase for ToT strategy
                if self.config.reasoning_strategy == ReasoningStrategy.TREE_OF_THOUGHT:
                    max_tokens = max(max_tokens, 4096)
                    logger.debug(f"[VISION] Increased to {max_tokens} for Tree-of-Thought")
                
                # Validate image
                if not screenshot_bytes or len(screenshot_bytes) == 0:
                    logger.error("[VISION] Screenshot is empty! Falling back to text-only")
                    raise ValueError("Empty screenshot")
                
                logger.debug(
                    f"[VISION] Sending structured request: "
                    f"image={len(screenshot_bytes)//1024}KB, max_tokens={max_tokens}"
                )
                
                # ALWAYS use structured output with vision
                structured_data = await self.llm.generate_structured_with_vision(
                    prompt=prompts["user"],
                    image_data=screenshot_bytes,
                    schema=REACT_RESPONSE_SCHEMA,
                    system_prompt=prompts["system"],
                    temperature=self.config.llm.reasoning_temperature,
                    max_tokens=max_tokens,
                )
                
                # Check for error in response
                if "error" in structured_data:
                    logger.error(f"[VISION] Structured output error: {structured_data.get('error')}")
                    raise ValueError(structured_data.get('error', 'Unknown error'))
                
                # Validate and repair if needed
                structured_data = await self._validate_and_repair_response(
                    structured_data=structured_data,
                    original_prompt=prompts["user"],
                    system_prompt=prompts["system"],
                    max_attempts=max_repair_attempts,
                )
                
                logger.debug(f"[VISION] Structured response keys: {list(structured_data.keys())}")
                return StructuredResponse(content=json.dumps(structured_data))
                
            except Exception as e:
                logger.warning(f"[VISION] Failed: {e}, falling back to text-only structured")
                # Fall through to text-only structured generation
        
        # Text-only structured generation
        if self.config.llm.enable_dynamic_tokens:
            from flybrowser.agents.config import calculate_max_tokens_for_response, estimate_tokens
            system_tokens = estimate_tokens(prompts["system"])
            user_tokens = estimate_tokens(prompts["user"])
            max_tokens = calculate_max_tokens_for_response(
                system_prompt_tokens=system_tokens,
                user_prompt_tokens=user_tokens,
                context_tokens=0,
                safety_margin=self.config.llm.token_safety_margin
            )
            logger.debug(f"[TEXT] Calculated max_tokens={max_tokens}")
        else:
            max_tokens = self.config.llm.reasoning_max_tokens
            logger.debug(f"[TEXT] Using configured max_tokens={max_tokens}")
        
        if self.config.reasoning_strategy == ReasoningStrategy.TREE_OF_THOUGHT:
            max_tokens = max(max_tokens, 4096)
            logger.debug(f"[ToT] Increased to {max_tokens}")
        
        # ALWAYS use structured output for text
        logger.debug("[TEXT] Using structured output with JSON mode")
        structured_data = await self.llm.generate_structured(
            prompt=prompts["user"],
            schema=REACT_RESPONSE_SCHEMA,
            system_prompt=prompts["system"],
            temperature=self.config.llm.reasoning_temperature,
        )
        
        # Validate and repair if needed
        structured_data = await self._validate_and_repair_response(
            structured_data=structured_data,
            original_prompt=prompts["user"],
            system_prompt=prompts["system"],
            max_attempts=max_repair_attempts,
        )
        
        logger.debug(f"[TEXT] Structured response keys: {list(structured_data.keys())}")
        return StructuredResponse(content=json.dumps(structured_data))
    
    def _auto_fix_action_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically fix common action format errors.
        
        The LLM often produces incorrect formats like:
          {"action": {"click": {"selector": "..."}}}
        
        When it should be:
          {"action": {"tool": "click", "parameters": {"selector": "..."}}}
        
        This method detects and fixes these errors without LLM calls.
        
        Args:
            data: Raw structured data from LLM
            
        Returns:
            Fixed data (or original if no fix needed)
        """
        if "action" not in data:
            return data
        
        action = data["action"]
        
        # Already correct format
        if "tool" in action and "parameters" in action:
            return data
        
        # Detect common error: {"action": {"tool_name": {...params}}}
        # Known tool names that might appear incorrectly
        known_tools = {
            "click", "type_text", "navigate", "extract_text", "scroll",
            "screenshot", "wait", "complete", "fail", "get_page_state",
            "go_back", "go_forward", "refresh", "hover", "select",
            "extract_table", "get_attribute", "page_explorer"
        }
        
        # Check if any key in action is a known tool name
        for key in action:
            if key in known_tools:
                # Found incorrect format - fix it
                params = action[key]
                if not isinstance(params, dict):
                    params = {}
                
                logger.info(f"[AUTO-FIX] Fixed action format: {{{key}: ...}} -> {{tool: '{key}', parameters: ...}}")
                data["action"] = {
                    "tool": key,
                    "parameters": params
                }
                return data
        
        return data
    
    async def _validate_and_repair_response(
        self,
        structured_data: Dict[str, Any],
        original_prompt: str,
        system_prompt: str,
        max_attempts: int = 2,
    ) -> Dict[str, Any]:
        """
        Validate structured response and attempt repair if malformed.
        
        When the LLM returns JSON that doesn't match the expected schema,
        we first try automatic fixes for common errors, then ask the LLM
        to repair if auto-fix doesn't work.
        
        Args:
            structured_data: The response data to validate
            original_prompt: Original user prompt for context
            system_prompt: System prompt for context  
            max_attempts: Maximum repair attempts
            
        Returns:
            Validated (and possibly repaired) response data
            
        Raises:
            ValueError: If validation fails after all repair attempts
        """
        import json
        
        # First try auto-fix for common format errors
        structured_data = self._auto_fix_action_format(structured_data)
        
        # Then validate
        is_valid, errors = validate_react_response(structured_data)
        
        if is_valid:
            return structured_data
        
        # Need repair
        logger.warning(f"[REPAIR] Response validation failed: {errors}")
        
        for attempt in range(max_attempts):
            logger.info(f"[REPAIR] Attempting repair {attempt + 1}/{max_attempts}")
            
            # Build repair prompt with context
            malformed_output = json.dumps(structured_data, indent=2)
            repair_prompt = build_repair_prompt(
                original_prompt=original_prompt,
                malformed_output=malformed_output,
                validation_errors=errors,
                schema=REACT_RESPONSE_SCHEMA,
            )
            
            # Ask LLM to repair (use lower temperature for more deterministic fix)
            try:
                repaired_data = await self.llm.generate_structured(
                    prompt=repair_prompt,
                    schema=REACT_RESPONSE_SCHEMA,
                    system_prompt="You are a JSON repair assistant. Fix the malformed JSON to match the required schema.",
                    temperature=self.config.llm.repair_temperature,
                )
                
                # Validate repaired response
                is_valid, errors = validate_react_response(repaired_data)
                
                if is_valid:
                    logger.info(f"[REPAIR] Successfully repaired on attempt {attempt + 1}")
                    return repaired_data
                else:
                    logger.warning(f"[REPAIR] Attempt {attempt + 1} still invalid: {errors}")
                    structured_data = repaired_data  # Use for next attempt
                    
            except Exception as e:
                logger.error(f"[REPAIR] Attempt {attempt + 1} failed with error: {e}")
        
        # All repair attempts failed
        error_msg = f"Response validation failed after {max_attempts} repair attempts. Errors: {errors}"
        logger.error(f"[REPAIR] {error_msg}")
        raise ValueError(error_msg)
    
    def _should_use_vision(self, iteration: int) -> bool:
        """
        Decide whether to include vision (screenshot) for this iteration.
        
        Priority order:
        1. Vision override (from SDK method's use_vision parameter)
        2. Model capability check
        3. Operation mode strategy
        
        Mode-aware strategy optimized for performance and cost:
        - NAVIGATE: High frequency (exploration focus)
        - EXECUTE: Minimal (speed focus)
        - SCRAPE: Balanced (structure understanding)
        - RESEARCH: Balanced (content discovery)
        - AUTO: Default balanced approach
        
        Args:
            iteration: Current iteration number
            
        Returns:
            True if screenshot should be captured and sent to LLM
        """
        # Check for explicit vision override from SDK method
        vision_override = getattr(self, '_vision_override', None)
        if vision_override is not None:
            if vision_override:
                # User wants vision - check if model supports it
                if ModelCapability.VISION not in self.model_info.capabilities:
                    logger.debug("[VISION] Override requested but model doesn't support vision")
                    return False
                logger.debug(f"[VISION] Using override: enabled (iteration {iteration})")
                return True
            else:
                # User explicitly disabled vision
                logger.debug(f"[VISION] Using override: disabled (iteration {iteration})")
                return False
        
        # Check if vision is available on the model
        if ModelCapability.VISION not in self.model_info.capabilities:
            return False
        
        # First iteration - most modes use vision to understand initial state
        if iteration == 1:
            logger.debug(f"[VISION:{self.operation_mode.value}] Trigger: first iteration")
            return True
        
        # Mode-specific vision strategies
        if self.operation_mode == OperationMode.NAVIGATE:
            # NAVIGATE: High frequency for comprehensive exploration
            # Use vision every 2 iterations or after navigation
            if iteration % 2 == 0:
                logger.debug(f"[VISION:NAVIGATE] Trigger: periodic (iteration {iteration})")
                return True
            if self._last_action_was_navigation():
                logger.debug("[VISION:NAVIGATE] Trigger: after navigation")
                return True
        
        elif self.operation_mode == OperationMode.EXECUTE:
            # EXECUTE: Minimal vision for speed (only failures)
            # Skip periodic checks, only use on errors
            if self._consecutive_failures >= 2:
                logger.debug(f"[VISION:EXECUTE] Trigger: after {self._consecutive_failures} failures")
                return True
            # No periodic vision for EXECUTE mode
            return False
        
        elif self.operation_mode == OperationMode.SCRAPE:
            # SCRAPE: Vision for page structure, not every iteration
            # Less frequent than NAVIGATE, focus on content regions
            if self._last_action_was_navigation():
                logger.debug("[VISION:SCRAPE] Trigger: after navigation for structure")
                return True
            if iteration % 5 == 0:
                logger.debug(f"[VISION:SCRAPE] Trigger: periodic structure check (iteration {iteration})")
                return True
        
        elif self.operation_mode == OperationMode.RESEARCH:
            # RESEARCH: Balanced vision for content discovery
            # Similar to default but slightly more frequent
            if iteration % 3 == 0:
                logger.debug(f"[VISION:RESEARCH] Trigger: periodic (iteration {iteration})")
                return True
            if self._last_action_was_navigation():
                logger.debug("[VISION:RESEARCH] Trigger: after navigation")
                return True
        
        else:  # OperationMode.AUTO
            # AUTO: Default balanced approach
            if iteration % 3 == 0:
                logger.debug(f"[VISION:AUTO] Trigger: periodic (iteration {iteration})")
                return True
            if self._last_action_was_navigation():
                logger.debug("[VISION:AUTO] Trigger: after navigation")
                return True
        
        # Universal triggers across all modes
        # After consecutive failures - visual analysis might help (except EXECUTE which checked above)
        if self.operation_mode != OperationMode.EXECUTE and self._consecutive_failures >= 2:
            logger.debug(f"[VISION:{self.operation_mode.value}] Trigger: after {self._consecutive_failures} failures")
            return True
        
        return False
    
    def _last_action_was_navigation(self) -> bool:
        """
        Check if the last action was a navigation action.
        
        Returns:
            True if last action changed the page (navigate, click with navigation, etc.)
        """
        if not self._steps:
            return False
        
        last_step = self._steps[-1]
        if not last_step.action:
            return False
        
        # Navigation-related tools that change the page
        navigation_tools = {
            "navigate", "go_back", "go_forward", "refresh",
            "click",  # Clicks often trigger navigation
        }
        
        return last_step.action.tool_name in navigation_tools
    
    def _extract_final_result(self) -> Any:
        """
        Extract final result from execution history.
        
        Compiles observations and memory into a final result object.
        Looks for explicit results from 'complete' tool or compiles from observations.
        
        Returns:
            Final result data or compiled summary
        """
        # Check if last step had explicit result (from 'complete' tool)
        if self._steps:
            last_step = self._steps[-1]
            if last_step.observation and last_step.observation.result:
                # Check if complete tool was called with result
                if last_step.action and last_step.action.tool_name == "complete":
                    explicit_result = last_step.action.parameters.get("result")
                    if explicit_result:
                        logger.info("Extracted result from 'complete' tool call")
                        return explicit_result
                
                # Check if last observation has useful data
                if last_step.observation.result.success and last_step.observation.parsed_data:
                    return last_step.observation.parsed_data
        
        # Compile from memory and execution history
        compiled_result = {
            "status": "completed",
            "summary": self.memory.get_task_summary() if hasattr(self.memory, 'get_task_summary') else "Task completed",
            "observations": [],
            "actions_taken": len(self._steps),
        }
        
        # Add key observations from execution
        for step in self._steps:
            if step.observation and step.observation.result.success:
                compiled_result["observations"].append({
                    "action": step.action.tool_name if step.action else "unknown",
                    "summary": step.observation.summary[:200] if step.observation.summary else "",
                    "data": step.observation.parsed_data
                })
        
        # Limit observations to most relevant (last 10)
        if len(compiled_result["observations"]) > 10:
            compiled_result["observations"] = compiled_result["observations"][-10:]
        
        logger.info(f"Compiled final result from {len(self._steps)} steps")
        return compiled_result

    async def _execute_action(self, action: Action, step: ReActStep) -> Observation:
        """
        Execute an action using the appropriate tool.

        Args:
            action: The action to execute
            step: Current step for recording

        Returns:
            Observation from action execution
        """
        self._state = ExecutionState.ACTING

        # Get tool from registry (pass page controller and element detector)
        tool = self.tool_registry.get_tool(
            action.tool_name,
            page_controller=self.page,
            element_detector=self.element_detector,
        )
        if tool is None:
            error_result = ToolResult.error_result(
                error=f"Tool '{action.tool_name}' not found",
                error_code="TOOL_NOT_FOUND"
            )
            return Observation(
                result=error_result,
                source=action.tool_name,
                raw_output=f"Tool '{action.tool_name}' not found",
                summary=f"Tool '{action.tool_name}' not found",
            )
        
        # Inject LLM provider and config into tool for intelligent features (e.g. obstacle detection)
        tool.llm_provider = self.llm
        tool.agent_config = self.config
        
        # Validate URL parameters for navigation/browse tools
        validator = get_scope_validator()
        if action.tool_name in ("navigate", "goto"):
            url_param = action.parameters.get("url")
            if url_param:
                is_valid, error = validator.validate_url(url_param)
                if not is_valid:
                    error_result = ToolResult.error_result(
                        error=f"Invalid URL: {error}",
                        error_code="INVALID_URL"
                    )
                    return Observation(
                        result=error_result,
                        source=action.tool_name,
                        raw_output=f"Invalid URL: {error}",
                        summary=f"Invalid URL: {error}",
                    )

        # Check if approval required for dangerous actions
        if (
            self.config.require_approval_for_dangerous
            and tool.metadata.safety_level in (SafetyLevel.SENSITIVE, SafetyLevel.DANGEROUS)
        ):
            self._state = ExecutionState.WAITING_APPROVAL

            if self.approval_callback:
                approved = self.approval_callback(action)
                if not approved:
                    error_result = ToolResult.error_result(
                        error="User rejected dangerous action",
                        error_code="APPROVAL_DENIED"
                    )
                    return Observation(
                        result=error_result,
                        source=action.tool_name,
                        raw_output="Action rejected by user",
                        summary="Action rejected by user",
                    )
            else:
                logger.warning(
                    f"Dangerous action '{action.tool_name}' requires approval but no callback provided"
                )

        # Execute the tool
        self._state = ExecutionState.ACTING
        try:
            # Execute tool with parameters as kwargs
            result = await tool.execute(**action.parameters)

            self._state = ExecutionState.OBSERVING
            
            # Store current URL in memory after successful navigation
            if result.success and action.tool_name in ("navigate", "goto"):
                try:
                    current_url = await self.page.get_url()
                    if current_url:
                        self.memory.working.set_scratch("current_url", current_url)
                        logger.debug(f"Stored current URL: {current_url}")
                except Exception as e:
                    logger.debug(f"Could not store current URL: {e}")
            
            # Store extraction results in working memory for persistence
            # This is CRITICAL for text-only mode to have actual data available
            # across LLM reasoning iterations (prevents hallucination)
            if result.success and action.tool_name in (
                "evaluate_javascript", "extract_text", "extract_data", 
                "extract_structured_data", "get_page_content"
            ):
                if result.data:
                    # Use a timestamped key to avoid overwriting previous extractions
                    key = f"extracted_{action.tool_name}_{int(time.time())}"
                    self.memory.working.set_scratch(key, result.data)
                    logger.info(f"Stored extraction result in working memory: {key}")
            
            if result.success:
                self._consecutive_failures = 0  # Reset on success
            else:
                self._consecutive_failures += 1
                
                # Intelligent failure recovery: suggest alternatives
                if self._consecutive_failures >= 2:
                    alternative_suggestion = self._suggest_alternative_approach(action, result)
                    if alternative_suggestion:
                        logger.info(f" Failure recovery: {alternative_suggestion}")
                        result.metadata['alternative_suggestion'] = alternative_suggestion

            # Format the result into a string summary
            if result.data:
                raw_output = str(result.data)
                summary = f"Success: {str(result.data)[:200]}" if result.success else f"Failed: {result.error}"
            else:
                raw_output = result.error or "No output"
                summary = result.error or "Action completed"
            
            # Add alternative suggestion to summary if present
            if not result.success and result.metadata.get('alternative_suggestion'):
                summary += f"\n Alternative: {result.metadata['alternative_suggestion']}"

            return Observation(
                result=result,
                source=action.tool_name,
                raw_output=raw_output,
                parsed_data=result.data,
                summary=summary,
                metadata=result.metadata,
            )
        except Exception as e:
            self._consecutive_failures += 1
            logger.exception(f"Tool execution failed: {e}")
            error_result = ToolResult.error_result(
                error=f"Execution error: {str(e)}",
                error_code="EXECUTION_ERROR"
            )
            
            # Suggest alternatives on exception too
            if self._consecutive_failures >= 2:
                alternative_suggestion = self._suggest_alternative_approach(action, error_result)
                if alternative_suggestion:
                    logger.info(f" Failure recovery: {alternative_suggestion}")
                    error_result.metadata['alternative_suggestion'] = alternative_suggestion
            
            summary = f"Tool execution failed: {str(e)}"
            if error_result.metadata.get('alternative_suggestion'):
                summary += f"\n Alternative: {error_result.metadata['alternative_suggestion']}"
            
            return Observation(
                result=error_result,
                source=action.tool_name,
                raw_output=f"Execution error: {str(e)}",
                summary=summary,
            )

    def _suggest_alternative_approach(self, failed_action: Action, result: ToolResult) -> Optional[str]:
        """
        Suggest alternative approaches when actions fail repeatedly.
        
        Provides intelligent suggestions for:
        - Alternative tools (e.g., search_human instead of search_api)
        - Different parameters
        - Recovery strategies
        
        Args:
            failed_action: The action that failed
            result: The failed result
            
        Returns:
            Suggestion string or None
        """
        tool_name = failed_action.tool_name
        error_code = result.error_code
        error_msg = result.error or ""
        
        # Search tool alternatives
        if tool_name == "search_api":
            if "API" in error_msg or "NO_RESULTS" in error_code or "provider" in error_msg.lower():
                return "Try using 'search_human' tool instead - it doesn't require API keys and simulates human browsing"
        
        elif tool_name == "search_human":
            if "bot" in error_msg.lower() or "captcha" in error_msg.lower() or "sorry" in error_msg.lower():
                return "Google detected automation. Try 'search_human' with engine='duckduckgo' or engine='bing' which are less restrictive"
            if "obstacle" in error_msg.lower() or "timeout" in error_msg.lower():
                return "The page has obstacles. Try navigating directly to a known URL or use a different search engine parameter"
            if "parse" in error_msg.lower() or "extract" in error_msg.lower():
                return "Try changing the 'engine' parameter (google/duckduckgo/bing) as page structure may differ"
        
        # Navigation alternatives
        elif tool_name == "navigate":
            if "timeout" in error_msg.lower():
                return "Page load timeout. Try using 'wait' tool after navigation or increase timeout"
            if "blocked" in error_msg.lower() or "refused" in error_msg.lower():
                return "Navigation blocked. The site may be detecting automation. Consider using human-like tools"
        
        # Click alternatives  
        elif tool_name == "click":
            if "not found" in error_msg.lower() or "timeout" in error_msg.lower():
                return "Element not found. Try using 'get_page_state' first to see available elements, or wait for page to load"
            if "intercepts" in error_msg.lower():
                return "Another element is blocking the click. Check for modals/overlays with 'screenshot' or 'get_page_state'"
        
        # Extraction alternatives
        elif tool_name == "extract_text":
            if "empty" in error_msg.lower() or "no content" in error_msg.lower():
                return "No content found. Try 'screenshot' to see what's on the page, or check if JavaScript rendering is needed"
        
        # Generic suggestions based on consecutive failures
        if self._consecutive_failures >= 3:
            return "Multiple failures detected. Consider: 1) Take a screenshot to see current state, 2) Get page state to understand available elements, 3) Re-plan approach with 'fail' tool and new strategy"
        
        return None
    
    def _check_completion(self, action: Optional[Action]) -> bool:
        """
        Check if the action signals task completion.

        Args:
            action: The action that was executed

        Returns:
            True if task is complete
        """
        if action is None:
            return False

        if action.tool_name == "complete":
            self._task_completed = True
            self._state = ExecutionState.COMPLETED
            self._final_result = action.parameters.get("result")
            logger.info("Task completed successfully")
            return True

        if action.tool_name == "fail":
            self._task_failed = True
            self._state = ExecutionState.FAILED
            self._error_message = action.parameters.get("reason", "Task failed")
            logger.info(f"Task failed: {self._error_message}")
            return True

        return False

    def _should_explore_page(self, task: str) -> bool:
        """
        Determine if current page should be explored after navigation.
        
        ALWAYS explores when vision is enabled to build complete PageMap context.
        This ensures the agent has comprehensive page understanding (sections,
        navigation, content structure) before planning actions.
        
        The operation mode affects HOW we explore (depth/scope), not WHETHER we explore.
        Exploration scope is determined by the mode:
        - NAVIGATE: Full exploration (comprehensive)
        - EXECUTE: Viewport only (fast)
        - SCRAPE: Content-focused (data regions)
        - AUTO: Smart adaptive
        
        Args:
            task: Task description (not used for decision, kept for compatibility)
            
        Returns:
            True if page exploration should be triggered (always when vision enabled)
        """
        # Must have explorer and analyzer initialized (vision-capable models only)
        if not self._page_explorer or not self._page_analyzer:
            return False
        
        # Vision capability is required for page exploration
        if ModelCapability.VISION not in self.model_info.capabilities:
            logger.debug("[PageExploration] Skipped - vision capability required")
            return False
        
        # ALWAYS explore when vision is enabled - mode just affects depth/scope
        logger.debug(
            f"[PageExploration:{self.operation_mode.value}] Enabled - "
            f"building PageMap with {self._get_exploration_scope_description()} scope"
        )
        return True
    
    def _get_exploration_scope_description(self) -> str:
        """Get human-readable description of exploration scope for current mode."""
        scope_descriptions = {
            OperationMode.NAVIGATE: "FULL (comprehensive)",
            OperationMode.EXECUTE: "VIEWPORT (fast)",
            OperationMode.SCRAPE: "CONTENT (data-focused)",
            OperationMode.RESEARCH: "SMART (adaptive)",
            OperationMode.AUTO: "SMART (adaptive)",
        }
        return scope_descriptions.get(self.operation_mode, "SMART")
    
    async def _explore_current_page(self) -> bool:
        """
        Perform systematic page exploration and store understanding in memory.
        
        Returns:
            True if exploration succeeded, False otherwise
        """
        try:
            # Get current URL
            current_url = await self.page.get_url()
            if not current_url:
                logger.warning("Cannot explore: no page loaded")
                return False
            
            # Check if we already have a PageMap for this URL
            if self.memory.has_page_map(current_url):
                logger.info(f"PageMap already exists for {current_url}")
                return True
            
            # Execute page exploration with mode-specific scope
            # Mode determines exploration depth: NAVIGATE=FULL, EXECUTE=VIEWPORT, etc.
            logger.info(
                f"[PageExplorer] Starting {self._get_exploration_scope_description()} "
                f"exploration of {current_url}"
            )
            result = await self._page_explorer.execute(operation_mode=self.operation_mode)
            
            if not result.success:
                logger.warning(f"Page exploration failed: {result.error}")
                return False
            
            # Get PageMap from result metadata
            page_map = result.metadata.get("page_map")
            if not page_map or not page_map.screenshots:
                logger.warning("Page exploration produced no screenshots")
                return False
            
            logger.info(
                f"[PageExplorer] Captured {len(page_map.screenshots)} screenshots, "
                f"coverage: {page_map.get_coverage_percentage():.1f}%"
            )
            
            # Analyze screenshots to build understanding
            logger.info(f"[PageAnalyzer] Analyzing {len(page_map.screenshots)} screenshots...")
            analyzed_map = await self._page_analyzer.analyze_page_map(page_map)
            
            if not analyzed_map:
                logger.warning("Page analysis failed to produce structured understanding")
                # Still store the raw PageMap
                self.memory.store_page_map(current_url, page_map)
                return True
            
            # Store analyzed PageMap in memory
            self.memory.store_page_map(current_url, analyzed_map)
            
            # Phase 2 of sitemap: If this is the FIRST page and site exploration is enabled,
            # analyze intent WITH page context to determine informed limits
            if self._is_site_exploration and not self.memory.has_sitemap_graph():
                await self._analyze_and_init_sitemap_with_context_async(current_url, analyzed_map)
            
            # Update SitemapGraph for site exploration tracking (filtered for language variants)
            await self._update_sitemap_after_navigation_async(current_url, analyzed_map)
            
            # Check if we should trigger parallel exploration for remaining pages
            # This happens AFTER the first page (usually homepage) is fully explored
            if await self._should_trigger_parallel_exploration():
                self._report_progress(" Starting parallel exploration of remaining pages...")
                await self._execute_parallel_site_exploration()
            
            # Log summary
            logger.info(
                f"[PageAnalyzer] Page understanding complete: "
                f"{len(analyzed_map.sections)} sections, "
                f"{len(analyzed_map.navigation_links)} navigation links"
            )
            if analyzed_map.summary:
                logger.info(f"[PageAnalyzer] Summary: {analyzed_map.summary[:150]}...")
            
            return True
            
        except Exception as e:
            logger.exception(f"Page exploration failed: {e}")
            return False
    
    async def _should_trigger_parallel_exploration(self) -> bool:
        """
        Check if parallel exploration should be triggered.
        
        Parallel exploration is triggered when:
        1. This is a site exploration task
        2. Parallel explorer is configured and available
        3. Sitemap has pending pages to explore
        4. We've just finished exploring the first page (homepage)
        
        Returns:
            True if parallel exploration should start
        """
        # Must be site exploration with parallel explorer available
        if not self._is_site_exploration:
            return False
        
        if not self._parallel_explorer:
            return False
        
        if not self.config.parallel_exploration.enable_parallel:
            return False
        
        # Must have sitemap initialized with pending pages
        graph = self.memory.get_sitemap_graph()
        if not graph:
            return False
        
        # Only trigger if we have multiple pending pages
        # (single page doesn't benefit from parallelization)
        if graph.pending_count < 2:
            return False
        
        # Only trigger once - check if we've already done parallel exploration
        if hasattr(self, '_parallel_exploration_done') and self._parallel_exploration_done:
            return False
        
        logger.info(
            f"[ParallelExploration] Triggering: {graph.pending_count} pages pending, "
            f"parallelism factor: {graph.get_parallelism_factor()}"
        )
        return True
    
    async def _execute_parallel_site_exploration(self) -> Optional[ParallelExplorationStats]:
        """
        Execute parallel exploration of all pending pages in the sitemap.
        
        This takes over the site exploration from the normal ReAct loop,
        exploring multiple pages concurrently for significant speedup.
        
        Returns:
            Statistics from the parallel exploration, or None if failed
        """
        if not self._parallel_explorer:
            logger.warning("[ParallelExploration] No parallel explorer available")
            return None
        
        graph = self.memory.get_sitemap_graph()
        if not graph:
            logger.warning("[ParallelExploration] No sitemap graph available")
            return None
        
        # Mark that we've started parallel exploration (prevent re-triggering)
        self._parallel_exploration_done = True
        
        # Set progress callback for parallel explorer
        self._parallel_explorer._progress_callback = self._progress_callback
        
        try:
            stats = await self._parallel_explorer.explore_site(
                sitemap_graph=graph,
                task=self._current_task,
                operation_mode=self.operation_mode,
            )
            
            self._report_progress(
                f" Parallel exploration complete: {stats.pages_explored} pages, "
                f"{stats.pages_failed} failed, "
                f"parallelism: {stats.parallelism_achieved:.1f}x"
            )
            
            return stats
            
        except Exception as e:
            logger.exception(f"[ParallelExploration] Failed: {e}")
            self._report_progress(f" Parallel exploration failed: {e}")
            return None
    
    def _create_result(self, iteration: int, execution_time: float) -> AgentResult:
        """
        Create the final agent result.

        Args:
            iteration: Total iterations completed
            execution_time: Total execution time in ms

        Returns:
            AgentResult with execution details
        """
        if self._task_completed:
            return AgentResult.success_result(
                result=self._final_result,
                steps=self._steps,
                total_iterations=iteration,
                execution_time_ms=execution_time,
                final_state=self._state,
            )
        else:
            return AgentResult.failure_result(
                error=self._error_message or "Task did not complete",
                steps=self._steps,
                total_iterations=iteration,
                execution_time_ms=execution_time,
            )
