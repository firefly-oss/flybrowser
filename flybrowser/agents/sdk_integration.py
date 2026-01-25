# Copyright 2026 Firefly Software Solutions Inc
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SDK Integration for ReAct Framework.

This module provides high-level wrappers and utilities for integrating
the ReAct framework with the FlyBrowser SDK, supporting embedded, server,
and cluster modes.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from flybrowser.agents.react_agent import ReActAgent
from flybrowser.agents.orchestrator import AgentOrchestrator, ExecutionMode
from flybrowser.agents.tools.registry import ToolRegistry
from flybrowser.agents.config import AgentConfig
from flybrowser.agents.memory import AgentMemory

# Import tools directly to avoid circular import
from flybrowser.agents.tools.navigation import (
    NavigateTool,
    GoBackTool,
    GoForwardTool,
    RefreshTool,
)
from flybrowser.agents.tools.interaction import (
    ClickTool,
    TypeTool,
    ScrollTool,
    HoverTool,
    PressKeyTool,
    SelectOptionTool,
    CheckboxTool,
    FocusTool,
    FillTool,
    WaitForSelectorTool,
    DoubleClickTool,
    RightClickTool,
    DragAndDropTool,
    UploadFileTool,
    EvaluateJavaScriptTool,
    GetAttributeTool,
    ClearInputTool,
)
from flybrowser.agents.tools.extraction import (
    ExtractTextTool,
    ScreenshotTool,
    GetPageStateTool,
)
from flybrowser.agents.tools.system import (
    CompleteTool,
    FailTool,
    WaitTool,
    AskUserTool,
)
from flybrowser.agents.tools.search_api import SearchAPITool
from flybrowser.agents.tools.search_human import SearchHumanTool
from flybrowser.agents.tools.search_rank import SearchRankTool
from flybrowser.agents.tools.search import SearchAgentTool
from flybrowser.agents.tools.captcha import (
    DetectCaptchaTool,
    SolveCaptchaTool,
    WaitForCaptchaResolvedTool,
)
from flybrowser.agents.tools.page_analyzer import PageAnalyzer
from flybrowser.agents.strategy_selector import StrategySelector
from flybrowser.agents.config import PageAnalysisConfig
from flybrowser.utils.logger import logger

if TYPE_CHECKING:
    from flybrowser.core.page import PageController
    from flybrowser.core.element import ElementDetector
    from flybrowser.llm.base import BaseLLMProvider
    from flybrowser.prompts.manager import PromptManager


class ReActBrowserAgent:
    """
    High-level ReAct agent for browser automation in the SDK.
    
    This class provides a unified interface for using the ReAct framework
    in FlyBrowser, with automatic tool registration, memory management,
    and orchestration capabilities.
    
    Attributes:
        agent: The core ReActAgent
        orchestrator: Agent orchestrator with safety mechanisms
        registry: Tool registry with all browser tools
        memory: Agent memory system
        
    Example:
        >>> react_agent = ReActBrowserAgent(page, element_detector, llm)
        >>> await react_agent.initialize()
        >>> result = await react_agent.execute_autonomous(
        ...     \"Navigate to example.com and extract the title\"
        ... )
    """
    
    def __init__(
        self,
        page_controller: "PageController",
        element_detector: Optional["ElementDetector"],
        llm_provider: "BaseLLMProvider",
        agent_config: Optional[AgentConfig] = None,
        execution_mode: ExecutionMode = ExecutionMode.AUTONOMOUS,
        enable_memory: bool = True,
    ) -> None:
        """
        Initialize the ReAct browser agent.
        
        Args:
            page_controller: Browser page controller
            element_detector: Element detection system (optional)
            llm_provider: LLM provider for reasoning
            agent_config: Agent configuration (uses defaults if not provided)
            execution_mode: Orchestrator execution mode
            enable_memory: Enable memory system (default: True)
        """
        self.page = page_controller
        self.element_detector = element_detector
        self.llm = llm_provider
        self.config = agent_config or AgentConfig()
        self.execution_mode = execution_mode
        
        # Initialize memory if enabled
        self.memory = AgentMemory() if enable_memory else None
        
        # Stealth components (can be set after construction)
        self.captcha_solver = None
        
        # Initialize intelligent strategy selector for autonomous mode
        self.strategy_selector = StrategySelector()
        
        # Will be initialized in initialize()
        self.registry: Optional[ToolRegistry] = None
        self.agent: Optional[ReActAgent] = None
        self.orchestrator: Optional[AgentOrchestrator] = None
        self.page_analyzer: Optional[PageAnalyzer] = None
        
        # Page analysis configuration
        self.analysis_config = PageAnalysisConfig(
            enable_llm_html_analysis=True,  # Enable by default
            min_elements_for_heuristic_success=5,
        )
        
        logger.info("ReActBrowserAgent created (not yet initialized)")
    
    async def initialize(self) -> None:
        """
        Initialize the agent with tools and orchestrator.
        
        This must be called after construction before using the agent.
        """
        # Create and populate tool registry
        self.registry = ToolRegistry()
        await self._register_default_tools()
        
        # Initialize PageAnalyzer BEFORE creating the ReActAgent
        # because the agent creates a filtered copy of the registry
        if self.analysis_config.enable_llm_html_analysis:
            # We need a temporary prompt manager for PageAnalyzer
            # It will be replaced with the agent's one after agent creation
            from flybrowser.prompts.manager import PromptManager
            temp_prompt_manager = PromptManager()
            self.page_analyzer = PageAnalyzer(
                llm_provider=self.llm,
                prompt_manager=temp_prompt_manager,
                config=self.analysis_config,
            )
            logger.info("[PAGE ANALYSIS] LLM-based page analyzer initialized")
        
        # Register GetPageStateTool with page_analyzer BEFORE creating ReActAgent
        # This is critical because ReActAgent creates a filtered copy of the registry
        page_state_tool = GetPageStateTool(
            page_controller=self.page,
            page_analyzer=self.page_analyzer,
        )
        self.registry.register_instance(page_state_tool)
        logger.info("[PAGE ANALYSIS] GetPageStateTool registered with LLM analyzer")
        
        # Create ReAct agent with element detector for AI-based element finding
        self.agent = ReActAgent(
            page_controller=self.page,
            llm_provider=self.llm,
            tool_registry=self.registry,
            memory=self.memory,
            config=self.config,
            element_detector=self.element_detector,
        )
        
        # Update PageAnalyzer's prompt_manager with the agent's one for consistency
        if self.page_analyzer:
            self.page_analyzer.prompt_manager = self.agent.prompt_manager
        
        # Create orchestrator
        self.orchestrator = AgentOrchestrator(
            react_agent=self.agent,
            execution_mode=self.execution_mode,
        )
        
        logger.info(
            f"ReActBrowserAgent initialized with {len(self.registry)} tools, "
            f"mode={self.execution_mode.value}"
        )
    
    async def _register_default_tools(self) -> None:
        """Register all default browser automation tools."""
        # Navigation tools
        self.registry.register(NavigateTool)
        self.registry.register(GoBackTool)
        self.registry.register(GoForwardTool)
        self.registry.register(RefreshTool)
        
        # Interaction tools - comprehensive set for full browser automation
        self.registry.register(ClickTool)
        self.registry.register(TypeTool)
        self.registry.register(ScrollTool)
        self.registry.register(HoverTool)
        self.registry.register(PressKeyTool)
        self.registry.register(SelectOptionTool)
        self.registry.register(CheckboxTool)
        self.registry.register(FocusTool)
        self.registry.register(FillTool)
        self.registry.register(WaitForSelectorTool)
        self.registry.register(DoubleClickTool)
        self.registry.register(RightClickTool)
        self.registry.register(DragAndDropTool)
        self.registry.register(UploadFileTool)
        self.registry.register(EvaluateJavaScriptTool)
        self.registry.register(GetAttributeTool)
        self.registry.register(ClearInputTool)
        
        # Extraction tools
        self.registry.register(ExtractTextTool)
        self.registry.register(ScreenshotTool)
        # Note: GetPageStateTool will be registered later in initialize() with page_analyzer
        # self.registry.register(GetPageStateTool)
        
        # System tools
        self.registry.register(CompleteTool)
        self.registry.register(FailTool)
        self.registry.register(WaitTool)
        self.registry.register(AskUserTool)
        
        # Search tools - register the new unified SearchAgentTool if API keys are available
        # This provides LLM-powered intent detection and multi-provider search
        search_agent_tool = SearchAgentTool()
        if search_agent_tool.has_providers_configured():
            # Create and register the configured SearchAgentTool
            configured_search_tool = SearchAgentTool.create(
                llm_provider=self.llm,
                config=self.config.search_providers if hasattr(self.config, 'search_providers') else None,
            )
            self.registry.register_instance(configured_search_tool)
            logger.info("Registered SearchAgentTool (API keys detected)")
        else:
            # Fall back to legacy SearchAPITool if available
            search_api_tool = SearchAPITool()
            if search_api_tool.has_api_keys_configured():
                self.registry.register(SearchAPITool)
                logger.info("Registered SearchAPITool (legacy, API keys detected)")
            else:
                logger.info(
                    "No search API tools registered (no API keys). "
                    "Using SearchHumanTool as fallback. "
                    "Set SERPER_API_KEY, GOOGLE_CUSTOM_SEARCH_API_KEY/CX, or BING_SEARCH_API_KEY to enable API search."
                )
        
        # Always register human-like search as fallback for browser-based search
        self.registry.register(SearchHumanTool)
        
        # Register search result analysis tool for ranking
        self.registry.register(SearchRankTool)
        
        # CAPTCHA tools - always register detection, solving only if configured
        detect_captcha_tool = DetectCaptchaTool(page_controller=self.page)
        self.registry.register_instance(detect_captcha_tool)
        
        # Wait for CAPTCHA resolution (useful for Cloudflare challenges)
        wait_captcha_tool = WaitForCaptchaResolvedTool(page_controller=self.page)
        self.registry.register_instance(wait_captcha_tool)
        
        # CAPTCHA solving tool - only if solver is configured
        if self.captcha_solver:
            solve_captcha_tool = SolveCaptchaTool(
                page_controller=self.page,
                captcha_solver=self.captcha_solver,
            )
            self.registry.register_instance(solve_captcha_tool)
            logger.info("[CAPTCHA] SolveCaptchaTool registered with solver")
        else:
            logger.info("[CAPTCHA] CAPTCHA detection tools registered (solver not configured)")
        
        logger.info(f"Registered {len(self.registry)} default tools")
    
    async def execute_autonomous(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 50,
        max_time_seconds: float = 1800.0,
        auto_select_strategy: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a goal autonomously using the ReAct framework.
        
        In autonomous mode, the system intelligently selects the optimal
        reasoning strategy (Standard/CoT/ToT) based on task complexity and
        execution context.
        
        Args:
            goal: High-level goal description
            context: Optional context dictionary
            max_iterations: Maximum execution iterations
            max_time_seconds: Maximum execution time
            auto_select_strategy: Auto-select reasoning strategy (default: True)
            
        Returns:
            Dictionary with execution results
        """
        if not self.agent or not self.orchestrator:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        logger.info(f"[AUTONOMOUS MODE] Executing goal: {goal[:100]}...")
        
        # AUTONOMOUS MODE: Intelligently select reasoning strategy
        if auto_select_strategy:
            original_strategy = self.config.reasoning_strategy
            selected_strategy = self.strategy_selector.select_strategy(
                task=goal,
                memory=self.memory,
                force_autonomous=True,
                context=context,
            )
            
            # Temporarily override the strategy
            self.config.reasoning_strategy = selected_strategy
            logger.info(
                f"[AUTONOMOUS MODE] Auto-selected strategy: {selected_strategy.value}"
            )
        
        # Prepare task with context
        task = goal
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            task = f"{goal} (Context: {context_str})"
        
        try:
            # Execute through orchestrator
            result = await self.orchestrator.execute(task=task)
            
            # Record outcome for strategy learning
            if auto_select_strategy:
                self.strategy_selector.record_outcome(
                    strategy=selected_strategy,
                    success=result.get("success", False),
                )
            
            return {
                "success": result.get("success", False),
                "result": result.get("result"),
                "error": result.get("error"),
                "steps": result.get("steps", []),
                "stop_reason": result.get("stop_reason"),
                "circuit_breaker_report": result.get("circuit_breaker_report"),
                "execution_mode": result.get("execution_mode"),
                "reasoning_strategy": selected_strategy.value if auto_select_strategy else self.config.reasoning_strategy.value,
            }
        
        finally:
            # Restore original strategy if we changed it
            if auto_select_strategy:
                self.config.reasoning_strategy = original_strategy
    
    async def execute(
        self,
        task: str,
        step_callback: Optional[Callable] = None,
        operation_mode: Optional["OperationMode"] = None,
        vision_override: Optional[bool] = None,
    ) -> "AgentResult":
        """
        Execute a task using the ReAct agent (direct passthrough).

        This is the primary execution method that uses the agent's configured
        max_iterations and settings.

        Args:
            task: Task description
            step_callback: Optional callback for each step
            operation_mode: Optional operation mode (NAVIGATE, EXECUTE, SCRAPE, AUTO)
            vision_override: Override vision behavior for this execution.
                - None: Use model capabilities and operation mode to decide (default)
                - True: Force vision on (if model supports it)
                - False: Force vision off (text-only mode)

        Returns:
            AgentResult object
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        logger.info(f"Executing task: {task[:100]}...")
        return await self.agent.execute(task, step_callback, operation_mode, vision_override)
    
    async def execute_task(
        self,
        task: str,
        max_iterations: int = 25,
        operation_mode: Optional["OperationMode"] = None,
    ) -> Dict[str, Any]:
        """
        Execute a specific task using the ReAct agent with custom iteration limit.

        Args:
            task: Task description
            max_iterations: Maximum iterations
            operation_mode: Optional operation mode (NAVIGATE, EXECUTE, SCRAPE, AUTO)

        Returns:
            AgentResult dictionary
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        logger.info(f"Executing task: {task[:100]}...")

        # Temporarily override config max_iterations if needed
        original_max = self.config.max_iterations
        self.config.max_iterations = max_iterations

        try:
            result = await self.agent.execute(task, operation_mode=operation_mode)

            return {
                "success": result.success,
                "result": result.result,
                "error": result.error,
                "steps": [step.to_dict() for step in result.steps],
                "total_iterations": result.total_iterations,
                "execution_time_ms": result.execution_time_ms,
                "final_state": result.final_state.value,
                "metadata": result.metadata,
            }
        finally:
            # Restore original config
            self.config.max_iterations = original_max
    
    async def execute_with_approval(
        self,
        task: str,
        approval_callback: Any,
        max_iterations: int = 25,
    ) -> Dict[str, Any]:
        """
        Execute a task with human-in-the-loop approval for dangerous actions.
        
        Args:
            task: Task description
            approval_callback: Async callback for approval requests
            max_iterations: Maximum iterations
            
        Returns:
            Execution result dictionary
        """
        if not self.orchestrator:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        # Temporarily set supervised mode
        original_mode = self.orchestrator.execution_mode
        self.orchestrator.execution_mode = ExecutionMode.SUPERVISED
        self.orchestrator.approval_callback = approval_callback
        
        try:
            result = await self.orchestrator.execute(task=task)
            return result
        finally:
            # Restore original mode
            self.orchestrator.execution_mode = original_mode
            self.orchestrator.approval_callback = None
    
    def get_tool_list(self) -> List[str]:
        """Get list of registered tool names."""
        if not self.registry:
            return []
        return self.registry.list_tools()
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get memory system status."""
        if not self.memory:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "cycle_count": self.memory.cycle_count,
            "current_goal": self.memory.current_goal,
            "failure_count": self.memory.get_failure_count(),
        }


def create_react_agent_for_sdk(
    page_controller: "PageController",
    element_detector: Optional["ElementDetector"],
    llm_provider: "BaseLLMProvider",
    **kwargs: Any,
) -> ReActBrowserAgent:
    """
    Factory function to create a ReAct agent for SDK use.
    
    Args:
        page_controller: Browser page controller
        element_detector: Element detection system
        llm_provider: LLM provider
        **kwargs: Additional configuration options
        
    Returns:
        Configured ReActBrowserAgent
    """
    return ReActBrowserAgent(
        page_controller=page_controller,
        element_detector=element_detector,
        llm_provider=llm_provider,
        **kwargs,
    )
