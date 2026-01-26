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
Task Planning System for Autonomous Execution.

This module provides intelligent task decomposition and planning capabilities
for the ReAct agent, enabling autonomous execution similar to Warp Agent,
Augmented AI, and JetBrains Junie.

The planner:
- Analyzes task complexity to determine if planning is needed
- Decomposes tasks into phases and goals
- Stores plans in working memory for state management
- Adapts plans dynamically based on execution outcomes
- Provides transparent progress tracking

Example:
    >>> planner = TaskPlanner(llm_provider, prompt_manager)
    >>> plan = await planner.create_plan("Extract all products and prices")
    >>> # Plan has phases: Navigation → Extraction → Validation
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from flybrowser.agents.structured_llm import StructuredLLMWrapper
from flybrowser.agents.schemas import PLAN_SCHEMA, PLAN_ADAPTATION_SCHEMA

if TYPE_CHECKING:
    from flybrowser.llm.base import BaseLLMProvider
    from flybrowser.prompts import PromptManager
    from .memory import AgentMemory

logger = logging.getLogger(__name__)


class PhaseStatus(str, Enum):
    """Status of a plan phase."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GoalStatus(str, Enum):
    """Status of a phase goal."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REPLANNING = "replanning"


@dataclass
class Goal:
    """
    A specific goal within a phase.
    
    Represents a concrete objective that must be achieved
    to complete the phase.
    
    Attributes:
        description: What needs to be accomplished
        success_criteria: How to determine if goal is met
        status: Current status
        attempts: Number of execution attempts
        error_message: Error if failed
    """
    description: str
    success_criteria: str
    status: GoalStatus = GoalStatus.PENDING
    attempts: int = 0
    error_message: Optional[str] = None
    completed_at: Optional[float] = None
    
    def mark_in_progress(self) -> None:
        """Mark goal as in progress."""
        self.status = GoalStatus.IN_PROGRESS
        self.attempts += 1
    
    def mark_completed(self) -> None:
        """Mark goal as completed."""
        self.status = GoalStatus.COMPLETED
        self.completed_at = time.time()
        logger.info(f"[ok] Goal completed: {self.description}")
    
    def mark_failed(self, error: str) -> None:
        """Mark goal as failed."""
        self.status = GoalStatus.FAILED
        self.error_message = error
        logger.warning(f"[fail] Goal failed: {self.description} - {error}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "description": self.description,
            "success_criteria": self.success_criteria,
            "status": self.status.value,
            "attempts": self.attempts,
            "error_message": self.error_message,
            "completed_at": self.completed_at,
        }
    
    def format_for_prompt(self) -> str:
        """Format goal for LLM prompt."""
        status_symbol = {
            GoalStatus.PENDING: "○",
            GoalStatus.IN_PROGRESS: "[partial]",
            GoalStatus.COMPLETED: "[ok]",
            GoalStatus.FAILED: "[fail]",
            GoalStatus.REPLANNING: "",
        }[self.status]
        
        return f"{status_symbol} {self.description}"


@dataclass
class Phase:
    """
    A major step in the execution plan.
    
    Represents a cohesive group of related goals that together
    accomplish a significant part of the overall task.
    
    Attributes:
        name: Phase name (e.g., "Navigation", "Data Extraction")
        description: What this phase accomplishes
        goals: List of goals to achieve in this phase
        dependencies: Phase names that must complete before this
        status: Current status
        started_at: When phase started
        completed_at: When phase completed
    """
    name: str
    description: str
    goals: List[Goal] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: PhaseStatus = PhaseStatus.PENDING
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    
    def mark_in_progress(self) -> None:
        """Mark phase as in progress."""
        self.status = PhaseStatus.IN_PROGRESS
        self.started_at = time.time()
        logger.info(f"> Phase started: {self.name}")
    
    def mark_completed(self) -> None:
        """Mark phase as completed."""
        self.status = PhaseStatus.COMPLETED
        self.completed_at = time.time()
        logger.info(f"[ok] Phase completed: {self.name}")
    
    def mark_failed(self, error: str) -> None:
        """Mark phase as failed."""
        self.status = PhaseStatus.FAILED
        self.error_message = error
        logger.warning(f"[fail] Phase failed: {self.name} - {error}")
    
    def get_current_goal(self) -> Optional[Goal]:
        """Get the current active goal."""
        for goal in self.goals:
            if goal.status in (GoalStatus.PENDING, GoalStatus.IN_PROGRESS, GoalStatus.REPLANNING):
                return goal
        return None
    
    def are_all_goals_completed(self) -> bool:
        """Check if all goals are completed."""
        return all(g.status == GoalStatus.COMPLETED for g in self.goals)
    
    def has_failed_goals(self) -> bool:
        """Check if any goals have failed."""
        return any(g.status == GoalStatus.FAILED for g in self.goals)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "goals": [g.to_dict() for g in self.goals],
            "dependencies": self.dependencies,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
        }
    
    def format_for_prompt(self) -> str:
        """Format phase for LLM prompt."""
        status_symbol = {
            PhaseStatus.PENDING: "○",
            PhaseStatus.IN_PROGRESS: "[partial]",
            PhaseStatus.COMPLETED: "[ok]",
            PhaseStatus.FAILED: "[fail]",
            PhaseStatus.SKIPPED: "−",
        }[self.status]
        
        lines = [f"{status_symbol} **{self.name}**: {self.description}"]
        if self.goals:
            lines.append("  Goals:")
            for goal in self.goals:
                lines.append(f"    {goal.format_for_prompt()}")
        
        return "\n".join(lines)


@dataclass
class ExecutionPlan:
    """
    Complete execution plan for a task.
    
    Represents the full strategy for accomplishing a complex task,
    broken down into phases and goals with progress tracking.
    
    Attributes:
        task: Original task description
        phases: List of phases to execute
        current_phase_index: Index of currently executing phase
        created_at: When plan was created
        updated_at: When plan was last modified
        metadata: Additional context and settings
    """
    task: str
    phases: List[Phase] = field(default_factory=list)
    current_phase_index: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_current_phase(self) -> Optional[Phase]:
        """Get the currently executing phase."""
        if 0 <= self.current_phase_index < len(self.phases):
            return self.phases[self.current_phase_index]
        return None
    
    def advance_to_next_phase(self) -> bool:
        """
        Advance to the next phase.
        
        Returns:
            True if advanced, False if no more phases
        """
        if self.current_phase_index < len(self.phases) - 1:
            self.current_phase_index += 1
            self.updated_at = time.time()
            logger.info(f"Advanced to phase {self.current_phase_index + 1}/{len(self.phases)}")
            return True
        return False
    
    def is_complete(self) -> bool:
        """Check if all phases are completed."""
        return all(
            p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
            for p in self.phases
        )
    
    def has_failures(self) -> bool:
        """Check if any phases have failed."""
        return any(p.status == PhaseStatus.FAILED for p in self.phases)
    
    def get_progress_summary(self) -> str:
        """Get a summary of plan progress."""
        completed = sum(1 for p in self.phases if p.status == PhaseStatus.COMPLETED)
        total = len(self.phases)
        current = self.get_current_phase()
        
        lines = [
            f"Progress: {completed}/{total} phases completed",
        ]
        
        if current:
            lines.append(f"Current: {current.name} ({current.status.value})")
            current_goal = current.get_current_goal()
            if current_goal:
                lines.append(f"  Goal: {current_goal.description}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task": self.task,
            "phases": [p.to_dict() for p in self.phases],
            "current_phase_index": self.current_phase_index,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }
    
    def format_for_prompt(
        self,
        available_data: Optional[List[str]] = None,
        completed_actions: Optional[List[str]] = None,
    ) -> str:
        """
        Format plan for LLM prompt.
        
        Provides structured context about the current execution plan,
        highlighting the current phase and goal, and what data is already
        available to avoid redundant work.
        
        Args:
            available_data: List of extracted data keys available in memory
            completed_actions: List of actions completed in previous phases
        """
        current = self.get_current_phase()
        
        lines = [
            "## Execution Plan",
            f"Task: {self.task}",
            f"Progress: Phase {self.current_phase_index + 1}/{len(self.phases)}",
            "",
            "### Phases:",
        ]
        
        for i, phase in enumerate(self.phases):
            prefix = "→ " if i == self.current_phase_index else "  "
            lines.append(f"{prefix}{phase.format_for_prompt()}")
        
        if current:
            lines.append("")
            lines.append("### Current Focus:")
            lines.append(f"Phase: {current.name}")
            lines.append(f"Description: {current.description}")
            
            current_goal = current.get_current_goal()
            if current_goal:
                lines.append(f"Current Goal: {current_goal.description}")
                lines.append(f"Success Criteria: {current_goal.success_criteria}")
        
        # CRITICAL: Show what data is already available to prevent redundant work
        if available_data:
            lines.append("")
            lines.append("### ⚠️ DATA ALREADY AVAILABLE (check before acting!):")
            lines.append("The following data has been extracted in previous steps:")
            for data_key in available_data:
                # Extract meaningful description from key (e.g., extracted_extract_text_123 -> text extraction)
                desc = data_key.replace("extracted_", "").rsplit("_", 1)[0].replace("_", " ")
                lines.append(f"  • {data_key} ({desc})")
            lines.append("")
            lines.append("**→ If this data satisfies your current goal, call `complete` immediately!**")
            lines.append("**→ Do NOT extract the same data again - use what's in Extracted Data section below.**")
        
        # Show what actions were completed previously
        if completed_actions:
            lines.append("")
            lines.append("### Previous Phase Actions:")
            for action in completed_actions[-5:]:  # Last 5 actions
                lines.append(f"  ✓ {action}")
        
        return "\n".join(lines)


class TaskPlanner:
    """
    Intelligent task planner for autonomous execution.
    
    Analyzes tasks to determine complexity, creates structured
    execution plans, and manages plan lifecycle including adaptation.
    
    Attributes:
        llm: LLM provider for plan generation
        prompt_manager: Prompt manager for templates
    """
    
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        prompt_manager: PromptManager,
        model_capabilities: Optional[List[Any]] = None,
        config: Optional[Any] = None,
        conversation_manager: Optional["ConversationManager"] = None,
    ) -> None:
        """
        Initialize the task planner.
        
        Args:
            llm_provider: LLM for plan generation
            prompt_manager: Prompt manager for templates
            model_capabilities: Model capabilities for capability-aware planning
            config: AgentConfig for planning parameters (temperature, tokens)
            conversation_manager: Optional shared ConversationManager for unified
                                  token tracking (if not provided, one will be created)
        """
        self.llm = llm_provider
        self.prompt_manager = prompt_manager
        self.model_capabilities = model_capabilities or []
        self.config = config
        self._conversation_manager = conversation_manager
    
    def _format_capability_context(self) -> Dict[str, Any]:
        """
        Format model capabilities for use in planning prompts.
        
        Returns:
            Dictionary with 'text' (formatted string) and 'vision_enabled' (bool)
        """
        # Import here to avoid circular dependency
        from flybrowser.llm.base import ModelCapability
        
        has_vision = ModelCapability.VISION in self.model_capabilities
        has_tool_calling = ModelCapability.TOOL_CALLING in self.model_capabilities
        has_structured = ModelCapability.STRUCTURED_OUTPUT in self.model_capabilities
        
        lines = ["Your execution capabilities:"]
        
        # Vision capability
        if has_vision:
            lines.append("- **Vision**: YES - Can analyze screenshots for complex page layouts")
            lines.append("  Strategy: Use visual verification, leverage screenshots for navigation")
        else:
            lines.append("- **Vision**: NO - Text-only navigation")
            lines.append("  Strategy: Use element IDs, selectors, and structured page state queries")
        
        # Tool calling (should always be true for ReAct agents)
        if has_tool_calling:
            lines.append("- **Tool Calling**: YES")
        
        # Structured output
        if has_structured:
            lines.append("- **Structured Output**: YES")
        
        capability_text = "\n".join(lines)
        
        return {
            "text": capability_text,
            "vision_enabled": has_vision,
        }
    
    def should_create_plan(self, task: str) -> bool:
        """
        Determine if a task requires planning.
        
        Simple tasks can execute directly, while complex tasks
        benefit from structured planning.
        
        Args:
            task: Task description
            
        Returns:
            True if planning is recommended
        """
        task_lower = task.lower()
        
        # Keywords indicating complexity
        complexity_indicators = [
            "all", "every", "multiple", "extract", "compare", "validate",
            "scrape", "collect", "pages", "then", "after", "complex",
            "comprehensive", "detailed", "analyze"
        ]
        
        # Keywords indicating simplicity
        simplicity_indicators = [
            "click", "type", "navigate to", "go to", "open",
            "close", "scroll", "hover"
        ]
        
        # Count indicators
        complexity_score = sum(1 for word in complexity_indicators if word in task_lower)
        simplicity_score = sum(1 for word in simplicity_indicators if word in task_lower)
        
        # Length factor
        word_count = len(task.split())
        
        # Decision logic
        if complexity_score >= 2:
            return True
        if word_count > 15 and simplicity_score == 0:
            return True
        if "and" in task_lower and word_count > 10:
            return True
        
        return False
    
    async def create_plan(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        memory: Optional[Any] = None,
    ) -> ExecutionPlan:
        """
        Create a structured execution plan for a task.
        
        Uses LLM to decompose the task into phases and goals,
        creating a comprehensive execution strategy.
        
        Args:
            task: Task description
            context: Optional context information
            memory: Optional memory for accessing PageMaps
            
        Returns:
            ExecutionPlan with phases and goals
        """
        logger.info(f"Creating execution plan for: {task[:100]}...")
        
        try:
            # Enhance context with PageMap if available
            enhanced_context = context or {}
            if memory:
                page_context = self._get_page_context(memory)
                if page_context:
                    enhanced_context["page_understanding"] = page_context
                    logger.info(f"Planning with page context: {len(page_context.get('sections', []))} sections")
            
            # Format capability context for planning
            capability_context = self._format_capability_context()
            
            # Build prompt for plan generation
            prompts = self.prompt_manager.get_prompt(
                "task_planner",
                task=task,
                context=json.dumps(enhanced_context, indent=2),
                capabilities=capability_context["text"],
                vision_enabled=capability_context["vision_enabled"],
            )
            
            # Get LLM to generate plan
            planning_temp = self.config.planning_temperature if self.config else 0.3
            
            # Calculate max_tokens dynamically if enabled
            if self.config and self.config.llm.enable_dynamic_tokens:
                from flybrowser.agents.config import calculate_max_tokens_for_response, estimate_tokens
                system_tokens = estimate_tokens(prompts["system"])
                user_tokens = estimate_tokens(prompts["user"])
                planning_tokens = calculate_max_tokens_for_response(
                    system_prompt_tokens=system_tokens,
                    user_prompt_tokens=user_tokens,
                    context_tokens=0,
                    safety_margin=self.config.llm.token_safety_margin
                )
                logger.debug(
                    f"[PLANNING DYNAMIC] max_tokens={planning_tokens} "
                    f"(system: {system_tokens}, user: {user_tokens})"
                )
            else:
                planning_tokens = self.config.planning_max_tokens if self.config else 2048
                logger.debug(f"[PLANNING STATIC] max_tokens={planning_tokens}")
            
            # Use StructuredLLMWrapper for reliable JSON output with repair
            # Share ConversationManager with ReActAgent if provided for unified token tracking
            wrapper = StructuredLLMWrapper(
                self.llm, 
                max_repair_attempts=2,
                conversation_manager=self._conversation_manager,
            )
            
            try:
                plan_data = await wrapper.generate_structured(
                    prompt=prompts["user"],
                    schema=PLAN_SCHEMA,
                    system_prompt=prompts["system"],
                    temperature=planning_temp,
                    max_tokens=planning_tokens,
                )
                
                # Parse structured response into ExecutionPlan
                plan = self._parse_structured_plan(task, plan_data)
            except ValueError as e:
                logger.warning(f"Structured plan generation failed: {e}, using fallback")
                return self._create_fallback_plan(task)
            
            # Log the plan in a nice formatted way
            self._log_plan(plan, "Plan Created")
            
            return plan
        
        except Exception as e:
            logger.warning(f"Plan generation failed: {e}, creating fallback plan")
            return self._create_fallback_plan(task)
    
    def _get_page_context(self, memory: Any) -> Optional[Dict[str, Any]]:
        """
        Extract page understanding from memory for planning context.
        
        Retrieves PageMap from memory and formats it for inclusion
        in planning prompts, enabling spatially-aware plan generation.
        
        Args:
            memory: AgentMemory instance
            
        Returns:
            Dictionary with page structure information, or None
        """
        try:
            # Try to get current URL from memory scratch pad
            current_url = memory.working.get_scratch("current_url")
            if not current_url:
                return None
            
            # Get PageMap for current URL
            page_map = memory.get_page_map(current_url)
            if not page_map:
                return None
            
            # Format for planning prompt
            page_context = {
                "url": current_url,
                "total_height": page_map.total_page_height,
                "coverage": f"{page_map.get_coverage_percentage():.1f}%",
                "sections": [],
                "navigation": [],
            }
            
            # Add section information
            for section in page_map.sections:
                page_context["sections"].append({
                    "type": section.section_type.value if hasattr(section.section_type, 'value') else str(section.section_type),
                    "title": section.title,
                    "scroll_range": f"{section.scroll_start}-{section.scroll_end}px",
                    "description": section.description,
                })
            
            # Add navigation links
            for nav_link in page_map.navigation_links[:10]:  # Limit to 10 most relevant
                page_context["navigation"].append({
                    "text": nav_link.get("text", "Unknown"),
                    "url": nav_link.get("likely_url", "Unknown"),
                    "location": nav_link.get("location", "Unknown"),
                })
            
            # Add summary if available
            if page_map.summary:
                page_context["summary"] = page_map.summary
            
            return page_context
            
        except Exception as e:
            logger.debug(f"Could not extract page context: {e}")
            return None
    
    def _parse_structured_plan(self, task: str, plan_data: Dict[str, Any]) -> ExecutionPlan:
        """
        Parse structured plan data (already validated JSON dict) into ExecutionPlan.
        
        Args:
            task: Original task
            plan_data: Validated plan dictionary from StructuredLLMWrapper
            
        Returns:
            Parsed ExecutionPlan
        """
        phases = []
        for phase_data in plan_data.get("phases", []):
            goals = [
                Goal(
                    description=g["description"],
                    success_criteria=g.get("success_criteria", "Goal completed successfully"),
                )
                for g in phase_data.get("goals", [])
            ]
            
            phase = Phase(
                name=phase_data["name"],
                description=phase_data.get("description", ""),
                goals=goals,
                dependencies=phase_data.get("dependencies", []),
            )
            phases.append(phase)
        
        return ExecutionPlan(
            task=task,
            phases=phases,
            metadata={"source": "llm_structured"},
        )
    
    def _parse_plan_response(self, task: str, response: str) -> ExecutionPlan:
        """
        Parse LLM response into ExecutionPlan.
        
        DEPRECATED: Use _parse_structured_plan with StructuredLLMWrapper instead.
        
        Expects JSON structure:
        {
            "phases": [
                {
                    "name": "Phase Name",
                    "description": "What this phase does",
                    "goals": [
                        {
                            "description": "Goal description",
                            "success_criteria": "How to know it's done"
                        }
                    ]
                }
            ]
        }
        
        Args:
            task: Original task
            response: LLM response
            
        Returns:
            Parsed ExecutionPlan
        """
        try:
            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                plan_data = json.loads(json_str)
            else:
                # Fallback: try parsing entire response
                plan_data = json.loads(response)
            
            return self._parse_structured_plan(task, plan_data)
        
        except Exception as e:
            logger.warning(f"Failed to parse plan response: {e}")
            return self._create_fallback_plan(task)
    
    def _create_fallback_plan(self, task: str) -> ExecutionPlan:
        """
        Create a simple fallback plan when LLM generation fails.
        
        Args:
            task: Task description
            
        Returns:
            Basic ExecutionPlan
        """
        logger.info("Creating fallback plan with single phase")
        
        phase = Phase(
            name="Task Execution",
            description=f"Execute: {task}",
            goals=[
                Goal(
                    description=task,
                    success_criteria="Task completed successfully",
                )
            ],
        )
        
        return ExecutionPlan(
            task=task,
            phases=[phase],
            metadata={"source": "fallback"},
        )
    
    async def adapt_plan(
        self,
        plan: ExecutionPlan,
        failure_context: Dict[str, Any],
    ) -> ExecutionPlan:
        """
        Adapt an execution plan based on failure.
        
        Analyzes the failure and regenerates the plan with
        adjustments to avoid repeating the same mistakes.
        
        Args:
            plan: Current execution plan
            failure_context: Information about the failure
            
        Returns:
            Updated ExecutionPlan
        """
        logger.info("Adapting plan based on failure...")
        
        current_phase = plan.get_current_phase()
        if not current_phase:
            return plan
        
        try:
            # Build adaptation prompt
            prompts = self.prompt_manager.get_prompt(
                "task_planner",
                task=plan.task,
                context=json.dumps({
                    "original_plan": plan.to_dict(),
                    "failure": failure_context,
                    "instruction": "Regenerate plan avoiding the failure",
                }, indent=2),
            )
            
            # Get adapted plan from LLM
            adaptation_temp = self.config.plan_adaptation_temperature if self.config else 0.4
            
            # Calculate max_tokens dynamically if enabled
            if self.config and self.config.llm.enable_dynamic_tokens:
                from flybrowser.agents.config import calculate_max_tokens_for_response, estimate_tokens
                system_tokens = estimate_tokens(prompts["system"])
                user_tokens = estimate_tokens(prompts["user"])
                adaptation_tokens = calculate_max_tokens_for_response(
                    system_prompt_tokens=system_tokens,
                    user_prompt_tokens=user_tokens,
                    context_tokens=0,
                    safety_margin=self.config.llm.token_safety_margin
                )
                logger.debug(
                    f"[ADAPTATION DYNAMIC] max_tokens={adaptation_tokens} "
                    f"(system: {system_tokens}, user: {user_tokens})"
                )
            else:
                adaptation_tokens = self.config.planning_max_tokens if self.config else 2048
                logger.debug(f"[ADAPTATION STATIC] max_tokens={adaptation_tokens}")
            
            # Use StructuredLLMWrapper for reliable JSON output with repair
            wrapper = StructuredLLMWrapper(self.llm, max_repair_attempts=2)
            
            try:
                plan_data = await wrapper.generate_structured(
                    prompt=prompts["user"],
                    schema=PLAN_SCHEMA,
                    system_prompt=prompts["system"],
                    temperature=adaptation_temp,
                    max_tokens=adaptation_tokens,
                )
                
                adapted_plan = self._parse_structured_plan(plan.task, plan_data)
            except ValueError as e:
                logger.warning(f"Structured plan adaptation failed: {e}, keeping original")
                return plan
            adapted_plan.current_phase_index = plan.current_phase_index
            adapted_plan.metadata["adapted"] = True
            adapted_plan.metadata["adaptation_reason"] = failure_context.get("reason")
            
            # Log the adapted plan
            self._log_plan(adapted_plan, "Plan Adapted")
            
            return adapted_plan
        
        except Exception as e:
            logger.warning(f"Plan adaptation failed: {e}, keeping original")
            return plan
    
    def store_plan_in_memory(self, plan: ExecutionPlan, memory: AgentMemory) -> None:
        """
        Store the execution plan in agent memory.
        
        Uses working memory scratch pad for plan state management.
        
        Args:
            plan: Execution plan to store
            memory: Agent memory system
        """
        memory.working.set_scratch("execution_plan", plan)
        memory.working.set_scratch("plan_created_at", plan.created_at)
        logger.debug("Plan stored in working memory")
    
    def get_plan_from_memory(self, memory: AgentMemory) -> Optional[ExecutionPlan]:
        """
        Retrieve execution plan from memory.
        
        Args:
            memory: Agent memory system
            
        Returns:
            ExecutionPlan if found, None otherwise
        """
        return memory.working.get_scratch("execution_plan")
    
    def update_plan_in_memory(self, plan: ExecutionPlan, memory: AgentMemory) -> None:
        """
        Update the stored plan in memory.
        
        Args:
            plan: Updated execution plan
            memory: Agent memory system
        """
        plan.updated_at = time.time()
        memory.working.set_scratch("execution_plan", plan)
        logger.debug("Plan updated in working memory")
    
    def _log_plan(self, plan: ExecutionPlan, title: str = "Execution Plan") -> None:
        """
        Log the execution plan in a nice, readable format.
        
        Args:
            plan: Execution plan to log
            title: Title for the log output
        """
        logger.info(f"\n{'='*60}")
        logger.info(f" {title}")
        logger.info(f"{'='*60}")
        logger.info(f"Task: {plan.task[:80]}{'...' if len(plan.task) > 80 else ''}")
        logger.info(f"Phases: {len(plan.phases)} | Total Goals: {sum(len(p.goals) for p in plan.phases)}")
        
        if plan.metadata.get('adapted'):
            logger.info(f"  Adapted due to: {plan.metadata.get('adaptation_reason', 'failure')}")
        
        logger.info("")
        
        for phase_idx, phase in enumerate(plan.phases, 1):
            status_icon = {
                PhaseStatus.PENDING: "⏳",
                PhaseStatus.IN_PROGRESS: ">",
                PhaseStatus.COMPLETED: "",
                PhaseStatus.FAILED: "",
            }.get(phase.status, "📌")
            
            logger.info(f"{status_icon} Phase {phase_idx}/{len(plan.phases)}: {phase.name}")
            logger.info(f"   {phase.description}")
            
            for goal_idx, goal in enumerate(phase.goals, 1):
                goal_status = {
                    GoalStatus.PENDING: "◯",
                    GoalStatus.IN_PROGRESS: "[partial]",
                    GoalStatus.COMPLETED: "●",
                    GoalStatus.FAILED: "[fail]",
                }.get(goal.status, "○")
                
                logger.info(f"   {goal_status} Goal {goal_idx}: {goal.description}")
                if goal.status == GoalStatus.FAILED and goal.error_message:
                    logger.info(f"      ↳ Error: {goal.error_message}")
            
            logger.info("")  # Empty line between phases
        
        logger.info(f"{'='*60}\n")
