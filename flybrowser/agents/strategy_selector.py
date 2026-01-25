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
Intelligent Strategy Selector for ReAct Agent.

Automatically selects the optimal reasoning strategy based on:
- Task complexity
- Execution context
- Previous failures
- Resource constraints
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, TYPE_CHECKING

from .types import ReasoningStrategy
from flybrowser.utils.logger import logger

if TYPE_CHECKING:
    from .memory import AgentMemory


class TaskComplexity:
    """Task complexity indicators."""
    
    SIMPLE_KEYWORDS = {
        "click", "goto", "navigate", "scroll", "wait", "get", "take screenshot",
        "go to", "go back", "refresh"
    }
    
    MEDIUM_KEYWORDS = {
        "extract", "find", "search for", "locate", "read", "get data",
        "fill form", "type", "enter", "submit"
    }
    
    COMPLEX_KEYWORDS = {
        "compare", "analyze", "if", "when", "until", "verify", "validate",
        "ensure", "check if", "multiple", "all", "each"
    }
    
    VERY_COMPLEX_KEYWORDS = {
        "scrape", "collect all", "iterate", "loop", "every", "across pages",
        "follow links", "recursive", "comprehensive"
    }


class StrategySelector:
    """
    Intelligently selects reasoning strategy for autonomous execution.
    
    This class analyzes the task, context, and execution history to
    automatically choose the best reasoning strategy without user input.
    
    Strategies:
    - REACT_STANDARD: Simple, direct tasks
    - CHAIN_OF_THOUGHT: Medium complexity, multi-step tasks
    - TREE_OF_THOUGHT: Complex tasks with multiple approaches or after failures
    - SELF_REFLECTION: After repeated failures
    """
    
    def __init__(self) -> None:
        """Initialize the strategy selector."""
        self.last_strategy: Optional[ReasoningStrategy] = None
        self.strategy_success_count: Dict[ReasoningStrategy, int] = {}
        self.strategy_failure_count: Dict[ReasoningStrategy, int] = {}
    
    def select_strategy(
        self,
        task: str,
        memory: Optional[AgentMemory] = None,
        force_autonomous: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningStrategy:
        """
        Select the optimal reasoning strategy for a task.
        
        Args:
            task: The task description
            memory: Agent memory for failure history
            force_autonomous: If True, always select intelligently (ignore manual config)
            context: Additional context about the task
            
        Returns:
            Selected reasoning strategy
        """
        # Analyze task complexity
        complexity_score = self._analyze_task_complexity(task)
        
        # Check failure history
        failure_count = 0
        recent_failures = 0
        if memory:
            failure_count = memory.get_failure_count()
            # Check if recent attempts failed
            recent_steps = memory.short_term.get_recent(n=3)
            from .types import ExecutionOutcome
            recent_failures = sum(
                1 for step in recent_steps 
                if step.outcome == ExecutionOutcome.FAILURE
            )
        
        # Decision logic
        strategy = self._decide_strategy(
            complexity_score=complexity_score,
            total_failures=failure_count,
            recent_failures=recent_failures,
            task_length=len(task),
            context=context,
        )
        
        logger.info(
            f"[Strategy Selector] Selected {strategy.value} "
            f"(complexity={complexity_score:.2f}, failures={failure_count})"
        )
        
        self.last_strategy = strategy
        return strategy
    
    def _analyze_task_complexity(self, task: str) -> float:
        """
        Analyze task complexity based on keywords and structure.
        
        Returns:
            Complexity score from 0.0 (trivial) to 1.0 (very complex)
        """
        task_lower = task.lower()
        score = 0.0
        
        # Base complexity from task length
        # Longer tasks tend to be more complex
        word_count = len(task.split())
        length_score = min(word_count / 50.0, 0.3)  # Max 0.3 from length
        score += length_score
        
        # Keyword-based scoring
        for keyword in TaskComplexity.SIMPLE_KEYWORDS:
            if keyword in task_lower:
                score += 0.05
        
        for keyword in TaskComplexity.MEDIUM_KEYWORDS:
            if keyword in task_lower:
                score += 0.15
        
        for keyword in TaskComplexity.COMPLEX_KEYWORDS:
            if keyword in task_lower:
                score += 0.25
        
        for keyword in TaskComplexity.VERY_COMPLEX_KEYWORDS:
            if keyword in task_lower:
                score += 0.35
        
        # Multi-step indicators
        if any(sep in task_lower for sep in ["and then", "after", "once", "followed by"]):
            score += 0.2
        
        # Conditional logic indicators
        if any(word in task_lower for word in ["if", "when", "unless", "until"]):
            score += 0.25
        
        # Quantifiers indicating iteration
        if any(word in task_lower for word in ["all", "every", "each", "multiple"]):
            score += 0.2
        
        # Comparison/analysis indicators
        if any(word in task_lower for word in ["compare", "analyze", "evaluate", "assess"]):
            score += 0.2
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def _decide_strategy(
        self,
        complexity_score: float,
        total_failures: int,
        recent_failures: int,
        task_length: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningStrategy:
        """
        Decide the reasoning strategy based on multiple factors.
        
        Decision rules:
        1. If recent failures >= 2: Use TREE_OF_THOUGHT (explore alternatives)
        2. If total failures >= 5: Use SELF_REFLECTION (learn from mistakes)
        3. If complexity >= 0.7: Use TREE_OF_THOUGHT (complex problem)
        4. If complexity >= 0.4: Use CHAIN_OF_THOUGHT (multi-step)
        5. Otherwise: Use REACT_STANDARD (simple task)
        
        Args:
            complexity_score: Task complexity (0.0-1.0)
            total_failures: Total failure count
            recent_failures: Recent failure count
            task_length: Length of task description
            context: Additional context
            
        Returns:
            Selected reasoning strategy
        """
        # Rule 1: Recent failures suggest we need to explore alternatives
        if recent_failures >= 2:
            logger.debug(
                f"Selecting TREE_OF_THOUGHT due to {recent_failures} recent failures"
            )
            return ReasoningStrategy.TREE_OF_THOUGHT
        
        # Rule 2: Many total failures suggests we need self-reflection
        if total_failures >= 5:
            logger.debug(
                f"Selecting SELF_REFLECTION due to {total_failures} total failures"
            )
            return ReasoningStrategy.SELF_REFLECTION
        
        # Rule 3: High complexity tasks benefit from exploring multiple paths
        if complexity_score >= 0.7:
            logger.debug(
                f"Selecting TREE_OF_THOUGHT for high complexity ({complexity_score:.2f})"
            )
            return ReasoningStrategy.TREE_OF_THOUGHT
        
        # Rule 4: Medium complexity tasks benefit from explicit reasoning
        if complexity_score >= 0.4:
            logger.debug(
                f"Selecting CHAIN_OF_THOUGHT for medium complexity ({complexity_score:.2f})"
            )
            return ReasoningStrategy.CHAIN_OF_THOUGHT
        
        # Rule 5: Simple tasks use standard ReAct
        logger.debug(
            f"Selecting REACT_STANDARD for low complexity ({complexity_score:.2f})"
        )
        return ReasoningStrategy.REACT_STANDARD
    
    def record_outcome(self, strategy: ReasoningStrategy, success: bool) -> None:
        """
        Record the outcome of a strategy usage for learning.
        
        Args:
            strategy: The strategy that was used
            success: Whether it was successful
        """
        if success:
            self.strategy_success_count[strategy] = (
                self.strategy_success_count.get(strategy, 0) + 1
            )
        else:
            self.strategy_failure_count[strategy] = (
                self.strategy_failure_count.get(strategy, 0) + 1
            )
    
    def get_best_strategy_stats(self) -> Dict[str, Any]:
        """
        Get statistics about strategy performance.
        
        Returns:
            Dictionary with strategy performance stats
        """
        stats = {}
        for strategy in ReasoningStrategy:
            successes = self.strategy_success_count.get(strategy, 0)
            failures = self.strategy_failure_count.get(strategy, 0)
            total = successes + failures
            
            if total > 0:
                stats[strategy.value] = {
                    "successes": successes,
                    "failures": failures,
                    "total": total,
                    "success_rate": successes / total,
                }
        
        return stats
