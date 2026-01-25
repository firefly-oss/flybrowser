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
Memory system for the ReAct agent framework.

This module provides a comprehensive memory system with:
- Short-term memory: Current task execution context
- Working memory: Active reasoning and action cycles
- Long-term memory: Persistent patterns and learned knowledge
- Context window management: Token-aware memory pruning
- Relevance-based retrieval: Finding pertinent memories

The memory system integrates with the ReAct Thought-Action-Observation cycle
to provide contextual awareness and avoid repeated failures.

Example:
    >>> memory = AgentMemory()
    >>> memory.set_user_context({"preference": "value"})
    >>>
    >>> # Record a cycle
    >>> memory.record_cycle(thought, action, observation)
    >>>
    >>> # Get context for next reasoning
    >>> context = memory.get_reasoning_context()
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from .types import (
    Action,
    ExecutionOutcome,
    MemoryPriority,
    Observation,
    OperationMode,
    ReasoningStrategy,
    Thought,
)

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """
    A single memory entry in the agent's memory system.

    Represents a complete Thought-Action-Observation cycle with
    metadata for relevance scoring and retention decisions.
    """
    entry_id: str
    timestamp: float
    thought: Thought
    action: Action
    observation: Optional[Observation]
    outcome: ExecutionOutcome
    priority: MemoryPriority = MemoryPriority.NORMAL

    # Context at time of action
    url: str = ""
    page_title: str = ""

    # Execution metadata
    duration_ms: float = 0.0
    retry_count: int = 0
    error_message: Optional[str] = None

    # Token tracking for context window management
    estimated_tokens: int = 0

    # Tags for filtering
    tags: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Calculate estimated tokens if not provided."""
        if self.estimated_tokens == 0:
            self.estimated_tokens = self._estimate_tokens()

    def _estimate_tokens(self) -> int:
        """Estimate token count for this entry."""
        # Rough estimation: ~4 chars per token
        total_chars = 0
        
        # Handle None thought (when LLM doesn't provide reasoning)
        if self.thought and hasattr(self.thought, 'content'):
            total_chars += len(self.thought.content)
        
        total_chars += len(self.action.tool_name) + len(str(self.action.parameters))
        if self.observation:
            total_chars += len(self.observation.summary) + len(self.observation.raw_output)
        if self.error_message:
            total_chars += len(self.error_message)
        return max(1, total_chars // 4)

    @property
    def action_signature(self) -> str:
        """Generate a unique signature for this action."""
        sig_parts = [
            self.action.tool_name,
            str(sorted(self.action.parameters.items())),
            self.url,
        ]
        return hashlib.md5(":".join(sig_parts).encode()).hexdigest()[:16]

    def format_for_prompt(self, include_observation: bool = True) -> str:
        """Format this entry for inclusion in a prompt."""
        lines = []
        
        # Handle None thought
        if self.thought and hasattr(self.thought, 'content'):
            lines.append(f"**Thought**: {self.thought.content}")
        
        lines.append(f"**Action**: {self.action.tool_name}({self.action.parameters})")
        if include_observation and self.observation:
            obs_content = self.observation.summary or self.observation.raw_output
            if len(obs_content) > 500:
                obs_content = obs_content[:500] + "..."
            lines.append(f"**Observation**: {obs_content}")
        if self.outcome == ExecutionOutcome.FAILURE and self.error_message:
            lines.append(f"**Error**: {self.error_message}")
        return "\n".join(lines)


@dataclass
class StateSnapshot:
    """
    Snapshot of browser state for backtracking.

    Captures the essential page state at a point in time,
    allowing the agent to reason about navigation history.
    """
    snapshot_id: str
    timestamp: float
    url: str
    page_title: str
    visible_elements_summary: str
    form_state: Dict[str, Any] = field(default_factory=dict)
    scroll_position: Tuple[int, int] = (0, 0)
    associated_entry_id: Optional[str] = None

    def format_brief(self) -> str:
        """Brief format for navigation history."""
        return f"{self.page_title} ({self.url})"



class ContextStore:
    """
    Manages user-provided and inferred context.

    Stores context that helps the agent make informed decisions,
    including user preferences, extracted information, and cached data.
    """

    def __init__(self) -> None:
        """Initialize the context store."""
        self._user_context: Dict[str, Any] = {}
        self._inferred_context: Dict[str, Any] = {}
        self._context_history: List[Dict[str, Any]] = []
        self._extraction_cache: Dict[str, Any] = {}

    def set_user_context(self, context: Dict[str, Any]) -> None:
        """Set user-provided context."""
        self._user_context.update(context)
        self._context_history.append({
            "timestamp": time.time(),
            "type": "user_provided",
            "context": context.copy(),
        })
        logger.debug(f"User context updated with {len(context)} items")

    def add_inferred_context(self, key: str, value: Any, source: str = "") -> None:
        """Add context inferred during execution."""
        self._inferred_context[key] = {
            "value": value,
            "source": source,
            "timestamp": time.time(),
        }
        logger.debug(f"Inferred context: {key}={value} (from {source})")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        if key in self._user_context:
            return self._user_context[key]
        if key in self._inferred_context:
            return self._inferred_context[key]["value"]
        return default

    def get_all(self) -> Dict[str, Any]:
        """Get all context (user + inferred)."""
        result = self._user_context.copy()
        for key, item in self._inferred_context.items():
            if key not in result:
                result[key] = item["value"]
        return result

    def get_relevant_context(self, task_description: str) -> Dict[str, Any]:
        """Get context relevant to a specific task using keyword matching."""
        task_lower = task_description.lower()
        keywords = set(task_lower.split())

        relevant = {}
        for key, value in self.get_all().items():
            key_words = set(key.lower().replace("_", " ").split())
            if keywords & key_words:
                relevant[key] = value
            elif isinstance(value, str) and any(kw in value.lower() for kw in keywords):
                relevant[key] = value

        return relevant if relevant else self.get_all()

    def cache_extraction(self, key: str, data: Any) -> None:
        """Cache extracted data."""
        self._extraction_cache[key] = {
            "data": data,
            "timestamp": time.time(),
        }

    def get_cached_extraction(self, key: str, max_age_seconds: float = 300) -> Optional[Any]:
        """Get cached extraction if still valid."""
        if key not in self._extraction_cache:
            return None
        cached = self._extraction_cache[key]
        if time.time() - cached["timestamp"] > max_age_seconds:
            del self._extraction_cache[key]
            return None
        return cached["data"]

    def format_for_prompt(self) -> str:
        """Format context for inclusion in LLM prompts."""
        context = self.get_all()
        if not context:
            return "No context provided."
        lines = ["## User Context"]
        for key, value in context.items():
            display_key = key.replace("_", " ").title()
            lines.append(f"- {display_key}: {value}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all context."""
        self._user_context.clear()
        self._inferred_context.clear()
        self._extraction_cache.clear()


class ShortTermMemory:
    """
    Short-term memory for current task execution.

    Stores recent TAO cycles with fast access for immediate
    decision-making and failure avoidance.
    """

    def __init__(self, max_entries: int = 50) -> None:
        """Initialize short-term memory."""
        self.max_entries = max_entries
        self._entries: Deque[MemoryEntry] = deque(maxlen=max_entries)
        self._failed_signatures: Set[str] = set()
        self._entry_counter: int = 0
        self._action_frequency: Dict[str, int] = {}  # Track repeated actions

    def add(self, entry: MemoryEntry) -> None:
        """Add an entry to short-term memory."""
        self._entries.append(entry)
        if entry.outcome == ExecutionOutcome.FAILURE:
            self._failed_signatures.add(entry.action_signature)
        
        # Track action frequency for loop detection
        action_sig = entry.action_signature
        self._action_frequency[action_sig] = self._action_frequency.get(action_sig, 0) + 1
        
        logger.debug(f"STM: Added entry {entry.entry_id}")

    def has_failed(self, action_signature: str) -> bool:
        """Check if an action signature has failed before."""
        return action_signature in self._failed_signatures

    def get_recent(self, n: int = 5) -> List[MemoryEntry]:
        """Get the n most recent entries."""
        return list(self._entries)[-n:]

    def get_by_outcome(self, outcome: ExecutionOutcome) -> List[MemoryEntry]:
        """Get entries by outcome type."""
        return [e for e in self._entries if e.outcome == outcome]

    def generate_entry_id(self) -> str:
        """Generate a unique entry ID."""
        self._entry_counter += 1
        return f"stm_{self._entry_counter}_{uuid.uuid4().hex[:8]}"

    def total_tokens(self) -> int:
        """Get total estimated tokens in short-term memory."""
        return sum(e.estimated_tokens for e in self._entries)
    
    def get_action_frequency(self, tool_name: str, params: Dict[str, Any]) -> int:
        """Get how many times a specific action has been executed.
        
        Args:
            tool_name: Name of the tool
            params: Parameters dict
            
        Returns:
            Number of times this action was executed in recent history
        """
        # Create action signature from tool + key params
        import json
        import hashlib
        params_str = json.dumps(params, sort_keys=True)
        sig = f"{tool_name}_{hashlib.md5(params_str.encode()).hexdigest()[:8]}"
        return self._action_frequency.get(sig, 0)
    
    def get_loop_warnings(self, threshold: int = 3) -> List[str]:
        """Detect repeated actions that might indicate a loop.
        
        Args:
            threshold: Minimum repetition count to trigger warning
            
        Returns:
            List of warning messages about repeated actions
        """
        warnings = []
        recent = self.get_recent(10)  # Check last 10 actions
        
        # Count actions in recent history
        action_counts = {}
        for entry in recent:
            # Create readable action description
            tool = entry.action.tool_name
            params = entry.action.parameters or {}
            
            # Skip complete() calls - they legitimately repeat when advancing through
            # different goals/phases in planning mode
            if tool == "complete":
                continue
            
            # Create key based on tool and important params
            if tool == "scroll_page":
                direction = params.get("direction", "down")
                amount = params.get("amount", 0)
                key = f"scroll({direction}, {amount}px)"
            elif tool == "navigate":
                url = params.get("url", "")
                # Normalize URL
                key = f"navigate({url})"
            elif tool == "extract_text":
                selector = params.get("selector", "body")
                key = f"extract_text({selector})"
            elif tool == "click":
                selector = params.get("selector", params.get("text", "unknown"))
                key = f"click({selector})"
            else:
                key = f"{tool}()"
            
            action_counts[key] = action_counts.get(key, 0) + 1
        
        # Generate warnings for repeated actions
        for action, count in action_counts.items():
            if count >= threshold:
                warnings.append(f" LOOP DETECTED: {action} repeated {count} times in last 10 actions!")
        
        return warnings

    def format_history(self, last_n: int = 5) -> str:
        """Format recent history for prompts."""
        recent = self.get_recent(last_n)
        if not recent:
            return "No previous actions taken."

        lines = ["## Recent Actions"]
        
        # Check for loops and add prominent warnings
        loop_warnings = self.get_loop_warnings(threshold=3)
        if loop_warnings:
            lines.append("\n".join(loop_warnings))
            lines.append("")
        for i, entry in enumerate(recent, 1):
            outcome_symbol = "[ok]" if entry.outcome == ExecutionOutcome.SUCCESS else "[fail]"
            # Handle None thought
            thought_text = entry.thought.content[:100] if entry.thought and hasattr(entry.thought, 'content') else "[No reasoning provided]"
            
            # Format action with parameters - use larger limit for important params
            params_str = str(entry.action.parameters)[:200] if entry.action.parameters else "{}"
            action_line = f"{i}. [{outcome_symbol}] {entry.action.tool_name}({params_str})"
            lines.append(action_line)
            
            # Include observation result if available and successful
            if entry.observation and entry.outcome == ExecutionOutcome.SUCCESS:
                # Special formatting for page_state results (extract links)
                if entry.action.tool_name == "get_page_state" and entry.observation.result:
                    tool_result = entry.observation.result
                    result = tool_result.data if hasattr(tool_result, 'data') else tool_result
                    # Extract navigation links if available
                    if isinstance(result, dict):
                        nav_links = result.get("navigation_links", [])
                        other_links = result.get("other_links", [])
                        total_links = result.get("links_count", len(nav_links) + len(other_links))
                        buttons = result.get("buttons", [])
                        hidden_links = result.get("hidden_links", [])
                        
                        # Check for potential issues
                        warnings = []
                        
                        # Warning: Only language switcher links detected
                        if nav_links and len(nav_links) <= 3:
                            link_texts = [l.get('text', '').upper() for l in nav_links]
                            if all(text in ['EN', 'ES', 'PT', 'FR', 'DE', 'IT', 'JA', 'ZH', 'KO', 'RU'] for text in link_texts if text):
                                warnings.append(" Only language links found!")
                        
                        # Helpful: Menu buttons detected
                        menu_buttons = [b for b in buttons if b.get('type') == 'menu']
                        if menu_buttons:
                            menu_names = [b.get('text', b.get('ariaLabel', 'Menu'))[:20] for b in menu_buttons[:3]]
                            warnings.append(f" Menu buttons available: {', '.join(menu_names)}")
                        
                        # Warning: Hidden navigation detected
                        if hidden_links and len(hidden_links) > len(nav_links):
                            warnings.append(f" {len(hidden_links)} hidden nav links (menu closed)")
                        
                        if nav_links or other_links:
                            # Format links with text and URL for easy use
                            link_list = []
                            for l in nav_links[:5]:
                                text = l.get('text', 'Link').strip()[:30]
                                url = l.get('url', l.get('href', 'N/A'))
                                link_list.append(f"{text} ({url})")
                            
                            if link_list:
                                lines.append(f"   Found {total_links} total links, {len(nav_links)} navigation:")
                                lines.append(f"   Links: {' | '.join(link_list)}")
                                # Add warnings if any
                                if warnings:
                                    lines.append(f"   {' '.join(warnings)}")
                                
                                # Show analysis method and cost if LLM was used
                                analysis_method = result.get('analysis_method', 'heuristic')
                                if analysis_method != 'heuristic':
                                    llm_cost = result.get('llm_cost_usd', 0.0)
                                    lines.append(f"   (Detected by: {analysis_method}, cost: ${llm_cost:.4f})")
                                    
                                    # Show LLM suggestions if any
                                    llm_suggestions = result.get('llm_suggestions', [])
                                    if llm_suggestions:
                                        for suggestion in llm_suggestions[:2]:  # Max 2 suggestions
                                            lines.append(f"    {suggestion}")
                            else:
                                lines.append(f"   Found {total_links} links on page")
                        else:
                            lines.append(f"   Current URL: {result.get('url', 'N/A')}, Title: {result.get('title', 'N/A')}")
                            if warnings:
                                lines.append(f"   {' '.join(warnings)}")
                    else:
                        result_preview = str(result)[:300]
                        lines.append(f"   Result: {result_preview}")
                # Navigate results
                elif entry.action.tool_name == "navigate" and entry.observation.result:
                    tool_result = entry.observation.result
                    result = tool_result.data if hasattr(tool_result, 'data') else tool_result
                    if isinstance(result, dict):
                        lines.append(f"   Navigated to: {result.get('url', 'N/A')}")
                    else:
                        lines.append(f"   Navigation successful")
                # Search results - format ALL results clearly with ranking info
                elif entry.action.tool_name == "search" and entry.observation.result:
                    tool_result = entry.observation.result
                    result = tool_result.data if hasattr(tool_result, 'data') else tool_result
                    if isinstance(result, dict) and 'results' in result:
                        query = result.get('query', 'N/A')
                        search_results = result.get('results', [])
                        result_count = result.get('result_count', len(search_results))
                        ranking_applied = result.get('ranking_applied', False)
                        
                        lines.append(f"   Query: '{query}' - Found {result_count} results (ranking: {'applied' if ranking_applied else 'none'})")
                        lines.append(f"   Results (sorted by relevance_score):")
                        
                        # Show ALL results with full URLs and ranking info
                        for r in search_results:
                            title = r.get('title', 'N/A')[:80]
                            url = r.get('url', 'N/A')
                            snippet = r.get('snippet', '')[:150]
                            score = r.get('relevance_score', 0)
                            position = r.get('position', 0)
                            domain = r.get('domain', '')
                            ranking_signals = r.get('ranking_signals', {})
                            
                            # Format ranking signals compactly
                            signals_str = ', '.join([f"{k}:{v:.2f}" for k, v in ranking_signals.items()]) if ranking_signals else ''
                            
                            lines.append(f"   {position}. [{score:.3f}] {title}")
                            lines.append(f"      URL: {url}")
                            lines.append(f"      Snippet: {snippet}")
                            if signals_str:
                                lines.append(f"      Signals: {signals_str}")
                        
                        # Add answer box if present
                        if result.get('answer_box'):
                            lines.append(f"   Answer Box: {str(result['answer_box'])[:200]}")
                    else:
                        result_preview = str(result)[:500]
                        lines.append(f"   Result: {result_preview}")
                # Other successful actions - especially extraction/JS results
                else:
                    tool_result = entry.observation.result
                    result_data = tool_result.data if hasattr(tool_result, 'data') else tool_result
                    
                    # For extraction tools, store the FULL result - this is critical data!
                    # Truncation should only happen at prompt formatting time based on token budget,
                    # not at storage time where we lose information permanently.
                    # The LLM needs this data to avoid hallucinating results.
                    if entry.action.tool_name in ("evaluate_javascript", "extract_text", "extract_data", "extract_structured_data", "get_page_content"):
                        # Store full extraction result - let prompt formatting handle truncation
                        result_preview = str(result_data) if result_data else "[Success]"
                    else:
                        # Non-extraction actions can be summarized more aggressively
                        result_preview = str(result_data)[:500] if result_data else "[Success]"
                    lines.append(f"   Result: {result_preview}")
            elif entry.observation and entry.outcome == ExecutionOutcome.FAILURE:
                error_msg = entry.observation.error or "Unknown error"
                lines.append(f"   Error: {error_msg}")
        
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear short-term memory."""
        self._entries.clear()
        self._failed_signatures.clear()
        self._action_frequency.clear()



@dataclass
class LearnedPattern:
    """
    Site-specific pattern learned during execution.

    Stores reusable knowledge about how to interact with
    specific sites or accomplish specific tasks.
    """
    pattern_id: str
    domain: str
    description: str
    pattern_type: str  # e.g., "login_flow", "navigation", "form_fill"
    steps: List[Dict[str, Any]]
    success_count: int = 0
    failure_count: int = 0
    last_used: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0




class WorkingMemory:
    """
    Working memory for active reasoning.

    Tracks the current goal, active TAO cycle, and navigation
    path during task execution.
    """

    def __init__(self, token_budget: int = 4000) -> None:
        """Initialize working memory."""
        self.token_budget = token_budget
        self._current_goal: Optional[str] = None
        self._active_thought: Optional[Thought] = None
        self._active_action: Optional[Action] = None
        self._navigation_path: List[StateSnapshot] = []
        self._scratch_pad: Dict[str, Any] = {}
        self._cycle_count: int = 0

    def set_goal(self, goal: str) -> None:
        """Set the current task goal."""
        self._current_goal = goal
        self._cycle_count = 0
        logger.info(f"Working memory: Goal set - {goal[:100]}")

    def start_cycle(self, thought: Thought) -> None:
        """Start a new TAO cycle."""
        self._active_thought = thought
        self._active_action = None
        self._cycle_count += 1
        logger.debug(f"Working memory: Cycle {self._cycle_count} started")

    def set_action(self, action: Action) -> None:
        """Set the active action."""
        self._active_action = action

    def record_navigation(self, snapshot: StateSnapshot) -> None:
        """Record a navigation state."""
        self._navigation_path.append(snapshot)
        # Keep navigation history manageable
        if len(self._navigation_path) > 20:
            self._navigation_path = self._navigation_path[-15:]

    def get_navigation_history(self) -> List[StateSnapshot]:
        """Get navigation history."""
        return self._navigation_path.copy()

    def set_scratch(self, key: str, value: Any) -> None:
        """Store temporary data in scratch pad."""
        self._scratch_pad[key] = value

    def get_scratch(self, key: str, default: Any = None) -> Any:
        """Get temporary data from scratch pad."""
        return self._scratch_pad.get(key, default)

    @property
    def current_goal(self) -> Optional[str]:
        """Get the current goal."""
        return self._current_goal

    @property
    def cycle_count(self) -> int:
        """Get the current cycle count."""
        return self._cycle_count

    def format_for_prompt(self) -> str:
        """Format working memory for prompt."""
        lines = []
        if self._current_goal:
            lines.append(f"## Current Goal\n{self._current_goal}")
        
        # Include PageMap data prominently if available (critical for agent efficiency!)
        page_maps_formatted = []
        for key, value in self._scratch_pad.items():
            if key.startswith("page_map:") and hasattr(value, 'format_for_prompt'):
                # This is a PageMap object - use its dedicated formatting
                page_maps_formatted.append(value.format_for_prompt(include_screenshots=False, include_navigation=True))
        
        if page_maps_formatted:
            lines.append("\n## 📍 AVAILABLE PAGE DATA (use instead of re-extracting!)")
            lines.extend(page_maps_formatted)
        
        # Include other scratchpad data (useful for tracking state across iterations)
        # IMPORTANT: Extraction data is stored WITHOUT truncation here.
        # Truncation based on token budget happens in AgentMemory.format_for_prompt()
        other_scratch_data = []
        for key, value in self._scratch_pad.items():
            # Skip page_map entries since we handled them above
            if key.startswith("page_map:"):
                continue
            # Extraction data gets full storage - it's critical for avoiding hallucination
            if key.startswith("extracted_"):
                # Full extraction data - no truncation
                other_scratch_data.append(f"- {key}: {value}")
            elif isinstance(value, list):
                if len(value) > 10:
                    other_scratch_data.append(f"- {key}: [{len(value)} items] {value[:5]}...")
                else:
                    other_scratch_data.append(f"- {key}: {value}")
            elif isinstance(value, dict):
                other_scratch_data.append(f"- {key}: {len(value)} entries")
            else:
                # Regular values can be summarized
                value_str = str(value)[:200]
                other_scratch_data.append(f"- {key}: {value_str}")
        
        if other_scratch_data:
            lines.append("\n## Working Data")
            lines.extend(other_scratch_data)
        
        if self._navigation_path:
            lines.append("\n## Navigation Path")
            for i, snap in enumerate(self._navigation_path[-5:], 1):
                lines.append(f"{i}. {snap.format_brief()}")
        return "\n".join(lines) if lines else "No active context."

    def clear(self) -> None:
        """Clear working memory."""
        self._current_goal = None
        self._active_thought = None
        self._active_action = None
        self._navigation_path.clear()
        self._scratch_pad.clear()
        self._cycle_count = 0


class LongTermMemory:
    """
    Long-term memory for persistent patterns.

    Stores learned patterns and site-specific knowledge
    that persists across task executions.
    """

    def __init__(self, max_patterns: int = 100) -> None:
        """Initialize long-term memory."""
        self.max_patterns = max_patterns
        self._patterns: Dict[str, LearnedPattern] = {}
        self._domain_index: Dict[str, List[str]] = {}  # domain -> pattern_ids

    def add_pattern(self, pattern: LearnedPattern) -> None:
        """Add or update a learned pattern."""
        self._patterns[pattern.pattern_id] = pattern

        # Update domain index
        if pattern.domain not in self._domain_index:
            self._domain_index[pattern.domain] = []
        if pattern.pattern_id not in self._domain_index[pattern.domain]:
            self._domain_index[pattern.domain].append(pattern.pattern_id)

        logger.info(f"LTM: Pattern added/updated - {pattern.pattern_id}")

        # Prune if over limit
        if len(self._patterns) > self.max_patterns:
            self._prune_patterns()

    def _prune_patterns(self) -> None:
        """Remove least useful patterns."""
        # Sort by success rate and recency
        sorted_patterns = sorted(
            self._patterns.values(),
            key=lambda p: (p.success_rate, p.last_used),
        )
        # Remove bottom 20%
        to_remove = len(self._patterns) - int(self.max_patterns * 0.8)
        for pattern in sorted_patterns[:to_remove]:
            self.remove_pattern(pattern.pattern_id)

    def remove_pattern(self, pattern_id: str) -> None:
        """Remove a pattern."""
        if pattern_id in self._patterns:
            pattern = self._patterns[pattern_id]
            if pattern.domain in self._domain_index:
                self._domain_index[pattern.domain] = [
                    pid for pid in self._domain_index[pattern.domain]
                    if pid != pattern_id
                ]
            del self._patterns[pattern_id]

    def get_patterns_for_domain(self, domain: str) -> List[LearnedPattern]:
        """Get patterns for a specific domain."""
        pattern_ids = self._domain_index.get(domain, [])
        return [self._patterns[pid] for pid in pattern_ids if pid in self._patterns]

    def get_pattern(self, pattern_id: str) -> Optional[LearnedPattern]:
        """Get a specific pattern."""
        return self._patterns.get(pattern_id)

    def record_usage(self, pattern_id: str, success: bool) -> None:
        """Record pattern usage outcome."""
        if pattern_id in self._patterns:
            pattern = self._patterns[pattern_id]
            if success:
                pattern.success_count += 1
            else:
                pattern.failure_count += 1
            pattern.last_used = time.time()

    def format_relevant_patterns(self, domain: str, max_patterns: int = 3) -> str:
        """Format relevant patterns for prompt."""
        patterns = self.get_patterns_for_domain(domain)
        if not patterns:
            return ""

        # Sort by success rate
        patterns.sort(key=lambda p: p.success_rate, reverse=True)
        patterns = patterns[:max_patterns]

        lines = ["## Learned Patterns for This Site"]
        for p in patterns:
            lines.append(f"- {p.description} (success rate: {p.success_rate:.0%})")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear long-term memory."""
        self._patterns.clear()
        self._domain_index.clear()



class AgentMemory:
    """
    Main memory class combining all memory types.

    Provides a unified interface for memory operations during
    ReAct agent execution, managing context window limits
    and coordinating between memory subsystems.
    """

    def __init__(
        self,
        context_window_budget: int = 64000,  # Modern LLMs have 128K+ context windows
        short_term_max_entries: int = 50,  # Keep more history
        long_term_max_patterns: int = 100,
    ) -> None:
        """Initialize agent memory system."""
        self.context_window_budget = context_window_budget

        # Initialize memory subsystems
        self.short_term = ShortTermMemory(max_entries=short_term_max_entries)
        self.working = WorkingMemory(token_budget=context_window_budget // 4)
        self.long_term = LongTermMemory(max_patterns=long_term_max_patterns)
        self.context_store = ContextStore()
        
        # Track visited URLs to prevent revisiting
        self._visited_urls: Set[str] = set()
        
        # Operation mode tracking for mode-aware memory management
        self._operation_mode: Optional[OperationMode] = None

        logger.info("Agent memory system initialized")

    def start_task(self, goal: str, user_context: Optional[Dict[str, Any]] = None) -> None:
        """Start a new task with the given goal."""
        # Preserve page context that was set by SDK BEFORE this call
        # The SDK sets current_url/current_title so the agent knows what page it's on
        preserved_url = self.working.get_scratch("current_url")
        preserved_title = self.working.get_scratch("current_title")
        
        # Clear working memory to remove stale data from previous tasks
        # This prevents old extraction results from polluting the new task
        self.working.clear()
        
        # Restore the page context that was set just before this call
        # This is CRITICAL - without it the agent doesn't know what page it's on!
        if preserved_url:
            self.working.set_scratch("current_url", preserved_url)
        if preserved_title:
            self.working.set_scratch("current_title", preserved_title)
        
        # Now set the new goal
        self.working.set_goal(goal)
        
        # Clear other task-specific memory
        self.short_term.clear()
        self._visited_urls.clear()  # Clear visited URLs for new task

        if user_context:
            self.context_store.set_user_context(user_context)

        logger.info(f"Task started: {goal[:100]}")

    def record_cycle(
        self,
        thought: Thought,
        action: Action,
        observation: Observation,
        outcome: ExecutionOutcome = ExecutionOutcome.SUCCESS,
        priority: MemoryPriority = MemoryPriority.NORMAL,
        tags: Optional[List[str]] = None,
    ) -> MemoryEntry:
        """Record a complete TAO cycle."""
        entry = MemoryEntry(
            entry_id=self.short_term.generate_entry_id(),
            timestamp=time.time(),
            thought=thought,
            action=action,
            observation=observation,
            outcome=outcome,
            priority=priority,
            tags=set(tags) if tags else set(),
        )

        self.short_term.add(entry)
        self.working.start_cycle(thought)
        self.working.set_action(action)

        return entry

    def record_navigation_state(self, snapshot: StateSnapshot) -> None:
        """Record a navigation state snapshot."""
        self.working.record_navigation(snapshot)
        # Track visited URL
        if snapshot.url:
            self._visited_urls.add(snapshot.url)

    def get_relevant_context(self, query: str, max_tokens: int = 2000) -> str:
        """Get relevant context for a query."""
        relevant = self.context_store.get_relevant_context(query)
        # Format as string
        if not relevant:
            return ""
        lines = []
        for key, value in relevant.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def add_inferred_context(self, key: str, value: Any, source: str = "") -> None:
        """Add inferred context from observations."""
        self.context_store.add_inferred_context(key, value, source=source)

    def get_patterns_for_domain(self, domain: str) -> List[LearnedPattern]:
        """Get learned patterns for a domain."""
        return self.long_term.get_patterns_for_domain(domain)

    def learn_pattern(self, pattern: LearnedPattern) -> None:
        """Learn a new pattern from successful execution."""
        self.long_term.add_pattern(pattern)

    def is_repeated_failure(self, action: Action) -> bool:
        """Check if an action signature has failed before."""
        # Generate action signature
        import hashlib
        sig_parts = [
            action.tool_name,
            str(sorted(action.parameters.items())),
        ]
        action_signature = hashlib.md5(":".join(sig_parts).encode()).hexdigest()[:16]
        return self.short_term.has_failed(action_signature)
    
    def has_visited_url(self, url: str) -> bool:
        """Check if a URL has been visited in this task."""
        return url in self._visited_urls
    
    def get_visited_urls(self) -> List[str]:
        """Get list of visited URLs."""
        return list(self._visited_urls)
    
    def mark_url_visited(self, url: str) -> None:
        """Mark a URL as visited."""
        self._visited_urls.add(url)
        logger.debug(f"Marked URL as visited: {url}")

    def format_for_prompt(self, domain: Optional[str] = None, token_budget: Optional[int] = None) -> str:
        """
        Format all memory for inclusion in LLM prompt.

        Prioritizes extraction data to prevent LLM hallucination in text-only mode.
        Combines working memory, recent history, learned patterns,
        and relevant context within token budget.
        
        Priority order (highest to lowest):
        1. Current goal (essential)
        2. Extraction data from tools (CRITICAL - prevents hallucination)
        3. Recent action history
        4. Navigation/visited URLs
        5. Learned patterns
        6. User context
        
        Args:
            domain: Optional domain for filtering learned patterns.
                    If not provided, extracts from current URL in working memory.
            token_budget: Optional override for context window budget.
        """
        sections = []
        remaining_budget = token_budget or self.context_window_budget
        
        def estimate_tokens(text: str) -> int:
            """Estimate tokens (~4 chars per token)."""
            return max(1, len(text) // 4)
        
        def add_section(content: str, priority: str = "normal") -> bool:
            """Add section if budget allows. Returns True if added."""
            nonlocal remaining_budget
            tokens = estimate_tokens(content)
            if tokens < remaining_budget:
                sections.append(content)
                remaining_budget -= tokens
                return True
            elif priority == "critical":
                # For critical content, truncate to fit but don't skip
                available_chars = remaining_budget * 4
                if available_chars > 500:  # Only add if we can fit meaningful content
                    truncated = content[:available_chars - 50] + "\n... [truncated]"
                    sections.append(truncated)
                    remaining_budget = 0
                    return True
            return False

        # 1. CRITICAL: Current goal (always include)
        if self.working._current_goal:
            add_section(f"## Current Goal\n{self.working._current_goal}", priority="critical")
        
        # 1.5. CRITICAL: Current page context (so agent knows where it is!)
        # This prevents the agent from asking "what URL?" when already on a page
        current_url = self.working.get_scratch("current_url")
        current_title = self.working.get_scratch("current_title")
        if current_url:
            page_context = f"## Current Page\nYou are already on: {current_url}"
            if current_title:
                page_context += f"\nPage title: {current_title}"
            page_context += "\n\n**IMPORTANT: Extract data from THIS page. Do NOT ask for URL - you're already on the page!**"
            add_section(page_context, priority="critical")

        # 2. CRITICAL: Extraction data from working memory scratch pad
        # This is the most important data for text-only mode to avoid hallucination
        extraction_lines = []
        for key, value in self.working._scratch_pad.items():
            if key.startswith("extracted_"):
                extraction_lines.append(f"### {key}")
                extraction_lines.append(str(value))
        
        if extraction_lines:
            extraction_content = "## Extracted Data (from previous tool calls - USE THIS DATA!)\n" + "\n".join(extraction_lines)
            add_section(extraction_content, priority="critical")

        # 3. Recent execution history (includes tool results)
        history_content = self.short_term.format_history(last_n=5)
        if history_content:
            add_section(history_content)

        # 4. Working memory (navigation path, page maps, other scratch data)
        # Skip extraction data since we already added it with higher priority
        working_lines = []
        
        # Navigation path
        nav_path = self.working._navigation_path
        if nav_path:
            working_lines.append("## Navigation Path")
            for i, snap in enumerate(nav_path[-5:], 1):
                working_lines.append(f"{i}. {snap.format_brief()}")
        
        # Page maps (for vision mode)
        for key, value in self.working._scratch_pad.items():
            if key.startswith("page_map:") and hasattr(value, 'format_for_prompt'):
                working_lines.append(value.format_for_prompt(include_screenshots=False, include_navigation=True))
            elif not key.startswith("extracted_") and not key.startswith("page_map:") and key not in ("current_url", "current_title"):
                # Other scratch data (non-extraction, non-page context which is already shown above)
                if isinstance(value, (list, dict)):
                    working_lines.append(f"- {key}: {len(value)} items")
                else:
                    working_lines.append(f"- {key}: {str(value)[:200]}")
        
        if working_lines:
            add_section("\n".join(working_lines))
        
        # 5. Visited URLs (lower priority)
        if self._visited_urls and remaining_budget > 100:
            visited_content = f"## Visited URLs\n{', '.join(sorted(self._visited_urls))}"
            add_section(visited_content)

        # Extract domain from current URL if not provided
        if not domain:
            current_url = self.working.get_scratch("current_url")
            if current_url:
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(current_url)
                    domain = parsed.netloc
                except Exception:
                    pass

        # 6. Learned patterns for domain (lowest priority)
        if domain and remaining_budget > 200:
            patterns_content = self.long_term.format_relevant_patterns(domain)
            if patterns_content:
                add_section(patterns_content)
        
        # 7. User context if provided (lowest priority)
        user_context = self.context_store.format_for_prompt()
        if user_context and user_context != "No context provided." and remaining_budget > 100:
            add_section(user_context)

        return "\n\n".join(sections) if sections else ""

    def get_failure_count(self) -> int:
        """Get the count of failed entries."""
        return len(self.short_term.get_by_outcome(ExecutionOutcome.FAILURE))
    
    def get_task_summary(self) -> str:
        """
        Get a brief summary of the current task execution.
        
        Compiles information from working memory, recent actions,
        and execution statistics into a readable summary.
        
        Returns:
            Human-readable task summary string
        """
        parts = []
        
        # Goal
        if self.working.current_goal:
            parts.append(f"Goal: {self.working.current_goal}")
        
        # Execution stats
        total_actions = len(self.short_term._entries)
        successful = len(self.short_term.get_by_outcome(ExecutionOutcome.SUCCESS))
        failed = len(self.short_term.get_by_outcome(ExecutionOutcome.FAILURE))
        
        parts.append(f"Actions: {total_actions} total ({successful} successful, {failed} failed)")
        
        # Current URL
        current_url = self.working.get_scratch("current_url")
        if current_url:
            parts.append(f"Current page: {current_url}")
        
        # Navigation history
        nav_path = self.working.get_navigation_history()
        if nav_path:
            parts.append(f"Pages visited: {len(nav_path)}")
        
        # Include stored PageMaps summary (for site exploration tasks)
        page_maps = self.get_all_page_maps()
        if page_maps:
            parts.append(f"\nExplored pages ({len(page_maps)} total):")
            for url, page_map in page_maps.items():
                title = getattr(page_map, 'title', 'Untitled')
                summary = getattr(page_map, 'summary', '')[:150]
                parts.append(f"  - {title}: {summary}...")
        
        # Key observations from successful actions
        recent_successes = self.short_term.get_by_outcome(ExecutionOutcome.SUCCESS)[-3:]
        if recent_successes:
            recent_actions = [e.action.tool_name for e in recent_successes]
            parts.append(f"\nRecent successful actions: {', '.join(recent_actions)}")
        
        return "\n".join(parts) if parts else "Task in progress"

    @property
    def cycle_count(self) -> int:
        """Get the current cycle count."""
        return self.working.cycle_count

    @property
    def current_goal(self) -> Optional[str]:
        """Get the current goal."""
        return self.working.current_goal

    def clear_task(self) -> None:
        """Clear task-specific memory (short-term and working)."""
        self.short_term.clear()
        self.working.clear()
        logger.info("Task memory cleared")

    def clear_all(self) -> None:
        """Clear all memory including long-term patterns."""
        self.short_term.clear()
        self.working.clear()
        self.long_term.clear()
        self.context_store.clear()
        logger.info("All memory cleared")
    
    # PageMap storage methods for spatial page understanding
    def store_page_map(self, url: str, page_map: Any) -> None:
        """
        Store PageMap in working memory with TTL.
        
        PageMaps contain comprehensive spatial understanding of web pages
        including screenshots, sections, and navigation structure.
        
        Args:
            url: URL of the page
            page_map: PageMap object with page understanding
        """
        self.working.set_scratch(f"page_map:{url}", page_map)
        logger.info(
            f"[Memory] Stored PageMap for {url}: "
            f"{len(page_map.screenshots)} screenshots, "
            f"{len(page_map.sections)} sections, "
            f"{page_map.get_coverage_percentage():.1f}% coverage"
        )
    
    def get_page_map(self, url: str) -> Any:
        """
        Retrieve PageMap from memory.
        
        Args:
            url: URL of the page
            
        Returns:
            PageMap object if found, None otherwise
        """
        return self.working.get_scratch(f"page_map:{url}")
    
    def get_page_summary(self, url: str) -> str:
        """
        Get quick page summary from stored PageMap.
        
        Args:
            url: URL of the page
            
        Returns:
            Page summary string, or empty string if not found
        """
        page_map = self.get_page_map(url)
        if page_map and hasattr(page_map, 'summary'):
            return page_map.summary
        return ""
    
    def has_page_map(self, url: str) -> bool:
        """
        Check if PageMap exists for URL.
        
        Args:
            url: URL to check
            
        Returns:
            True if PageMap is stored
        """
        return self.get_page_map(url) is not None
    
    def get_all_page_maps(self) -> Dict[str, Any]:
        """
        Get all stored PageMaps for site exploration aggregation.
        
        Returns:
            Dictionary of URL -> PageMap for all stored pages
        """
        page_maps = {}
        for key, value in self.working._scratch_pad.items():
            if key.startswith("page_map:"):
                url = key[9:]  # Remove 'page_map:' prefix
                page_maps[url] = value
        return page_maps
    
    def get_all_page_summaries(self) -> Dict[str, str]:
        """
        Get summaries from all stored PageMaps.
        
        Useful for final aggregation in site exploration tasks.
        
        Returns:
            Dictionary of URL -> summary string for all stored pages
        """
        summaries = {}
        for url, page_map in self.get_all_page_maps().items():
            if hasattr(page_map, 'summary') and page_map.summary:
                summaries[url] = page_map.summary
            elif hasattr(page_map, 'title') and page_map.title:
                summaries[url] = f"Page: {page_map.title}"
        return summaries
    
    def format_site_exploration_summary(self) -> str:
        """
        Format a comprehensive summary of all visited pages for site exploration.
        
        This is used when the agent needs to aggregate content from multiple
        visited pages into a final summary.
        
        Returns:
            Formatted string with summaries from all visited pages
        """
        page_maps = self.get_all_page_maps()
        if not page_maps:
            return "No pages have been explored yet."
        
        lines = [f"## Site Exploration Summary ({len(page_maps)} pages visited)"]
        
        for url, page_map in page_maps.items():
            # Extract key info from each PageMap
            title = getattr(page_map, 'title', 'Untitled')
            summary = getattr(page_map, 'summary', '')
            sections = getattr(page_map, 'sections', [])
            
            lines.append(f"\n### {title}")
            lines.append(f"URL: {url}")
            
            if sections:
                section_names = [s.name for s in sections if hasattr(s, 'name')]
                if section_names:
                    lines.append(f"Sections: {', '.join(section_names)}")
            
            if summary:
                # Truncate long summaries
                if len(summary) > 300:
                    summary = summary[:297] + "..."
                lines.append(f"Summary: {summary}")
        
        return "\n".join(lines)
    
    # Operation mode tracking for mode-aware memory management
    def set_operation_mode(self, mode: OperationMode) -> None:
        """
        Set the current operation mode.
        
        This affects memory retention strategies, pruning priorities,
        and context formatting for optimal performance per mode.
        
        Args:
            mode: The detected operation mode for current task
        """
        self._operation_mode = mode
        logger.info(f"[Memory] Operation mode set: {mode.value}")
    
    def get_operation_mode(self) -> Optional[OperationMode]:
        """
        Get the current operation mode.
        
        Returns:
            Current OperationMode or None if not set
        """
        return self._operation_mode
    
    # SitemapGraph integration for site exploration tracking
    def init_sitemap_graph(self, homepage_url: str, limits: Optional[Any] = None) -> Any:
        """
        Initialize SitemapGraph for multi-page site exploration.
        
        This creates a graph-based tracking structure that maintains:
        - Discovered vs visited page states
        - Depth-limited traversal enforcement
        - Parent-child navigation hierarchy
        - Real-time exploration status
        
        Args:
            homepage_url: Starting URL (Level 0)
            limits: Optional SitemapLimits configuration
            
        Returns:
            Initialized SitemapGraph instance
        """
        from .sitemap_graph import SitemapGraph, SitemapLimits
        
        if limits is None:
            limits = SitemapLimits()
        
        graph = SitemapGraph(homepage_url, limits=limits)
        self.working.set_scratch("sitemap_graph", graph)
        logger.info(f"[Memory] SitemapGraph initialized for {homepage_url}")
        return graph
    
    def get_sitemap_graph(self) -> Optional[Any]:
        """
        Get the current SitemapGraph if one exists.
        
        Returns:
            SitemapGraph instance or None
        """
        return self.working.get_scratch("sitemap_graph")
    
    def has_sitemap_graph(self) -> bool:
        """Check if a SitemapGraph exists for current exploration."""
        return self.get_sitemap_graph() is not None
    
    def update_sitemap_visited(
        self,
        url: str,
        title: str = "",
        summary: str = "",
        section_count: int = 0
    ) -> bool:
        """
        Mark a page as visited in the SitemapGraph.
        
        Automatically syncs with PageMap storage.
        
        Args:
            url: Page URL
            title: Page title
            summary: Page summary from analysis
            section_count: Number of sections found
            
        Returns:
            True if updated, False if no SitemapGraph exists
        """
        graph = self.get_sitemap_graph()
        if not graph:
            return False
        
        graph.mark_visited(
            url=url,
            title=title,
            summary=summary,
            section_count=section_count,
            page_map_stored=self.has_page_map(url)
        )
        return True
    
    def add_sitemap_links(
        self,
        parent_url: str,
        links: List[Dict[str, str]],
        link_type: str = "unknown"
    ) -> int:
        """
        Add discovered links to the SitemapGraph.
        
        Args:
            parent_url: URL where links were discovered
            links: List of dicts with 'url' and 'text' keys
            link_type: Type of navigation (main_nav, footer, content, etc.)
            
        Returns:
            Number of new links added
        """
        graph = self.get_sitemap_graph()
        if not graph:
            return 0
        
        from .sitemap_graph import LinkType
        try:
            lt = LinkType(link_type)
        except ValueError:
            lt = LinkType.UNKNOWN
        
        return graph.add_discovered_links(parent_url, links, lt)
    
    def get_sitemap_status(self) -> str:
        """
        Get formatted exploration status for agent context.
        
        Returns:
            Formatted status string or empty if no SitemapGraph
        """
        graph = self.get_sitemap_graph()
        if not graph:
            return ""
        return graph.format_exploration_status()
    
    def get_next_page_to_visit(self) -> Optional[str]:
        """
        Get the next pending page URL from SitemapGraph.
        
        Returns:
            URL to visit next, or None if exploration complete
        """
        graph = self.get_sitemap_graph()
        if not graph:
            return None
        
        node = graph.get_next_pending()
        return node.url if node else None
    
    def is_sitemap_exploration_complete(self) -> bool:
        """
        Check if site exploration is complete.
        
        Returns:
            True if all pages visited or limits reached
        """
        graph = self.get_sitemap_graph()
        if not graph:
            return True  # No graph = nothing to explore
        return graph.is_exploration_complete()
