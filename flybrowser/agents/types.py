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
Core types and data structures for the ReAct agentic framework.

This module provides the foundational types for implementing the ReAct
(Reasoning and Acting) paradigm with explicit Thought-Action-Observation cycles.

Key Components:
- ReActStep: Complete reasoning step with thought, action, and observation
- Thought: Explicit reasoning with confidence scoring
- Action: Tool invocation with parameters
- Observation: Result from action execution
- ToolResult: Standardized tool response format
- ExecutionState: Agent execution lifecycle states
- ReasoningStrategy: Available reasoning strategies (CoT, ToT, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid


class ReasoningStrategy(str, Enum):
    """Available reasoning strategies for the ReAct agent."""
    
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Linear step-by-step reasoning
    TREE_OF_THOUGHT = "tree_of_thought"    # Explore multiple reasoning paths
    REACT_STANDARD = "react_standard"       # Standard ReAct loop
    SELF_REFLECTION = "self_reflection"     # Reflect on previous actions
    PLAN_AND_SOLVE = "plan_and_solve"       # Plan first, then execute


class ExecutionState(str, Enum):
    """States of agent execution lifecycle."""
    
    IDLE = "idle"                    # Agent is idle, waiting for task
    THINKING = "thinking"            # Agent is reasoning about next action
    ACTING = "acting"                # Agent is executing an action
    OBSERVING = "observing"          # Agent is processing observation
    WAITING_APPROVAL = "waiting_approval"  # Waiting for human approval
    PAUSED = "paused"                # Execution paused
    COMPLETED = "completed"          # Task completed successfully
    FAILED = "failed"                # Task failed
    CANCELLED = "cancelled"          # Task was cancelled


class ExecutionOutcome(str, Enum):
    """Outcome of a ReAct step execution."""

    SUCCESS = "success"              # Step executed successfully
    FAILURE = "failure"              # Step execution failed
    PARTIAL = "partial"              # Step partially completed
    SKIPPED = "skipped"              # Step was skipped
    PENDING = "pending"              # Step pending execution
    CANCELLED = "cancelled"          # Step was cancelled


class MemoryPriority(int, Enum):
    """Priority levels for memory retention."""

    CRITICAL = 4  # Never prune (errors, key discoveries)
    HIGH = 3      # Keep as long as possible (successful patterns)
    NORMAL = 2    # Standard retention
    LOW = 1       # First to prune when space needed


class ToolCategory(str, Enum):
    """Categories of tools for organization and access control."""
    
    NAVIGATION = "navigation"        # Browser navigation tools
    INTERACTION = "interaction"      # Page interaction (click, type, etc.)
    EXTRACTION = "extraction"        # Data extraction tools
    SEARCH = "search"                # Search tools (web search, ranking, etc.)
    SYSTEM = "system"                # System tools (complete, fail, etc.)
    UTILITY = "utility"              # Helper utilities
    DANGEROUS = "dangerous"          # Tools requiring approval


class SafetyLevel(str, Enum):
    """Safety classification for operations."""
    
    SAFE = "safe"                    # No confirmation needed
    MODERATE = "moderate"            # May need logging
    SENSITIVE = "sensitive"          # Requires extra caution
    DANGEROUS = "dangerous"          # Requires human approval


class ErrorCode(str, Enum):
    """Standardized error codes for tool execution failures."""
    
    # Parameter/Input Errors
    INVALID_PARAMS = "INVALID_PARAMS"              # Invalid or missing parameters
    INVALID_SELECTOR = "INVALID_SELECTOR"          # Invalid CSS/XPath selector
    INVALID_URL = "INVALID_URL"                    # Malformed or invalid URL
    
    # Execution Errors
    TIMEOUT = "TIMEOUT"                            # Operation timed out
    EXECUTION_ERROR = "EXECUTION_ERROR"            # General execution failure
    NOT_FOUND = "NOT_FOUND"                        # Element/resource not found
    
    # Network/Connection Errors
    NETWORK_ERROR = "NETWORK_ERROR"                # Network connectivity issue
    NAVIGATION_ERROR = "NAVIGATION_ERROR"          # Navigation failed
    LOAD_FAILED = "LOAD_FAILED"                    # Page load failed
    
    # Permission/Security Errors
    PERMISSION_DENIED = "PERMISSION_DENIED"        # Insufficient permissions
    BLOCKED = "BLOCKED"                            # Action blocked (CORS, CSP, etc.)
    
    # Content/Parsing Errors
    PARSE_ERROR = "PARSE_ERROR"                    # Failed to parse content
    EXTRACTION_FAILED = "EXTRACTION_FAILED"        # Data extraction failed
    
    # State Errors
    INVALID_STATE = "INVALID_STATE"                # Invalid state for operation
    PAGE_NOT_LOADED = "PAGE_NOT_LOADED"            # No page loaded
    
    # Tool-Specific Errors
    SEARCH_FAILED = "SEARCH_FAILED"                # Search operation failed
    CAPTCHA_DETECTED = "CAPTCHA_DETECTED"          # CAPTCHA blocking action
    RATE_LIMITED = "RATE_LIMITED"                  # Rate limit exceeded


class OperationMode(str, Enum):
    """
    Operation modes that determine execution strategy across the entire ReAct framework.
    
    Each mode optimizes behavior for specific task types:
    - Vision strategy (frequency, scope)
    - Page exploration depth
    - Memory retention patterns
    - Prompt template selection
    - Planning approach
    - Performance/cost trade-offs
    """
    
    NAVIGATE = "navigate"
    # High-level exploration and navigation
    # Focus: Understanding entire page structure, discovering site architecture
    # Keywords: explore, browse, navigate, visit, discover, tour, look around
    # Characteristics: Full page exploration, comprehensive memory, detailed prompts
    # Vision: High frequency (every 2 iterations)
    # Exploration: FULL scope (8-10 screenshots)
    # Use Cases: "Explore the website", "Browse the catalog", "Tour the dashboard"
    
    EXECUTE = "execute"
    # Precise targeted interactions
    # Focus: Fast, specific actions with minimal overhead
    # Keywords: click, fill, submit, select, type, press, tap, enter, choose
    # Characteristics: Minimal exploration, focused vision, concise prompts
    # Vision: Minimal frequency (first iteration + failures only)
    # Exploration: VIEWPORT scope (1 screenshot)
    # Use Cases: "Click the login button", "Fill out the form", "Submit the order"
    
    SCRAPE = "scrape"
    # Data extraction and collection
    # Focus: Extracting structured data from pages efficiently
    # Keywords: extract, scrape, collect, gather, fetch, get data, list all, retrieve
    # Characteristics: Content-focused exploration, data memory, extraction prompts
    # Vision: Balanced frequency (page structure understanding)
    # Exploration: CONTENT scope (3-5 screenshots)
    # Use Cases: "Extract all product prices", "Scrape contact information", "Collect reviews"
    
    RESEARCH = "research"
    # Information discovery and research
    # Focus: Finding and understanding information, content analysis
    # Keywords: find, search, research, locate, discover, investigate, look for
    # Characteristics: Smart exploration, summary memory, analytical prompts
    # Vision: Balanced frequency (every 3 iterations)
    # Exploration: SMART scope (adaptive 2-6 screenshots)
    # Use Cases: "Find pricing information", "Research company details", "Locate contact page"
    
    AUTO = "auto"
    # Automatic mode detection
    # Falls back to intelligent mode selection based on task analysis
    # Uses balanced settings when mode cannot be confidently determined
    # Default fallback for backward compatibility


@dataclass
class Thought:
    """
    Explicit reasoning step in the ReAct cycle.
    
    Represents the agent's internal reasoning about the current state,
    what action to take next, and why.
    """
    
    content: str                     # The reasoning content
    confidence: float = 0.0          # Confidence score (0.0 to 1.0)
    strategy: ReasoningStrategy = ReasoningStrategy.REACT_STANDARD
    reasoning_tokens: int = 0        # Tokens used for reasoning
    alternatives_considered: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate confidence is within bounds."""
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class Action:
    """
    Tool invocation in the ReAct cycle.
    
    Represents a specific tool to be called with its parameters.
    """
    
    tool_name: str                   # Name of the tool to invoke
    parameters: Dict[str, Any] = field(default_factory=dict)
    safety_level: SafetyLevel = SafetyLevel.SAFE
    requires_approval: bool = False
    timeout_seconds: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary for serialization."""
        return {
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "safety_level": self.safety_level.value,
            "requires_approval": self.requires_approval,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class ToolResult:
    """
    Standardized result from tool execution.
    
    All tools return this format to ensure consistent handling.
    """
    
    success: bool
    data: Any = None                 # Result data (varies by tool)
    error: Optional[str] = None      # Error message if failed
    error_code: Optional[str] = None # Machine-readable error code
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def success_result(cls, data: Any, **metadata) -> ToolResult:
        """Create a successful tool result."""
        return cls(success=True, data=data, metadata=metadata)
    
    @classmethod
    def error_result(
        cls,
        error: str,
        error_code: Optional[str] = None,
        data: Any = None,
        **metadata
    ) -> ToolResult:
        """Create an error tool result with optional context data."""
        return cls(success=False, error=error, error_code=error_code, data=data, metadata=metadata)


@dataclass
class Observation:
    """
    Result observation from executing an action.

    Captures both raw output and parsed/structured data from tool execution.
    """

    result: ToolResult                # The tool execution result
    source: str = ""                  # Source of observation (tool name)
    raw_output: str = ""              # Raw string output
    parsed_data: Any = None           # Parsed/structured data
    summary: str = ""                 # Human-readable summary
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success(self) -> bool:
        """Check if the observation indicates success."""
        return self.result.success

    @property
    def error(self) -> Optional[str]:
        """Get error message if observation indicates failure."""
        return self.result.error


@dataclass
class ReActStep:
    """
    Complete ReAct cycle step: Thought -> Action -> Observation.

    Represents one full iteration of the ReAct loop with all components.
    """

    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_number: int = 0
    thought: Optional[Thought] = None
    action: Optional[Action] = None
    observation: Optional[Observation] = None
    state: ExecutionState = ExecutionState.IDLE
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0

    def complete(self, state: ExecutionState = ExecutionState.COMPLETED) -> None:
        """Mark the step as complete."""
        self.completed_at = datetime.now()
        self.state = state
        if self.started_at:
            self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000

    @property
    def is_complete(self) -> bool:
        """Check if step has all components."""
        return all([self.thought, self.action, self.observation])

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "step_number": self.step_number,
            "thought": self.thought.content if self.thought else None,
            "action": self.action.to_dict() if self.action else None,
            "observation_success": self.observation.success if self.observation else None,
            "state": self.state.value,
            "duration_ms": self.duration_ms,
        }


class MemoryType(str, Enum):
    """Types of memory entries."""

    SHORT_TERM = "short_term"        # Recent context (current session)
    WORKING = "working"              # Active task context
    LONG_TERM = "long_term"          # Persistent knowledge
    EPISODIC = "episodic"            # Past experiences/actions


@dataclass
class MemoryEntry:
    """
    Entry in the agent's memory system.

    Stores information with metadata for retrieval and relevance scoring.
    """

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""                # The memory content
    memory_type: MemoryType = MemoryType.SHORT_TERM
    importance: float = 0.5          # Importance score (0.0 to 1.0)
    relevance_score: float = 0.0     # Current relevance (computed)
    embedding: Optional[List[float]] = None  # Vector embedding
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    expires_at: Optional[datetime] = None

    def access(self) -> None:
        """Record an access to this memory."""
        self.accessed_at = datetime.now()
        self.access_count += 1

    @property
    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


# AgentConfig has been moved to flybrowser.agents.config
# Import it from there:
#   from flybrowser.agents.config import AgentConfig


# JSON Schema for structured ReAct responses (used with generate_structured_with_vision)
REACT_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {
            "type": "string",
            "description": "Your reasoning about the current state and what action to take next"
        },
        "action": {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": "Name of the tool to execute"
                },
                "parameters": {
                    "type": "object",
                    "description": "Parameters for the tool",
                    "additionalProperties": True
                }
            },
            "required": ["tool", "parameters"],
            "additionalProperties": False
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in this action (0.0 to 1.0)",
            "minimum": 0.0,
            "maximum": 1.0
        }
    },
    "required": ["thought", "action"],
    "additionalProperties": False
}


def validate_react_response(data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate that a response conforms to REACT_RESPONSE_SCHEMA.
    
    Args:
        data: Dictionary to validate
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    if not isinstance(data, dict):
        return False, [f"Response must be a JSON object, got {type(data).__name__}"]
    
    # Check required fields
    if "thought" not in data:
        errors.append("Missing required field 'thought'")
    elif not isinstance(data["thought"], str):
        errors.append(f"'thought' must be a string, got {type(data['thought']).__name__}")
    
    if "action" not in data:
        errors.append("Missing required field 'action'")
    elif not isinstance(data["action"], dict):
        errors.append(f"'action' must be an object, got {type(data['action']).__name__}")
    else:
        action = data["action"]
        if "tool" not in action:
            errors.append("Missing required field 'action.tool'")
        elif not isinstance(action["tool"], str):
            errors.append(f"'action.tool' must be a string, got {type(action['tool']).__name__}")
        
        if "parameters" not in action:
            errors.append("Missing required field 'action.parameters'")
        elif not isinstance(action["parameters"], dict):
            errors.append(f"'action.parameters' must be an object, got {type(action['parameters']).__name__}")
    
    # Check confidence if present (optional)
    if "confidence" in data:
        conf = data["confidence"]
        if not isinstance(conf, (int, float)):
            errors.append(f"'confidence' must be a number, got {type(conf).__name__}")
        elif not (0.0 <= conf <= 1.0):
            errors.append(f"'confidence' must be between 0.0 and 1.0, got {conf}")
    
    return len(errors) == 0, errors


def build_repair_prompt(
    original_prompt: str,
    malformed_output: str,
    validation_errors: List[str],
    schema: Dict[str, Any] = REACT_RESPONSE_SCHEMA,
    available_tools: Optional[List[str]] = None,
) -> str:
    """
    Build a repair prompt to fix malformed LLM output.
    
    When structured output fails to match the schema, we ask the LLM to repair
    its response using the original context and the validation errors.
    
    Args:
        original_prompt: The original user prompt that produced the malformed output
        malformed_output: The malformed JSON output from the LLM
        validation_errors: List of validation error messages
        schema: The expected JSON schema
        available_tools: List of available tool names (to guide tool selection)
        
    Returns:
        A prompt asking the LLM to repair its output
    """
    import json
    
    schema_str = json.dumps(schema, indent=2)
    errors_str = "\n".join(f"- {err}" for err in validation_errors)
    
    # Truncate malformed output if too long
    max_output_len = 2000
    if len(malformed_output) > max_output_len:
        malformed_output = malformed_output[:max_output_len] + "... [truncated]"
    
    # Build tool guidance if available
    tool_guidance = ""
    if available_tools:
        tool_guidance = f"""\n## Available Tools (use ONLY these)
{', '.join(available_tools)}

**IMPORTANT**: For search operations, use 'search' (API-based, fast) NOT 'search_human' (browser-based, slow) unless 'search' is not in the available tools list.
"""
    
    return f"""Your previous response did not match the required JSON schema. Please fix it.

## Validation Errors
{errors_str}

## Your Malformed Output
```json
{malformed_output}
```

## Required Schema
```json
{schema_str}
```
{tool_guidance}
## Original Task Context
{original_prompt}

## Instructions
Please provide a corrected JSON response that:
1. Fixes all validation errors listed above
2. Maintains the same intent/action from your original response
3. Strictly follows the required schema
4. Uses ONLY tools from the available tools list

Respond ONLY with the corrected JSON object, no explanation."""


def parse_react_response(data: Dict[str, Any]) -> tuple[Optional[Thought], Optional[Action]]:
    """
    Parse a structured ReAct response into Thought and Action objects.
    
    Args:
        data: Dictionary from structured LLM response
        
    Returns:
        Tuple of (Thought, Action) - either may be None if parsing fails
    """
    thought = None
    action = None
    
    # Parse thought
    thought_content = data.get("thought")
    if thought_content:
        thought = Thought(
            content=str(thought_content),
            confidence=data.get("confidence", 0.8),
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
        )
    
    # Parse action
    action_data = data.get("action")
    if action_data and isinstance(action_data, dict):
        tool_name = action_data.get("tool")
        if tool_name:
            action = Action(
                tool_name=str(tool_name),
                parameters=action_data.get("parameters", {}),
                requires_approval=False,
            )
    
    return thought, action

