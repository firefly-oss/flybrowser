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
Shared JSON Schemas for Multi-Agent Communication.

This module provides centralized schema definitions that ensure consistent
data structures across all agents. By using these schemas:

1. Agents produce outputs in predictable formats
2. Agents can consume outputs from other agents reliably
3. Prompts can reference standardized schemas
4. Validation can be performed consistently

Schema Categories:
- Element schemas: Describing DOM elements and selectors
- Action schemas: Describing browser actions
- Plan schemas: Describing execution plans and steps
- Result schemas: Describing operation outcomes
- Page schemas: Describing page state and analysis

Usage:
    >>> from flybrowser.prompts.schemas import (
    ...     ELEMENT_SCHEMA,
    ...     ACTION_SCHEMA,
    ...     format_schema_for_prompt,
    ... )
    >>> 
    >>> # Include schema in prompt
    >>> prompt = f"Return JSON matching: {format_schema_for_prompt(ELEMENT_SCHEMA)}"
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

# =============================================================================
# ELEMENT SCHEMAS
# =============================================================================

ELEMENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "A DOM element identified on the page",
    "properties": {
        "selector": {
            "type": "string",
            "description": "CSS selector or XPath to locate the element",
        },
        "selector_type": {
            "type": "string",
            "enum": ["css", "xpath"],
            "description": "Type of selector",
        },
        "tag_name": {
            "type": "string",
            "description": "HTML tag name (e.g., 'button', 'input', 'a')",
        },
        "text": {
            "type": "string",
            "description": "Visible text content of the element",
        },
        "attributes": {
            "type": "object",
            "description": "Key HTML attributes (id, class, name, etc.)",
            "additionalProperties": {"type": "string"},
        },
        "is_visible": {
            "type": "boolean",
            "description": "Whether the element is visible in the viewport",
        },
        "is_interactive": {
            "type": "boolean",
            "description": "Whether the element can be interacted with",
        },
        "bounding_box": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "width": {"type": "number"},
                "height": {"type": "number"},
            },
        },
    },
    "required": ["selector", "selector_type"],
}

ELEMENT_DETECTION_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "Result of element detection operation",
    "properties": {
        "selector": {
            "type": "string",
            "description": "Primary CSS selector or XPath",
        },
        "selector_type": {
            "type": "string",
            "enum": ["css", "xpath"],
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence in selector reliability (0.0-1.0)",
        },
        "reasoning": {
            "type": "string",
            "description": "Chain-of-thought analysis explaining selection",
        },
        "fallback_selectors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "type": {"type": "string", "enum": ["css", "xpath"]},
                    "confidence": {"type": "number"},
                },
            },
            "description": "Alternative selectors if primary fails",
        },
        "element_properties": {
            "type": "object",
            "properties": {
                "tag": {"type": "string"},
                "visible": {"type": "boolean"},
                "interactive": {"type": "boolean"},
                "aria_label": {"type": ["string", "null"]},
            },
        },
    },
    "required": ["selector", "selector_type", "confidence", "reasoning"],
}


# =============================================================================
# ACTION SCHEMAS
# =============================================================================

ACTION_TYPES = [
    "navigate",
    "click",
    "type",
    "select",
    "scroll",
    "wait",
    "extract",
    "screenshot",
    "hover",
    "press",
    "focus",
    "clear",
    "check",
    "uncheck",
    "drag",
    "upload",
]

ACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "A browser action to execute",
    "properties": {
        "action": {
            "type": "string",
            "enum": ACTION_TYPES,
            "description": "The action type to perform",
        },
        "params": {
            "type": "object",
            "description": "Action-specific parameters",
            "properties": {
                "selector": {"type": "string"},
                "url": {"type": "string"},
                "text": {"type": "string"},
                "value": {"type": "string"},
                "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                "amount": {"type": "number"},
                "key": {"type": "string"},
                "timeout": {"type": "number"},
                "condition": {"type": "string"},
            },
        },
        "description": {
            "type": "string",
            "description": "Human-readable description of what this action does",
        },
        "wait_after": {
            "type": "number",
            "description": "Milliseconds to wait after action completes",
        },
        "retry_on_failure": {
            "type": "boolean",
            "description": "Whether to retry if action fails",
        },
        "timeout": {
            "type": "number",
            "description": "Action timeout in milliseconds",
        },
    },
    "required": ["action"],
}


# =============================================================================
# PLAN SCHEMAS
# =============================================================================

PLAN_STEP_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "A single step in an execution plan",
    "properties": {
        "id": {
            "type": "string",
            "description": "Unique identifier for this step",
        },
        "action": {
            "type": "string",
            "enum": ACTION_TYPES,
            "description": "The action type",
        },
        "params": {
            "type": "object",
            "description": "Action parameters",
        },
        "description": {
            "type": "string",
            "description": "What this step accomplishes",
        },
        "agent": {
            "type": "string",
            "description": "Agent responsible for this step",
            "enum": [
                "action_agent",
                "extraction_agent",
                "navigation_agent",
                "page_analyzer_agent",
                "monitoring_agent",
                "workflow_agent",
            ],
        },
        "dependencies": {
            "type": "array",
            "items": {"type": "string"},
            "description": "IDs of steps that must complete before this one",
        },
        "expected_outcome": {
            "type": "string",
            "description": "Expected result of this step",
        },
        "verification": {
            "type": "object",
            "properties": {
                "method": {"type": "string"},
                "criteria": {"type": "string"},
            },
            "description": "How to verify step success",
        },
        "fallback": {
            "type": "object",
            "description": "Alternative step if this one fails",
        },
        "wait_after": {
            "type": "number",
            "description": "Wait time after step in ms",
        },
    },
    "required": ["id", "action", "description"],
}

EXECUTION_PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "Complete execution plan for a task",
    "properties": {
        "plan": {
            "type": "array",
            "items": PLAN_STEP_SCHEMA,
            "description": "Ordered list of steps to execute",
        },
        "reasoning": {
            "type": "string",
            "description": "Explanation of planning decisions",
        },
        "estimated_duration": {
            "type": "number",
            "description": "Estimated total duration in ms",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence in plan success",
        },
        "complexity": {
            "type": "string",
            "enum": ["trivial", "simple", "moderate", "complex", "very_complex"],
            "description": "Task complexity assessment",
        },
        "parallel_groups": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "string"},
            },
            "description": "Groups of step IDs that can execute in parallel",
        },
    },
    "required": ["plan", "confidence"],
}


# =============================================================================
# EXTRACTION SCHEMAS
# =============================================================================

EXTRACTION_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "Result of data extraction operation",
    "properties": {
        "data": {
            "description": "Extracted data matching the requested schema",
        },
        "source": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "selector": {"type": "string"},
                "timestamp": {"type": "string"},
            },
            "description": "Where the data was extracted from",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence in extraction accuracy",
        },
        "completeness": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "How complete the extraction is (1.0 = all requested fields found)",
        },
        "issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Any issues encountered during extraction",
        },
    },
    "required": ["data"],
}


# =============================================================================
# VERIFICATION SCHEMAS
# =============================================================================

VERIFICATION_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "Result of action/step verification",
    "properties": {
        "success": {
            "type": "boolean",
            "description": "Whether the action succeeded",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence in the verification",
        },
        "evidence": {
            "type": "string",
            "description": "Specific evidence supporting the determination",
        },
        "observations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "What was observed on the page",
        },
        "issues_detected": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Problems found during verification",
        },
        "page_changed": {
            "type": "boolean",
            "description": "Whether the page state changed",
        },
        "recovery_suggestions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Suggestions if verification failed",
        },
    },
    "required": ["success", "confidence", "evidence"],
}


# =============================================================================
# OBSTACLE SCHEMAS
# =============================================================================

OBSTACLE_TYPES = [
    "cookie_consent",
    "newsletter_modal",
    "login_prompt",
    "age_verification",
    "notification_permission",
    "paywall",
    "overlay_ad",
    "generic_modal",
    "loading_spinner",
    "captcha",
]

OBSTACLE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "An obstacle blocking page interaction",
    "properties": {
        "type": {
            "type": "string",
            "enum": OBSTACLE_TYPES,
            "description": "Type of obstacle",
        },
        "description": {
            "type": "string",
            "description": "Human-readable description",
        },
        "selector": {
            "type": "string",
            "description": "CSS selector to locate the obstacle",
        },
        "is_blocking": {
            "type": "boolean",
            "description": "Whether it blocks interaction with main content",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "dismiss_strategies": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ordered list of dismissal strategies",
        },
    },
    "required": ["type", "is_blocking", "confidence"],
}

OBSTACLE_DETECTION_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "Result of obstacle detection analysis",
    "properties": {
        "obstacles": {
            "type": "array",
            "items": OBSTACLE_SCHEMA,
            "description": "Detected obstacles",
        },
        "page_interactable": {
            "type": "boolean",
            "description": "Whether the main page content can be interacted with",
        },
        "recommended_action": {
            "type": "string",
            "enum": ["proceed", "dismiss_obstacles", "wait", "manual_intervention"],
            "description": "Recommended next action",
        },
    },
    "required": ["obstacles", "page_interactable", "recommended_action"],
}


# =============================================================================
# PAGE STATE SCHEMAS
# =============================================================================

PAGE_STATE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "Current state of the web page",
    "properties": {
        "url": {
            "type": "string",
            "description": "Current page URL",
        },
        "title": {
            "type": "string",
            "description": "Page title",
        },
        "page_type": {
            "type": "string",
            "enum": [
                "landing",
                "login",
                "form",
                "list",
                "detail",
                "search",
                "checkout",
                "confirmation",
                "error",
                "unknown",
            ],
            "description": "Detected page type",
        },
        "loading_state": {
            "type": "string",
            "enum": ["loading", "interactive", "complete"],
            "description": "Page loading state",
        },
        "interactive_elements": {
            "type": "array",
            "items": ELEMENT_SCHEMA,
            "description": "Key interactive elements on the page",
        },
        "forms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "required": {"type": "boolean"},
                                "selector": {"type": "string"},
                            },
                        },
                    },
                    "submit_button": {"type": "string"},
                },
            },
            "description": "Forms detected on the page",
        },
        "obstacles": {
            "type": "array",
            "items": OBSTACLE_SCHEMA,
            "description": "Detected obstacles",
        },
        "scroll_position": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "max_x": {"type": "number"},
                "max_y": {"type": "number"},
            },
        },
    },
    "required": ["url", "title"],
}


# =============================================================================
# NAVIGATION SCHEMAS
# =============================================================================

NAVIGATION_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "Result of navigation planning",
    "properties": {
        "navigation_type": {
            "type": "string",
            "enum": ["url", "link", "button", "menu", "search", "back", "forward", "refresh"],
            "description": "Type of navigation to perform",
        },
        "target": {
            "type": "string",
            "description": "Target URL or element description",
        },
        "selector": {
            "type": "string",
            "description": "CSS selector if clicking an element",
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "target": {"type": "string"},
                    "wait_after": {"type": "number"},
                },
            },
            "description": "Steps to complete navigation",
        },
        "reasoning": {
            "type": "string",
            "description": "Analysis of navigation approach",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "required": ["navigation_type", "target", "confidence"],
}


# =============================================================================
# INTER-AGENT COMMUNICATION SCHEMAS
# =============================================================================

AGENT_TASK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "Task passed between agents",
    "properties": {
        "task_id": {
            "type": "string",
            "description": "Unique task identifier",
        },
        "task_type": {
            "type": "string",
            "enum": [
                "extract",
                "navigate",
                "interact",
                "analyze",
                "verify",
                "monitor",
                "plan",
            ],
            "description": "Type of task",
        },
        "instruction": {
            "type": "string",
            "description": "What to accomplish",
        },
        "context": {
            "type": "object",
            "description": "Context from previous steps",
        },
        "constraints": {
            "type": "object",
            "properties": {
                "timeout_ms": {"type": "number"},
                "max_retries": {"type": "integer"},
                "required_confidence": {"type": "number"},
            },
        },
        "source_agent": {
            "type": "string",
            "description": "Agent that created this task",
        },
    },
    "required": ["task_id", "task_type", "instruction"],
}

AGENT_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "Result returned by an agent",
    "properties": {
        "task_id": {
            "type": "string",
            "description": "ID of the completed task",
        },
        "success": {
            "type": "boolean",
            "description": "Whether the task succeeded",
        },
        "result": {
            "description": "Task-specific result data",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "duration_ms": {
            "type": "number",
            "description": "Time taken in milliseconds",
        },
        "agent": {
            "type": "string",
            "description": "Agent that processed the task",
        },
        "error": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
                "recoverable": {"type": "boolean"},
            },
        },
        "next_steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Suggested follow-up actions",
        },
    },
    "required": ["task_id", "success"],
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_schema_for_prompt(
    schema: Dict[str, Any],
    indent: int = 2,
) -> str:
    """
    Format a schema for inclusion in a prompt.
    
    Args:
        schema: JSON Schema dict
        indent: JSON indentation level
        
    Returns:
        Formatted JSON string
    """
    return json.dumps(schema, indent=indent)


def get_schema_properties(schema: Dict[str, Any]) -> List[str]:
    """
    Get required and optional property names from a schema.
    
    Args:
        schema: JSON Schema dict
        
    Returns:
        List of property names
    """
    return list(schema.get("properties", {}).keys())


def get_required_properties(schema: Dict[str, Any]) -> List[str]:
    """
    Get required property names from a schema.
    
    Args:
        schema: JSON Schema dict
        
    Returns:
        List of required property names
    """
    return schema.get("required", [])


def create_example_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an example object from a schema.
    
    Args:
        schema: JSON Schema dict
        
    Returns:
        Example dict with placeholder values
    """
    result = {}
    schema_type = schema.get("type")
    
    if schema_type == "object":
        for prop, prop_schema in schema.get("properties", {}).items():
            prop_type = prop_schema.get("type")
            
            if prop_type == "string":
                if "enum" in prop_schema:
                    result[prop] = prop_schema["enum"][0]
                else:
                    result[prop] = f"<{prop}>"
            elif prop_type == "number":
                result[prop] = 0.0
            elif prop_type == "integer":
                result[prop] = 0
            elif prop_type == "boolean":
                result[prop] = True
            elif prop_type == "array":
                result[prop] = []
            elif prop_type == "object":
                result[prop] = {}
            elif prop_type is None and "description" in prop_schema:
                result[prop] = "<any>"
    
    return result


# =============================================================================
# SCHEMA REGISTRY
# =============================================================================

# Central registry of all schemas for easy lookup
SCHEMA_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Element schemas
    "element": ELEMENT_SCHEMA,
    "element_detection_result": ELEMENT_DETECTION_RESULT_SCHEMA,
    # Action schemas
    "action": ACTION_SCHEMA,
    # Plan schemas
    "plan_step": PLAN_STEP_SCHEMA,
    "execution_plan": EXECUTION_PLAN_SCHEMA,
    # Extraction schemas
    "extraction_result": EXTRACTION_RESULT_SCHEMA,
    # Verification schemas
    "verification_result": VERIFICATION_RESULT_SCHEMA,
    # Obstacle schemas
    "obstacle": OBSTACLE_SCHEMA,
    "obstacle_detection_result": OBSTACLE_DETECTION_RESULT_SCHEMA,
    # Page schemas
    "page_state": PAGE_STATE_SCHEMA,
    # Navigation schemas
    "navigation_result": NAVIGATION_RESULT_SCHEMA,
    # Inter-agent schemas
    "agent_task": AGENT_TASK_SCHEMA,
    "agent_result": AGENT_RESULT_SCHEMA,
}


def get_schema(name: str) -> Dict[str, Any]:
    """
    Get a schema by name from the registry.
    
    Args:
        name: Schema name
        
    Returns:
        Schema dict
        
    Raises:
        KeyError: If schema not found
    """
    if name not in SCHEMA_REGISTRY:
        available = ", ".join(SCHEMA_REGISTRY.keys())
        raise KeyError(f"Schema '{name}' not found. Available: {available}")
    return SCHEMA_REGISTRY[name]


def list_schemas() -> List[str]:
    """Get all available schema names."""
    return list(SCHEMA_REGISTRY.keys())
