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
Prompt templates for LLM interactions.

This module provides backward-compatible prompt constants while also
integrating with the centralized prompt management system in flybrowser.prompts.

For new code, prefer using the PromptManager:
    >>> from flybrowser.prompts import PromptManager
    >>> manager = PromptManager()
    >>> prompts = manager.get_prompt("action_planning", instruction="...", url="...")

The constants below are maintained for backward compatibility.
"""

from typing import Any, Dict, Optional

# Lazy-loaded prompt manager for centralized prompts
_prompt_manager = None


def get_prompt_manager():
    """Get or create the global prompt manager."""
    global _prompt_manager
    if _prompt_manager is None:
        from flybrowser.prompts import PromptManager
        _prompt_manager = PromptManager()
    return _prompt_manager


def get_prompt(
    name: str,
    version: Optional[str] = None,
    **variables: Any,
) -> Dict[str, str]:
    """
    Get a rendered prompt from the centralized prompt system.

    Args:
        name: Template name (e.g., "action_planning", "data_extraction")
        version: Optional template version
        **variables: Variables to render the template

    Returns:
        Dictionary with 'system' and 'user' prompts

    Example:
        >>> prompts = get_prompt(
        ...     "action_planning",
        ...     instruction="Click the login button",
        ...     url="https://example.com",
        ...     title="Example",
        ...     visible_elements="[...]"
        ... )
        >>> print(prompts["system"])
        >>> print(prompts["user"])
    """
    manager = get_prompt_manager()
    return manager.get_prompt(name, version=version, **variables)


# ============================================================================
# BACKWARD COMPATIBLE CONSTANTS
# These are maintained for existing code that imports them directly.
# New code should use get_prompt() or PromptManager instead.
# ============================================================================

# System prompts
ELEMENT_DETECTION_SYSTEM = """You are an expert web automation assistant. Your task is to identify elements on a web page based on natural language descriptions.

You will be provided with:
1. A screenshot of the web page
2. The page's HTML structure
3. A description of the element to find

Analyze the page and identify the best selector for the described element. Consider:
- Element visibility and interactability
- Semantic meaning and context
- Reliability of the selector

CRITICAL: You MUST respond with ONLY a valid JSON object. Do NOT include:
- Explanations or reasoning outside the JSON
- Markdown code blocks (```json)
- Any text before or after the JSON object

Respond with ONLY this JSON structure:
{
  "selector": "CSS selector or XPath string",
  "selector_type": "css" or "xpath",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}

Start your response with { and end with } - nothing else.
"""

EXTRACTION_SYSTEM = """You are a data extraction specialist. Your task is to extract specific information from web pages.

You will be provided with:
1. A screenshot of the web page (optional)
2. The page's HTML content
3. The current page URL and title
4. A description of what data to extract

GUIDELINES:
- Extract actual visible content from the page (text, headings, links, data)
- DO NOT extract HTML attributes, element IDs, CSS classes, or technical metadata
- DO NOT extract URL parameters, query strings, or technical page data
- Focus on human-readable information that answers the query
- Structure data clearly in the "data" field
- Metadata fields (source_url, extraction_query) will be auto-populated - leave them empty or use placeholder values

CRITICAL: You MUST respond with ONLY valid JSON. Do NOT include:
- Explanations outside the JSON object
- Markdown code blocks (```json)
- Any text before or after the JSON

Your response must start with { and end with } - nothing else.
"""

ACTION_PLANNING_SYSTEM = """You are a web automation planner. Your task is to break down high-level instructions into specific browser actions.

You will be provided with:
1. Current page state and screenshot
2. A high-level instruction (e.g., "add item to cart")

Break down the instruction into atomic actions such as:
- click(selector)
- type(selector, text)
- scroll(direction)
- wait(condition)

CRITICAL: You MUST respond with ONLY a valid JSON object. Do NOT include:
- Explanations or reasoning outside the JSON
- Markdown code blocks
- Any text before or after the JSON

Return ONLY this JSON structure:
{
  "actions": [
    {"action_type": "click", "target": "description", "value": null, "options": {}},
    ...
  ],
  "reasoning": "Brief plan explanation"
}

Start your response with { and end with } - nothing else.
"""

NAVIGATION_SYSTEM = """You are a web navigation assistant. Your task is to help navigate web pages intelligently.

Analyze the current page and determine the best way to reach the desired state or page.
Consider:
- Available links and navigation elements
- Page structure and hierarchy
- Optimal path to the goal
"""

MONITORING_SYSTEM = """You are a page monitoring expert. Your task is to evaluate conditions on web pages.

Analyze the current page state and determine if the specified condition is met.
Consider:
- Element visibility and existence
- Text content and patterns
- Page state and attributes
"""

WORKFLOW_SYSTEM = """You are a workflow execution expert. Your task is to execute multi-step browser automation workflows.

Execute each step carefully, managing state between steps and handling errors appropriately.
Consider:
- Step dependencies and order
- Variable substitution
- Error recovery strategies
"""

# User prompt templates
ELEMENT_DETECTION_PROMPT = """Find the element that matches this description: {description}

Current page URL: {url}
Page title: {title}

HTML snippet:
{html_snippet}
"""

EXTRACTION_PROMPT = """Extract the following information from the page: {query}

Current page URL: {url}
Page title: {title}

IMPORTANT:
- Extract the actual visible content/text from the page, not HTML attributes or technical data
- Focus on human-readable information that answers the query
- Structure the extracted data in the "data" field
- Include metadata with the actual page URL: {url} and query: {query}
"""

ACTION_PLANNING_PROMPT = """Plan the steps to accomplish this task: {instruction}

Current page URL: {url}
Page title: {title}
Visible elements: {elements}
"""

NAVIGATION_PROMPT = """Navigate to accomplish this goal: {goal}

Current page URL: {url}
Available links: {links}
"""

MONITORING_PROMPT = """Check if this condition is met: {condition}

Current page URL: {url}
Page title: {title}
Page content: {content}
"""

WORKFLOW_PROMPT = """Execute this workflow step: {step}

Current state: {state}
Variables: {variables}
"""

