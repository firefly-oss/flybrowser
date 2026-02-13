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
Element detection and interaction using LLM-powered strategies.

This module provides the ElementDetector class which uses LLMs to intelligently
locate and interact with web page elements based on natural language descriptions.

The detector can work with or without vision capabilities, using either:
- Vision-based detection: Analyzes screenshots to locate elements visually
- Text-based detection: Analyzes HTML structure to find elements

This enables natural language element interaction like:
- "Find the login button"
- "Locate the search input field"
- "Click the submit button"
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any, Dict, Optional

from playwright.async_api import Page

from flybrowser.exceptions import ElementNotFoundError
from flybrowser.llm.base import BaseLLMProvider
from flybrowser.utils.logger import logger
from flybrowser.utils.timing import StepTimer
from flybrowser.agents.structured_llm import StructuredLLMWrapper
from flybrowser.agents.schemas import ELEMENT_DETECTION_SCHEMA


# =============================================================================
# Browser Action Logger - Colored logging for Playwright browser actions
# =============================================================================

class BrowserActionLogger:
    """
    Provides consistent colored logging for browser actions.
    
    This makes it easy to see when the browser is actually performing actions
    like clicking, typing, navigating, etc.
    """
    
    # ANSI color codes for browser actions
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Browser action specific colors
    BROWSER = "\033[38;5;75m"     # Light blue for browser actions
    SUCCESS = "\033[38;5;82m"     # Bright green
    ERROR = "\033[38;5;196m"      # Bright red
    WARNING = "\033[38;5;220m"    # Yellow
    
    def __init__(self):
        self._action_start_time: Optional[float] = None
        self._current_action: Optional[str] = None
    
    def start_action(self, action_type: str, description: str) -> float:
        """
        Log the start of a browser action.
        
        Args:
            action_type: Type of action (CLICK, TYPE, NAVIGATE, etc.)
            description: What is being acted on
            
        Returns:
            Start time for calculating duration
        """
        self._action_start_time = time.time()
        self._current_action = action_type
        logger.info(
            f"{self.BROWSER}{self.BOLD}> [BROWSER {action_type}]{self.RESET} "
            f"{self.BROWSER}{description}{self.RESET}"
        )
        return self._action_start_time
    
    def end_action(self, success: bool, details: Optional[str] = None) -> None:
        """
        Log the end of a browser action.
        
        Args:
            success: Whether the action succeeded
            details: Optional additional details
        """
        duration_ms = 0.0
        if self._action_start_time:
            duration_ms = (time.time() - self._action_start_time) * 1000
        
        status_icon = "[OK]" if success else "[FAIL]"
        status_color = self.SUCCESS if success else self.ERROR
        details_str = f" -> {details}" if details else ""
        
        logger.info(
            f"{status_color}{status_icon}{self.RESET} {self.BROWSER}[BROWSER]{self.RESET} "
            f"{self.DIM}{duration_ms:.0f}ms{self.RESET}{details_str}"
        )
        
        self._action_start_time = None
        self._current_action = None
    
    def log_step(self, message: str) -> None:
        """Log an intermediate step within an action."""
        logger.info(f"  {self.BROWSER}|-{self.RESET} {message}")
    
    def log_warning(self, message: str) -> None:
        """Log a warning during an action."""
        logger.warning(f"  {self.WARNING}[WARN]{self.RESET} {message}")
    
    def log_retry(self, attempt: int, reason: str) -> None:
        """Log a retry attempt."""
        logger.info(f"  {self.WARNING}[RETRY {attempt}]{self.RESET} {reason}")


# Global browser action logger instance
_browser_logger = BrowserActionLogger()


# Common element patterns for well-known sites and element types
COMMON_ELEMENT_PATTERNS = {
    # Search boxes across various sites
    "search": {
        "description_keywords": ["search", "buscar", "rechercher", "suchen"],
        "selectors": [
            # Google
            "textarea[name='q']",
            "input[name='q']",
            "input[title='Search']",
            "input[aria-label='Search']",
            # Generic patterns
            "input[type='search']",
            "input[placeholder*='search' i]",
            "input[placeholder*='buscar' i]",
            "input[name='search']",
            "input[name='query']",
            "input[id*='search' i]",
            "input[class*='search' i]",
            "[role='searchbox']",
            "[aria-label*='search' i]",
        ],
    },
    # Submit/Search buttons
    "submit_search": {
        "description_keywords": ["search button", "submit", "go", "find"],
        "selectors": [
            "button[type='submit']",
            "input[type='submit']",
            "button[aria-label*='search' i]",
            "button:has-text('Search')",
            "button:has-text('Go')",
            "[role='button'][aria-label*='search' i]",
        ],
    },
    # Login buttons
    "login": {
        "description_keywords": ["login", "log in", "sign in", "signin"],
        "selectors": [
            "button:has-text('Log in')",
            "button:has-text('Login')",
            "button:has-text('Sign in')",
            "a:has-text('Log in')",
            "a:has-text('Sign in')",
            "[data-testid*='login' i]",
            "#login-button",
            ".login-btn",
        ],
    },
    # Email/Username inputs
    "email_input": {
        "description_keywords": ["email", "username", "user"],
        "selectors": [
            "input[type='email']",
            "input[name='email']",
            "input[name='username']",
            "input[id*='email' i]",
            "input[placeholder*='email' i]",
            "input[autocomplete='email']",
            "input[autocomplete='username']",
        ],
    },
    # Password inputs
    "password_input": {
        "description_keywords": ["password", "contraseña"],
        "selectors": [
            "input[type='password']",
            "input[name='password']",
            "input[id*='password' i]",
            "input[autocomplete='current-password']",
        ],
    },
    # Search result links (Google, Bing, etc.)
    "search_result_link": {
        "description_keywords": ["search result", "first result", "result link"],
        "selectors": [
            # Google search results
            "#search a[href]:not([href*='google.com'])",
            "#rso a[href]:not([href*='google.com'])",
            "div.g a[href]:not([href*='google.com'])",
            "div[data-hveid] a[href]:not([href*='google.com'])",
            # Bing search results
            "#b_results a[href]:not([href*='bing.com'])",
            "li.b_algo a[href]",
            # DuckDuckGo
            ".result__a[href]",
            # Generic result containers
            "[data-testid*='result'] a[href]",
            "article a[href]",
        ],
    },
}


class ElementDetector:
    """
    Detects and interacts with page elements using LLM-enhanced strategies.

    This class combines traditional element detection with LLM intelligence
    to locate elements based on natural language descriptions. It supports
    both vision-based and text-based detection strategies.

    Attributes:
        page: Playwright Page instance for element interaction
        llm: LLM provider for intelligent element detection

    Example:
        >>> detector = ElementDetector(page, llm_provider)
        >>> element_info = await detector.find_element("the login button")
        >>> await detector.click(element_info["selector"])
    """

    # Human-like delay ranges (milliseconds)
    MIN_ACTION_DELAY = 100
    MAX_ACTION_DELAY = 300
    MIN_TYPING_DELAY = 50
    MAX_TYPING_DELAY = 150

    def __init__(self, page: Page, llm_provider: BaseLLMProvider) -> None:
        """
        Initialize the element detector.

        Args:
            page: Playwright Page instance for element operations
            llm_provider: LLM provider for intelligent element detection.
                Should support vision capabilities for best results.

        Example:
            >>> from flybrowser.llm.base import BaseLLMProvider
            >>> # Use a BaseLLMProvider implementation
            >>> detector = ElementDetector(page, llm_provider)
        """
        self.page = page
        self.llm = llm_provider
    
    async def _human_delay(self, min_ms: int = None, max_ms: int = None) -> None:
        """
        Add a random delay to simulate human-like behavior.
        
        This helps avoid bot detection by making actions appear more natural.
        
        Args:
            min_ms: Minimum delay in milliseconds (default: MIN_ACTION_DELAY)
            max_ms: Maximum delay in milliseconds (default: MAX_ACTION_DELAY)
        """
        min_delay = min_ms or self.MIN_ACTION_DELAY
        max_delay = max_ms or self.MAX_ACTION_DELAY
        delay = random.randint(min_delay, max_delay) / 1000.0
        await asyncio.sleep(delay)

    async def find_element(
        self, description: str, use_vision: bool = True
    ) -> Dict[str, Any]:
        """
        Find an element based on natural language description.

        This method uses an LLM to analyze the page and locate the element
        matching the description. It can use vision (screenshot analysis)
        or text-based (HTML analysis) detection.

        Args:
            description: Natural language description of the element to find.
                Examples:
                - "the login button"
                - "the email input field"
                - "the submit button in the form"
                - "the search icon in the header"
            use_vision: Whether to use vision-based detection. When True,
                sends a screenshot to a vision-capable LLM for better
                visual understanding. Default: True

        Returns:
            Dictionary containing element information:
            {
                "selector": "CSS selector or XPath",
                "selector_type": "css" or "xpath",
                "confidence": 0.0-1.0,
                "reasoning": "Why this element was selected",
                "timing": {"total_ms": float, "breakdown": {...}}
            }

        Raises:
            ElementNotFoundError: If element cannot be found or LLM fails

        Example:
            >>> element = await detector.find_element("the login button")
            >>> print(element["selector"])
            'button.login-btn'

            >>> element = await detector.find_element(
            ...     "the search input",
            ...     use_vision=False  # Use HTML-only detection
            ... )
        """
        # Start timing
        timer = StepTimer()
        timer.start()
        
        try:
            logger.info(f"Finding element: {description}")

            # First, try common patterns for well-known element types
            timer.start_step("common_patterns")
            common_result = await self._try_common_patterns(description)
            timer.end_step("common_patterns")
            
            if common_result:
                common_result["timing"] = timer.get_timings().to_dict()
                return common_result

            # Get page information for context
            timer.start_step("get_page_context")
            url = self.page.url
            title = await self.page.title()
            html = await self.page.content()
            timer.end_step("get_page_context")

            # Extract relevant HTML for the LLM - focus on interactive elements
            html_snippet = await self._extract_relevant_html(html, description)

            # Use vision-based detection if enabled AND model supports it
            timer.start_step("llm_generate")
            llm_has_vision = getattr(self.llm, 'vision_enabled', False) or self.llm.supports_vision()
            use_vision_actual = use_vision and llm_has_vision
            if use_vision and not llm_has_vision:
                logger.debug(f"Vision requested but model {self.llm.model} doesn't support it, using text-only")
            
            # Build element detection prompts inline
            prompts = self._build_element_detection_prompts(
                description=description,
                url=url,
                title=title,
                html_snippet=html_snippet,
                screenshot_available=use_vision_actual,
            )
            
            # Use StructuredLLMWrapper for reliable JSON output with repair
            wrapper = StructuredLLMWrapper(
                llm_provider=self.llm,
                max_repair_attempts=2,
                repair_temperature=0.1,
            )
            
            try:
                if use_vision_actual:
                    screenshot = await self.page.screenshot(type="png")
                    result = await wrapper.generate_structured_with_vision(
                        prompt=prompts["user"],
                        image_data=screenshot,
                        schema=ELEMENT_DETECTION_SCHEMA,
                        system_prompt=prompts["system"],
                        temperature=0.3,
                    )
                else:
                    result = await wrapper.generate_structured(
                        prompt=prompts["user"],
                        schema=ELEMENT_DETECTION_SCHEMA,
                        system_prompt=prompts["system"],
                        temperature=0.3,
                    )
            except ValueError as e:
                # Structured output validation failed after repair attempts
                raise ElementNotFoundError(f"Could not parse element detection response: {e}")
            timer.end_step("llm_generate")
            
            # Validate that the selector actually exists on the page
            timer.start_step("validate_selector")
            selector = result.get("selector", "")
            selector_type = result.get("selector_type", "css")
            
            validated_selector = await self._validate_selector(selector, selector_type)
            if validated_selector:
                result["selector"] = validated_selector["selector"]
                result["selector_type"] = validated_selector["selector_type"]
                result["validated"] = True
            else:
                # LLM selector didn't work, try fallback strategies
                logger.warning(f"LLM selector '{selector}' not found, trying fallbacks")
                fallback = await self._try_fallback_selectors(description)
                if fallback:
                    result = fallback
                else:
                    result["validated"] = False
                    result["warning"] = "Selector not validated - may not exist on page"
            timer.end_step("validate_selector")
            
            logger.info(f"Found element with selector: {result.get('selector')}")
            
            # Add timing information to result
            result["timing"] = timer.get_timings().to_dict()
            return result

        except Exception as e:
            logger.error(f"Element detection failed: {e}")
            raise ElementNotFoundError(f"Failed to find element '{description}': {e}") from e
    
    def _build_element_detection_prompts(
        self,
        description: str,
        url: str,
        title: str,
        html_snippet: str,
        screenshot_available: bool,
    ) -> Dict[str, str]:
        """
        Build system and user prompts for element detection.

        Returns a dict with 'system' and 'user' keys containing the rendered
        prompt strings.  This replaces the previous PromptManager template
        lookup with an equivalent inline implementation.
        """
        system_prompt = (
            "You are an expert web automation assistant specializing in precise element identification.\n"
            "Your task is to identify interactive elements on web pages with high accuracy.\n"
            "\n"
            "## Your Capabilities:\n"
            "- Analyze HTML structure and visual layout\n"
            "- Understand semantic meaning and context\n"
            "- Identify the most reliable selectors\n"
            "- Consider accessibility attributes\n"
            "- Evaluate element visibility and interactability\n"
            "\n"
            "## Analysis Process (Chain-of-Thought):\n"
            "1. **Visual Analysis**: Examine the screenshot to locate the element visually\n"
            "2. **HTML Analysis**: Study the HTML structure around the element\n"
            "3. **Selector Strategy**: Determine the best selector type (ID, class, data-*, aria-*, text content)\n"
            "4. **Reliability Check**: Assess selector uniqueness and stability\n"
            "5. **Fallback Options**: Identify alternative selectors if primary fails\n"
            "\n"
            "## Output Requirements:\n"
            "You MUST respond with valid JSON matching this exact structure:\n"
            '```json\n'
            '{\n'
            '  "selector": "string - the primary CSS selector or XPath",\n'
            '  "selector_type": "css" or "xpath",\n'
            '  "confidence": 0.0 to 1.0,\n'
            '  "reasoning": "string - your chain-of-thought analysis",\n'
            '  "fallback_selectors": [\n'
            '    {"selector": "string", "type": "css|xpath", "confidence": 0.0-1.0}\n'
            '  ],\n'
            '  "element_properties": {\n'
            '    "tag": "string",\n'
            '    "visible": boolean,\n'
            '    "interactive": boolean,\n'
            '    "aria_label": "string or null"\n'
            '  }\n'
            '}\n'
            '```\n'
            "\n"
            "## Best Practices:\n"
            "- Prefer data-testid, aria-label, or unique IDs when available\n"
            "- Avoid fragile selectors based on nth-child or complex hierarchies\n"
            "- Consider text content for buttons and links\n"
            "- Verify element is visible and interactive\n"
            "- Provide confidence score based on selector reliability\n"
            "\n"
            "## CRITICAL: Invalid CSS Selectors to AVOID\n"
            "NEVER use these non-standard CSS pseudo-selectors (they cause errors):\n"
            "- `:contains()` - NOT valid CSS! Use XPath instead: `//element[contains(text(), 'text')]`\n"
            "- `:has()` with text - Limited browser support\n"
            "- jQuery-specific selectors\n"
            "\n"
            "For text-based matching, ALWAYS prefer:\n"
            "1. XPath: `//button[contains(text(), 'Click Me')]` or `//a[text()='Link Text']`\n"
            "2. CSS with attributes: `[aria-label='Button label']` or `[title='tooltip']`\n"
            "3. Simple class/ID selectors when available\n"
        )

        user_prompt = (
            f"## Task\n"
            f"Find the element matching this description: **{description}**\n"
            f"\n"
            f"## Page Context\n"
            f"- URL: {url}\n"
            f"- Title: {title}\n"
            f"\n"
            f"## HTML Structure (relevant section)\n"
            f"```html\n"
            f"{html_snippet}\n"
            f"```\n"
        )
        if screenshot_available:
            user_prompt += (
                "\n"
                "## Visual Context\n"
                "A screenshot of the page is provided. Use it to visually locate the element.\n"
            )
        user_prompt += (
            "\n"
            "## Instructions\n"
            "1. Analyze both the HTML and visual layout\n"
            "2. Identify the element matching the description\n"
            "3. Generate the most reliable selector\n"
            "4. Provide your reasoning and confidence score\n"
            "5. Include fallback selectors\n"
            "\n"
            "Respond with JSON only, no additional text.\n"
        )

        return {"system": system_prompt, "user": user_prompt}

    async def _try_common_patterns(self, description: str) -> Optional[Dict[str, Any]]:
        """
        Try common element patterns before falling back to LLM.
        
        This provides fast, reliable detection for well-known element types.
        Uses conservative matching to avoid false positives.
        """
        description_lower = description.lower()
        
        for pattern_type, pattern_info in COMMON_ELEMENT_PATTERNS.items():
            # Skip the search input pattern if looking for search results
            # "search result" should NOT match the search input box
            search_result_keywords = ["search result", "result link", "first result", "click result"]
            if pattern_type == "search" and any(kw in description_lower for kw in search_result_keywords):
                logger.debug(f"Skipping 'search' pattern for result-related query: {description}")
                continue
            # Check if description matches any keywords for this pattern
            if any(keyword in description_lower for keyword in pattern_info["description_keywords"]):
                # Try each selector for this pattern
                for selector in pattern_info["selectors"]:
                    try:
                        elem = await self.page.query_selector(selector)
                        if elem:
                            is_visible = await elem.is_visible()
                            if is_visible:
                                # Check if multiple elements match - if so, mark it
                                # The _get_locator method will use .first to handle this
                                all_elements = await self.page.query_selector_all(selector)
                                element_count = len(all_elements)
                                if element_count > 1:
                                    logger.info(f"Found element via common pattern: {selector} (will use .first of {element_count})")
                                    return {
                                        "selector": selector,
                                        "selector_type": "css",
                                        "confidence": 0.95,
                                        "reasoning": f"Matched common {pattern_type} pattern (first of {element_count} matches)",
                                        "validated": True,
                                        "use_first": True,  # Signal to use .first locator
                                        "element_count": element_count,
                                    }
                                else:
                                    logger.info(f"Found element via common pattern: {selector}")
                                    return {
                                        "selector": selector,
                                        "selector_type": "css",
                                        "confidence": 0.95,
                                        "reasoning": f"Matched common {pattern_type} pattern",
                                        "validated": True,
                                    }
                    except Exception:
                        continue
        
        return None
    
    async def _validate_selector(self, selector: str, selector_type: str) -> Optional[Dict[str, Any]]:
        """
        Validate that a selector exists and is visible on the page.
        """
        try:
            if selector_type == "xpath":
                elem = await self.page.query_selector(f"xpath={selector}")
            else:
                elem = await self.page.query_selector(selector)
            
            if elem:
                is_visible = await elem.is_visible()
                if is_visible:
                    return {"selector": selector, "selector_type": selector_type}
        except Exception as e:
            logger.debug(f"Selector validation failed for '{selector}': {e}")
        
        return None
    
    async def _extract_relevant_html(self, full_html: str, description: str) -> str:
        """
        Extract relevant HTML sections for element detection.
        
        Instead of just truncating, we extract the most relevant parts:
        - All <input>, <textarea>, <select> elements
        - All <button> elements
        - All <a> elements (limited)
        - Elements with role attributes
        - Forms and their contents
        """
        import re
        
        description_lower = description.lower()
        relevant_parts = []
        
        # Extract <head> section for meta info (truncated)
        head_match = re.search(r'<head[^>]*>.*?</head>', full_html, re.DOTALL | re.IGNORECASE)
        if head_match:
            head_content = head_match.group(0)
            # Only keep title and relevant meta tags
            title_match = re.search(r'<title[^>]*>.*?</title>', head_content, re.DOTALL | re.IGNORECASE)
            if title_match:
                relevant_parts.append(title_match.group(0))
        
        # Extract all form elements
        forms = re.findall(r'<form[^>]*>.*?</form>', full_html, re.DOTALL | re.IGNORECASE)
        for form in forms[:3]:  # Limit to 3 forms
            relevant_parts.append(form[:2000])  # Limit form size
        
        # Extract all input elements with context
        inputs = re.findall(r'<input[^>]*/?>', full_html, re.IGNORECASE)
        relevant_parts.extend(inputs[:30])  # Limit to 30 inputs
        
        # Extract all textarea elements
        textareas = re.findall(r'<textarea[^>]*>.*?</textarea>', full_html, re.DOTALL | re.IGNORECASE)
        relevant_parts.extend(textareas[:10])
        
        # Extract all select elements
        selects = re.findall(r'<select[^>]*>.*?</select>', full_html, re.DOTALL | re.IGNORECASE)
        relevant_parts.extend(s[:500] for s in selects[:10])  # Truncate long selects
        
        # Extract all button elements
        buttons = re.findall(r'<button[^>]*>.*?</button>', full_html, re.DOTALL | re.IGNORECASE)
        relevant_parts.extend(buttons[:20])
        
        # Extract elements with role attribute
        role_elements = re.findall(r'<[^>]+role=["\'][^"\'>]+["\'][^>]*>.*?</[^>]+>', full_html, re.DOTALL | re.IGNORECASE)
        relevant_parts.extend(r[:300] for r in role_elements[:15])
        
        # If searching for search-related elements
        if any(word in description_lower for word in ['search', 'buscar', 'find', 'query']):
            # Look for search-specific patterns
            search_patterns = [
                r'<[^>]*(?:search|query|buscar)[^>]*>.*?</[^>]+>',
                r'<[^>]*name=["\']q["\'][^>]*>',
                r'<[^>]*aria-label=["\'][^"\'>]*search[^"\'>]*["\'][^>]*>',
            ]
            for pattern in search_patterns:
                matches = re.findall(pattern, full_html, re.DOTALL | re.IGNORECASE)
                relevant_parts.extend(matches[:5])
        
        # If searching for links
        if any(word in description_lower for word in ['link', 'click', 'navigate', 'go to']):
            links = re.findall(r'<a[^>]*>.*?</a>', full_html, re.DOTALL | re.IGNORECASE)
            relevant_parts.extend(l[:200] for l in links[:20])
        
        # Combine and deduplicate
        combined = "\n".join(dict.fromkeys(relevant_parts))  # Preserve order, remove duplicates
        
        # If still too long, truncate
        if len(combined) > 8000:
            combined = combined[:8000] + "\n... [truncated]"
        
        # If we didn't extract much, fall back to first part of HTML
        if len(combined) < 500:
            combined = full_html[:5000] + "\n... [truncated]"
        
        return combined
    
    async def _try_fallback_selectors(self, description: str) -> Optional[Dict[str, Any]]:
        """
        Try various fallback strategies when the primary selector fails.
        """
        description_lower = description.lower()
        
        # Generate fallback selectors based on description
        fallback_selectors = []
        
        # Text-based fallbacks
        if any(word in description_lower for word in ["button", "click", "submit"]):
            words = description.replace("the", "").replace("a ", "").strip().split()
            for word in words:
                if word.lower() not in ["button", "click", "submit", "the", "a", "an"]:
                    fallback_selectors.extend([
                        f"button:has-text('{word}')",
                        f"a:has-text('{word}')",
                        f"[role='button']:has-text('{word}')",
                    ])
        
        # Input field fallbacks
        if any(word in description_lower for word in ["input", "field", "textbox", "type"]):
            fallback_selectors.extend([
                "input:visible",
                "textarea:visible",
                "[contenteditable='true']:visible",
            ])
        
        # Try all common pattern selectors as last resort
        for pattern_info in COMMON_ELEMENT_PATTERNS.values():
            fallback_selectors.extend(pattern_info["selectors"])
        
        # Try each fallback
        for selector in fallback_selectors:
            try:
                elem = await self.page.query_selector(selector)
                if elem:
                    is_visible = await elem.is_visible()
                    if is_visible:
                        logger.info(f"Found element via fallback: {selector}")
                        return {
                            "selector": selector,
                            "selector_type": "css",
                            "confidence": 0.7,
                            "reasoning": "Found via fallback selector",
                            "validated": True,
                        }
            except Exception:
                continue
        
        return None

    async def click(
        self,
        selector: str,
        selector_type: str = "css",
        timeout: int = 10000,
        force: bool = False,
        human_like: bool = True,
        use_first: bool = False,
    ) -> None:
        """
        Click an element with robust error handling and fallbacks.

        Args:
            selector: Element selector
            selector_type: Type of selector (css or xpath)
            timeout: Maximum time to wait for element in milliseconds
            force: Whether to force click even if element is obscured
            human_like: Whether to add human-like delay before clicking
            use_first: If True, use .first to avoid strict mode violations when
                       multiple elements match the selector
        """
        # Truncate selector for display
        selector_display = selector[:60] + "..." if len(selector) > 60 else selector
        _browser_logger.start_action("CLICK", selector_display)
        
        try:
            # Add human-like delay before action
            if human_like:
                await self._human_delay()
            
            locator = self._get_locator(selector, selector_type, use_first=use_first)
            
            # Wait for element to be visible with custom timeout
            await locator.wait_for(state="visible", timeout=timeout)
            
            # Try normal click first
            try:
                await locator.click(timeout=timeout, force=force)
                _browser_logger.end_action(True, "clicked")
                return
            except Exception as click_error:
                error_msg = str(click_error).lower()
                # Detect Playwright strict mode violation and auto-retry with .first
                if "strict mode violation" in error_msg and not use_first:
                    _browser_logger.log_retry(1, "strict mode violation, using .first")
                    locator = self._get_locator(selector, selector_type, use_first=True)
                    try:
                        await locator.click(timeout=timeout, force=force)
                        _browser_logger.end_action(True, "clicked (first match)")
                        return
                    except Exception as retry_error:
                        logger.debug(f"Click with .first also failed: {retry_error}")
                _browser_logger.log_step(f"Normal click failed, trying alternatives")
            
            # Try scrolling into view and clicking
            try:
                _browser_logger.log_step("Scrolling into view")
                await locator.scroll_into_view_if_needed()
                await locator.click(timeout=5000)
                _browser_logger.end_action(True, "clicked (after scroll)")
                return
            except Exception:
                pass
            
            # Try force click as last resort
            if not force:
                try:
                    _browser_logger.log_step("Attempting force click")
                    await locator.click(force=True, timeout=5000)
                    _browser_logger.end_action(True, "force clicked")
                    return
                except Exception:
                    pass
            
            # If all else fails, try JavaScript click
            try:
                _browser_logger.log_step("Attempting JS click")
                await locator.evaluate("el => el.click()")
                _browser_logger.end_action(True, "JS clicked")
                return
            except Exception as e:
                _browser_logger.end_action(False, f"all methods failed: {e}")
                raise ElementNotFoundError(f"All click methods failed for {selector}: {e}")
                
        except Exception as e:
            _browser_logger.end_action(False, str(e)[:50])
            raise ElementNotFoundError(f"Failed to click element {selector}: {e}") from e

    async def type_text(
        self,
        selector: str,
        text: str,
        selector_type: str = "css",
        delay: int = 75,  # Human-like typing delay (50-150ms is natural)
        clear_first: bool = True,
        timeout: int = 10000,
        human_like: bool = True,
        use_first: bool = False,
    ) -> None:
        """
        Type text into an element with robust error handling.

        Args:
            selector: Element selector
            text: Text to type
            selector_type: Type of selector (css or xpath)
            delay: Base delay between keystrokes in milliseconds
            clear_first: Whether to clear existing content first
            timeout: Maximum time to wait for element in milliseconds
            human_like: Whether to add random variation to typing delay
            use_first: If True, use .first to avoid strict mode violations when
                       multiple elements match the selector
        """
        # Truncate selector and text for display
        selector_display = selector[:50] + "..." if len(selector) > 50 else selector
        text_display = text[:20] + "..." if len(text) > 20 else text
        _browser_logger.start_action("TYPE", f"'{text_display}' -> {selector_display}")
        
        try:
            # Add human-like delay before typing
            if human_like:
                await self._human_delay()
            
            locator = self._get_locator(selector, selector_type, use_first=use_first)
            
            # Wait for element to be visible
            await locator.wait_for(state="visible", timeout=timeout)
            
            # Add random variation to typing delay if human_like
            actual_delay = delay
            if human_like:
                actual_delay = random.randint(self.MIN_TYPING_DELAY, self.MAX_TYPING_DELAY)
            
            # Try fill first (fastest method) - but less human-like
            # Only use fill() if not trying to be human-like
            if not human_like:
                try:
                    if clear_first:
                        await locator.clear()
                    await locator.fill(text)
                    _browser_logger.end_action(True, f"filled {len(text)} chars")
                    return
                except Exception as fill_error:
                    _browser_logger.log_step(f"Fill failed, trying type()")
            
            # Use type() for human-like character-by-character input
            try:
                await locator.click()  # Focus the element
                if clear_first:
                    await locator.press("Control+a")
                    await locator.press("Backspace")
                await locator.type(text, delay=actual_delay)
                _browser_logger.end_action(True, f"typed {len(text)} chars")
                return
            except Exception as type_error:
                error_msg = str(type_error).lower()
                # Detect Playwright strict mode violation and auto-retry with .first
                if "strict mode violation" in error_msg and not use_first:
                    _browser_logger.log_retry(1, "strict mode violation, using .first")
                    locator = self._get_locator(selector, selector_type, use_first=True)
                    try:
                        await locator.click()  # Focus
                        if clear_first:
                            await locator.press("Control+a")
                            await locator.press("Backspace")
                        await locator.type(text, delay=actual_delay)
                        _browser_logger.end_action(True, f"typed {len(text)} chars (first match)")
                        return
                    except Exception as retry_error:
                        logger.debug(f"Type with .first also failed: {retry_error}")
                _browser_logger.log_step("Type failed, trying JS")
            
            # Last resort: JavaScript (not human-like but works)
            try:
                _browser_logger.log_step("Attempting JS input")
                await locator.evaluate(f"el => {{ el.value = '{text}'; el.dispatchEvent(new Event('input', {{ bubbles: true }})); }}")
                _browser_logger.end_action(True, f"JS filled {len(text)} chars")
                return
            except Exception as e:
                _browser_logger.end_action(False, f"all methods failed: {e}")
                raise ElementNotFoundError(f"All type methods failed for {selector}: {e}")
                
        except Exception as e:
            _browser_logger.end_action(False, str(e)[:50])
            raise ElementNotFoundError(f"Failed to type into element {selector}: {e}") from e
    
    def _get_locator(self, selector: str, selector_type: str, use_first: bool = False):
        """
        Get a Playwright locator for the given selector.
        
        Args:
            selector: Element selector
            selector_type: Type of selector (css or xpath)
            use_first: If True, use .first to avoid strict mode violations when
                       multiple elements match. This is the best-in-class Playwright
                       approach for handling selectors that match multiple elements.
        """
        if selector_type == "xpath":
            locator = self.page.locator(f"xpath={selector}")
        else:
            locator = self.page.locator(selector)
        
        # Use .first to get only the first matching element
        # This is Playwright's recommended approach for handling multiple matches
        if use_first:
            return locator.first
        return locator

    async def get_text(self, selector: str, selector_type: str = "css") -> str:
        """
        Get text content of an element.

        Args:
            selector: Element selector
            selector_type: Type of selector (css or xpath)

        Returns:
            Text content
        """
        try:
            if selector_type == "xpath":
                return await self.page.locator(f"xpath={selector}").text_content() or ""
            else:
                return await self.page.locator(selector).text_content() or ""
        except Exception as e:
            logger.error(f"Get text failed: {e}")
            raise ElementNotFoundError(f"Failed to get text from element {selector}: {e}") from e

