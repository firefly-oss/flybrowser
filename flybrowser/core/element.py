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

import json
from typing import Any, Dict, Optional

from playwright.async_api import Page

from flybrowser.exceptions import ElementNotFoundError
from flybrowser.llm.base import BaseLLMProvider
from flybrowser.llm.prompts import ELEMENT_DETECTION_PROMPT, ELEMENT_DETECTION_SYSTEM
from flybrowser.utils.logger import logger


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

    def __init__(self, page: Page, llm_provider: BaseLLMProvider) -> None:
        """
        Initialize the element detector.

        Args:
            page: Playwright Page instance for element operations
            llm_provider: LLM provider for intelligent element detection.
                Should support vision capabilities for best results.

        Example:
            >>> from flybrowser.llm.factory import LLMProviderFactory
            >>> llm = LLMProviderFactory.create("openai", api_key="sk-...")
            >>> detector = ElementDetector(page, llm)
        """
        self.page = page
        self.llm = llm_provider

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
                "reasoning": "Why this element was selected"
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
        try:
            logger.info(f"Finding element: {description}")

            # Get page information for context
            url = self.page.url
            title = await self.page.title()
            html = await self.page.content()

            # Truncate HTML for prompt to avoid token limits
            # Keep first 5000 characters which usually includes key page structure
            html_snippet = html[:5000] + "..." if len(html) > 5000 else html

            # Build prompt with page context
            prompt = ELEMENT_DETECTION_PROMPT.format(
                description=description,
                url=url,
                title=title,
                html_snippet=html_snippet,
            )

            # Use vision-based detection if enabled
            if use_vision:
                screenshot = await self.page.screenshot(type="png")
                response = await self.llm.generate_with_vision(
                    prompt=prompt,
                    image_data=screenshot,
                    system_prompt=ELEMENT_DETECTION_SYSTEM,
                    temperature=0.3,
                )
            else:
                response = await self.llm.generate(
                    prompt=prompt,
                    system_prompt=ELEMENT_DETECTION_SYSTEM,
                    temperature=0.3,
                )

            # Parse response
            try:
                result = json.loads(response.content)
                logger.info(f"Found element with selector: {result.get('selector')}")
                return result
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response: {response.content}")
                raise ElementNotFoundError(
                    f"Failed to parse element detection response for: {description}"
                )

        except Exception as e:
            logger.error(f"Element detection failed: {e}")
            raise ElementNotFoundError(f"Failed to find element '{description}': {e}") from e

    async def click(self, selector: str, selector_type: str = "css") -> None:
        """
        Click an element.

        Args:
            selector: Element selector
            selector_type: Type of selector (css or xpath)
        """
        try:
            if selector_type == "xpath":
                await self.page.locator(f"xpath={selector}").click()
            else:
                await self.page.locator(selector).click()
            logger.info(f"Clicked element: {selector}")
        except Exception as e:
            logger.error(f"Click failed: {e}")
            raise ElementNotFoundError(f"Failed to click element {selector}: {e}") from e

    async def type_text(
        self, selector: str, text: str, selector_type: str = "css", delay: int = 0
    ) -> None:
        """
        Type text into an element.

        Args:
            selector: Element selector
            text: Text to type
            selector_type: Type of selector (css or xpath)
            delay: Delay between keystrokes in milliseconds
        """
        try:
            if selector_type == "xpath":
                await self.page.locator(f"xpath={selector}").fill(text)
            else:
                await self.page.locator(selector).fill(text)
            logger.info(f"Typed text into element: {selector}")
        except Exception as e:
            logger.error(f"Type text failed: {e}")
            raise ElementNotFoundError(f"Failed to type into element {selector}: {e}") from e

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

