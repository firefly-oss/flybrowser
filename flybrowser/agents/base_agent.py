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
Base agent class for FlyBrowser agents.

This module provides the BaseAgent abstract class which serves as the foundation
for all specialized agents in FlyBrowser. Agents are responsible for executing
specific tasks like data extraction, form filling, navigation, etc.

All agents have access to:
- PageController: For page navigation and operations
- ElementDetector: For finding and interacting with elements
- BaseLLMProvider: For LLM-powered intelligence

Subclasses must implement the execute() method to define their specific behavior.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

from flybrowser.core.element import ElementDetector
from flybrowser.core.page import PageController
from flybrowser.llm.base import BaseLLMProvider
from flybrowser.utils.logger import logger

if TYPE_CHECKING:
    from flybrowser.security.pii_handler import PIIHandler


class BaseAgent(ABC):
    """
    Abstract base class for all FlyBrowser agents.

    This class provides the common interface and utilities that all agents share.
    Agents are specialized components that perform specific browser automation tasks
    using LLM intelligence.

    Attributes:
        page: PageController instance for page operations
        detector: ElementDetector instance for element location and interaction
        llm: BaseLLMProvider instance for LLM-powered operations
        pii_handler: Optional PIIHandler for secure handling of sensitive data

    Example:
        Creating a custom agent:

        >>> class MyCustomAgent(BaseAgent):
        ...     async def execute(self, task: str) -> Dict[str, Any]:
        ...         # Get page context
        ...         context = await self.get_page_context()
        ...
        ...         # Use LLM to process task
        ...         response = await self.llm.generate(
        ...             f"Task: {task}\\nPage: {context['url']}"
        ...         )
        ...
        ...         return {"result": response.content}
        >>>
        >>> agent = MyCustomAgent(page_controller, element_detector, llm_provider)
        >>> result = await agent.execute("Summarize this page")

        With PII handling:

        >>> from flybrowser.security.pii_handler import PIIHandler
        >>> pii_handler = PIIHandler()
        >>> agent = MyCustomAgent(page_controller, element_detector, llm_provider, pii_handler=pii_handler)
    """

    def __init__(
        self,
        page_controller: PageController,
        element_detector: ElementDetector,
        llm_provider: BaseLLMProvider,
        pii_handler: Optional["PIIHandler"] = None,
    ) -> None:
        """
        Initialize the base agent with required components.

        Args:
            page_controller: PageController instance for page navigation and operations.
                Provides methods like goto(), screenshot(), get_html(), etc.
            element_detector: ElementDetector instance for finding and interacting
                with page elements using natural language descriptions.
            llm_provider: BaseLLMProvider instance for LLM-powered operations.
                Can be OpenAI, Anthropic, Ollama, or any other provider.
            pii_handler: Optional PIIHandler for secure handling of sensitive data.
                When provided, enables secure_fill/secure_type methods and
                automatic masking of PII in prompts and logs.

        Example:
            >>> from flybrowser.core.page import PageController
            >>> from flybrowser.core.element import ElementDetector
            >>> from flybrowser.llm.factory import LLMProviderFactory
            >>>
            >>> llm = LLMProviderFactory.create("openai", api_key="sk-...")
            >>> agent = MyAgent(page_controller, element_detector, llm)
            >>>
            >>> # With PII handling
            >>> from flybrowser.security.pii_handler import PIIHandler
            >>> pii_handler = PIIHandler()
            >>> agent = MyAgent(page_controller, element_detector, llm, pii_handler=pii_handler)
        """
        self.page = page_controller
        self.detector = element_detector
        self.llm = llm_provider
        self.pii_handler = pii_handler

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the agent's primary task.

        This is an abstract method that must be implemented by all subclasses.
        It defines the main behavior of the agent.

        Args:
            *args: Positional arguments specific to the agent's task
            **kwargs: Keyword arguments specific to the agent's task

        Returns:
            Dictionary containing the results of the agent's execution.
            Structure depends on the specific agent implementation.

        Raises:
            NotImplementedError: If subclass doesn't implement this method

        Example:
            In a subclass:

            >>> async def execute(self, query: str, use_vision: bool = False) -> Dict[str, Any]:
            ...     context = await self.get_page_context()
            ...     result = await self.llm.generate(query)
            ...     return {"data": result.content}
        """
        pass

    async def get_page_context(self) -> Dict[str, Any]:
        """
        Get current page context for use in LLM prompts.

        This helper method gathers key information about the current page
        that can be included in prompts to provide context to the LLM.

        Returns:
            Dictionary containing:
            - url: Current page URL
            - title: Page title
            - html: Full HTML content of the page

        Example:
            >>> context = await agent.get_page_context()
            >>> print(context["url"])
            'https://example.com'
            >>> print(context["title"])
            'Example Domain'
            >>>
            >>> # Use in prompt
            >>> prompt = f"Extract data from {context['url']}"
        """
        return {
            "url": await self.page.get_url(),
            "title": await self.page.get_title(),
            "html": await self.page.get_html(),
        }

    def mask_for_llm(self, text: str) -> str:
        """
        Mask PII in text before sending to LLM.

        If a PIIHandler is configured, replaces stored credential values with
        placeholders (e.g., {{CREDENTIAL:email}}) and masks other PII patterns.
        Otherwise, returns the text unchanged.

        Args:
            text: Text that may contain PII

        Returns:
            Text with PII masked/replaced with placeholders (if handler configured)
        """
        if self.pii_handler:
            # Use placeholder-based replacement for stored credentials
            return self.pii_handler.replace_values_with_placeholders(text)
        return text

    def mask_for_log(self, text: str) -> str:
        """
        Mask PII in text before logging.

        If a PIIHandler is configured, masks sensitive data in the text.
        Otherwise, returns the text unchanged.

        Args:
            text: Text that may contain PII

        Returns:
            Text with PII masked (if handler configured) or original text
        """
        if self.pii_handler:
            return self.pii_handler.mask_for_log(text)
        return text

    def mask_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mask sensitive values in a dictionary.

        If a PIIHandler is configured, masks sensitive values.
        Otherwise, returns the dictionary unchanged.

        Args:
            data: Dictionary that may contain sensitive values

        Returns:
            Dictionary with sensitive values masked (if handler configured)
        """
        if self.pii_handler:
            return self.pii_handler.mask_dict(data)
        return data

    def resolve_placeholders(self, text: str) -> str:
        """
        Resolve credential placeholders back to actual values.

        This is used for browser execution: the LLM's output contains
        placeholders which are resolved to real values just before
        the browser action is performed.

        Args:
            text: Text containing placeholders like {{CREDENTIAL:email}}

        Returns:
            Text with placeholders replaced by actual values
        """
        if self.pii_handler:
            return self.pii_handler.resolve_placeholders(text)
        return text

    def has_placeholders(self, text: str) -> bool:
        """Check if text contains any credential placeholders."""
        if self.pii_handler:
            return self.pii_handler.has_placeholders(text)
        return False

    async def secure_fill(
        self,
        selector: str,
        credential_id: str,
        clear_first: bool = True,
    ) -> bool:
        """
        Securely fill a form field with a stored credential.

        The credential value is retrieved from the PIIHandler and filled
        directly into the form without ever being exposed to the LLM or logged.

        Args:
            selector: CSS selector for the input field
            credential_id: ID of the stored credential in PIIHandler
            clear_first: Whether to clear the field before filling

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If no PIIHandler is configured

        Example:
            >>> # Store credential first
            >>> cred_id = pii_handler.store_credential("password", "secret123")
            >>> # Then use secure_fill
            >>> await agent.secure_fill("#password-input", cred_id)
        """
        if not self.pii_handler:
            raise ValueError("PIIHandler is required for secure_fill. Initialize agent with pii_handler parameter.")

        return await self.pii_handler.secure_fill(
            self.page.page,
            selector,
            credential_id,
            clear_first=clear_first,
        )

    async def secure_type(
        self,
        selector: str,
        credential_id: str,
        delay: int = 50,
    ) -> bool:
        """
        Securely type a credential into a form field character by character.

        The credential value is retrieved from the PIIHandler and typed
        directly into the form without ever being exposed to the LLM or logged.
        Useful for fields that don't work well with fill().

        Args:
            selector: CSS selector for the input field
            credential_id: ID of the stored credential in PIIHandler
            delay: Delay between keystrokes in milliseconds

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If no PIIHandler is configured

        Example:
            >>> # Store credential first
            >>> cred_id = pii_handler.store_credential("password", "secret123")
            >>> # Then use secure_type
            >>> await agent.secure_type("#password-input", cred_id, delay=100)
        """
        if not self.pii_handler:
            raise ValueError("PIIHandler is required for secure_type. Initialize agent with pii_handler parameter.")

        return await self.pii_handler.secure_type(
            self.page.page,
            selector,
            credential_id,
            delay=delay,
        )

