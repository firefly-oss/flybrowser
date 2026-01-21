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
Extraction agent for intelligent data extraction from web pages.

This module provides the ExtractionAgent class which uses LLMs to extract
structured or unstructured data from web pages based on natural language queries.

The agent supports:
- Natural language extraction queries
- Vision-based extraction using screenshots
- Structured extraction with JSON schemas
- Text-based extraction from HTML

Example:
    >>> agent = ExtractionAgent(page_controller, element_detector, llm_provider)
    >>> data = await agent.execute("Extract all product names and prices")
    >>> print(data)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from flybrowser.agents.base_agent import BaseAgent
from flybrowser.agents.validation_agent import ResponseValidator
from flybrowser.exceptions import ExtractionError
from flybrowser.llm.prompts import EXTRACTION_PROMPT, EXTRACTION_SYSTEM
from flybrowser.utils.logger import logger
from flybrowser.utils.timing import StepTimer


class ExtractionAgent(BaseAgent):
    """
    Agent specialized in extracting data from web pages using LLMs.

    This agent uses LLMs to understand natural language extraction queries
    and extract relevant data from web pages. It supports multiple extraction
    modes:

    1. Text-based: Analyzes HTML content
    2. Vision-based: Analyzes page screenshots
    3. Structured: Returns data matching a JSON schema

    The agent inherits from BaseAgent and has access to:
    - page_controller: For page operations
    - element_detector: For element location
    - llm: For intelligent extraction

    Example:
        >>> agent = ExtractionAgent(page_controller, element_detector, llm)
        >>>
        >>> # Simple extraction
        >>> data = await agent.execute("What is the main heading?")
        >>>
        >>> # Structured extraction
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "title": {"type": "string"},
        ...         "price": {"type": "number"}
        ...     }
        ... }
        >>> data = await agent.execute("Extract product info", schema=schema)
    """
    
    def __init__(self, page_controller, element_detector, llm_provider, pii_handler=None) -> None:
        """Initialize the extraction agent."""
        super().__init__(page_controller, element_detector, llm_provider, pii_handler=pii_handler)
        self.validator = ResponseValidator(llm_provider)

    async def execute(
        self, query: str, use_vision: bool = False, schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract data from the current page based on a natural language query.

        This method uses an LLM to understand the query and extract the requested
        information from the page. It can work with HTML content, screenshots,
        or both, and can return structured data matching a JSON schema.

        Args:
            query: Natural language query describing what to extract.
                Examples:
                - "What is the main heading?"
                - "Extract all product names and prices"
                - "Get the article title, author, and publication date"
                - "Find all links in the navigation menu"
            use_vision: Whether to use vision-based extraction. When True,
                captures a screenshot and sends it to a vision-capable LLM
                for better understanding of visual layout. Default: False
            schema: Optional JSON schema for structured extraction. When provided,
                the LLM will return data matching this schema structure.
                Example:
                {
                    "type": "object",
                    "properties": {
                        "products": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "price": {"type": "number"}
                                }
                            }
                        }
                    }
                }

        Returns:
            Dictionary containing the extracted data. Structure depends on
            the query and schema (if provided).

        Raises:
            ExtractionError: If extraction fails or LLM returns invalid data

        Example:
            Simple extraction:
            >>> data = await agent.execute("What is the page title?")
            >>> print(data)
            {'title': 'Example Domain'}

            Structured extraction:
            >>> schema = {
            ...     "type": "object",
            ...     "properties": {
            ...         "products": {
            ...             "type": "array",
            ...             "items": {
            ...                 "type": "object",
            ...                 "properties": {
            ...                     "name": {"type": "string"},
            ...                     "price": {"type": "number"}
            ...                 }
            ...             }
            ...         }
            ...     }
            ... }
            >>> data = await agent.execute(
            ...     "Extract all products",
            ...     schema=schema
            ... )

            Vision-based extraction:
            >>> data = await agent.execute(
            ...     "What colors are used in the header?",
            ...     use_vision=True
            ... )
        """
        # Start timing
        timer = StepTimer()
        timer.start()
        
        try:
            # Mask query for logging to avoid exposing PII
            logger.info(f"Extracting data: {self.mask_for_log(query)}")

            # Get page context (URL, title, HTML)
            timer.start_step("get_page_context")
            context = await self.get_page_context()
            timer.end_step("get_page_context")

            # Mask query for LLM to avoid exposing PII
            safe_query = self.mask_for_llm(query)

            # Build extraction prompt with page context
            prompt = EXTRACTION_PROMPT.format(
                query=safe_query,
                url=context["url"],
                title=context["title"],
            )

            # Add HTML context (truncated to avoid token limits)
            # Keep first 8000 characters which usually includes key content
            html_snippet = context["html"][:8000] + "..." if len(context["html"]) > 8000 else context["html"]
            prompt += f"\n\nHTML Content:\n{html_snippet}"

            # Use standardized default schema if none provided
            # This provides consistent structure for any extraction query
            if not schema:
                schema = {
                    "type": "object",
                    "properties": {
                        "extracted_data": {
                            "type": ["object", "array"],
                            "description": "The extracted information - use object for single items, array for lists"
                        }
                    },
                    "required": ["extracted_data"],
                    "additionalProperties": False
                }

            # Use vision-based extraction if enabled
            timer.start_step("llm_generate")
            if use_vision:
                screenshot = await self.page.screenshot()
                response = await self.llm.generate_with_vision(
                    prompt=prompt,
                    image_data=screenshot,
                    system_prompt=EXTRACTION_SYSTEM,
                    temperature=0.3,
                )
            else:
                response = await self.llm.generate(
                    prompt=prompt,
                    system_prompt=EXTRACTION_SYSTEM,
                    temperature=0.3,
                )
            timer.end_step("llm_generate")

            # Validate and fix response to ensure JSON format
            timer.start_step("validate_response")
            result = await self.validator.validate_and_fix(
                response.content,
                schema,
                context=f"Extracting data for query: {safe_query}"
            )
            timer.end_step("validate_response")
            
            # Enrich metadata with accurate values using base method
            timer.start_step("enrich_metadata")
            result = await self.enrich_response_metadata(
                result,
                query_or_instruction=query,
                include_page_context=True
            )
            timer.end_step("enrich_metadata")

            logger.info("Data extracted successfully")
            return {
                "success": True,
                "data": result,
                "query": query,
                "timing": timer.get_timings().to_dict(),
            }

        except Exception as e:
            logger.error(f"Extraction failed: {self.mask_for_log(str(e))}")
            # Return error dict instead of raising, for better error handling
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "query": query,
                "exception_type": type(e).__name__,
                "timing": timer.get_timings().to_dict(),
            }

