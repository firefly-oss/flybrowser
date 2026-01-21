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
Response validation agent for ensuring LLM responses match expected formats.

This module provides the ResponseValidator class which validates LLM responses
against JSON schemas and attempts to fix malformed responses by asking the LLM
to correct them.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from flybrowser.llm.base import BaseLLMProvider
from flybrowser.utils.logger import logger


class ResponseValidator:
    """
    Validates and fixes LLM responses to ensure they match expected formats.
    
    This validator acts as a "judge" that ensures LLM responses are properly
    formatted JSON matching the expected schema. If a response is malformed,
    it will attempt to extract or fix the JSON, or ask the LLM to correct it.
    
    Example:
        >>> validator = ResponseValidator(llm_provider)
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "selector": {"type": "string"},
        ...         "confidence": {"type": "number"}
        ...     },
        ...     "required": ["selector"]
        ... }
        >>> result = await validator.validate_and_fix(
        ...     response_text,
        ...     schema,
        ...     max_attempts=3
        ... )
    """
    
    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        """
        Initialize the response validator.
        
        Args:
            llm_provider: LLM provider for fixing invalid responses
        """
        self.llm = llm_provider
        
    async def validate_and_fix(
        self,
        response_text: str,
        schema: Dict[str, Any],
        context: Optional[str] = None,
        max_attempts: int = 3,
    ) -> Dict[str, Any]:
        """
        Validate a response against a schema and fix if needed.
        
        This method will:
        1. Try to parse the response as JSON
        2. If parsing fails, attempt to extract JSON from the text
        3. If extraction fails, ask the LLM to fix the response
        4. Validate the result against the schema
        5. Repeat up to max_attempts times
        
        Args:
            response_text: The LLM response to validate
            schema: JSON schema the response should match
            context: Optional context about what the response should contain
            max_attempts: Maximum validation/fix attempts (default: 3)
            
        Returns:
            Validated and parsed JSON object
            
        Raises:
            ValueError: If validation fails after max_attempts
        """
        current_text = response_text
        
        for attempt in range(max_attempts):
            try:
                # Try to parse as JSON
                parsed = self._try_parse_json(current_text)
                
                if parsed is not None:
                    # Validate against schema
                    if self._validate_schema(parsed, schema):
                        logger.debug(f"Response validated successfully on attempt {attempt + 1}")
                        return parsed
                    else:
                        logger.warning(f"Response doesn't match schema on attempt {attempt + 1}")
                        
                # If we're here, parsing failed or schema didn't match
                if attempt < max_attempts - 1:
                    logger.info(f"Attempting to fix response (attempt {attempt + 1}/{max_attempts})")
                    current_text = await self._ask_llm_to_fix(
                        current_text,
                        schema,
                        context,
                        attempt + 1
                    )
                else:
                    # Last attempt - try best-effort extraction
                    logger.warning("Max attempts reached, trying best-effort extraction")
                    parsed = self._extract_json_best_effort(current_text)
                    if parsed and self._validate_schema(parsed, schema):
                        return parsed
                    raise ValueError(
                        f"Failed to validate response after {max_attempts} attempts. "
                        f"Last response: {current_text[:200]}..."
                    )
                    
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    current_text = await self._ask_llm_to_fix(
                        current_text,
                        schema,
                        context,
                        attempt + 1
                    )
                else:
                    raise ValueError(
                        f"Failed to parse JSON after {max_attempts} attempts: {e}"
                    )
                    
        raise ValueError(f"Validation failed after {max_attempts} attempts")
        
    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Try to parse text as JSON, with multiple strategies.
        
        Args:
            text: Text to parse
            
        Returns:
            Parsed JSON object or None if parsing fails
        """
        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        # Strategy 2: Extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
                
        # Strategy 3: Find first { to last } and try to parse
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
            
        return None
        
    def _extract_json_best_effort(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Best-effort JSON extraction from text.
        
        This tries various heuristics to extract valid JSON from
        potentially malformed or verbose responses.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Extracted JSON or None
        """
        # Try all JSON objects in the text
        for match in re.finditer(r'\{[^{}]*\}', text):
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
                
        return None
    
    def _validate_schema(self, data: Any, schema: Dict[str, Any], path: str = "root") -> bool:
        """
        Validate data against JSON schema with recursive validation.
        
        Enhanced validator that handles:
        - Required properties
        - Type checking with proper type coercion
        - Nested objects (recursive)
        - Arrays with item schemas
        - Min/max properties
        - Additional properties
        - Rejection of schema-like responses
        
        Args:
            data: Data to validate
            schema: JSON schema to validate against
            path: Current path in data (for error messages)
            
        Returns:
            True if valid, False otherwise
        """
        # First check: Reject if response is itself a schema definition
        if isinstance(data, dict) and self._is_schema_definition(data):
            logger.warning(f"{path}: Response appears to be a schema definition, not actual data")
            return False
        # Handle type validation
        expected_type = schema.get("type")
        if expected_type:
            # Handle multiple allowed types (e.g., ["object", "array"])
            if isinstance(expected_type, list):
                type_valid = any(self._check_type(data, t, path, silent=True) for t in expected_type)
                if not type_valid:
                    logger.debug(f"{path}: Expected one of types {expected_type}, got {type(data).__name__}")
                    return False
            else:
                if not self._check_type(data, expected_type, path):
                    return False
        
        # For objects, validate properties
        if isinstance(data, dict):
            # Check required properties
            required = schema.get("required", [])
            for prop in required:
                if prop not in data:
                    logger.debug(f"{path}: Missing required property '{prop}'")
                    return False
            
            # Check min/max properties
            if "minProperties" in schema:
                if len(data) < schema["minProperties"]:
                    logger.debug(f"{path}: Too few properties (need at least {schema['minProperties']})")
                    return False
            
            if "maxProperties" in schema:
                if len(data) > schema["maxProperties"]:
                    logger.debug(f"{path}: Too many properties (max {schema['maxProperties']})")
                    return False
            
            # Validate each property
            properties = schema.get("properties", {})
            additional_allowed = schema.get("additionalProperties", True)
            
            for prop, value in data.items():
                prop_path = f"{path}.{prop}"
                
                if prop in properties:
                    # Validate against property schema (recursive)
                    if not self._validate_schema(value, properties[prop], prop_path):
                        return False
                elif not additional_allowed:
                    logger.debug(f"{prop_path}: Additional property not allowed")
                    return False
        
        # For arrays, validate items
        elif isinstance(data, list):
            items_schema = schema.get("items")
            if items_schema:
                for i, item in enumerate(data):
                    item_path = f"{path}[{i}]"
                    if not self._validate_schema(item, items_schema, item_path):
                        return False
            
            # Check min/max items
            if "minItems" in schema and len(data) < schema["minItems"]:
                logger.debug(f"{path}: Too few items (need at least {schema['minItems']})")
                return False
            
            if "maxItems" in schema and len(data) > schema["maxItems"]:
                logger.debug(f"{path}: Too many items (max {schema['maxItems']})")
                return False
        
        return True
    
    def _check_type(self, value: Any, expected_type: str, path: str, silent: bool = False) -> bool:
        """
        Check if value matches expected type.
        
        Args:
            value: Value to check
            expected_type: Expected type name
            path: Path for error messages
            silent: If True, don't log errors (for trying multiple types)
            
        Returns:
            True if type matches
        """
        type_checks = {
            "string": lambda v: isinstance(v, str),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
            "null": lambda v: v is None,
        }
        
        check = type_checks.get(expected_type)
        if not check:
            logger.warning(f"Unknown type '{expected_type}' in schema")
            return True  # Be permissive for unknown types
        
        if not check(value):
            if not silent:
                actual_type = type(value).__name__
                logger.debug(f"{path}: Expected type '{expected_type}', got '{actual_type}'")
            return False
        
        return True
    
    def _is_schema_definition(self, data: Dict[str, Any]) -> bool:
        """
        Detect if a response is actually a JSON schema definition instead of data.
        
        Args:
            data: Dictionary to check
            
        Returns:
            True if data looks like a schema definition
        """
        # Check for schema keywords at the root level
        schema_keywords = ["type", "properties", "required", "items", "additionalProperties", "$schema"]
        schema_type_values = ["object", "array", "string", "number", "boolean", "null", "integer"]
        
        # If 'type' key exists with a schema type value, it's likely a schema
        if "type" in data:
            type_val = data["type"]
            if isinstance(type_val, str) and type_val in schema_type_values:
                # Check if it has other schema keywords
                if "properties" in data or "items" in data or "required" in data:
                    return True
        
        # Check if 'properties' contains nested type definitions
        if "properties" in data and isinstance(data["properties"], dict):
            for prop_val in data["properties"].values():
                if isinstance(prop_val, dict) and "type" in prop_val:
                    return True
        
        return False
    
    def _describe_expected_data(self, schema: Dict[str, Any], context: Optional[str]) -> str:
        """
        Describe what data is expected without showing the schema structure.
        
        Args:
            schema: JSON schema
            context: Optional context about the task
            
        Returns:
            Human-readable description of expected data
        """
        description = "What you need to return:\n"
        
        if context:
            description += f"Task: {context}\n"
        
        # Check for common patterns
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        if "extracted_data" in properties:
            description += "- A JSON object with 'extracted_data' containing the actual information from the page\n"
        elif "actions" in properties:
            description += "- A JSON object with 'actions' array containing the specific actions to take\n"
        elif "conditions" in properties:
            description += "- A JSON object with 'conditions' array describing what to monitor\n"
        else:
            description += f"- A JSON object with these fields: {', '.join(required)}\n"
        
        description += "\nRemember: Use REAL data from the page, not placeholders or schema definitions."
        return description
        
        
    async def _ask_llm_to_fix(
        self,
        malformed_response: str,
        schema: Dict[str, Any],
        context: Optional[str],
        attempt: int
    ) -> str:
        """
        Ask the LLM to fix a malformed response.
        
        Args:
            malformed_response: The malformed response to fix
            schema: Expected JSON schema
            context: Optional context about the task
            attempt: Current attempt number
            
        Returns:
            Fixed response from LLM
        """
        # Extract just the data requirements from schema, not the structure
        data_description = self._describe_expected_data(schema, context)
        
        fix_prompt = f"""Your previous response was not valid. You returned schema/structure instead of actual data.

Previous attempt:
{malformed_response[:500]}

{data_description}

CRITICAL INSTRUCTIONS:
1. Return ACTUAL DATA from the page, NOT a schema definition
2. Do NOT return anything with "type": "object" or "properties"
3. Do NOT return angle brackets like <placeholder>
4. Do NOT return example/dummy data
5. Extract REAL values visible on the current page

Respond with ONLY valid JSON containing actual data.
Start with {{ and end with }}. No markdown, no explanations.

Attempt {attempt}/3 - Return real data now."""

        try:
            response = await self.llm.generate(
                prompt=fix_prompt,
                system_prompt="You are a JSON formatting expert. Return ONLY valid JSON with no additional text.",
                temperature=0.1,  # Very low temperature for precise formatting
            )
            return response.content
        except Exception as e:
            logger.error(f"Failed to ask LLM to fix response: {e}")
            return malformed_response
