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
Structured LLM wrapper for reliable JSON responses.

This module provides a centralized wrapper for LLM calls that:
- Enforces structured JSON output
- Validates responses against schemas
- Automatically repairs malformed outputs
- Provides consistent error handling

Used by tools and components that need reliable LLM-generated JSON.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, TypeVar, Union

from flybrowser.agents.config import (
    estimate_tokens,
    calculate_max_tokens_for_response,
    calculate_max_tokens_for_vision_response,
)

if TYPE_CHECKING:
    from flybrowser.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

# Default max tokens for structured output (generous to avoid truncation)
DEFAULT_STRUCTURED_MAX_TOKENS = 8192
DEFAULT_STRUCTURED_VISION_MAX_TOKENS = 8192

T = TypeVar('T')


def validate_json_schema(
    data: Dict[str, Any],
    schema: Dict[str, Any],
    path: str = "",
) -> tuple[bool, List[str]]:
    """
    Validate data against a JSON schema.
    
    Simple validation that checks:
    - Required fields
    - Types
    - Nested objects
    
    Args:
        data: Data to validate
        schema: JSON schema to validate against
        path: Current path (for error messages)
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    if not isinstance(data, dict):
        return False, [f"{path or 'root'}: expected object, got {type(data).__name__}"]
    
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    # Check required fields
    for field in required:
        if field not in data:
            errors.append(f"{path}.{field}" if path else f"Missing required field '{field}'")
    
    # Validate each property
    for field, value in data.items():
        if field not in properties:
            continue  # Allow additional properties
        
        prop_schema = properties[field]
        field_path = f"{path}.{field}" if path else field
        
        # Handle oneOf at property level (e.g., value can be string or object)
        if "oneOf" in prop_schema:
            if not _validate_one_of(value, prop_schema["oneOf"], field_path):
                errors.append(f"'{field_path}' does not match any oneOf schema")
        elif expected_type := prop_schema.get("type"):
            type_valid, type_errors = _validate_type(value, expected_type, prop_schema, field_path)
            if not type_valid:
                errors.extend(type_errors)
    
    return len(errors) == 0, errors


def _validate_type(
    value: Any,
    expected_type: str,
    prop_schema: Dict[str, Any],
    path: str,
) -> tuple[bool, List[str]]:
    """Validate a value against an expected type."""
    errors = []
    
    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    
    if expected_type == "object" and isinstance(value, dict):
        # Recursively validate nested objects
        if "properties" in prop_schema:
            nested_valid, nested_errors = validate_json_schema(value, prop_schema, path)
            if not nested_valid:
                errors.extend(nested_errors)
    elif expected_type == "array" and isinstance(value, list):
        # Validate array items if items schema provided
        if "items" in prop_schema:
            items_schema = prop_schema["items"]
            for i, item in enumerate(value):
                item_path = f"{path}[{i}]"
                # Handle oneOf in items
                if "oneOf" in items_schema:
                    if not _validate_one_of(item, items_schema["oneOf"], item_path):
                        errors.append(f"'{item_path}' does not match any oneOf schema")
                elif items_schema.get("type") == "object":
                    item_valid, item_errors = validate_json_schema(item, items_schema, item_path)
                    if not item_valid:
                        errors.extend(item_errors)
    elif expected_type in type_map:
        if not isinstance(value, type_map[expected_type]):
            errors.append(f"'{path}' must be {expected_type}, got {type(value).__name__}")
    
    return len(errors) == 0, errors


def _validate_one_of(
    value: Any,
    one_of_schemas: List[Dict[str, Any]],
    path: str,
) -> bool:
    """
    Validate that a value matches at least one of the given schemas.
    
    Args:
        value: Value to validate
        one_of_schemas: List of schemas, value must match one
        path: Current path (for error messages)
        
    Returns:
        True if value matches at least one schema
    """
    for schema in one_of_schemas:
        schema_type = schema.get("type")
        
        if schema_type == "string" and isinstance(value, str):
            return True
        elif schema_type == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        elif schema_type == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        elif schema_type == "boolean" and isinstance(value, bool):
            return True
        elif schema_type == "null" and value is None:
            return True
        elif schema_type == "array" and isinstance(value, list):
            return True
        elif schema_type == "object" and isinstance(value, dict):
            # Check if the object matches the schema's properties
            if "properties" in schema:
                is_valid, _ = validate_json_schema(value, schema, path)
                if is_valid:
                    return True
            else:
                # No properties defined, any object is valid
                return True
    
    return False


def build_repair_prompt(
    original_prompt: str,
    malformed_output: str,
    validation_errors: List[str],
    schema: Dict[str, Any],
) -> str:
    """
    Build a repair prompt to fix malformed LLM output.
    
    Args:
        original_prompt: Original user prompt context
        malformed_output: The malformed JSON output
        validation_errors: List of validation error messages
        schema: Expected JSON schema
        
    Returns:
        Prompt asking LLM to repair the output
    """
    schema_str = json.dumps(schema, indent=2)
    errors_str = "\n".join(f"- {err}" for err in validation_errors)
    
    # Truncate if too long
    max_output_len = 2000
    if len(malformed_output) > max_output_len:
        malformed_output = malformed_output[:max_output_len] + "... [truncated]"
    
    max_prompt_len = 1000
    if len(original_prompt) > max_prompt_len:
        original_prompt = original_prompt[:max_prompt_len] + "... [truncated]"
    
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

## Original Context (abbreviated)
{original_prompt}

## Instructions
Provide a corrected JSON response that:
1. Fixes all validation errors listed above
2. Maintains the same intent from your original response
3. Strictly follows the required schema

Respond ONLY with the corrected JSON object."""


class StructuredLLMWrapper:
    """
    Wrapper for reliable structured LLM responses.
    
    Provides:
    - Structured JSON output enforcement
    - Schema validation
    - Automatic repair for malformed responses
    - Consistent error handling
    
    Example:
        >>> wrapper = StructuredLLMWrapper(llm_provider)
        >>> result = await wrapper.generate_structured(
        ...     prompt="Analyze this data",
        ...     schema={"type": "object", "properties": {...}},
        ...     system_prompt="You are an analyst",
        ... )
    """
    
    def __init__(
        self,
        llm_provider: "BaseLLMProvider",
        max_repair_attempts: int = 2,
        repair_temperature: float = 0.1,
    ):
        """
        Initialize the wrapper.
        
        Args:
            llm_provider: LLM provider to wrap
            max_repair_attempts: Maximum repair attempts for malformed output
            repair_temperature: Temperature for repair attempts (low = deterministic)
        """
        self.llm = llm_provider
        self.max_repair_attempts = max_repair_attempts
        self.repair_temperature = repair_temperature
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        custom_validator: Optional[Callable[[Dict[str, Any]], tuple[bool, List[str]]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON response with validation and repair.
        
        Args:
            prompt: User prompt
            schema: JSON schema for response
            system_prompt: System prompt (if None, uses a default)
            temperature: Generation temperature
            max_tokens: Maximum tokens (if None, calculated dynamically)
            custom_validator: Optional custom validation function
            
        Returns:
            Validated JSON response
            
        Raises:
            ValueError: If response cannot be validated after repair attempts
        """
        # Calculate max_tokens dynamically if not provided
        effective_max_tokens = max_tokens
        if effective_max_tokens is None:
            system_tokens = estimate_tokens(system_prompt or "")
            prompt_tokens = estimate_tokens(prompt)
            schema_tokens = estimate_tokens(json.dumps(schema))
            effective_max_tokens = calculate_max_tokens_for_response(
                system_prompt_tokens=system_tokens,
                user_prompt_tokens=prompt_tokens + schema_tokens,
                context_tokens=0,
                safety_margin=1.5,  # 50% buffer for structured output
            )
            # Ensure minimum for complex JSON structures
            effective_max_tokens = max(effective_max_tokens, 4096)
            logger.debug(f"[STRUCTURED] Dynamic max_tokens={effective_max_tokens}")
        
        # Use the provider's built-in structured output if available
        if hasattr(self.llm, 'generate_structured'):
            try:
                structured_data = await self.llm.generate_structured(
                    prompt=prompt,
                    schema=schema,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=effective_max_tokens,
                )
            except Exception as e:
                logger.warning(f"[STRUCTURED] generate_structured failed: {e}, falling back to regular generation")
                structured_data = await self._generate_and_parse(
                    prompt, schema, system_prompt, temperature, effective_max_tokens
                )
        else:
            structured_data = await self._generate_and_parse(
                prompt, schema, system_prompt, temperature, effective_max_tokens
            )
        
        # Validate
        validator = custom_validator or (lambda d: validate_json_schema(d, schema))
        is_valid, errors = validator(structured_data)
        
        if is_valid:
            return structured_data
        
        # Repair loop
        logger.warning(f"[STRUCTURED] Validation failed: {errors}")
        return await self._repair_response(
            structured_data=structured_data,
            original_prompt=prompt,
            system_prompt=system_prompt,
            schema=schema,
            validator=validator,
        )
    
    async def generate_structured_with_vision(
        self,
        prompt: str,
        image_data: Union[bytes, List[bytes]],
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        custom_validator: Optional[Callable[[Dict[str, Any]], tuple[bool, List[str]]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON response with vision and validation.
        
        Supports single image (bytes) or multiple images (List[bytes]).
        
        Args:
            prompt: User prompt
            image_data: Single image bytes or list of image bytes for multi-image
            schema: JSON schema for response
            system_prompt: System prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens (if None, calculated dynamically)
            custom_validator: Optional custom validation function
            
        Returns:
            Validated JSON response
        """
        # Calculate max_tokens dynamically if not provided
        effective_max_tokens = max_tokens
        if effective_max_tokens is None:
            # Calculate total image size
            if isinstance(image_data, list):
                total_image_size = sum(len(img) for img in image_data)
            else:
                total_image_size = len(image_data)
            
            effective_max_tokens = calculate_max_tokens_for_vision_response(
                system_prompt=system_prompt or "",
                user_prompt=prompt + json.dumps(schema),
                image_size_bytes=total_image_size,
                context_tokens=0,
                safety_margin=1.5,  # 50% buffer
            )
            # Ensure minimum for vision + JSON structures
            effective_max_tokens = max(effective_max_tokens, 4096)
            logger.debug(f"[STRUCTURED VISION] Dynamic max_tokens={effective_max_tokens}")
        
        # Use provider's structured vision if available
        if hasattr(self.llm, 'generate_structured_with_vision'):
            try:
                structured_data = await self.llm.generate_structured_with_vision(
                    prompt=prompt,
                    image_data=image_data,
                    schema=schema,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=effective_max_tokens,
                )
            except Exception as e:
                logger.warning(f"[STRUCTURED] generate_structured_with_vision failed: {e}, using regular vision")
                structured_data = await self._generate_vision_and_parse(
                    prompt, image_data, schema, system_prompt, temperature, effective_max_tokens
                )
        else:
            structured_data = await self._generate_vision_and_parse(
                prompt, image_data, schema, system_prompt, temperature, effective_max_tokens
            )
        
        # Validate
        validator = custom_validator or (lambda d: validate_json_schema(d, schema))
        is_valid, errors = validator(structured_data)
        
        if is_valid:
            return structured_data
        
        # Repair (without vision - just fix JSON)
        logger.warning(f"[STRUCTURED VISION] Validation failed: {errors}")
        return await self._repair_response(
            structured_data=structured_data,
            original_prompt=prompt,
            system_prompt=system_prompt,
            schema=schema,
            validator=validator,
        )
    
    async def _generate_and_parse(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        """Generate and parse JSON from regular generation."""
        # Add JSON instruction to prompt
        json_prompt = f"""{prompt}

IMPORTANT: Respond ONLY with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Do not include any explanation or markdown, just the JSON object."""
        
        response = await self.llm.generate(
            prompt=json_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return self._extract_json(response.content)
    
    async def _generate_vision_and_parse(
        self,
        prompt: str,
        image_data: Union[bytes, List[bytes]],
        schema: Dict[str, Any],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        """Generate and parse JSON from vision generation (single or multi-image)."""
        json_prompt = f"""{prompt}

IMPORTANT: Respond ONLY with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Do not include any explanation or markdown, just the JSON object."""
        
        response = await self.llm.generate_with_vision(
            prompt=json_prompt,
            image_data=image_data,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return self._extract_json(response.content)
    
    async def _repair_response(
        self,
        structured_data: Dict[str, Any],
        original_prompt: str,
        system_prompt: Optional[str],
        schema: Dict[str, Any],
        validator: Callable[[Dict[str, Any]], tuple[bool, List[str]]],
    ) -> Dict[str, Any]:
        """Attempt to repair malformed response."""
        is_valid, errors = validator(structured_data)
        
        for attempt in range(self.max_repair_attempts):
            logger.info(f"[REPAIR] Attempt {attempt + 1}/{self.max_repair_attempts}")
            
            malformed_output = json.dumps(structured_data, indent=2)
            repair_prompt = build_repair_prompt(
                original_prompt=original_prompt,
                malformed_output=malformed_output,
                validation_errors=errors,
                schema=schema,
            )
            
            try:
                if hasattr(self.llm, 'generate_structured'):
                    repaired_data = await self.llm.generate_structured(
                        prompt=repair_prompt,
                        schema=schema,
                        system_prompt="Fix the malformed JSON to match the required schema.",
                        temperature=self.repair_temperature,
                    )
                else:
                    response = await self.llm.generate(
                        prompt=repair_prompt,
                        system_prompt="Fix the malformed JSON to match the required schema.",
                        temperature=self.repair_temperature,
                    )
                    repaired_data = self._extract_json(response.content)
                
                is_valid, errors = validator(repaired_data)
                
                if is_valid:
                    logger.info(f"[REPAIR] Successfully repaired on attempt {attempt + 1}")
                    return repaired_data
                else:
                    logger.warning(f"[REPAIR] Attempt {attempt + 1} still invalid: {errors}")
                    structured_data = repaired_data
                    
            except Exception as e:
                logger.error(f"[REPAIR] Attempt {attempt + 1} failed: {e}")
        
        # All attempts failed
        error_msg = f"Validation failed after {self.max_repair_attempts} repair attempts. Errors: {errors}"
        logger.error(f"[REPAIR] {error_msg}")
        raise ValueError(error_msg)
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text that may contain markdown or other content."""
        import re
        
        # Try direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code block
        patterns = [
            r'```(?:json)?\s*(\{.*?\})\s*```',  # ```json {...}```
            r'```(?:json)?\s*(\[.*?\])\s*```',  # ```json [...]```
            r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',  # Nested object
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        
        # Try finding JSON object boundaries
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not extract valid JSON from response: {text[:200]}...")


# Convenience function for one-off structured calls
async def generate_structured_response(
    llm_provider: "BaseLLMProvider",
    prompt: str,
    schema: Dict[str, Any],
    system_prompt: Optional[str] = None,
    temperature: float = 0.3,
    max_repair_attempts: int = 2,
) -> Dict[str, Any]:
    """
    Convenience function for generating structured responses.
    
    Args:
        llm_provider: LLM provider
        prompt: User prompt
        schema: JSON schema
        system_prompt: Optional system prompt
        temperature: Generation temperature
        max_repair_attempts: Max repair attempts
        
    Returns:
        Validated JSON response
    """
    wrapper = StructuredLLMWrapper(
        llm_provider=llm_provider,
        max_repair_attempts=max_repair_attempts,
    )
    return await wrapper.generate_structured(
        prompt=prompt,
        schema=schema,
        system_prompt=system_prompt,
        temperature=temperature,
    )
