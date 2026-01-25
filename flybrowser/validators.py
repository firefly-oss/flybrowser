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
Built-in validators for data extraction.

This module provides common validation functions for use with FlyBrowser's
scraping and extraction capabilities.

Usage:
    >>> from flybrowser.validators import not_empty, min_items, has_required_fields
    >>> 
    >>> result = await browser.scrape(
    ...     goal="Extract products",
    ...     target_schema=product_schema,
    ...     validators=[not_empty, min_items(5), has_required_fields(["name", "price"])]
    ... )
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union


def not_empty(data: Any) -> bool:
    """
    Validate that data is not empty.
    
    Works with lists, dicts, strings, and other collections.
    
    Args:
        data: The data to validate
        
    Returns:
        True if data is not empty, False otherwise
        
    Example:
        >>> from flybrowser.validators import not_empty
        >>> not_empty([1, 2, 3])
        True
        >>> not_empty([])
        False
    """
    if data is None:
        return False
    if isinstance(data, (list, dict, str, set, tuple)):
        return len(data) > 0
    return True


def min_items(count: int) -> Callable[[Any], bool]:
    """
    Create a validator that checks for minimum number of items.
    
    Args:
        count: Minimum number of items required
        
    Returns:
        A validator function
        
    Example:
        >>> from flybrowser.validators import min_items
        >>> validator = min_items(5)
        >>> validator([1, 2, 3, 4, 5])
        True
        >>> validator([1, 2])
        False
    """
    def validator(data: Any) -> bool:
        if data is None:
            return False
        if isinstance(data, (list, tuple, set)):
            return len(data) >= count
        if isinstance(data, dict):
            return len(data) >= count
        return True
    
    validator.__name__ = f"min_items({count})"
    return validator


def max_items(count: int) -> Callable[[Any], bool]:
    """
    Create a validator that checks for maximum number of items.
    
    Args:
        count: Maximum number of items allowed
        
    Returns:
        A validator function
        
    Example:
        >>> from flybrowser.validators import max_items
        >>> validator = max_items(100)
        >>> validator([1, 2, 3])
        True
    """
    def validator(data: Any) -> bool:
        if data is None:
            return True
        if isinstance(data, (list, tuple, set)):
            return len(data) <= count
        if isinstance(data, dict):
            return len(data) <= count
        return True
    
    validator.__name__ = f"max_items({count})"
    return validator


def has_required_fields(fields: List[str]) -> Callable[[Any], bool]:
    """
    Create a validator that checks if all items have required fields.
    
    For lists, checks each item. For dicts, checks the dict directly.
    
    Args:
        fields: List of required field names
        
    Returns:
        A validator function
        
    Example:
        >>> from flybrowser.validators import has_required_fields
        >>> validator = has_required_fields(["name", "price"])
        >>> validator([{"name": "A", "price": 10}])
        True
        >>> validator([{"name": "A"}])  # missing price
        False
    """
    def validator(data: Any) -> bool:
        if data is None:
            return False
        
        if isinstance(data, list):
            # Check each item in the list
            for item in data:
                if isinstance(item, dict):
                    if not all(field in item for field in fields):
                        return False
            return True
        
        if isinstance(data, dict):
            return all(field in data for field in fields)
        
        return False
    
    validator.__name__ = f"has_required_fields({fields})"
    return validator


def no_null_values(fields: Optional[List[str]] = None) -> Callable[[Any], bool]:
    """
    Create a validator that checks for no null/None values in specified fields.
    
    Args:
        fields: Optional list of field names to check. If None, checks all fields.
        
    Returns:
        A validator function
        
    Example:
        >>> from flybrowser.validators import no_null_values
        >>> validator = no_null_values(["name", "price"])
        >>> validator([{"name": "A", "price": 10}])
        True
        >>> validator([{"name": "A", "price": None}])
        False
    """
    def validator(data: Any) -> bool:
        if data is None:
            return False
        
        def check_item(item: Dict[str, Any]) -> bool:
            if not isinstance(item, dict):
                return True
            
            check_fields = fields or list(item.keys())
            for field in check_fields:
                if field in item and item[field] is None:
                    return False
            return True
        
        if isinstance(data, list):
            return all(check_item(item) for item in data)
        
        return check_item(data)
    
    validator.__name__ = f"no_null_values({fields})"
    return validator


def unique_values(field: str) -> Callable[[Any], bool]:
    """
    Create a validator that checks if a field has unique values across all items.
    
    Useful for ensuring no duplicates in extracted data.
    
    Args:
        field: Field name that should have unique values
        
    Returns:
        A validator function
        
    Example:
        >>> from flybrowser.validators import unique_values
        >>> validator = unique_values("id")
        >>> validator([{"id": 1}, {"id": 2}])
        True
        >>> validator([{"id": 1}, {"id": 1}])  # duplicate id
        False
    """
    def validator(data: Any) -> bool:
        if data is None or not isinstance(data, list):
            return True
        
        values = []
        for item in data:
            if isinstance(item, dict) and field in item:
                value = item[field]
                if value in values:
                    return False
                values.append(value)
        
        return True
    
    validator.__name__ = f"unique_values({field})"
    return validator


def matches_pattern(field: str, pattern: str) -> Callable[[Any], bool]:
    """
    Create a validator that checks if a field matches a regex pattern.
    
    Args:
        field: Field name to check
        pattern: Regex pattern to match
        
    Returns:
        A validator function
        
    Example:
        >>> from flybrowser.validators import matches_pattern
        >>> validator = matches_pattern("email", r".+@.+\\..+")
        >>> validator([{"email": "test@example.com"}])
        True
    """
    import re
    compiled = re.compile(pattern)
    
    def validator(data: Any) -> bool:
        if data is None:
            return True
        
        def check_item(item: Dict[str, Any]) -> bool:
            if not isinstance(item, dict):
                return True
            if field not in item:
                return True
            value = item[field]
            if value is None:
                return True
            return bool(compiled.match(str(value)))
        
        if isinstance(data, list):
            return all(check_item(item) for item in data)
        
        return check_item(data)
    
    validator.__name__ = f"matches_pattern({field}, {pattern})"
    return validator


def in_range(field: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Callable[[Any], bool]:
    """
    Create a validator that checks if numeric values are within a range.
    
    Args:
        field: Field name to check
        min_val: Minimum allowed value (inclusive), or None for no minimum
        max_val: Maximum allowed value (inclusive), or None for no maximum
        
    Returns:
        A validator function
        
    Example:
        >>> from flybrowser.validators import in_range
        >>> validator = in_range("price", min_val=0, max_val=1000)
        >>> validator([{"price": 50}])
        True
        >>> validator([{"price": -1}])
        False
    """
    def validator(data: Any) -> bool:
        if data is None:
            return True
        
        def check_item(item: Dict[str, Any]) -> bool:
            if not isinstance(item, dict):
                return True
            if field not in item:
                return True
            value = item[field]
            if value is None:
                return True
            if not isinstance(value, (int, float)):
                return True
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False
            return True
        
        if isinstance(data, list):
            return all(check_item(item) for item in data)
        
        return check_item(data)
    
    validator.__name__ = f"in_range({field}, min={min_val}, max={max_val})"
    return validator


def schema_compliant(schema: Dict[str, Any]) -> Callable[[Any], bool]:
    """
    Create a validator that checks data against a JSON Schema.
    
    Requires the `jsonschema` package to be installed for full validation.
    Falls back to basic type checking if not available.
    
    Args:
        schema: JSON Schema to validate against
        
    Returns:
        A validator function
        
    Example:
        >>> from flybrowser.validators import schema_compliant
        >>> product_schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"},
        ...         "price": {"type": "number"}
        ...     },
        ...     "required": ["name", "price"]
        ... }
        >>> validator = schema_compliant(product_schema)
        >>> validator({"name": "Widget", "price": 9.99})
        True
    """
    def validator(data: Any) -> bool:
        if data is None:
            return False
        
        try:
            import jsonschema
            jsonschema.validate(instance=data, schema=schema)
            return True
        except ImportError:
            # Fallback: basic type checking
            return _basic_schema_check(data, schema)
        except Exception:
            return False
    
    validator.__name__ = f"schema_compliant({schema.get('title', 'schema')})"
    return validator


def _basic_schema_check(data: Any, schema: Dict[str, Any]) -> bool:
    """Basic schema validation without jsonschema library."""
    schema_type = schema.get("type")
    
    if schema_type == "array":
        if not isinstance(data, list):
            return False
        if "items" in schema and data:
            return all(_basic_schema_check(item, schema["items"]) for item in data)
        return True
    
    elif schema_type == "object":
        if not isinstance(data, dict):
            return False
        required = schema.get("required", [])
        return all(prop in data for prop in required)
    
    elif schema_type == "string":
        return isinstance(data, str)
    
    elif schema_type == "number":
        return isinstance(data, (int, float))
    
    elif schema_type == "integer":
        return isinstance(data, int)
    
    elif schema_type == "boolean":
        return isinstance(data, bool)
    
    return True


def combine(*validators: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """
    Combine multiple validators into one that passes only if all pass.
    
    Args:
        *validators: Validator functions to combine
        
    Returns:
        A combined validator function
        
    Example:
        >>> from flybrowser.validators import not_empty, min_items, combine
        >>> validator = combine(not_empty, min_items(5))
        >>> validator([1, 2, 3, 4, 5])
        True
    """
    def combined_validator(data: Any) -> bool:
        return all(v(data) for v in validators)
    
    names = [getattr(v, '__name__', str(v)) for v in validators]
    combined_validator.__name__ = f"combine({', '.join(names)})"
    return combined_validator


def any_of(*validators: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """
    Combine validators so that at least one must pass.
    
    Args:
        *validators: Validator functions to combine
        
    Returns:
        A validator that passes if any validator passes
        
    Example:
        >>> from flybrowser.validators import min_items, any_of
        >>> validator = any_of(min_items(10), min_items(5))
        >>> validator([1, 2, 3, 4, 5])  # Passes min_items(5)
        True
    """
    def any_validator(data: Any) -> bool:
        return any(v(data) for v in validators)
    
    names = [getattr(v, '__name__', str(v)) for v in validators]
    any_validator.__name__ = f"any_of({', '.join(names)})"
    return any_validator


# Convenience aliases
required = not_empty
minimum = min_items
maximum = max_items
unique = unique_values
