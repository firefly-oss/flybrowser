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
Professional Context System for FlyBrowser Agent Actions.

This module provides a standardized, type-safe context system for passing
structured data to agent actions. It includes:

- Standardized context types and schemas
- Builder pattern for type-safe context construction
- Runtime validation with detailed error messages
- Tool integration for context-aware behavior

The context system enables powerful features like:
- Form filling with structured data
- File uploads with metadata
- Filtered extraction and search
- Conditional navigation
- Preference-based automation

Example:
    >>> from flybrowser.agents.context import ContextBuilder
    >>> 
    >>> # Form filling context
    >>> context = ContextBuilder()\\
    ...     .with_form_data({
    ...         "email": "user@example.com",
    ...         "password": "***",
    ...         "remember_me": True
    ...     })\\
    ...     .build()
    >>> 
    >>> await browser.act("Fill and submit login form", context=context)
    >>> 
    >>> # File upload context
    >>> context = ContextBuilder()\\
    ...     .with_file("resume", "/path/to/resume.pdf", "application/pdf")\\
    ...     .with_file("cover_letter", "/path/to/letter.docx")\\
    ...     .build()
    >>> 
    >>> await browser.act("Upload application documents", context=context)
    >>> 
    >>> # Extraction with filters
    >>> context = ContextBuilder()\\
    ...     .with_filters({"price_max": 100, "category": "electronics"})\\
    ...     .with_preferences({"sort_by": "price", "limit": 10})\\
    ...     .build()
    >>> 
    >>> result = await browser.extract("Get product listings", context=context)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ContextType(str, Enum):
    """Standardized context types supported by the system."""
    
    # Form and Input Context
    FORM_DATA = "form_data"
    """Form field values for automated filling. Schema: {field_name: value}"""
    
    # File Upload Context
    FILES = "files"
    """File upload specifications. Schema: [{field, path, mime_type?, name?}]"""
    
    # Extraction and Search Context
    FILTERS = "filters"
    """Data filtering criteria. Schema: {filter_name: filter_value}"""
    
    PREFERENCES = "preferences"
    """User preferences for behavior. Schema: {pref_name: pref_value}"""
    
    # Navigation Context
    CONDITIONS = "conditions"
    """Conditions that must be met. Schema: {condition_name: expected_value}"""
    
    # General Context
    CONSTRAINTS = "constraints"
    """General constraints or limits. Schema: {constraint_name: value}"""
    
    METADATA = "metadata"
    """Additional metadata for tools. Schema: {key: value}"""


@dataclass
class FileUploadSpec:
    """Specification for a file upload.
    
    Attributes:
        field: Form field name or selector for the file input
        path: Absolute or relative path to the file
        mime_type: MIME type of the file (auto-detected if not provided)
        name: Custom filename to use (defaults to actual filename)
        verify_exists: Whether to verify file exists before upload
    """
    field: str
    path: str
    mime_type: Optional[str] = None
    name: Optional[str] = None
    verify_exists: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "field": self.field,
            "path": self.path,
        }
        if self.mime_type:
            result["mime_type"] = self.mime_type
        if self.name:
            result["name"] = self.name
        return result
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate file upload specification.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.field:
            return False, "File field name is required"
        
        if not self.path:
            return False, "File path is required"
        
        if self.verify_exists:
            file_path = Path(self.path).expanduser()
            if not file_path.exists():
                return False, f"File not found: {self.path}"
            if not file_path.is_file():
                return False, f"Path is not a file: {self.path}"
        
        return True, None


@dataclass
class ActionContext:
    """
    Structured context for agent actions.
    
    Provides type-safe access to different context types with validation.
    Use ContextBuilder to construct instances.
    
    Attributes:
        form_data: Form field values for automated filling
        files: File upload specifications
        filters: Data filtering criteria
        preferences: User preferences
        conditions: Navigation/action conditions
        constraints: General constraints
        metadata: Additional tool-specific metadata
    """
    form_data: Dict[str, Any] = field(default_factory=dict)
    files: List[FileUploadSpec] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary format for serialization."""
        result = {}
        
        if self.form_data:
            result["form_data"] = self.form_data
        if self.files:
            result["files"] = [f.to_dict() for f in self.files]
        if self.filters:
            result["filters"] = self.filters
        if self.preferences:
            result["preferences"] = self.preferences
        if self.conditions:
            result["conditions"] = self.conditions
        if self.constraints:
            result["constraints"] = self.constraints
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result
    
    def is_empty(self) -> bool:
        """Check if context is empty."""
        return not any([
            self.form_data,
            self.files,
            self.filters,
            self.preferences,
            self.conditions,
            self.constraints,
            self.metadata,
        ])
    
    def has_type(self, context_type: ContextType) -> bool:
        """Check if context contains a specific type."""
        if context_type == ContextType.FORM_DATA:
            return bool(self.form_data)
        elif context_type == ContextType.FILES:
            return bool(self.files)
        elif context_type == ContextType.FILTERS:
            return bool(self.filters)
        elif context_type == ContextType.PREFERENCES:
            return bool(self.preferences)
        elif context_type == ContextType.CONDITIONS:
            return bool(self.conditions)
        elif context_type == ContextType.CONSTRAINTS:
            return bool(self.constraints)
        elif context_type == ContextType.METADATA:
            return bool(self.metadata)
        return False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionContext":
        """Create ActionContext from dictionary.
        
        Args:
            data: Dictionary representation of context
            
        Returns:
            ActionContext instance
        """
        files_data = data.get("files", [])
        files = [
            FileUploadSpec(
                field=f.get("field", ""),
                path=f.get("path", ""),
                mime_type=f.get("mime_type"),
                name=f.get("name"),
            )
            for f in files_data
        ]
        
        return cls(
            form_data=data.get("form_data", {}),
            files=files,
            filters=data.get("filters", {}),
            preferences=data.get("preferences", {}),
            conditions=data.get("conditions", {}),
            constraints=data.get("constraints", {}),
            metadata=data.get("metadata", {}),
        )


class ContextValidator:
    """
    Validator for ActionContext instances.
    
    Performs validation against defined schemas and business rules.
    """
    
    @staticmethod
    def validate(context: ActionContext) -> tuple[bool, List[str]]:
        """Validate an ActionContext instance.
        
        Args:
            context: Context to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate form_data
        if context.form_data:
            if not isinstance(context.form_data, dict):
                errors.append("form_data must be a dictionary")
            else:
                for key, value in context.form_data.items():
                    if not isinstance(key, str):
                        errors.append(f"form_data key must be string, got {type(key)}")
        
        # Validate files
        if context.files:
            if not isinstance(context.files, list):
                errors.append("files must be a list")
            else:
                for i, file_spec in enumerate(context.files):
                    if not isinstance(file_spec, FileUploadSpec):
                        errors.append(f"files[{i}] must be FileUploadSpec instance")
                        continue
                    
                    is_valid, error = file_spec.validate()
                    if not is_valid:
                        errors.append(f"files[{i}]: {error}")
        
        # Validate filters
        if context.filters:
            if not isinstance(context.filters, dict):
                errors.append("filters must be a dictionary")
        
        # Validate preferences
        if context.preferences:
            if not isinstance(context.preferences, dict):
                errors.append("preferences must be a dictionary")
        
        # Validate conditions
        if context.conditions:
            if not isinstance(context.conditions, dict):
                errors.append("conditions must be a dictionary")
        
        # Validate constraints
        if context.constraints:
            if not isinstance(context.constraints, dict):
                errors.append("constraints must be a dictionary")
        
        # Validate metadata
        if context.metadata:
            if not isinstance(context.metadata, dict):
                errors.append("metadata must be a dictionary")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_for_tool(
        context: ActionContext,
        expected_types: List[ContextType]
    ) -> tuple[bool, List[str]]:
        """Validate context for a specific tool's requirements.
        
        Args:
            context: Context to validate
            expected_types: Context types the tool expects
            
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        # Check if context has expected types
        provided_types = []
        if context.form_data:
            provided_types.append(ContextType.FORM_DATA)
        if context.files:
            provided_types.append(ContextType.FILES)
        if context.filters:
            provided_types.append(ContextType.FILTERS)
        if context.preferences:
            provided_types.append(ContextType.PREFERENCES)
        if context.conditions:
            provided_types.append(ContextType.CONDITIONS)
        if context.constraints:
            provided_types.append(ContextType.CONSTRAINTS)
        if context.metadata:
            provided_types.append(ContextType.METADATA)
        
        # Check for unexpected context types
        unexpected = set(provided_types) - set(expected_types)
        if unexpected:
            warnings.append(
                f"Context contains unexpected types that may be ignored: "
                f"{[t.value for t in unexpected]}"
            )
        
        return len(warnings) == 0, warnings


class ContextBuilder:
    """
    Builder for constructing ActionContext instances with validation.
    
    Provides a fluent interface for type-safe context construction.
    
    Example:
        >>> context = ContextBuilder()\\
        ...     .with_form_data({"username": "user", "password": "***"})\\
        ...     .with_file("resume", "/path/to/resume.pdf")\\
        ...     .with_filters({"price_max": 100})\\
        ...     .build()
    """
    
    def __init__(self):
        """Initialize empty context builder."""
        self._form_data: Dict[str, Any] = {}
        self._files: List[FileUploadSpec] = []
        self._filters: Dict[str, Any] = {}
        self._preferences: Dict[str, Any] = {}
        self._conditions: Dict[str, Any] = {}
        self._constraints: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}
    
    def with_form_data(self, form_data: Dict[str, Any]) -> "ContextBuilder":
        """Add form field data for automated filling.
        
        Args:
            form_data: Dictionary of field_name -> value
            
        Returns:
            Self for chaining
        """
        self._form_data.update(form_data)
        return self
    
    def with_form_field(self, field: str, value: Any) -> "ContextBuilder":
        """Add a single form field.
        
        Args:
            field: Field name or selector
            value: Field value
            
        Returns:
            Self for chaining
        """
        self._form_data[field] = value
        return self
    
    def with_file(
        self,
        field: str,
        path: str,
        mime_type: Optional[str] = None,
        name: Optional[str] = None,
        verify_exists: bool = True,
    ) -> "ContextBuilder":
        """Add a file upload specification.
        
        Args:
            field: Form field name or selector
            path: Path to the file
            mime_type: Optional MIME type
            name: Optional custom filename
            verify_exists: Whether to verify file exists
            
        Returns:
            Self for chaining
        """
        self._files.append(FileUploadSpec(
            field=field,
            path=path,
            mime_type=mime_type,
            name=name,
            verify_exists=verify_exists,
        ))
        return self
    
    def with_files(self, files: List[Dict[str, Any]]) -> "ContextBuilder":
        """Add multiple file upload specifications.
        
        Args:
            files: List of file spec dictionaries
            
        Returns:
            Self for chaining
        """
        for file_dict in files:
            self._files.append(FileUploadSpec(
                field=file_dict.get("field", ""),
                path=file_dict.get("path", ""),
                mime_type=file_dict.get("mime_type"),
                name=file_dict.get("name"),
                verify_exists=file_dict.get("verify_exists", True),
            ))
        return self
    
    def with_filters(self, filters: Dict[str, Any]) -> "ContextBuilder":
        """Add data filtering criteria.
        
        Args:
            filters: Dictionary of filter_name -> filter_value
            
        Returns:
            Self for chaining
        """
        self._filters.update(filters)
        return self
    
    def with_filter(self, name: str, value: Any) -> "ContextBuilder":
        """Add a single filter.
        
        Args:
            name: Filter name
            value: Filter value
            
        Returns:
            Self for chaining
        """
        self._filters[name] = value
        return self
    
    def with_preferences(self, preferences: Dict[str, Any]) -> "ContextBuilder":
        """Add user preferences.
        
        Args:
            preferences: Dictionary of pref_name -> pref_value
            
        Returns:
            Self for chaining
        """
        self._preferences.update(preferences)
        return self
    
    def with_preference(self, name: str, value: Any) -> "ContextBuilder":
        """Add a single preference.
        
        Args:
            name: Preference name
            value: Preference value
            
        Returns:
            Self for chaining
        """
        self._preferences[name] = value
        return self
    
    def with_conditions(self, conditions: Dict[str, Any]) -> "ContextBuilder":
        """Add navigation/action conditions.
        
        Args:
            conditions: Dictionary of condition_name -> expected_value
            
        Returns:
            Self for chaining
        """
        self._conditions.update(conditions)
        return self
    
    def with_condition(self, name: str, value: Any) -> "ContextBuilder":
        """Add a single condition.
        
        Args:
            name: Condition name
            value: Expected value
            
        Returns:
            Self for chaining
        """
        self._conditions[name] = value
        return self
    
    def with_constraints(self, constraints: Dict[str, Any]) -> "ContextBuilder":
        """Add general constraints.
        
        Args:
            constraints: Dictionary of constraint_name -> value
            
        Returns:
            Self for chaining
        """
        self._constraints.update(constraints)
        return self
    
    def with_constraint(self, name: str, value: Any) -> "ContextBuilder":
        """Add a single constraint.
        
        Args:
            name: Constraint name
            value: Constraint value
            
        Returns:
            Self for chaining
        """
        self._constraints[name] = value
        return self
    
    def with_metadata(self, metadata: Dict[str, Any]) -> "ContextBuilder":
        """Add tool-specific metadata.
        
        Args:
            metadata: Dictionary of key -> value
            
        Returns:
            Self for chaining
        """
        self._metadata.update(metadata)
        return self
    
    def build(self, validate: bool = True) -> ActionContext:
        """Build and optionally validate the ActionContext.
        
        Args:
            validate: Whether to validate the context
            
        Returns:
            ActionContext instance
            
        Raises:
            ValueError: If validation fails
        """
        context = ActionContext(
            form_data=self._form_data,
            files=self._files,
            filters=self._filters,
            preferences=self._preferences,
            conditions=self._conditions,
            constraints=self._constraints,
            metadata=self._metadata,
        )
        
        if validate:
            is_valid, errors = ContextValidator.validate(context)
            if not is_valid:
                raise ValueError(f"Context validation failed: {'; '.join(errors)}")
        
        return context


# Convenience functions for common patterns

def create_form_context(form_data: Dict[str, Any]) -> ActionContext:
    """Create a context for form filling.
    
    Args:
        form_data: Form field values
        
    Returns:
        ActionContext with form_data
    """
    return ContextBuilder().with_form_data(form_data).build()


def create_upload_context(
    files: List[Union[Dict[str, Any], FileUploadSpec]]
) -> ActionContext:
    """Create a context for file uploads.
    
    Args:
        files: List of file specifications
        
    Returns:
        ActionContext with files
    """
    builder = ContextBuilder()
    
    for file in files:
        if isinstance(file, FileUploadSpec):
            builder._files.append(file)
        else:
            builder.with_file(
                field=file.get("field", ""),
                path=file.get("path", ""),
                mime_type=file.get("mime_type"),
                name=file.get("name"),
            )
    
    return builder.build()


def create_filter_context(
    filters: Dict[str, Any],
    preferences: Optional[Dict[str, Any]] = None
) -> ActionContext:
    """Create a context for filtered extraction/search.
    
    Args:
        filters: Filter criteria
        preferences: Optional user preferences
        
    Returns:
        ActionContext with filters and preferences
    """
    builder = ContextBuilder().with_filters(filters)
    if preferences:
        builder.with_preferences(preferences)
    return builder.build()
