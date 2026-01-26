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
Base tool class for the ReAct agentic framework.

This module provides the abstract BaseTool class that all tools must inherit from.
Tools are the primary way the agent interacts with the browser and external systems.

Key Features:
- Type-safe parameter validation via JSON schema
- Async execution with timeout support
- Standardized ToolResult return format
- Tool metadata for registry and documentation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from flybrowser.agents.types import (
    SafetyLevel,
    ToolCategory,
    ToolResult,
)
from flybrowser.llm.base import ModelCapability

if TYPE_CHECKING:
    from flybrowser.core.page import PageController


@dataclass
class ToolParameter:
    """Definition of a tool parameter for schema validation."""
    
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str = ""
    required: bool = False
    default: Any = None
    enum: Optional[List[Any]] = None  # Allowed values
    items_type: Optional[str] = None  # For array types
    properties: Optional[Dict[str, Any]] = None  # For object types


@dataclass
class ToolMetadata:
    """Metadata describing a tool's capabilities and requirements."""
    
    name: str
    description: str
    category: ToolCategory = ToolCategory.UTILITY
    safety_level: SafetyLevel = SafetyLevel.SAFE
    parameters: List[ToolParameter] = field(default_factory=list)
    returns_description: str = ""
    examples: List[str] = field(default_factory=list)
    requires_page: bool = False  # Whether tool needs PageController
    timeout_seconds: float = 30.0
    is_terminal: bool = False  # Whether this tool ends the task
    # Model capability requirements
    required_capabilities: List[ModelCapability] = field(default_factory=list)  # Must have these
    optimal_capabilities: List[ModelCapability] = field(default_factory=list)  # Works better with these
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Generate JSON schema for tool parameters."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop: Dict[str, Any] = {"type": param.type}
            if param.description:
                prop["description"] = param.description
            if param.enum:
                prop["enum"] = param.enum
            if param.items_type:
                prop["items"] = {"type": param.items_type}
            if param.properties:
                prop["properties"] = param.properties
            if param.default is not None:
                prop["default"] = param.default
                
            properties[param.name] = prop
            if param.required:
                required.append(param.name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "safety_level": self.safety_level.value,
            "parameters": self.to_json_schema(),
            "returns": self.returns_description,
            "examples": self.examples,
            "requires_page": self.requires_page,
            "timeout_seconds": self.timeout_seconds,
            "is_terminal": self.is_terminal,
        }


class BaseTool(ABC):
    """
    Abstract base class for all tools in the ReAct framework.
    
    Tools are the primary mechanism for the agent to take actions.
    Each tool must define its metadata and implement the execute method.
    
    Tools can access user-provided context through the memory system,
    enabling context-aware behavior like form filling and file uploads.
    
    Example:
        >>> class ClickTool(BaseTool):
        ...     metadata = ToolMetadata(
        ...         name="click",
        ...         description="Click on an element",
        ...         category=ToolCategory.INTERACTION,
        ...         parameters=[
        ...             ToolParameter("selector", "string", required=True),
        ...         ],
        ...     )
        ...     
        ...     async def execute(self, selector: str, **kwargs) -> ToolResult:
        ...         # Implementation
        ...         return ToolResult.success_result({"clicked": selector})
    """
    
    metadata: ToolMetadata  # Must be defined by subclass
    
    def __init__(self, page_controller: Optional["PageController"] = None) -> None:
        """Initialize the tool with optional page controller."""
        self.page = page_controller
        self._memory: Optional[Any] = None  # Set by tool registry
    
    def set_memory(self, memory: Any) -> None:
        """Set memory reference for context access."""
        self._memory = memory
    
    def get_user_context(self, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get user-provided context from memory.
        
        This allows tools to access context passed via SDK methods like:
        - browser.act("fill form", context={"form_data": {...}})
        - browser.act("upload file", context={"files": [...]})
        
        Args:
            key: Context key to retrieve. If None, returns all context.
            default: Default value if key not found.
            
        Returns:
            Context value or default.
            
        Example:
            >>> # In a tool's execute method:
            >>> user_context = self.get_user_context()
            >>> form_data = user_context.get("form_data", {})
            >>> files = user_context.get("files", [])
        """
        if not self._memory or not hasattr(self._memory, 'working'):
            return {} if key is None else default
        
        user_context = self._memory.working.get_scratch("user_context", {})
        
        if key is None:
            return user_context
        return user_context.get(key, default)
    
    @property
    def name(self) -> str:
        """Get the tool name."""
        return self.metadata.name
    
    @property
    def description(self) -> str:
        """Get the tool description."""
        return self.metadata.description

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters."""
        return self.metadata.to_json_schema()

    async def execute_safe(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with automatic parameter validation.
        
        This method validates parameters before calling execute() and handles
        validation errors gracefully by returning an error ToolResult.
        
        This is the RECOMMENDED way to call tools from the agent framework,
        as it ensures parameter validation is always performed.
        
        Args:
            **kwargs: Tool-specific parameters as defined in metadata
            
        Returns:
            ToolResult indicating success or failure
        """
        # Import ErrorCode from types
        from flybrowser.agents.types import ErrorCode
        
        # Validate parameters first
        is_valid, error = self.validate_parameters(kwargs)
        if not is_valid:
            return ToolResult.error_result(
                error=error,
                error_code=ErrorCode.INVALID_PARAMS,
                metadata={"tool": self.name, "params": kwargs}
            )
        
        # Execute the tool
        try:
            return await self.execute(**kwargs)
        except Exception as e:
            # Catch any unexpected exceptions and return as ToolResult
            return ToolResult.error_result(
                error=f"Tool execution failed: {str(e)}",
                error_code=ErrorCode.EXECUTION_ERROR,
                metadata={"tool": self.name, "exception_type": type(e).__name__}
            )
    
    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with the given parameters.

        This is the main entry point for tool execution. Subclasses must
        implement this method to define the tool's behavior.
        
        NOTE: Consider using execute_safe() instead, which automatically
        validates parameters before execution.

        Args:
            **kwargs: Tool-specific parameters as defined in metadata

        Returns:
            ToolResult indicating success or failure with data/error message

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement execute()")

    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate parameters against the tool's JSON schema.

        Args:
            params: Dictionary of parameter name to value

        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
        """
        schema = self.metadata.to_json_schema()
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check required parameters
        for param_name in required:
            if param_name not in params:
                return False, f"Missing required parameter: {param_name}"

        # Validate parameter types
        for param_name, value in params.items():
            if param_name not in properties:
                continue  # Allow extra parameters

            prop_schema = properties[param_name]
            expected_type = prop_schema.get("type")

            # Type validation
            if not self._validate_type(value, expected_type):
                return False, f"Parameter '{param_name}' has invalid type. Expected {expected_type}"

            # Enum validation
            if "enum" in prop_schema and value not in prop_schema["enum"]:
                return False, f"Parameter '{param_name}' must be one of: {prop_schema['enum']}"

        return True, None

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate a value against an expected JSON schema type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        if expected_type not in type_map:
            return True  # Unknown type, allow

        return isinstance(value, type_map[expected_type])

    def __repr__(self) -> str:
        """Return string representation of the tool."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"category={self.metadata.category.value!r}, "
            f"safety={self.metadata.safety_level.value!r})"
        )

