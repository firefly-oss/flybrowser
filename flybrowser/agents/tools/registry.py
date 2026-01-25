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
Tool registry for dynamic tool management in the ReAct framework.

This module provides the ToolRegistry class for registering, discovering,
and managing tools. It supports:
- Dynamic tool registration
- Tool lookup by name
- Category-based filtering
- JSON schema generation for LLM prompts
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

from flybrowser.agents.types import SafetyLevel, ToolCategory
from flybrowser.llm.base import ModelCapability, ModelInfo
from flybrowser.utils.logger import logger

if TYPE_CHECKING:
    from flybrowser.agents.tools.base import BaseTool, ToolMetadata
    from flybrowser.core.page import PageController


class ToolRegistry:
    """
    Registry for managing available tools.
    
    Thread-safe registry that maintains a catalog of all registered tools
    and provides methods for tool discovery and instantiation.
    
    Attributes:
        _tools: Mapping of tool name to tool class
        _category_index: Index of category to tool names
        _lock: Thread lock for concurrent access
        
    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(ClickTool)
        >>> registry.register(NavigateTool)
        >>> 
        >>> # Get all tools
        >>> all_tools = registry.list_tools()
        >>> 
        >>> # Get by category
        >>> nav_tools = registry.get_by_category(ToolCategory.NAVIGATION)
        >>> 
        >>> # Instantiate tool
        >>> click_tool = registry.get_tool("click", page_controller=page)
    """
    
    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._tools: Dict[str, Type["BaseTool"]] = {}
        self._instances: Dict[str, "BaseTool"] = {}  # For pre-instantiated tools
        self._category_index: Dict[ToolCategory, List[str]] = {}
        self._lock = threading.RLock()
    
    def register(self, tool_class: Type["BaseTool"]) -> None:
        """
        Register a tool class.
        
        Args:
            tool_class: The tool class to register (must have metadata attribute)
            
        Raises:
            ValueError: If tool has no metadata or name conflicts
        """
        with self._lock:
            metadata_attr = getattr(tool_class, "metadata", None)
            if metadata_attr is None:
                raise ValueError(
                    f"Tool {tool_class.__name__} has no metadata attribute. "
                    "Define a 'metadata' class attribute or property with ToolMetadata."
                )
            
            # Handle both class attribute and property
            # For properties, we need to instantiate the tool to get metadata
            if isinstance(metadata_attr, property):
                # Create a temporary instance to get metadata
                try:
                    temp_instance = tool_class(page_controller=None)
                    metadata = temp_instance.metadata
                except Exception as e:
                    raise ValueError(
                        f"Tool {tool_class.__name__} metadata property requires initialization: {e}"
                    )
            else:
                metadata = metadata_attr
            
            name = metadata.name
            if name in self._tools:
                raise ValueError(
                    f"Tool '{name}' is already registered. "
                    "Use a unique name or unregister the existing tool first."
                )
            
            # Register tool
            self._tools[name] = tool_class
            
            # Index by category
            category = metadata.category
            if category not in self._category_index:
                self._category_index[category] = []
            if name not in self._category_index[category]:
                self._category_index[category].append(name)
    
    def register_instance(self, tool_instance: "BaseTool") -> None:
        """
        Register a pre-instantiated tool instance.
        
        This is useful when tools need specific initialization parameters
        that cannot be provided through the standard instantiation pattern.
        
        Args:
            tool_instance: Pre-instantiated tool object
            
        Raises:
            ValueError: If tool name conflicts
        """
        with self._lock:
            metadata = tool_instance.metadata
            name = metadata.name
            
            if name in self._tools or name in self._instances:
                raise ValueError(
                    f"Tool '{name}' is already registered. "
                    "Use a unique name or unregister the existing tool first."
                )
            
            # Register instance
            self._instances[name] = tool_instance
            
            # Index by category
            category = metadata.category
            if category not in self._category_index:
                self._category_index[category] = []
            if name not in self._category_index[category]:
                self._category_index[category].append(name)
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name.
        
        Args:
            name: The tool name to unregister
            
        Returns:
            True if tool was unregistered, False if not found
        """
        with self._lock:
            # Check instances first
            if name in self._instances:
                instance = self._instances[name]
                category = instance.metadata.category
                del self._instances[name]
                
                # Remove from category index
                if category in self._category_index:
                    self._category_index[category] = [
                        n for n in self._category_index[category] if n != name
                    ]
                return True
            
            # Check classes
            if name not in self._tools:
                return False
            
            tool_class = self._tools[name]
            # Handle property-based metadata
            metadata_attr = tool_class.metadata
            if isinstance(metadata_attr, property):
                temp_instance = tool_class(page_controller=None)
                category = temp_instance.metadata.category
            else:
                category = metadata_attr.category
            
            # Remove from tools
            del self._tools[name]
            
            # Remove from category index
            if category in self._category_index:
                self._category_index[category] = [
                    n for n in self._category_index[category] if n != name
                ]
            
            return True
    
    def get_tool(
        self,
        name: str,
        page_controller: Optional["PageController"] = None,
        element_detector: Optional[Any] = None,
    ) -> Optional["BaseTool"]:
        """
        Get an instantiated tool by name.
        
        Args:
            name: The tool name
            page_controller: Optional page controller to inject
            element_detector: Optional element detector for AI-based element finding
            
        Returns:
            Instantiated tool or None if not found
        """
        with self._lock:
            # Check for pre-instantiated tool first
            if name in self._instances:
                instance = self._instances[name]
                # Update element_detector if provided and tool supports it
                if element_detector and hasattr(instance, '_element_detector'):
                    instance._element_detector = element_detector
                return instance
            
            # Otherwise instantiate from class
            tool_class = self._tools.get(name)
            if tool_class is None:
                return None
            
            # Try to instantiate with element_detector if the tool supports it
            import inspect
            sig = inspect.signature(tool_class.__init__)
            params = sig.parameters
            
            if 'element_detector' in params and element_detector:
                return tool_class(page_controller=page_controller, element_detector=element_detector)
            else:
                return tool_class(page_controller=page_controller)
    
    def get_tool_class(self, name: str) -> Optional[Type["BaseTool"]]:
        """Get a tool class by name without instantiating."""
        with self._lock:
            return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered (class or instance)."""
        with self._lock:
            return name in self._tools or name in self._instances

    def list_tools(self) -> List[str]:
        """Get a list of all registered tool names (classes and instances)."""
        with self._lock:
            # Combine tool classes and pre-instantiated instances
            return list(set(self._tools.keys()) | set(self._instances.keys()))

    def get_by_category(self, category: ToolCategory) -> List[str]:
        """Get all tool names in a category."""
        with self._lock:
            return self._category_index.get(category, []).copy()

    def get_by_safety_level(self, safety_level: SafetyLevel) -> List[str]:
        """Get all tool names at or below a safety level."""
        with self._lock:
            result = []
            for name, tool_class in self._tools.items():
                # Handle property-based metadata
                metadata_attr = tool_class.metadata
                if isinstance(metadata_attr, property):
                    temp_instance = tool_class(page_controller=None)
                    tool_safety = temp_instance.metadata.safety_level
                else:
                    tool_safety = metadata_attr.safety_level
                    
                if tool_safety.value <= safety_level.value:
                    result.append(name)
            return result

    def get_metadata(self, name: str) -> Optional["ToolMetadata"]:
        """Get tool metadata by name."""
        with self._lock:
            tool_class = self._tools.get(name)
            if tool_class:
                metadata_attr = tool_class.metadata
                # Handle property-based metadata
                if isinstance(metadata_attr, property):
                    temp_instance = tool_class(page_controller=None)
                    return temp_instance.metadata
                return metadata_attr
            return None

    def get_filtered_registry(
        self,
        model_capabilities: List[ModelCapability],
        warn_suboptimal: bool = True,
    ) -> "ToolRegistry":
        """
        Create a new registry with tools filtered by model capabilities.
        
        Args:
            model_capabilities: List of capabilities the model supports
            warn_suboptimal: Whether to warn about suboptimal tool availability
            
        Returns:
            New ToolRegistry with only compatible tools
        """
        filtered_registry = ToolRegistry()
        warnings = []
        
        with self._lock:
            # Process tool classes
            for name, tool_class in self._tools.items():
                # Get metadata
                metadata_attr = tool_class.metadata
                if isinstance(metadata_attr, property):
                    temp_instance = tool_class(page_controller=None)
                    metadata = temp_instance.metadata
                else:
                    metadata = metadata_attr
                
                # Check required capabilities
                if metadata.required_capabilities:
                    has_required = all(
                        cap in model_capabilities
                        for cap in metadata.required_capabilities
                    )
                    if not has_required:
                        missing = [c.value for c in metadata.required_capabilities if c not in model_capabilities]
                        warnings.append(f"Excluding '{name}': missing required capabilities {missing}")
                        continue
                
                # Register tool in filtered registry
                filtered_registry.register(tool_class)
                
                # Check optimal capabilities
                if warn_suboptimal and metadata.optimal_capabilities:
                    has_optimal = all(
                        cap in model_capabilities
                        for cap in metadata.optimal_capabilities
                    )
                    if not has_optimal:
                        missing = [c.value for c in metadata.optimal_capabilities if c not in model_capabilities]
                        warnings.append(f"Tool '{name}' works better with: {missing}")
            
            # Process pre-instantiated tools (like SearchAgentTool)
            for name, tool_instance in self._instances.items():
                metadata = tool_instance.metadata
                
                # Check required capabilities
                if metadata.required_capabilities:
                    has_required = all(
                        cap in model_capabilities
                        for cap in metadata.required_capabilities
                    )
                    if not has_required:
                        missing = [c.value for c in metadata.required_capabilities if c not in model_capabilities]
                        warnings.append(f"Excluding instance '{name}': missing required capabilities {missing}")
                        continue
                
                # Register instance in filtered registry
                filtered_registry.register_instance(tool_instance)
                
                # Check optimal capabilities
                if warn_suboptimal and metadata.optimal_capabilities:
                    has_optimal = all(
                        cap in model_capabilities
                        for cap in metadata.optimal_capabilities
                    )
                    if not has_optimal:
                        missing = [c.value for c in metadata.optimal_capabilities if c not in model_capabilities]
                        warnings.append(f"Tool instance '{name}' works better with: {missing}")
        
        # Log warnings
        if warnings:
            logger.info(f"Tool capability filtering: {len(warnings)} note(s)")
            for warning in warnings:
                logger.debug(f"  {warning}")
        
        return filtered_registry
    
    def get_tools_for_model(self, model_info: ModelInfo) -> Dict[str, "BaseTool"]:
        """
        Get instantiated tools compatible with model capabilities.
        
        Args:
            model_info: Model information including capabilities
            
        Returns:
            Dictionary of tool_name -> instantiated tool
        """
        filtered_registry = self.get_filtered_registry(model_info.capabilities)
        tools = {}
        
        for name in filtered_registry.list_tools():
            tool = filtered_registry.get_tool(name)
            if tool:
                tools[name] = tool
        
        return tools
    
    def check_tool_compatibility(
        self,
        tool_name: str,
        model_capabilities: List[ModelCapability],
    ) -> tuple[bool, List[str]]:
        """
        Check if a specific tool is compatible with model capabilities.
        
        Args:
            tool_name: Name of the tool to check
            model_capabilities: List of model capabilities
            
        Returns:
            Tuple of (is_compatible, list_of_warnings)
        """
        metadata = self.get_metadata(tool_name)
        if not metadata:
            return False, [f"Tool '{tool_name}' not found"]
        
        warnings = []
        
        # Check required capabilities
        if metadata.required_capabilities:
            missing_required = [
                cap.value for cap in metadata.required_capabilities
                if cap not in model_capabilities
            ]
            if missing_required:
                return False, [f"Missing required capabilities: {missing_required}"]
        
        # Check optimal capabilities
        if metadata.optimal_capabilities:
            missing_optimal = [
                cap.value for cap in metadata.optimal_capabilities
                if cap not in model_capabilities
            ]
            if missing_optimal:
                warnings.append(f"Tool works better with: {missing_optimal}")
        
        return True, warnings
    
    def generate_tools_prompt(
        self,
        categories: Optional[List[ToolCategory]] = None,
        max_safety_level: Optional[SafetyLevel] = None,
    ) -> str:
        """
        Generate a prompt describing available tools for the LLM.

        Args:
            categories: Optional list of categories to include
            max_safety_level: Optional maximum safety level to include

        Returns:
            Formatted string describing available tools
        """
        with self._lock:
            lines = ["Available tools:\n"]

            for name, tool_class in sorted(self._tools.items()):
                # Handle property-based metadata
                metadata_attr = tool_class.metadata
                if isinstance(metadata_attr, property):
                    temp_instance = tool_class(page_controller=None)
                    metadata = temp_instance.metadata
                else:
                    metadata = metadata_attr

                # Filter by category
                if categories and metadata.category not in categories:
                    continue

                # Filter by safety level
                if max_safety_level and metadata.safety_level.value > max_safety_level.value:
                    continue

                # Format tool description
                lines.append(f"## {name}")
                lines.append(f"Description: {metadata.description}")
                lines.append(f"Category: {metadata.category.value}")

                # Parameters
                if metadata.parameters:
                    lines.append("Parameters:")
                    for param in metadata.parameters:
                        req = "(required)" if param.required else "(optional)"
                        lines.append(f"  - {param.name}: {param.type} {req}")
                        if param.description:
                            lines.append(f"    {param.description}")

                # Examples
                if metadata.examples:
                    lines.append("Examples:")
                    for ex in metadata.examples:
                        lines.append(f"  - {ex}")

                lines.append("")

            return "\n".join(lines)

    def get_all_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get JSON schemas for all registered tools."""
        with self._lock:
            schemas = {}
            for name, tool_class in self._tools.items():
                # Handle property-based metadata
                metadata_attr = tool_class.metadata
                if isinstance(metadata_attr, property):
                    temp_instance = tool_class(page_controller=None)
                    metadata = temp_instance.metadata
                else:
                    metadata = metadata_attr
                schemas[name] = metadata.to_dict()
            return schemas

    def __len__(self) -> int:
        """Return the number of registered tools (classes and instances)."""
        with self._lock:
            return len(self._tools) + len(self._instances)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return self.has_tool(name)

