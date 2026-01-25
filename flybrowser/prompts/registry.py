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
Prompt registry for managing templates.

This module provides the PromptRegistry class which serves as a central repository
for all prompt templates. It handles:
- Template storage and retrieval
- Loading templates from YAML/JSON files
- Version management
- Variant management for A/B testing
- Template validation

The registry supports loading templates from a directory structure:
```
templates/
  ├── data_extraction.yaml
  ├── element_detection.yaml
  ├── navigation_planning.yaml
  └── variants/
      ├── data_extraction_v2.yaml
      └── element_detection_vision.yaml
```

Example:
    >>> from pathlib import Path
    >>> registry = PromptRegistry(templates_dir=Path("./prompts/templates"))
    >>>
    >>> # Get a template
    >>> template = registry.get("data_extraction", version="1.0.0")
    >>>
    >>> # Register a new template
    >>> new_template = PromptTemplate(
    ...     name="custom_extraction",
    ...     user_template="Extract {{ data_type }} from {{ content }}"
    ... )
    >>> registry.register(new_template)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from flybrowser.exceptions import ConfigurationError
from flybrowser.prompts.template import PromptTemplate
from flybrowser.utils.logger import logger


# Singleton instance for the default registry
_default_registry: Optional["PromptRegistry"] = None


def get_default_registry() -> "PromptRegistry":
    """
    Get or create the default prompt registry singleton.
    
    This ensures templates are loaded only once across the entire application.
    
    Returns:
        The shared PromptRegistry instance
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = PromptRegistry(_use_singleton=False)
    return _default_registry


class PromptRegistry:
    """
    Registry for managing and storing prompt templates.

    This class provides a centralized repository for all prompt templates.
    It supports loading templates from files, version management, and
    variant management for A/B testing.
    
    By default, PromptRegistry() returns a shared singleton instance to avoid
    loading templates multiple times. Use PromptRegistry(templates_dir=custom_path)
    to create a separate instance with custom templates.

    Attributes:
        templates: Dictionary mapping template keys to PromptTemplate instances
        templates_dir: Directory containing template YAML/JSON files

    Example:
        >>> registry = PromptRegistry()  # Returns shared singleton
        >>>
        >>> # Get a template
        >>> template = registry.get("data_extraction")
        >>>
        >>> # List all templates
        >>> all_templates = registry.list_templates()
        >>> for name in all_templates:
        ...     print(name)
        >>>
        >>> # Register custom template
        >>> custom = PromptTemplate(
        ...     name="my_template",
        ...     user_template="Do {{ task }}"
        ... )
        >>> registry.register(custom)
    """

    def __new__(cls, templates_dir: Optional[Path] = None, _use_singleton: bool = True):
        """
        Create or return the singleton instance.
        
        Args:
            templates_dir: If provided, creates a new separate instance.
            _use_singleton: Internal flag to control singleton behavior.
        """
        # If a custom templates_dir is provided, create a new instance
        if templates_dir is not None:
            instance = super().__new__(cls)
            return instance
        
        # Return singleton for default usage
        if _use_singleton:
            return get_default_registry()
        
        # Create new instance (used internally for singleton creation)
        return super().__new__(cls)

    def __init__(self, templates_dir: Optional[Path] = None, _use_singleton: bool = True) -> None:
        """
        Initialize the prompt registry.

        Args:
            templates_dir: Directory containing template files (YAML or JSON).
                If not provided, uses the default templates directory
                (flybrowser/prompts/templates/).
            _use_singleton: Internal flag, do not use directly.

        Example:
            Default templates directory (singleton):
            >>> registry = PromptRegistry()

            Custom templates directory (new instance):
            >>> from pathlib import Path
            >>> registry = PromptRegistry(
            ...     templates_dir=Path("./my_prompts")
            ... )
        """
        # Skip init if already initialized (singleton pattern)
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._initialized = True
        self.templates: Dict[str, PromptTemplate] = {}
        self.templates_dir = templates_dir or Path(__file__).parent / "templates"

        # Load templates if directory exists
        if self.templates_dir.exists():
            self.load_templates()

    def register(self, template: PromptTemplate) -> None:
        """
        Register a prompt template.

        Args:
            template: Template to register
        """
        key = self._make_key(template.name, template.version)
        self.templates[key] = template
        logger.debug(f"Registered template: {key}")

    def get(
        self, name: str, version: Optional[str] = None, variant: Optional[str] = None
    ) -> PromptTemplate:
        """
        Get a prompt template.

        Args:
            name: Template name
            version: Template version (latest if not specified)
            variant: Variant identifier for A/B testing

        Returns:
            PromptTemplate instance

        Raises:
            ConfigurationError: If template not found
        """
        # If variant specified, try to find it first
        if variant:
            variant_key = f"{name}:{variant}"
            if variant_key in self.templates:
                return self.templates[variant_key]

        # Find by name and version
        if version:
            key = self._make_key(name, version)
            if key not in self.templates:
                raise ConfigurationError(f"Template not found: {key}")
            return self.templates[key]

        # Find latest version
        matching = [k for k in self.templates.keys() if k.startswith(f"{name}:")]
        if not matching:
            raise ConfigurationError(f"No templates found for: {name}")

        # Sort by version and return latest
        latest_key = sorted(matching)[-1]
        return self.templates[latest_key]

    def list_templates(self) -> List[str]:
        """
        List all registered templates.

        Returns:
            List of template keys
        """
        return list(self.templates.keys())

    def load_templates(self) -> None:
        """Load templates from the templates directory (recursively including subdirectories)."""
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return

        # Load YAML templates (recursively search subdirectories)
        for yaml_file in self.templates_dir.glob("**/*.yaml"):
            try:
                self._load_yaml_template(yaml_file)
            except Exception as e:
                logger.error(f"Failed to load template {yaml_file}: {e}")

        # Load JSON templates (recursively search subdirectories)
        for json_file in self.templates_dir.glob("**/*.json"):
            try:
                self._load_json_template(json_file)
            except Exception as e:
                logger.error(f"Failed to load template {json_file}: {e}")

        logger.info(f"Loaded {len(self.templates)} templates from {self.templates_dir}")

    def _load_yaml_template(self, file_path: Path) -> None:
        """Load a template from a YAML file."""
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        template = PromptTemplate(**data)
        self.register(template)

    def _load_json_template(self, file_path: Path) -> None:
        """Load a template from a JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)

        template = PromptTemplate(**data)
        self.register(template)

    def save_template(self, template: PromptTemplate, format: str = "yaml") -> None:
        """
        Save a template to disk.

        Args:
            template: Template to save
            format: File format ('yaml' or 'json')
        """
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{template.name}_{template.version}.{format}"
        file_path = self.templates_dir / filename

        data = template.model_dump(exclude={"usage_count", "success_count"})

        if format == "yaml":
            with open(file_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif format == "json":
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved template to {file_path}")

    def _make_key(self, name: str, version: str) -> str:
        """Create a registry key from name and version."""
        return f"{name}:{version}"

    def get_stats(self) -> Dict:
        """
        Get registry statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "total_templates": len(self.templates),
            "templates": [
                {
                    "name": t.name,
                    "version": t.version,
                    "usage_count": t.usage_count,
                    "success_rate": t.get_success_rate(),
                }
                for t in self.templates.values()
            ],
        }

