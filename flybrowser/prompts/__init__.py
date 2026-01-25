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

"""Professional prompt management system."""

from flybrowser.prompts.manager import PromptManager
from flybrowser.prompts.registry import PromptRegistry, get_default_registry
from flybrowser.prompts.template import PromptTemplate
from flybrowser.prompts.schemas import (
    # Core schemas
    ACTION_SCHEMA,
    ACTION_TYPES,
    AGENT_RESULT_SCHEMA,
    AGENT_TASK_SCHEMA,
    ELEMENT_DETECTION_RESULT_SCHEMA,
    ELEMENT_SCHEMA,
    EXECUTION_PLAN_SCHEMA,
    EXTRACTION_RESULT_SCHEMA,
    NAVIGATION_RESULT_SCHEMA,
    OBSTACLE_DETECTION_RESULT_SCHEMA,
    OBSTACLE_SCHEMA,
    OBSTACLE_TYPES,
    PAGE_STATE_SCHEMA,
    PLAN_STEP_SCHEMA,
    SCHEMA_REGISTRY,
    VERIFICATION_RESULT_SCHEMA,
    # Utility functions
    create_example_from_schema,
    format_schema_for_prompt,
    get_required_properties,
    get_schema,
    get_schema_properties,
    list_schemas,
)

__all__ = [
    # Prompt management
    "PromptManager",
    "PromptRegistry",
    "PromptTemplate",
    "get_default_registry",
    # Schemas
    "ACTION_SCHEMA",
    "ACTION_TYPES",
    "AGENT_RESULT_SCHEMA",
    "AGENT_TASK_SCHEMA",
    "ELEMENT_DETECTION_RESULT_SCHEMA",
    "ELEMENT_SCHEMA",
    "EXECUTION_PLAN_SCHEMA",
    "EXTRACTION_RESULT_SCHEMA",
    "NAVIGATION_RESULT_SCHEMA",
    "OBSTACLE_DETECTION_RESULT_SCHEMA",
    "OBSTACLE_SCHEMA",
    "OBSTACLE_TYPES",
    "PAGE_STATE_SCHEMA",
    "PLAN_STEP_SCHEMA",
    "SCHEMA_REGISTRY",
    "VERIFICATION_RESULT_SCHEMA",
    # Schema utilities
    "create_example_from_schema",
    "format_schema_for_prompt",
    "get_required_properties",
    "get_schema",
    "get_schema_properties",
    "list_schemas",
]

