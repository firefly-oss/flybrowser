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
FlyBrowser Agentic Framework.

This module provides the agent framework for browser automation.

New (fireflyframework-genai based) components:
    - BrowserAgent: Framework-native agent with ReAct reasoning
    - ToolKits: Navigation, Interaction, Extraction, System, Search, Captcha
    - BrowserMemoryManager: Page-snapshot based memory
    - Middleware: Obstacle detection, screenshot-on-error
    - Streaming: SSE event formatting

Legacy components (types, config, memory, response, context, strategy_selector)
are retained for backward compatibility.
"""

# Core types
from flybrowser.agents.types import (
    Action,
    ToolResult,
    Observation,
    ReActStep,
    ExecutionState,
    SafetyLevel,
    MemoryEntry,
    ReasoningStrategy,
)

# Configuration
from flybrowser.agents.config import AgentConfig

# Memory
from flybrowser.agents.memory import AgentMemory, WorkingMemory

# Strategy Selection
from flybrowser.agents.strategy_selector import StrategySelector

# Response Models
from flybrowser.agents.response import (
    AgentRequestResponse,
    create_response,
    LLMUsageInfo,
    ExecutionInfo,
)

# Context System
from flybrowser.agents.context import (
    ContextType,
    ActionContext,
    FileUploadSpec,
    ContextBuilder,
    ContextValidator,
    create_form_context,
    create_upload_context,
    create_filter_context,
)

__all__ = [
    # Core types
    "Action",
    "ToolResult",
    "Observation",
    "ReActStep",
    "ExecutionState",
    "SafetyLevel",
    "MemoryEntry",
    "AgentConfig",
    "ReasoningStrategy",
    # Memory
    "AgentMemory",
    "WorkingMemory",
    # Strategy Selection
    "StrategySelector",
    # Response Models
    "AgentRequestResponse",
    "create_response",
    "LLMUsageInfo",
    "ExecutionInfo",
    # Context System
    "ContextType",
    "ActionContext",
    "FileUploadSpec",
    "ContextBuilder",
    "ContextValidator",
    "create_form_context",
    "create_upload_context",
    "create_filter_context",
]
