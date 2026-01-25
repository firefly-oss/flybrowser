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
FlyBrowser - A browser automation and web scraping framework driven by LLM agents.

This package provides an easy-to-use SDK for browser automation powered by
Large Language Models, enabling natural language interactions with web pages.
"""

__version__ = "1.26.1"
__author__ = "Firefly Software Solutions Inc"
__license__ = "Apache-2.0"

from flybrowser.agents import (
    ReActAgent,
    AgentOrchestrator,
    ToolRegistry,
    AgentMemory,
    ReActParser,
    ExecutionMode,
    SafetyLevel,
    ExecutionState,
)
from flybrowser.client import FlyBrowserClient
from flybrowser.core.browser import BrowserManager
from flybrowser.core.proxy_rotator import (
    ProxyConfig,
    ProxyRotator,
    ProxyProtocol,
    ProxyStatus,
    RotationStrategy,
)
from flybrowser.core.browser_pool import BrowserPool, PoolConfig
from flybrowser.core.page import PageController
from flybrowser.core.performance import (
    PerformanceConfig,
    SpeedPreset,
    get_performance_config,
    set_performance_config,
    use_fast_mode,
    use_balanced_mode,
    use_thorough_mode,
)
from flybrowser.core.recording import (
    RecordingConfig,
    RecordingManager,
    Screenshot,
    ScreenshotCapture,
    ScreenshotFormat,
)
from flybrowser.sdk import FlyBrowser
from flybrowser.security.pii_handler import PIIConfig, PIIHandler, PIIType
from flybrowser.utils.execution_logger import LogVerbosity

__all__ = [
    # Agents - ReAct Framework
    "ReActAgent",
    "AgentOrchestrator",
    "ToolRegistry",
    "AgentMemory",
    "ReActParser",
    "ExecutionMode",
    "SafetyLevel",
    "ExecutionState",
    # Core
    "BrowserManager",
    "BrowserPool",
    "FlyBrowser",
    "FlyBrowserClient",
    "PageController",
    # Proxy Rotation
    "ProxyConfig",
    "ProxyRotator",
    "ProxyProtocol",
    "ProxyStatus",
    "RotationStrategy",
    # Logging
    "LogVerbosity",
    # Performance
    "PerformanceConfig",
    "SpeedPreset",
    "get_performance_config",
    "set_performance_config",
    "use_fast_mode",
    "use_balanced_mode",
    "use_thorough_mode",
    # Security
    "PIIConfig",
    "PIIHandler",
    "PIIType",
    # Recording
    "PoolConfig",
    "RecordingConfig",
    "RecordingManager",
    "Screenshot",
    "ScreenshotCapture",
    "ScreenshotFormat",
]

