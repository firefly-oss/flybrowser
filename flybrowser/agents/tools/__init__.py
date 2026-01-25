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
Tools module for the ReAct agentic framework.

This module provides the tool system architecture including:
- BaseTool: Abstract base class for all tools
- ToolRegistry: Registry for tool management
- ToolMetadata: Tool metadata and JSON schema generation
- ToolParameter: Parameter definition for validation

Browser-specific tools:
- navigation: NavigateTool, GoBackTool, GoForwardTool, RefreshTool
- interaction: ClickTool, TypeTool, ScrollTool, HoverTool
- extraction: ExtractTextTool, ScreenshotTool, GetPageStateTool
- system: CompleteTool, FailTool, WaitTool, AskUserTool
"""

from flybrowser.agents.tools.base import (
    BaseTool,
    ToolMetadata,
    ToolParameter,
)
from flybrowser.agents.tools.registry import ToolRegistry

# Navigation tools
from flybrowser.agents.tools.navigation import (
    NavigateTool,
    GoBackTool,
    GoForwardTool,
    RefreshTool,
)

# Interaction tools
from flybrowser.agents.tools.interaction import (
    ClickTool,
    TypeTool,
    ScrollTool,
    HoverTool,
    PressKeyTool,
    SelectOptionTool,
    CheckboxTool,
    FocusTool,
    FillTool,
    WaitForSelectorTool,
    DoubleClickTool,
    RightClickTool,
    DragAndDropTool,
    UploadFileTool,
    EvaluateJavaScriptTool,
    GetAttributeTool,
    ClearInputTool,
)

# Extraction tools
from flybrowser.agents.tools.extraction import (
    ExtractTextTool,
    ScreenshotTool,
    GetPageStateTool,
)

# System tools
from flybrowser.agents.tools.system import (
    CompleteTool,
    FailTool,
    WaitTool,
    AskUserTool,
)

# Legacy search tools
from flybrowser.agents.tools.search_api import SearchAPITool
from flybrowser.agents.tools.search_human import SearchHumanTool, SearchHumanAdvancedTool
from flybrowser.agents.tools.search_rank import SearchRankTool

# New search abstraction layer
from flybrowser.agents.tools.search import (
    # Types
    SearchOptions,
    SearchType,
    ProviderHealth,
    ProviderStatus,
    RankedSearchResult,
    SearchAgentResponse,
    ProviderCapabilities,
    # Providers
    BaseSearchProvider,
    ProviderRegistry,
    SerperProvider,
    GoogleSearchProvider,
    BingSearchProvider,
    # Rankers
    BaseRanker,
    BM25Ranker,
    FreshnessRanker,
    DomainAuthorityRanker,
    CompositeRanker,
    # Agent
    SearchAgent,
    # Tool integration
    SearchAgentTool,
    # Intent detection
    SearchIntent,
    SearchIntentDetector,
    detect_search_intent,
)

# Page exploration tools
from flybrowser.agents.tools.page_explorer import PageExplorerTool
from flybrowser.agents.tools.search_utils import (
    SearchResult,
    SearchResponse,
    SearchEngine,
    SearchProvider,
)

__all__ = [
    # Base classes
    "BaseTool",
    "ToolMetadata",
    "ToolParameter",
    # Registry
    "ToolRegistry",
    # Navigation tools
    "NavigateTool",
    "GoBackTool",
    "GoForwardTool",
    "RefreshTool",
    # Interaction tools
    "ClickTool",
    "TypeTool",
    "ScrollTool",
    "HoverTool",
    "PressKeyTool",
    "SelectOptionTool",
    "CheckboxTool",
    "FocusTool",
    "FillTool",
    "WaitForSelectorTool",
    "DoubleClickTool",
    "RightClickTool",
    "DragAndDropTool",
    "UploadFileTool",
    "EvaluateJavaScriptTool",
    "GetAttributeTool",
    "ClearInputTool",
    # Extraction tools
    "ExtractTextTool",
    "ScreenshotTool",
    "GetPageStateTool",
    # System tools
    "CompleteTool",
    "FailTool",
    "WaitTool",
    "AskUserTool",
    # Legacy search tools
    "SearchAPITool",
    "SearchHumanTool",
    "SearchHumanAdvancedTool",
    "SearchRankTool",
    "SearchResult",
    "SearchResponse",
    "SearchEngine",
    "SearchProvider",
    # New search abstraction layer
    "SearchOptions",
    "SearchType",
    "ProviderHealth",
    "ProviderStatus",
    "RankedSearchResult",
    "SearchAgentResponse",
    "ProviderCapabilities",
    "BaseSearchProvider",
    "ProviderRegistry",
    "SerperProvider",
    "GoogleSearchProvider",
    "BingSearchProvider",
    "BaseRanker",
    "BM25Ranker",
    "FreshnessRanker",
    "DomainAuthorityRanker",
    "CompositeRanker",
    "SearchAgent",
    "SearchAgentTool",
    "SearchIntent",
    "SearchIntentDetector",
    "detect_search_intent",
    # Page exploration tools
    "PageExplorerTool",
]
