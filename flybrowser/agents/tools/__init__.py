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
Tools module for the FlyBrowser agentic framework.

Old tool implementations (navigation, interaction, extraction, system, etc.)
have been removed in favor of fireflyframework-genai ToolKits.

Remaining modules:
- search_utils: Search result types (SearchResult, SearchResponse, etc.)
- search_human: Human-readable search tools (has transitive deps, import directly)
- base: BaseTool abstract class (has transitive deps, import directly)
- search/: Search abstraction layer (has transitive deps, import directly)
"""

from flybrowser.agents.tools.search_utils import (
    SearchResult,
    SearchResponse,
    SearchEngine,
    SearchProvider,
)

__all__ = [
    "SearchResult",
    "SearchResponse",
    "SearchEngine",
    "SearchProvider",
]
