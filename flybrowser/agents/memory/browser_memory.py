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

"""BrowserMemoryManager -- browser-specific memory extensions.

Delegates storage to :class:`fireflyframework_genai.memory.MemoryManager`
while maintaining full backward compatibility with the original API.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from fireflyframework_genai.memory import MemoryManager
from fireflyframework_genai.memory.store import InMemoryStore


@dataclass
class PageSnapshot:
    url: str
    title: str
    elements_summary: str
    timestamp: float = 0.0


@dataclass
class ObstacleInfo:
    obstacle_type: str
    resolution: str


_RESERVED_KEYS = frozenset({
    "page_history", "visited_urls", "current_page",
    "navigation_graph", "obstacle_cache",
})


class BrowserMemoryManager:
    """Browser-specific memory manager backed by the framework MemoryManager.

    All data is stored as key-value pairs in the framework's WorkingMemory,
    enabling future migration to persistent backends (File, Postgres, MongoDB)
    with zero code changes.  The public API is fully backward-compatible with
    the original in-memory implementation.
    """

    def __init__(self) -> None:
        self._framework_memory = MemoryManager(store=InMemoryStore())
        self._conversation_id: str = self._framework_memory.new_conversation()

        # Local caches that mirror what is stored in working memory.
        # We keep them so that properties return the exact same *object*
        # types (PageSnapshot, ObstacleInfo) that callers already rely on.
        self._page_history: List[PageSnapshot] = []
        self._navigation_graph: Dict[str, List[str]] = {}
        self._obstacle_cache: Dict[str, ObstacleInfo] = {}
        self._visited_urls: set[str] = set()
        self._facts: Dict[str, Any] = {}

    # -- Framework accessors -----------------------------------------------

    @property
    def conversation_id(self) -> str:
        """The conversation ID for the current browser session."""
        return self._conversation_id

    # -- Backward-compatible properties ------------------------------------

    @property
    def page_history(self) -> List[PageSnapshot]:
        return self._page_history

    @property
    def navigation_graph(self) -> Dict[str, List[str]]:
        return self._navigation_graph

    @property
    def obstacle_cache(self) -> Dict[str, ObstacleInfo]:
        return self._obstacle_cache

    # -- Mutators (update local cache + sync to framework) -----------------

    def _sync_to_framework(self, key: str, value: Any) -> None:
        """Persist a value in the framework's working memory."""
        self._framework_memory.set_fact(key, value)

    def record_page_state(self, url: str, title: str, elements_summary: str) -> None:
        snapshot = PageSnapshot(url=url, title=title, elements_summary=elements_summary)
        self._page_history.append(snapshot)
        self._visited_urls.add(url)
        self._facts["current_page"] = {"url": url, "title": title}

        # Sync to framework working memory
        self._sync_to_framework("page_history", [asdict(s) for s in self._page_history])
        self._sync_to_framework("visited_urls", list(self._visited_urls))
        self._sync_to_framework("current_page", self._facts["current_page"])

    def record_navigation(self, from_url: str, to_url: str, method: str) -> None:
        self._navigation_graph.setdefault(from_url, []).append(to_url)

        # Sync to framework working memory
        self._sync_to_framework("navigation_graph", self._navigation_graph)

    def record_obstacle(self, url: str, obstacle_type: str, resolution: str) -> None:
        self._obstacle_cache[url] = ObstacleInfo(obstacle_type=obstacle_type, resolution=resolution)

        # Sync to framework working memory (store as plain dicts for serialisability)
        self._sync_to_framework(
            "obstacle_cache",
            {k: asdict(v) for k, v in self._obstacle_cache.items()},
        )

    def has_visited_url(self, url: str) -> bool:
        return url in self._visited_urls

    def get_current_page(self) -> Optional[PageSnapshot]:
        return self._page_history[-1] if self._page_history else None

    def set_fact(self, key: str, value: Any) -> None:
        if key in _RESERVED_KEYS:
            raise ValueError(
                f"Key {key!r} is reserved for internal browser state. "
                f"Reserved keys: {', '.join(sorted(_RESERVED_KEYS))}"
            )
        self._facts[key] = value
        self._sync_to_framework(key, value)

    def get_fact(self, key: str, default: Any = None) -> Any:
        return self._facts.get(key, default)

    def format_for_prompt(self) -> str:
        parts: List[str] = []
        current = self.get_current_page()
        if current:
            parts.append(f"Current page: {current.url} - {current.title}")
            if current.elements_summary:
                parts.append(f"Page elements: {current.elements_summary}")
        if len(self._page_history) > 1:
            parts.append(f"Pages visited: {len(self._page_history)} ({len(self._visited_urls)} unique)")
            recent = self._page_history[-5:]
            parts.append("Recent history: " + " -> ".join(s.url.split("//")[-1][:40] for s in recent))
        if self._obstacle_cache:
            parts.append(f"Known obstacles: {', '.join(self._obstacle_cache.keys())}")

        # Append user-set facts from working memory (skip browser-managed keys)
        user_facts = {
            k: v for k, v in self._facts.items()
            if k not in _RESERVED_KEYS and k != "current_page"
        }
        if user_facts:
            lines = ["Additional facts:"]
            for key, value in user_facts.items():
                lines.append(f"- {key}: {value}")
            parts.append("\n".join(lines))

        return "\n".join(parts) if parts else "No browser memory recorded yet."

    def clear(self) -> None:
        self._page_history.clear()
        self._navigation_graph.clear()
        self._obstacle_cache.clear()
        self._visited_urls.clear()
        self._facts.clear()

        # Clear framework working memory and start a fresh conversation
        self._framework_memory.clear_working()
        self._framework_memory.clear_conversation(self._conversation_id)
        self._conversation_id = self._framework_memory.new_conversation()
