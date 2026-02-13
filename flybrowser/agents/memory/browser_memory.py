"""BrowserMemoryManager — browser-specific memory extensions."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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


class BrowserMemoryManager:
    def __init__(self) -> None:
        self._page_history: List[PageSnapshot] = []
        self._navigation_graph: Dict[str, List[str]] = {}
        self._obstacle_cache: Dict[str, ObstacleInfo] = {}
        self._visited_urls: set[str] = set()
        self._facts: Dict[str, Any] = {}

    @property
    def page_history(self) -> List[PageSnapshot]:
        return self._page_history

    @property
    def navigation_graph(self) -> Dict[str, List[str]]:
        return self._navigation_graph

    @property
    def obstacle_cache(self) -> Dict[str, ObstacleInfo]:
        return self._obstacle_cache

    def record_page_state(self, url: str, title: str, elements_summary: str) -> None:
        self._page_history.append(PageSnapshot(url=url, title=title, elements_summary=elements_summary))
        self._visited_urls.add(url)
        self._facts["current_page"] = {"url": url, "title": title}

    def record_navigation(self, from_url: str, to_url: str, method: str) -> None:
        self._navigation_graph.setdefault(from_url, []).append(to_url)

    def record_obstacle(self, url: str, obstacle_type: str, resolution: str) -> None:
        self._obstacle_cache[url] = ObstacleInfo(obstacle_type=obstacle_type, resolution=resolution)

    def has_visited_url(self, url: str) -> bool:
        return url in self._visited_urls

    def get_current_page(self) -> Optional[PageSnapshot]:
        return self._page_history[-1] if self._page_history else None

    def set_fact(self, key: str, value: Any) -> None:
        self._facts[key] = value

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
        return "\n".join(parts) if parts else "No browser memory recorded yet."

    def clear(self) -> None:
        self._page_history.clear()
        self._navigation_graph.clear()
        self._obstacle_cache.clear()
        self._visited_urls.clear()
        self._facts.clear()
