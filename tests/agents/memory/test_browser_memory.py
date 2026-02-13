"""Tests for BrowserMemoryManager."""
import pytest
from flybrowser.agents.memory.browser_memory import BrowserMemoryManager, PageSnapshot


class TestBrowserMemoryManager:
    def test_creation(self):
        mem = BrowserMemoryManager()
        assert mem.page_history == []
        assert mem.navigation_graph == {}
        assert mem.obstacle_cache == {}

    def test_record_page_state(self):
        mem = BrowserMemoryManager()
        mem.record_page_state("https://example.com", "Example", "5 links, 2 buttons")
        assert len(mem.page_history) == 1
        snap = mem.page_history[0]
        assert snap.url == "https://example.com"
        assert snap.title == "Example"
        assert snap.elements_summary == "5 links, 2 buttons"

    def test_record_navigation(self):
        mem = BrowserMemoryManager()
        mem.record_navigation("https://a.com", "https://b.com", "click")
        assert "https://a.com" in mem.navigation_graph
        assert "https://b.com" in mem.navigation_graph["https://a.com"]

    def test_record_obstacle(self):
        mem = BrowserMemoryManager()
        mem.record_obstacle("https://example.com", "cookie_banner", "clicked accept")
        assert "https://example.com" in mem.obstacle_cache
        obs = mem.obstacle_cache["https://example.com"]
        assert obs.obstacle_type == "cookie_banner"
        assert obs.resolution == "clicked accept"

    def test_has_visited_url(self):
        mem = BrowserMemoryManager()
        assert mem.has_visited_url("https://example.com") is False
        mem.record_page_state("https://example.com", "Example", "")
        assert mem.has_visited_url("https://example.com") is True

    def test_get_current_page(self):
        mem = BrowserMemoryManager()
        assert mem.get_current_page() is None
        mem.record_page_state("https://a.com", "A", "")
        mem.record_page_state("https://b.com", "B", "")
        current = mem.get_current_page()
        assert current is not None
        assert current.url == "https://b.com"
        assert current.title == "B"

    def test_format_for_prompt(self):
        mem = BrowserMemoryManager()
        # Empty state
        assert mem.format_for_prompt() == "No browser memory recorded yet."

        # Single page
        mem.record_page_state("https://example.com", "Example", "3 buttons")
        prompt = mem.format_for_prompt()
        assert "Current page: https://example.com - Example" in prompt
        assert "Page elements: 3 buttons" in prompt

        # Multiple pages
        mem.record_page_state("https://other.com", "Other", "")
        prompt = mem.format_for_prompt()
        assert "Pages visited:" in prompt
        assert "Recent history:" in prompt
