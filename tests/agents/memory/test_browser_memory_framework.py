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

"""Tests for BrowserMemoryManager framework integration.

These tests verify that BrowserMemoryManager correctly delegates to
fireflyframework-genai's MemoryManager for storage while maintaining
full backward compatibility with the original API.
"""
import pytest
from flybrowser.agents.memory.browser_memory import BrowserMemoryManager, PageSnapshot, _RESERVED_KEYS


class TestBrowserMemoryFrameworkIntegration:
    """Tests that validate the MemoryManager delegation layer."""

    def test_creation_with_framework_memory(self):
        """BrowserMemoryManager should create a framework MemoryManager on init."""
        mgr = BrowserMemoryManager()
        assert mgr._framework_memory is not None

    def test_conversation_tracking(self):
        """A new conversation should be created on init with a valid ID."""
        mgr = BrowserMemoryManager()
        assert mgr.conversation_id is not None
        assert isinstance(mgr.conversation_id, str)
        assert len(mgr.conversation_id) > 0

    def test_record_page_stores_in_working_memory(self):
        """record_page_state should persist page_history via the framework."""
        mgr = BrowserMemoryManager()
        mgr.record_page_state("https://example.com", "Example", "5 links")
        stored = mgr._framework_memory.get_fact("page_history")
        assert stored is not None
        assert len(stored) == 1
        assert stored[0]["url"] == "https://example.com"
        assert stored[0]["title"] == "Example"

    def test_record_navigation_in_working_memory(self):
        """record_navigation should persist navigation_graph via the framework."""
        mgr = BrowserMemoryManager()
        mgr.record_navigation("https://a.com", "https://b.com", "click")
        stored = mgr._framework_memory.get_fact("navigation_graph")
        assert stored is not None
        assert "https://a.com" in stored
        assert "https://b.com" in stored["https://a.com"]

    def test_record_obstacle_in_working_memory(self):
        """record_obstacle should persist obstacle_cache via the framework."""
        mgr = BrowserMemoryManager()
        mgr.record_obstacle("https://example.com", "cookie_banner", "clicked accept")
        stored = mgr._framework_memory.get_fact("obstacle_cache")
        assert stored is not None
        assert "https://example.com" in stored
        assert stored["https://example.com"]["obstacle_type"] == "cookie_banner"
        assert stored["https://example.com"]["resolution"] == "clicked accept"

    def test_format_for_prompt_includes_memory_context(self):
        """format_for_prompt should include page context and user facts."""
        mgr = BrowserMemoryManager()
        mgr.record_page_state("https://example.com", "Example", "3 buttons")
        mgr.set_fact("extraction_target", "prices")
        prompt = mgr.format_for_prompt()
        assert "Current page: https://example.com - Example" in prompt
        assert "Additional facts:" in prompt
        assert "extraction_target: prices" in prompt

    def test_format_for_prompt_no_duplicate_browser_keys(self):
        """format_for_prompt should not include browser-managed keys in the facts section."""
        mgr = BrowserMemoryManager()
        mgr.record_page_state("https://example.com", "Example", "3 buttons")
        prompt = mgr.format_for_prompt()
        # Should NOT have an Additional facts section with browser keys
        if "Additional facts:" in prompt:
            assert "page_history" not in prompt.split("Additional facts:")[1]

    def test_set_fact_rejects_reserved_keys(self):
        """set_fact should raise ValueError for reserved browser keys."""
        mgr = BrowserMemoryManager()
        for key in _RESERVED_KEYS:
            with pytest.raises(ValueError, match="reserved"):
                mgr.set_fact(key, "anything")

    def test_set_fact_syncs_to_framework(self):
        """set_fact should persist to both local dict and framework."""
        mgr = BrowserMemoryManager()
        mgr.set_fact("extraction_target", "price")
        assert mgr.get_fact("extraction_target") == "price"
        assert mgr._framework_memory.get_fact("extraction_target") == "price"

    def test_clear_resets_framework_memory(self):
        """clear() should reset the framework's working memory."""
        mgr = BrowserMemoryManager()
        mgr.record_page_state("https://example.com", "Example", "5 links")
        mgr.record_navigation("https://a.com", "https://b.com", "click")
        mgr.record_obstacle("https://ex.com", "popup", "dismissed")
        mgr.set_fact("key1", "value1")

        mgr.clear()

        assert mgr._framework_memory.get_fact("page_history") is None
        assert mgr._framework_memory.get_fact("navigation_graph") is None
        assert mgr._framework_memory.get_fact("obstacle_cache") is None
        assert mgr.page_history == []
        assert mgr.navigation_graph == {}
        assert mgr.obstacle_cache == {}

    def test_backward_compat_page_history_property(self):
        """page_history property should return a list of PageSnapshot objects."""
        mgr = BrowserMemoryManager()
        mgr.record_page_state("https://a.com", "A", "links")
        mgr.record_page_state("https://b.com", "B", "buttons")

        history = mgr.page_history
        assert len(history) == 2
        assert isinstance(history[0], PageSnapshot)
        assert history[0].url == "https://a.com"
        assert history[1].url == "https://b.com"

    def test_backward_compat_navigation_graph_property(self):
        """navigation_graph property should return a Dict[str, List[str]]."""
        mgr = BrowserMemoryManager()
        mgr.record_navigation("https://a.com", "https://b.com", "click")
        mgr.record_navigation("https://a.com", "https://c.com", "click")

        graph = mgr.navigation_graph
        assert "https://a.com" in graph
        assert "https://b.com" in graph["https://a.com"]
        assert "https://c.com" in graph["https://a.com"]

    def test_backward_compat_obstacle_cache_property(self):
        """obstacle_cache property should return a Dict[str, ObstacleInfo]."""
        from flybrowser.agents.memory.browser_memory import ObstacleInfo

        mgr = BrowserMemoryManager()
        mgr.record_obstacle("https://example.com", "cookie_banner", "clicked accept")

        cache = mgr.obstacle_cache
        assert "https://example.com" in cache
        obs = cache["https://example.com"]
        assert isinstance(obs, ObstacleInfo)
        assert obs.obstacle_type == "cookie_banner"
        assert obs.resolution == "clicked accept"
