# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SearchAgent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from flybrowser.agents.search_agent import (
    SearchAgent,
    SearchAgentResult,
)
from flybrowser.agents.search_types import (
    SearchEngine,
    SearchOptions,
    SearchResult,
    SearchResultItem,
)
from flybrowser.agents.search_engines import SEARCH_ENGINES


class TestSearchAgentResult:
    """Tests for SearchAgentResult dataclass."""

    def test_successful_result(self):
        """Test successful search result."""
        search_result = MagicMock(spec=SearchResult)
        search_result.to_dict.return_value = {"query": "test", "results": []}
        
        result = SearchAgentResult(
            success=True,
            search_result=search_result,
        )
        
        assert result.success is True
        assert result.search_result == search_result
        assert result.error is None

    def test_failed_result(self):
        """Test failed search result."""
        result = SearchAgentResult(
            success=False,
            error="Search engine blocked",
        )
        
        assert result.success is False
        assert result.error == "Search engine blocked"
        assert result.search_result is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        search_result = MagicMock(spec=SearchResult)
        search_result.to_dict.return_value = {"query": "test", "results": []}

        # Mock timing, llm_usage, and page_metrics with to_dict methods
        mock_timing = MagicMock()
        mock_timing.to_dict.return_value = {"start": 0, "end": 100}
        mock_llm_usage = MagicMock()
        mock_llm_usage.to_dict.return_value = {"tokens": 0}
        mock_page_metrics = MagicMock()
        mock_page_metrics.to_dict.return_value = {"page_load_ms": 100}

        result = SearchAgentResult(
            success=True,
            search_result=search_result,
            timing=mock_timing,
            llm_usage=mock_llm_usage,
            page_metrics=mock_page_metrics,
        )

        result_dict = result.to_dict()
        assert "search_result" in result_dict
        assert result_dict["success"] is True


class TestSearchAgentInit:
    """Tests for SearchAgent initialization."""

    def test_init_default_values(self):
        """Test default initialization values."""
        page = MagicMock()
        detector = MagicMock()
        llm = MagicMock()
        
        agent = SearchAgent(page, detector, llm)
        
        assert agent.default_engine == SearchEngine.GOOGLE
        assert agent.cache_ttl_seconds == 300
        assert agent.min_request_delay == 2.0
        assert agent.max_request_delay == 5.0
        assert agent.session is None

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        page = MagicMock()
        detector = MagicMock()
        llm = MagicMock()
        
        agent = SearchAgent(
            page, detector, llm,
            default_engine=SearchEngine.DUCKDUCKGO,
            cache_ttl_seconds=600,
            min_request_delay=3.0,
            max_request_delay=7.0,
        )
        
        assert agent.default_engine == SearchEngine.DUCKDUCKGO
        assert agent.cache_ttl_seconds == 600
        assert agent.min_request_delay == 3.0
        assert agent.max_request_delay == 7.0

    def test_lazy_loaded_properties(self):
        """Test that ranker, analyzer, and captcha resolver are lazy-loaded."""
        page = MagicMock()
        detector = MagicMock()
        llm = MagicMock()
        
        agent = SearchAgent(page, detector, llm)
        
        # Properties should be None initially
        assert agent._result_ranker is None
        assert agent._page_analyzer is None
        assert agent._captcha_resolver is None


class TestSearchMethod:
    """Tests for SearchAgent.search method."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock SearchAgent for testing."""
        page = MagicMock()
        page.url = "https://www.google.com"
        page.title = AsyncMock(return_value="Google Search")
        page.goto = AsyncMock()
        page.screenshot = AsyncMock(return_value=b"fake_screenshot")
        page.evaluate = AsyncMock(return_value="<html><body>Test</body></html>")
        page.wait_for_load_state = AsyncMock()
        page.locator = MagicMock()
        page.keyboard = MagicMock()
        page.keyboard.press = AsyncMock()
        
        detector = MagicMock()
        detector.find_elements = AsyncMock(return_value=[])
        
        llm = MagicMock()
        llm.complete = AsyncMock(return_value='{"optimized_query": "test query"}')
        llm.complete_with_vision = AsyncMock(return_value='[]')
        
        agent = SearchAgent(page, detector, llm)
        return agent

    @pytest.mark.asyncio
    async def test_search_uses_default_engine(self, mock_agent):
        """Test that search uses default engine when none specified."""
        # Mock the internal methods
        mock_agent._search_with_fallback = AsyncMock(return_value=SearchResult(
            query="test",
            engine=SearchEngine.GOOGLE,
            results=[],
            total_found=0,
        ))

        result = await mock_agent.search("test query")

        assert result.engine == SearchEngine.GOOGLE
        mock_agent._search_with_fallback.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_bing_engine(self, mock_agent):
        """Test search with Bing engine."""
        mock_agent._search_with_fallback = AsyncMock(return_value=SearchResult(
            query="test",
            engine=SearchEngine.BING,
            results=[],
            total_found=0,
        ))

        options = SearchOptions(engine=SearchEngine.BING)
        result = await mock_agent.search("test query", options=options)

        assert result.engine == SearchEngine.BING

    @pytest.mark.asyncio
    async def test_search_with_duckduckgo_engine(self, mock_agent):
        """Test search with DuckDuckGo engine."""
        mock_agent._search_with_fallback = AsyncMock(return_value=SearchResult(
            query="test",
            engine=SearchEngine.DUCKDUCKGO,
            results=[],
            total_found=0,
        ))

        options = SearchOptions(engine=SearchEngine.DUCKDUCKGO)
        result = await mock_agent.search("test query", options=options)

        assert result.engine == SearchEngine.DUCKDUCKGO


class TestExecuteMethod:
    """Tests for SearchAgent.execute method."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock SearchAgent for testing."""
        page = MagicMock()
        page.url = "https://www.google.com"
        page.title = AsyncMock(return_value="Google Search")
        page.goto = AsyncMock()
        page.screenshot = AsyncMock(return_value=b"fake_screenshot")
        page.evaluate = AsyncMock(return_value="<html><body>Test</body></html>")
        page.wait_for_load_state = AsyncMock()

        detector = MagicMock()
        llm = MagicMock()
        llm.complete = AsyncMock(return_value='{"optimized_query": "test"}')
        llm.complete_with_vision = AsyncMock(return_value='[]')
        # Mock the session usage chain properly to return an int
        llm.get_session_usage = MagicMock(return_value=MagicMock(get=MagicMock(return_value=0)))

        agent = SearchAgent(page, detector, llm)
        return agent

    @pytest.mark.asyncio
    async def test_execute_without_explore(self, mock_agent):
        """Test execute method without exploration."""
        mock_agent.search = AsyncMock(return_value=SearchResult(
            query="python tutorials",
            engine=SearchEngine.GOOGLE,
            results=[],
            total_found=0,
        ))

        result = await mock_agent.execute(
            query="python tutorials",
            explore=False,
        )

        assert result.success is True
        mock_agent.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_explore(self, mock_agent):
        """Test execute method with exploration enabled."""
        mock_result = SearchResultItem(
            title="Test",
            url="https://example.com",
            snippet="Test snippet",
            score=0.9,
            position=1,
        )

        mock_agent.search = AsyncMock(return_value=SearchResult(
            query="python tutorials",
            engine=SearchEngine.GOOGLE,
            results=[mock_result],
            total_found=1,
        ))
        mock_agent._explore_results = AsyncMock(return_value=[])

        result = await mock_agent.execute(
            query="python tutorials",
            explore=True,
        )

        assert result.success is True


class TestHumanLikeSearch:
    """Tests for SearchAgent._perform_human_like_search method."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock SearchAgent for testing human-like search."""
        page = MagicMock()
        page.url = "https://www.google.com"
        page.title = AsyncMock(return_value="Google")
        page.goto = AsyncMock()
        page.screenshot = AsyncMock(return_value=b"fake_screenshot")
        page.evaluate = AsyncMock(return_value="<html></html>")
        page.wait_for_load_state = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        page.locator = MagicMock()
        page.keyboard = MagicMock()
        page.keyboard.press = AsyncMock()
        page.keyboard.type = AsyncMock()

        detector = MagicMock()
        detector.find_elements = AsyncMock(return_value=[])

        llm = MagicMock()
        llm.complete = AsyncMock(return_value='{"action": "search"}')
        llm.complete_with_vision = AsyncMock(return_value='[]')

        agent = SearchAgent(page, detector, llm)
        return agent

    @pytest.mark.asyncio
    async def test_human_like_search_navigates_to_homepage(self, mock_agent):
        """Test that human-like search navigates to search engine homepage."""
        # Mock internal methods that are called before the main flow
        mock_agent._detect_and_handle_obstacles = AsyncMock()
        mock_input_element = MagicMock()
        mock_input_element.click = AsyncMock()
        mock_agent._find_search_input = AsyncMock(return_value=mock_input_element)

        # Mock the main flow methods
        mock_agent._navigate_to_search_engine = AsyncMock(return_value=True)
        mock_agent._type_query_human_like = AsyncMock(return_value=True)
        mock_agent._submit_search = AsyncMock(return_value=True)
        mock_agent._wait_for_results = AsyncMock(return_value=True)
        mock_agent._extract_results = AsyncMock(return_value=[])

        engine_config = SEARCH_ENGINES[SearchEngine.GOOGLE]
        await mock_agent._perform_human_like_search(engine_config, "test query", SearchOptions())

        mock_agent._navigate_to_search_engine.assert_called_once()

    @pytest.mark.asyncio
    async def test_human_like_search_types_query(self, mock_agent):
        """Test that human-like search types query with delays."""
        # Mock internal methods that are called before the main flow
        mock_agent._detect_and_handle_obstacles = AsyncMock()
        mock_input_element = MagicMock()
        mock_input_element.click = AsyncMock()
        mock_agent._find_search_input = AsyncMock(return_value=mock_input_element)

        # Mock the main flow methods
        mock_agent._navigate_to_search_engine = AsyncMock(return_value=True)
        mock_agent._type_query_human_like = AsyncMock(return_value=True)
        mock_agent._submit_search = AsyncMock(return_value=True)
        mock_agent._wait_for_results = AsyncMock(return_value=True)
        mock_agent._extract_results = AsyncMock(return_value=[])

        engine_config = SEARCH_ENGINES[SearchEngine.GOOGLE]
        await mock_agent._perform_human_like_search(engine_config, "test query", SearchOptions())

        mock_agent._type_query_human_like.assert_called_once()


class TestExtractResults:
    """Tests for SearchAgent._extract_results method."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock SearchAgent for testing result extraction."""
        # Provide long enough HTML content to pass the content length check
        long_html = "<html><head><title>Google Search</title></head><body>" + "x" * 1000 + "<div class='results'>Search results here</div></body></html>"

        page = MagicMock()
        page.url = "https://www.google.com/search?q=test"
        page.title = AsyncMock(return_value="test - Google Search")
        page.screenshot = AsyncMock(return_value=b"fake_screenshot")
        page.evaluate = AsyncMock(return_value=long_html)
        # Add missing page methods used by SearchAgent
        page.goto = AsyncMock()
        page.get_url = AsyncMock(return_value="https://www.google.com/search?q=test")
        page.get_title = AsyncMock(return_value="test - Google Search")
        page.get_html = AsyncMock(return_value=long_html)
        # Add page.page sub-object with Playwright page methods
        page.page = MagicMock()
        page.page.wait_for_load_state = AsyncMock()
        page.page.query_selector = AsyncMock(return_value=None)
        page.page.wait_for_selector = AsyncMock()
        page.page.evaluate = AsyncMock(return_value={})

        detector = MagicMock()

        llm = MagicMock()
        # LLM returns JSON with extracted results
        llm.complete_with_vision = AsyncMock(return_value='''
        [
            {"title": "Result 1", "url": "https://example1.com", "snippet": "Description 1"},
            {"title": "Result 2", "url": "https://example2.com", "snippet": "Description 2"}
        ]
        ''')

        agent = SearchAgent(page, detector, llm)
        return agent

    @pytest.mark.asyncio
    @patch('flybrowser.agents.search_agent.ResultRankerAgent')
    async def test_extract_results_uses_llm_vision(self, mock_ranker_class, mock_agent):
        """Test that result extraction uses LLM vision analysis."""
        mock_ranker = MagicMock()
        mock_ranker.extract_results = AsyncMock(return_value=[
            {"title": "Result 1", "url": "https://example1.com", "snippet": "Description 1"},
            {"title": "Result 2", "url": "https://example2.com", "snippet": "Description 2"}
        ])
        mock_ranker_class.return_value = mock_ranker

        results = await mock_agent._extract_results(SearchEngine.GOOGLE)

        # Should call result ranker extract_results
        mock_ranker.extract_results.assert_called_once()

    @pytest.mark.asyncio
    @patch('flybrowser.agents.search_agent.ResultRankerAgent')
    async def test_extract_results_returns_ranked_results(self, mock_ranker_class, mock_agent):
        """Test that extracted results are properly formatted."""
        mock_ranker = MagicMock()
        mock_ranker.extract_results = AsyncMock(return_value=[
            {"title": "Result 1", "url": "https://example1.com", "snippet": "Description 1"},
            {"title": "Result 2", "url": "https://example2.com", "snippet": "Description 2"}
        ])
        mock_ranker_class.return_value = mock_ranker

        results = await mock_agent._extract_results(SearchEngine.GOOGLE)

        assert len(results) == 2
        assert results[0]["title"] == "Result 1"
        assert results[1]["url"] == "https://example2.com"


class TestErrorHandling:
    """Tests for SearchAgent error handling."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock SearchAgent for testing error handling."""
        page = MagicMock()
        page.url = "https://www.google.com"
        page.goto = AsyncMock()
        page.screenshot = AsyncMock(return_value=b"fake_screenshot")

        detector = MagicMock()
        llm = MagicMock()
        llm.complete = AsyncMock(return_value='{}')
        # Mock the session usage chain properly to return an int
        llm.get_session_usage = MagicMock(return_value=MagicMock(get=MagicMock(return_value=0)))

        agent = SearchAgent(page, detector, llm)
        return agent

    @pytest.mark.asyncio
    async def test_search_handles_navigation_failure(self, mock_agent):
        """Test that search handles navigation failures gracefully."""
        mock_agent.page.goto = AsyncMock(side_effect=Exception("Navigation failed"))
        mock_agent._search_with_fallback = AsyncMock(side_effect=Exception("Search failed"))

        with pytest.raises(Exception):
            await mock_agent.search("test query")

    @pytest.mark.asyncio
    async def test_execute_returns_failure_on_error(self, mock_agent):
        """Test that execute returns failure result on error."""
        mock_agent.search = AsyncMock(side_effect=Exception("Search error"))

        result = await mock_agent.execute(query="test")

        assert result.success is False
        assert result.error is not None

