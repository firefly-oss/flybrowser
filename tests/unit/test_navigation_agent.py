# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NavigationAgent."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from flybrowser.agents.navigation_agent import (
    NavigationAgent,
    NavigationResult,
    NavigationType,
    WaitStrategy,
)
from flybrowser.exceptions import NavigationError


class TestWaitStrategy:
    """Tests for WaitStrategy enum."""

    def test_all_wait_strategies(self):
        """Test all wait strategies exist."""
        assert WaitStrategy.LOAD == "load"
        assert WaitStrategy.DOM_CONTENT_LOADED == "domcontentloaded"
        assert WaitStrategy.NETWORK_IDLE == "networkidle"
        assert WaitStrategy.COMMIT == "commit"
        assert WaitStrategy.CUSTOM == "custom"


class TestNavigationType:
    """Tests for NavigationType enum."""

    def test_all_navigation_types(self):
        """Test all navigation types exist."""
        assert NavigationType.URL == "url"
        assert NavigationType.LINK == "link"
        assert NavigationType.BACK == "back"
        assert NavigationType.FORWARD == "forward"
        assert NavigationType.REFRESH == "refresh"
        assert NavigationType.SEARCH == "search"


class TestNavigationResult:
    """Tests for NavigationResult dataclass."""

    def test_successful_result(self):
        """Test successful navigation result."""
        result = NavigationResult(
            success=True,
            url="https://example.com",
            title="Example",
            navigation_type=NavigationType.URL,
            wait_time=1.5
        )
        
        assert result.success is True
        assert result.url == "https://example.com"
        assert result.title == "Example"
        assert result.error is None

    def test_failed_result(self):
        """Test failed navigation result."""
        result = NavigationResult(
            success=False,
            navigation_type=NavigationType.LINK,
            error="Element not found"
        )
        
        assert result.success is False
        assert result.error == "Element not found"


class TestNavigationAgentInit:
    """Tests for NavigationAgent initialization."""

    def test_init_default_values(self):
        """Test default initialization values."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = NavigationAgent(mock_page, mock_detector, mock_llm)
        
        assert agent.page is mock_page
        assert agent.detector is mock_detector
        assert agent.llm is mock_llm
        assert agent.default_timeout == 30000
        assert agent.default_wait_strategy == WaitStrategy.DOM_CONTENT_LOADED

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = NavigationAgent(
            mock_page,
            mock_detector,
            mock_llm,
            default_timeout=60000,
            default_wait_strategy=WaitStrategy.NETWORK_IDLE
        )
        
        assert agent.default_timeout == 60000
        assert agent.default_wait_strategy == WaitStrategy.NETWORK_IDLE


class TestNavigationAgentGoto:
    """Tests for NavigationAgent.goto()."""

    @pytest.mark.asyncio
    async def test_goto_successful(self):
        """Test successful URL navigation."""
        mock_page = MagicMock()
        mock_page.goto = AsyncMock()
        mock_page.get_url = AsyncMock(return_value="https://example.com")
        mock_page.get_title = AsyncMock(return_value="Example")
        mock_page.evaluate = AsyncMock(return_value=False)
        mock_page.page = MagicMock()
        mock_page.page.wait_for_load_state = AsyncMock()
        
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = NavigationAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.goto("https://example.com")
        
        assert result.success is True
        assert result.url == "https://example.com"
        assert result.navigation_type == NavigationType.URL

    @pytest.mark.asyncio
    async def test_goto_adds_protocol(self):
        """Test goto adds https protocol if missing."""
        mock_page = MagicMock()
        mock_page.goto = AsyncMock()
        mock_page.get_url = AsyncMock(return_value="https://example.com")
        mock_page.get_title = AsyncMock(return_value="Example")
        mock_page.evaluate = AsyncMock(return_value=False)
        mock_page.page = MagicMock()
        mock_page.page.wait_for_load_state = AsyncMock()
        
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = NavigationAgent(mock_page, mock_detector, mock_llm)
        
        await agent.goto("example.com")
        
        mock_page.goto.assert_awaited_once()
        call_args = mock_page.goto.await_args
        assert call_args[0][0] == "https://example.com"

    @pytest.mark.asyncio
    async def test_goto_failed(self):
        """Test failed URL navigation."""
        mock_page = MagicMock()
        mock_page.goto = AsyncMock(side_effect=Exception("Navigation error"))
        
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = NavigationAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.goto("https://invalid.example")
        
        assert result.success is False
        assert result.error is not None


class TestNavigationAgentFollowLink:
    """Tests for NavigationAgent.follow_link()."""

    @pytest.mark.asyncio
    async def test_follow_link_successful(self):
        """Test successful link following."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(return_value="https://example.com/about")
        mock_page.get_title = AsyncMock(return_value="About Us")
        mock_page.page = MagicMock()
        mock_page.page.wait_for_load_state = AsyncMock()
        
        mock_detector = MagicMock()
        mock_detector.find_element = AsyncMock(return_value={
            "selector": "a.about-link",
            "selector_type": "css"
        })
        mock_detector.click = AsyncMock()
        
        mock_llm = MagicMock()
        
        agent = NavigationAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.follow_link("About Us link")
        
        assert result.success is True
        assert result.navigation_type == NavigationType.LINK

    @pytest.mark.asyncio
    async def test_follow_link_element_not_found(self):
        """Test follow_link when element not found."""
        from flybrowser.exceptions import ElementNotFoundError
        
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(return_value="https://example.com")
        mock_page.get_title = AsyncMock(return_value="Example")
        
        mock_detector = MagicMock()
        mock_detector.find_element = AsyncMock(
            side_effect=ElementNotFoundError("Not found")
        )
        
        mock_llm = MagicMock()
        
        agent = NavigationAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.follow_link("nonexistent link")
        
        assert result.success is False
        assert "Could not find link" in result.error


class TestNavigationAgentGoBack:
    """Tests for NavigationAgent.go_back()."""

    @pytest.mark.asyncio
    async def test_go_back_successful(self):
        """Test successful back navigation."""
        mock_page = MagicMock()
        mock_page.page = MagicMock()
        mock_page.page.go_back = AsyncMock()
        mock_page.page.wait_for_load_state = AsyncMock()
        mock_page.get_url = AsyncMock(return_value="https://example.com")
        mock_page.get_title = AsyncMock(return_value="Example")
        mock_page.evaluate = AsyncMock(return_value=False)
        
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = NavigationAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.go_back()
        
        assert result.success is True
        assert result.navigation_type == NavigationType.BACK


class TestNavigationAgentGoForward:
    """Tests for NavigationAgent.go_forward()."""

    @pytest.mark.asyncio
    async def test_go_forward_successful(self):
        """Test successful forward navigation."""
        mock_page = MagicMock()
        mock_page.page = MagicMock()
        mock_page.page.go_forward = AsyncMock()
        mock_page.page.wait_for_load_state = AsyncMock()
        mock_page.get_url = AsyncMock(return_value="https://example.com/page2")
        mock_page.get_title = AsyncMock(return_value="Page 2")
        mock_page.evaluate = AsyncMock(return_value=False)
        
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = NavigationAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.go_forward()
        
        assert result.success is True
        assert result.navigation_type == NavigationType.FORWARD


class TestNavigationAgentRefresh:
    """Tests for NavigationAgent.refresh()."""

    @pytest.mark.asyncio
    async def test_refresh_successful(self):
        """Test successful page refresh."""
        mock_page = MagicMock()
        mock_page.page = MagicMock()
        mock_page.page.reload = AsyncMock()
        mock_page.page.wait_for_load_state = AsyncMock()
        mock_page.get_url = AsyncMock(return_value="https://example.com")
        mock_page.get_title = AsyncMock(return_value="Example")
        mock_page.evaluate = AsyncMock(return_value=False)
        
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = NavigationAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.refresh()
        
        assert result.success is True
        assert result.navigation_type == NavigationType.REFRESH


class TestNavigationAgentGetCurrentLocation:
    """Tests for NavigationAgent.get_current_location()."""

    @pytest.mark.asyncio
    async def test_get_current_location(self):
        """Test getting current location details."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(
            return_value="https://example.com/path?query=value#section"
        )
        mock_page.get_title = AsyncMock(return_value="Example Page")
        
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = NavigationAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.get_current_location()
        
        assert result["url"] == "https://example.com/path?query=value#section"
        assert result["title"] == "Example Page"
        assert result["protocol"] == "https"
        assert result["host"] == "example.com"
        assert result["path"] == "/path"
        assert result["query"] == "query=value"
        assert result["fragment"] == "section"


class TestNavigationAgentExecute:
    """Tests for NavigationAgent.execute()."""

    @pytest.mark.asyncio
    async def test_execute_url_navigation(self):
        """Test execute with URL navigation."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(return_value="https://example.com")
        mock_page.get_title = AsyncMock(return_value="Example")
        mock_page.get_html = AsyncMock(return_value="<html></html>")
        mock_page.goto = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=b"screenshot")
        mock_page.evaluate = AsyncMock(return_value=[])
        mock_page.page = MagicMock()
        mock_page.page.wait_for_load_state = AsyncMock()
        
        mock_detector = MagicMock()
        
        mock_llm = MagicMock()
        mock_llm.generate_with_vision = AsyncMock(
            return_value=MagicMock(content=json.dumps({
                "type": "url",
                "url": "https://example.com/products"
            }))
        )
        
        agent = NavigationAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.execute("Go to the products page")
        
        assert result["navigation_type"] == "url"

    @pytest.mark.asyncio
    async def test_execute_returns_error_dict_on_failure(self):
        """Test execute returns error dict on failure instead of raising."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(side_effect=Exception("Page error"))
        
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = NavigationAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.execute("Go somewhere")
        
        # Should return error dict instead of raising
        assert result["success"] is False
        assert "error" in result
        assert "Page error" in result["error"]
        assert result["navigation_type"] == "unknown"
        assert result["exception_type"] == "Exception"
