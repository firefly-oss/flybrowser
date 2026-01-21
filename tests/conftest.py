# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""
Shared test fixtures for FlyBrowser test suite.

This module provides common fixtures used across all test categories:
- Mock LLM providers
- Mock browser/page objects
- Test server fixtures
- Cluster node fixtures
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ==================== Environment Setup ====================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ.setdefault("FLYBROWSER_LOG_LEVEL", "warning")
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    yield


# ==================== Mock LLM Provider ====================

@dataclass
class MockLLMResponse:
    """Mock LLM response."""
    content: str
    model: str = "mock-model"
    usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.usage is None:
            self.usage = {"prompt_tokens": 10, "completion_tokens": 20}


class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    def __init__(self, default_response: str = "Mock response"):
        self.default_response = default_response
        self.calls: List[Dict[str, Any]] = []
        self.responses: List[str] = []
        self._response_index = 0
    
    def set_responses(self, responses: List[str]) -> None:
        """Set a sequence of responses to return."""
        self.responses = responses
        self._response_index = 0
    
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> MockLLMResponse:
        """Mock completion."""
        self.calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            **kwargs
        })
        
        if self.responses and self._response_index < len(self.responses):
            response = self.responses[self._response_index]
            self._response_index += 1
        else:
            response = self.default_response
        
        return MockLLMResponse(content=response)
    
    async def complete_with_vision(
        self,
        prompt: str,
        images: List[bytes],
        **kwargs: Any
    ) -> MockLLMResponse:
        """Mock vision completion."""
        self.calls.append({
            "prompt": prompt,
            "images": len(images),
            "type": "vision",
            **kwargs
        })
        return MockLLMResponse(content=self.default_response)
    
    def get_model_name(self) -> str:
        return "mock-model"
    
    def reset(self) -> None:
        """Reset call history."""
        self.calls = []
        self.responses = []
        self._response_index = 0


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Create a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def mock_llm_with_json_response() -> MockLLMProvider:
    """Create a mock LLM that returns JSON."""
    return MockLLMProvider(default_response='{"result": "success", "data": []}')


# ==================== Mock Browser/Page ====================

class MockPage:
    """Mock Playwright page for testing."""
    
    def __init__(self):
        self.url = "about:blank"
        self.title = "Mock Page"
        self.content = "<html><body>Mock content</body></html>"
        self.screenshot_data = b"PNG_MOCK_DATA"
        self._locators: Dict[str, Any] = {}
    
    async def goto(self, url: str, **kwargs) -> None:
        """Mock navigation."""
        self.url = url
        self.title = f"Page: {url}"
    
    async def title(self) -> str:
        """Get page title."""
        return self.title
    
    async def url(self) -> str:
        """Get page URL."""
        return self.url
    
    async def content(self) -> str:
        """Get page content."""
        return self.content
    
    async def screenshot(self, **kwargs) -> bytes:
        """Take screenshot."""
        return self.screenshot_data
    
    async def evaluate(self, expression: str) -> Any:
        """Evaluate JavaScript."""
        return None
    
    async def wait_for_load_state(self, state: str = "load") -> None:
        """Wait for load state."""
        pass
    
    async def wait_for_selector(self, selector: str, **kwargs) -> MagicMock:
        """Wait for selector."""
        return MagicMock()
    
    def locator(self, selector: str) -> MagicMock:
        """Get locator."""
        if selector not in self._locators:
            loc = MagicMock()
            loc.click = AsyncMock()
            loc.fill = AsyncMock()
            loc.type = AsyncMock()
            loc.text_content = AsyncMock(return_value="Mock text")
            loc.is_visible = AsyncMock(return_value=True)
            loc.count = AsyncMock(return_value=1)
            self._locators[selector] = loc
        return self._locators[selector]
    
    async def close(self) -> None:
        """Close page."""
        pass


class MockBrowser:
    """Mock Playwright browser for testing."""
    
    def __init__(self):
        self.pages: List[MockPage] = []
        self._context = MagicMock()
    
    async def new_page(self) -> MockPage:
        """Create new page."""
        page = MockPage()
        self.pages.append(page)
        return page
    
    async def new_context(self, **kwargs) -> MagicMock:
        """Create new context."""
        context = MagicMock()
        context.new_page = AsyncMock(return_value=MockPage())
        context.close = AsyncMock()
        return context
    
    async def close(self) -> None:
        """Close browser."""
        self.pages = []


class MockPlaywright:
    """Mock Playwright instance."""
    
    def __init__(self):
        self.chromium = MagicMock()
        self.chromium.launch = AsyncMock(return_value=MockBrowser())
        self.firefox = MagicMock()
        self.firefox.launch = AsyncMock(return_value=MockBrowser())
        self.webkit = MagicMock()
        self.webkit.launch = AsyncMock(return_value=MockBrowser())
    
    async def stop(self) -> None:
        """Stop Playwright."""
        pass


@pytest.fixture
def mock_page() -> MockPage:
    """Create a mock page."""
    return MockPage()


@pytest.fixture
def mock_browser() -> MockBrowser:
    """Create a mock browser."""
    return MockBrowser()


@pytest.fixture
def mock_playwright() -> MockPlaywright:
    """Create a mock Playwright instance."""
    return MockPlaywright()


# ==================== Temporary Directory ====================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_data_dir(temp_dir):
    """Create a temporary data directory with subdirs."""
    raft_dir = os.path.join(temp_dir, "raft")
    os.makedirs(raft_dir)
    return temp_dir


# ==================== Session Fixtures ====================

@pytest.fixture
def session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())


@pytest.fixture
def session_metadata() -> Dict[str, Any]:
    """Create sample session metadata."""
    return {
        "llm_provider": "openai",
        "llm_model": "gpt-4",
        "browser_type": "chromium",
        "headless": True,
    }


# ==================== Cluster Fixtures ====================

@pytest.fixture
def raft_config(temp_data_dir):
    """Create a Raft configuration for testing."""
    from flybrowser.service.cluster.raft import RaftConfig
    
    return RaftConfig(
        node_id="test-node-1",
        bind_host="127.0.0.1",
        bind_port=14321,
        api_host="127.0.0.1",
        api_port=18000,
        cluster_nodes=[],
        data_dir=os.path.join(temp_data_dir, "raft"),
        election_timeout_min_ms=150,
        election_timeout_max_ms=300,
        heartbeat_interval_ms=50,
    )


@pytest.fixture
def ha_node_config(temp_data_dir):
    """Create an HA node configuration for testing."""
    from flybrowser.service.cluster.ha_node import HANodeConfig
    
    return HANodeConfig(
        node_id="test-ha-node-1",
        api_host="127.0.0.1",
        api_port=18001,
        raft_host="127.0.0.1",
        raft_port=14322,
        peers=[],
        data_dir=temp_data_dir,
        max_sessions=5,
    )


# ==================== HTTP Client Fixtures ====================

@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp session."""
    session = MagicMock()
    session.request = AsyncMock()
    session.close = AsyncMock()
    return session


# ==================== FastAPI Test Client ====================

@pytest_asyncio.fixture
async def test_app():
    """Create a test FastAPI app instance."""
    from flybrowser.service.app import app
    return app


@pytest_asyncio.fixture
async def async_client(test_app):
    """Create an async test client."""
    from httpx import ASGITransport, AsyncClient
    
    async with AsyncClient(
        transport=ASGITransport(app=test_app),
        base_url="http://test"
    ) as client:
        yield client


# ==================== Utility Functions ====================

def make_async(func):
    """Convert a sync function to async for testing."""
    async def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@pytest.fixture(scope="session")
def session_event_loop():
    """Create a session-scoped event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def ensure_event_loop(session_event_loop):
    """Ensure an event loop is set for all tests.
    
    This is needed because some classes create asyncio primitives
    (Lock, Queue, etc.) in their __init__ which requires an event loop.
    """
    asyncio.set_event_loop(session_event_loop)
    yield


# ==================== Markers ====================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "cluster: marks tests as cluster tests"
    )
    config.addinivalue_line(
        "markers", "requires_browser: marks tests that need a real browser"
    )
