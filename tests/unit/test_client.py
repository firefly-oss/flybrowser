# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FlyBrowserClient."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flybrowser.client import (
    CircuitBreaker,
    CircuitState,
    ClusterNode,
    FlyBrowserClient,
    RetryConfig,
)


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_all_states(self):
        """Test all circuit states exist."""
        assert CircuitState.CLOSED == "closed"
        assert CircuitState.OPEN == "open"
        assert CircuitState.HALF_OPEN == "half_open"


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_default_values(self):
        """Test default circuit breaker values."""
        cb = CircuitBreaker()
        
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 30.0
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_record_success(self):
        """Test recording success resets state."""
        cb = CircuitBreaker()
        cb.failure_count = 3
        cb.state = CircuitState.HALF_OPEN
        
        cb.record_success()
        
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_record_failure(self):
        """Test recording failure increments count."""
        cb = CircuitBreaker()
        
        cb.record_failure()
        
        assert cb.failure_count == 1
        assert cb.state == CircuitState.CLOSED

    def test_record_failure_opens_circuit(self):
        """Test recording failures opens circuit at threshold."""
        cb = CircuitBreaker(failure_threshold=3)
        
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_can_execute_closed(self):
        """Test can_execute when closed."""
        cb = CircuitBreaker()
        cb.state = CircuitState.CLOSED
        
        assert cb.can_execute() is True

    def test_can_execute_open(self):
        """Test can_execute when open."""
        cb = CircuitBreaker(recovery_timeout=10.0)
        cb.state = CircuitState.OPEN
        cb.last_failure_time = time.time()
        
        assert cb.can_execute() is False

    def test_can_execute_open_after_recovery(self):
        """Test can_execute transitions to half_open after recovery timeout."""
        cb = CircuitBreaker(recovery_timeout=0.1)
        cb.state = CircuitState.OPEN
        cb.last_failure_time = time.time() - 1.0  # 1 second ago
        
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_can_execute_half_open(self):
        """Test can_execute in half_open state."""
        cb = CircuitBreaker(half_open_max_calls=2)
        cb.state = CircuitState.HALF_OPEN
        cb.half_open_calls = 0
        
        assert cb.can_execute() is True
        assert cb.half_open_calls == 1
        
        assert cb.can_execute() is True
        assert cb.half_open_calls == 2
        
        assert cb.can_execute() is False


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_values(self):
        """Test default retry config values."""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.base_delay == 0.5
        assert config.max_delay == 10.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_get_delay(self):
        """Test get_delay calculation."""
        config = RetryConfig(base_delay=1.0, jitter=False)
        
        assert config.get_delay(0) == 1.0
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 4.0
        assert config.get_delay(3) == 8.0

    def test_get_delay_max_cap(self):
        """Test get_delay respects max_delay."""
        config = RetryConfig(base_delay=5.0, max_delay=10.0, jitter=False)
        
        # 5 * 2^3 = 40, but capped at 10
        assert config.get_delay(3) == 10.0

    def test_get_delay_with_jitter(self):
        """Test get_delay includes jitter."""
        config = RetryConfig(base_delay=1.0, jitter=True)
        
        delays = [config.get_delay(0) for _ in range(10)]
        
        # With jitter, not all delays should be the same
        assert len(set(delays)) > 1


class TestClusterNode:
    """Tests for ClusterNode."""

    def test_default_values(self):
        """Test default cluster node values."""
        node = ClusterNode(address="http://localhost:8000")
        
        assert node.address == "http://localhost:8000"
        assert node.is_leader is False
        assert node.is_healthy is True
        assert isinstance(node.circuit_breaker, CircuitBreaker)


class TestFlyBrowserClientInit:
    """Tests for FlyBrowserClient initialization."""

    def test_init_default_values(self):
        """Test default initialization values."""
        client = FlyBrowserClient("http://localhost:8000")
        
        assert client.endpoint == "http://localhost:8000"
        assert client.api_key is None
        assert client.timeout == 30.0
        assert client.auto_discover is True
        assert client._started is False

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        config = RetryConfig(max_retries=5)
        client = FlyBrowserClient(
            "http://localhost:8000",
            api_key="test-key",
            timeout=60.0,
            retry_config=config,
            auto_discover=False
        )
        
        assert client.api_key == "test-key"
        assert client.timeout == 60.0
        assert client.retry_config.max_retries == 5
        assert client.auto_discover is False

    def test_endpoint_trailing_slash_removed(self):
        """Test endpoint has trailing slash removed."""
        client = FlyBrowserClient("http://localhost:8000/")
        assert client.endpoint == "http://localhost:8000"


class TestFlyBrowserClientStartStop:
    """Tests for FlyBrowserClient start/stop."""

    @pytest.mark.asyncio
    async def test_start(self):
        """Test client start."""
        client = FlyBrowserClient("http://localhost:8000", auto_discover=False)
        
        with patch("aiohttp.ClientSession"):
            await client.start()
            
            assert client._started is True
            assert client.endpoint in client._nodes

    @pytest.mark.asyncio
    async def test_start_already_started(self):
        """Test start when already started."""
        client = FlyBrowserClient("http://localhost:8000", auto_discover=False)
        client._started = True
        
        # Should not raise
        await client.start()

    @pytest.mark.asyncio
    async def test_stop(self):
        """Test client stop."""
        client = FlyBrowserClient("http://localhost:8000")
        client._started = True
        mock_close = AsyncMock()
        mock_session = MagicMock()
        mock_session.close = mock_close
        client._session = mock_session
        
        await client.stop()
        
        assert client._started is False
        assert client._session is None
        mock_close.assert_awaited_once()


class TestFlyBrowserClientSessionAPI:
    """Tests for FlyBrowserClient session API."""

    @pytest.mark.asyncio
    async def test_create_session(self):
        """Test create_session."""
        client = FlyBrowserClient("http://localhost:8000")
        client._started = True
        
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "session_id": "sess-123",
                "node_address": "localhost:8000"
            }
            
            result = await client.create_session(llm_provider="openai")
            
            assert result["session_id"] == "sess-123"
            mock_request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_session(self):
        """Test close_session."""
        client = FlyBrowserClient("http://localhost:8000")
        client._started = True
        client._session_routes["sess-123"] = "http://localhost:8000"
        
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"success": True}
            
            result = await client.close_session("sess-123")
            
            assert result is True
            assert "sess-123" not in client._session_routes

    @pytest.mark.asyncio
    async def test_get_session(self):
        """Test get_session."""
        client = FlyBrowserClient("http://localhost:8000")
        client._started = True
        
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"session_id": "sess-123", "status": "active"}
            
            result = await client.get_session("sess-123")
            
            assert result["session_id"] == "sess-123"


class TestFlyBrowserClientNavigationAPI:
    """Tests for FlyBrowserClient navigation API."""

    @pytest.mark.asyncio
    async def test_navigate(self):
        """Test navigate."""
        client = FlyBrowserClient("http://localhost:8000")
        client._started = True
        
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"success": True, "url": "https://example.com"}
            
            result = await client.navigate("sess-123", "https://example.com")
            
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_extract(self):
        """Test extract."""
        client = FlyBrowserClient("http://localhost:8000")
        client._started = True
        
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"data": {"title": "Example"}}
            
            result = await client.extract("sess-123", "Get the title")
            
            assert result["data"]["title"] == "Example"

    @pytest.mark.asyncio
    async def test_action(self):
        """Test action."""
        client = FlyBrowserClient("http://localhost:8000")
        client._started = True
        
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"success": True}
            
            result = await client.action("sess-123", "Click button")
            
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_screenshot(self):
        """Test screenshot."""
        client = FlyBrowserClient("http://localhost:8000")
        client._started = True
        
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"data_base64": "abc123"}
            
            result = await client.screenshot("sess-123")
            
            assert "data_base64" in result


class TestFlyBrowserClientAutonomousAPI:
    """Tests for FlyBrowserClient autonomous mode API."""

    @pytest.mark.asyncio
    async def test_auto(self):
        """Test auto() method."""
        client = FlyBrowserClient("http://localhost:8000")
        client._started = True
        
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "success": True,
                "goal": "Fill out the form",
                "result_data": {"confirmation": "Form submitted"},
                "sub_goals_completed": 3,
                "total_sub_goals": 3,
                "iterations": 12,
                "duration_seconds": 45.2,
            }
            
            result = await client.auto(
                session_id="sess-123",
                goal="Fill out the form",
                context={"name": "John Doe", "email": "john@example.com"},
                max_iterations=30,
                max_time_seconds=300,
            )
            
            assert result["success"] is True
            assert result["goal"] == "Fill out the form"
            assert result["result_data"]["confirmation"] == "Form submitted"
            mock_request.assert_awaited_once()
            
            # Verify the request was made with correct parameters
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"
            assert "/sessions/sess-123/auto" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_auto_with_target_schema(self):
        """Test auto() method with target_schema for scraping."""
        client = FlyBrowserClient("http://localhost:8000")
        client._started = True
        
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "success": True,
                "result_data": [{"name": "Product", "price": 29.99}],
                "pages_scraped": 3,
                "items_extracted": 15,
            }
            
            result = await client.auto(
                session_id="sess-123",
                goal="Extract products",
                target_schema={"type": "array", "items": {"type": "object"}},
                max_pages=5,
            )
            
            assert result["success"] is True
            assert result["pages_scraped"] == 3

    @pytest.mark.asyncio
    async def test_scrape(self):
        """Test scrape() method."""
        client = FlyBrowserClient("http://localhost:8000")
        client._started = True
        
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "success": True,
                "result_data": [
                    {"name": "Widget", "price": 29.99},
                    {"name": "Gadget", "price": 49.99},
                ],
                "pages_scraped": 5,
                "items_extracted": 47,
                "validation_results": [
                    {"validator": "not_empty", "passed": True},
                    {"validator": "min_items_10", "passed": True},
                ],
                "schema_compliance": 0.98,
            }
            
            result = await client.scrape(
                session_id="sess-123",
                goal="Extract all products",
                target_schema={
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "number"},
                        },
                    },
                },
                validators=["not_empty", "min_items_10"],
                max_pages=5,
            )
            
            assert result["success"] is True
            assert result["pages_scraped"] == 5
            assert result["items_extracted"] == 47
            assert result["schema_compliance"] == 0.98
            assert len(result["validation_results"]) == 2
            mock_request.assert_awaited_once()
            
            # Verify the request was made with correct parameters
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"
            assert "/sessions/sess-123/scrape" in call_args[0][1]
            assert "validators" in call_args[1]["json"]
            assert call_args[1]["json"]["validators"] == ["not_empty", "min_items_10"]

    @pytest.mark.asyncio
    async def test_scrape_returns_empty_on_failure(self):
        """Test scrape() returns empty dict on failure."""
        client = FlyBrowserClient("http://localhost:8000")
        client._started = True
        
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = None
            
            result = await client.scrape(
                session_id="sess-123",
                goal="Extract products",
                target_schema={"type": "array"},
            )
            
            assert result == {}


class TestFlyBrowserClientClusterAPI:
    """Tests for FlyBrowserClient cluster API."""

    @pytest.mark.asyncio
    async def test_get_cluster_status(self):
        """Test get_cluster_status."""
        client = FlyBrowserClient("http://localhost:8000")
        client._started = True
        
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"is_leader": True, "node_id": "node-1"}
            
            result = await client.get_cluster_status()
            
            assert result["is_leader"] is True

    @pytest.mark.asyncio
    async def test_get_cluster_nodes(self):
        """Test get_cluster_nodes."""
        client = FlyBrowserClient("http://localhost:8000")
        client._started = True
        
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "nodes": [
                    {"node_id": "node-1", "api_address": "localhost:8000"},
                    {"node_id": "node-2", "api_address": "localhost:8001"}
                ]
            }
            
            result = await client.get_cluster_nodes()
            
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health_check."""
        client = FlyBrowserClient("http://localhost:8000")
        client._started = True
        
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"status": "healthy"}
            
            result = await client.health_check()
            
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health_check when unhealthy."""
        client = FlyBrowserClient("http://localhost:8000")
        client._started = True
        
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("Connection failed")
            
            result = await client.health_check()
            
            assert result is False


class TestFlyBrowserClientHelpers:
    """Tests for FlyBrowserClient helper methods."""

    def test_get_leader_url(self):
        """Test _get_leader_url."""
        client = FlyBrowserClient("http://localhost:8000")
        
        # Without leader set
        assert client._get_leader_url() == "http://localhost:8000"
        
        # With leader set
        client._leader_address = "http://localhost:8001"
        assert client._get_leader_url() == "http://localhost:8001"

    def test_get_session_url(self):
        """Test _get_session_url."""
        client = FlyBrowserClient("http://localhost:8000")
        client._leader_address = "http://localhost:8000"
        
        # Without route
        assert client._get_session_url("sess-123") == "http://localhost:8000"
        
        # With route
        client._session_routes["sess-123"] = "http://localhost:8001"
        assert client._get_session_url("sess-123") == "http://localhost:8001"


class TestFlyBrowserClientContextManager:
    """Tests for FlyBrowserClient async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        client = FlyBrowserClient("http://localhost:8000", auto_discover=False)
        
        with patch.object(client, "start", new_callable=AsyncMock) as mock_start:
            with patch.object(client, "stop", new_callable=AsyncMock) as mock_stop:
                async with client:
                    mock_start.assert_awaited_once()
                
                mock_stop.assert_awaited_once()
