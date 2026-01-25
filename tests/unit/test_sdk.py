# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FlyBrowser SDK."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flybrowser.sdk import FlyBrowser


class TestFlyBrowserInit:
    """Tests for FlyBrowser initialization."""

    def test_init_embedded_mode(self):
        """Test initialization in embedded mode."""
        browser = FlyBrowser(llm_provider="openai", api_key="test-key")
        
        assert browser.mode == "embedded"
        assert browser._endpoint is None
        assert browser._llm_provider == "openai"
        assert browser._api_key == "test-key"
        assert browser._started is False

    def test_init_server_mode(self):
        """Test initialization in server mode."""
        browser = FlyBrowser(endpoint="http://localhost:8000")
        
        assert browser.mode == "server"
        assert browser._endpoint == "http://localhost:8000"
        assert browser._started is False

    def test_init_default_values(self):
        """Test default initialization values."""
        browser = FlyBrowser()
        
        assert browser._headless is True
        assert browser._browser_type == "chromium"
        assert browser._recording_enabled is False
        assert browser._pii_masking_enabled is True
        assert browser._timeout == 30.0

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        browser = FlyBrowser(
            headless=False,
            browser_type="firefox",
            recording_enabled=True,
            timeout=60.0
        )
        
        assert browser._headless is False
        assert browser._browser_type == "firefox"
        assert browser._recording_enabled is True
        assert browser._timeout == 60.0


class TestFlyBrowserProperties:
    """Tests for FlyBrowser properties."""

    def test_mode_property(self):
        """Test mode property."""
        embedded = FlyBrowser()
        assert embedded.mode == "embedded"
        
        server = FlyBrowser(endpoint="http://localhost:8000")
        assert server.mode == "server"

    def test_session_id_property(self):
        """Test session_id property."""
        browser = FlyBrowser()
        assert browser.session_id is None
        
        browser._session_id = "test-session"
        assert browser.session_id == "test-session"

    def test_endpoint_property(self):
        """Test endpoint property."""
        embedded = FlyBrowser()
        assert embedded.endpoint is None
        
        server = FlyBrowser(endpoint="http://localhost:8000")
        assert server.endpoint == "http://localhost:8000"


class TestFlyBrowserEnsureStarted:
    """Tests for FlyBrowser._ensure_started()."""

    def test_ensure_started_raises_when_not_started(self):
        """Test _ensure_started raises when not started."""
        browser = FlyBrowser()
        
        with pytest.raises(RuntimeError, match="FlyBrowser not started"):
            browser._ensure_started()

    def test_ensure_started_ok_when_started(self):
        """Test _ensure_started passes when started."""
        browser = FlyBrowser()
        browser._started = True
        
        # Should not raise
        browser._ensure_started()


class TestFlyBrowserStartStop:
    """Tests for FlyBrowser start() and stop()."""

    @pytest.mark.asyncio
    async def test_start_server_mode(self):
        """Test start in server mode."""
        browser = FlyBrowser(endpoint="http://localhost:8000")
        
        with patch("flybrowser.client.FlyBrowserClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.create_session = AsyncMock(return_value={
                "session_id": "test-session"
            })
            mock_client_class.return_value = mock_client
            
            await browser.start()
            
            assert browser._started is True
            assert browser._session_id == "test-session"

    @pytest.mark.asyncio
    async def test_start_already_started(self):
        """Test start when already started."""
        browser = FlyBrowser()
        browser._started = True
        
        # Should not raise, just log warning
        await browser.start()

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self):
        """Test stop when not started."""
        browser = FlyBrowser()
        
        # Should not raise
        await browser.stop()

    @pytest.mark.asyncio
    async def test_stop_server_mode(self):
        """Test stop in server mode."""
        browser = FlyBrowser(endpoint="http://localhost:8000")
        browser._started = True
        browser._session_id = "test-session"
        browser._client = AsyncMock()
        browser._client.close_session = AsyncMock()
        browser._client.stop = AsyncMock()
        
        await browser.stop()
        
        assert browser._started is False
        browser._client.close_session.assert_awaited_once_with("test-session")


class TestFlyBrowserGoto:
    """Tests for FlyBrowser.goto()."""

    @pytest.mark.asyncio
    async def test_goto_requires_started(self):
        """Test goto raises when not started."""
        browser = FlyBrowser()
        
        with pytest.raises(RuntimeError, match="FlyBrowser not started"):
            await browser.goto("https://example.com")

    @pytest.mark.asyncio
    async def test_goto_server_mode(self):
        """Test goto in server mode."""
        browser = FlyBrowser(endpoint="http://localhost:8000")
        browser._started = True
        browser._session_id = "test-session"
        browser._client = AsyncMock()
        browser._client.navigate = AsyncMock(return_value={"success": True})
        
        await browser.goto("https://example.com")
        
        browser._client.navigate.assert_awaited_once_with(
            "test-session", "https://example.com"
        )

    @pytest.mark.asyncio
    async def test_goto_embedded_mode(self):
        """Test goto in embedded mode."""
        browser = FlyBrowser()
        browser._started = True
        browser.page_controller = MagicMock()
        browser.page_controller.goto = AsyncMock()
        
        await browser.goto("https://example.com")
        
        browser.page_controller.goto.assert_awaited_once()


class TestFlyBrowserExtract:
    """Tests for FlyBrowser.extract()."""

    @pytest.mark.asyncio
    async def test_extract_server_mode(self):
        """Test extract in server mode returns AgentRequestResponse by default."""
        browser = FlyBrowser(endpoint="http://localhost:8000")
        browser._started = True
        browser._session_id = "test-session"
        browser._client = AsyncMock()
        browser._client.extract = AsyncMock(return_value={"data": {"title": "Example"}})
        
        result = await browser.extract("Get the title")
        
        # Default: return_metadata=True returns AgentRequestResponse
        from flybrowser.agents.response import AgentRequestResponse
        assert isinstance(result, AgentRequestResponse)
        assert result.data == {"title": "Example"}

    @pytest.mark.asyncio
    async def test_extract_server_mode_raw(self):
        """Test extract in server mode with return_metadata=False returns raw data."""
        browser = FlyBrowser(endpoint="http://localhost:8000")
        browser._started = True
        browser._session_id = "test-session"
        browser._client = AsyncMock()
        browser._client.extract = AsyncMock(return_value={"data": {"title": "Example"}})
        
        result = await browser.extract("Get the title", return_metadata=False)
        
        assert result == {"title": "Example"}

    @pytest.mark.asyncio
    async def test_extract_embedded_mode(self):
        """Test extract in embedded mode returns AgentRequestResponse by default."""
        browser = FlyBrowser()
        browser._started = True
        browser.orchestrator = MagicMock()
        browser.orchestrator.execute_extract = AsyncMock(return_value={
            "success": True,
            "data": {"title": "Example"}
        })
        
        result = await browser.extract("Get the title")
        
        # Default: return_metadata=True returns AgentRequestResponse
        from flybrowser.agents.response import AgentRequestResponse
        assert isinstance(result, AgentRequestResponse)
        assert result.data == {"title": "Example"}
        browser.orchestrator.execute_extract.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_extract_embedded_mode_raw(self):
        """Test extract in embedded mode with return_metadata=False."""
        browser = FlyBrowser()
        browser._started = True
        browser.orchestrator = MagicMock()
        browser.orchestrator.execute_extract = AsyncMock(return_value={
            "success": True,
            "data": {"title": "Example"}
        })
        
        result = await browser.extract("Get the title", return_metadata=False)
        
        assert result == {"title": "Example"}

    @pytest.mark.asyncio
    async def test_extract_embedded_mode_direct(self):
        """Test extract in embedded mode without orchestrator."""
        browser = FlyBrowser()
        browser._started = True
        browser.extraction_agent = MagicMock()
        browser.extraction_agent.execute = AsyncMock(return_value={
            "success": True,
            "data": {"title": "Example"}
        })
        
        result = await browser.extract("Get the title", use_orchestrator=False, return_metadata=False)
        
        assert result == {"title": "Example"}
        browser.extraction_agent.execute.assert_awaited_once()


class TestFlyBrowserAct:
    """Tests for FlyBrowser.act()."""

    @pytest.mark.asyncio
    async def test_act_server_mode(self):
        """Test act in server mode returns AgentRequestResponse by default."""
        browser = FlyBrowser(endpoint="http://localhost:8000")
        browser._started = True
        browser._session_id = "test-session"
        browser._client = AsyncMock()
        browser._client.action = AsyncMock(return_value={"success": True})
        
        result = await browser.act("Click the button")
        
        # Default: return_metadata=True returns AgentRequestResponse
        from flybrowser.agents.response import AgentRequestResponse
        assert isinstance(result, AgentRequestResponse)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_act_server_mode_raw(self):
        """Test act in server mode with return_metadata=False returns raw dict."""
        browser = FlyBrowser(endpoint="http://localhost:8000")
        browser._started = True
        browser._session_id = "test-session"
        browser._client = AsyncMock()
        browser._client.action = AsyncMock(return_value={"success": True})
        
        result = await browser.act("Click the button", return_metadata=False)
        
        assert result == {"success": True}


class TestFlyBrowserScreenshot:
    """Tests for FlyBrowser.screenshot()."""

    @pytest.mark.asyncio
    async def test_screenshot_server_mode(self):
        """Test screenshot in server mode."""
        browser = FlyBrowser(endpoint="http://localhost:8000")
        browser._started = True
        browser._session_id = "test-session"
        browser._client = AsyncMock()
        browser._client.screenshot = AsyncMock(return_value={"data_base64": "abc123"})
        
        result = await browser.screenshot()
        
        assert "data_base64" in result


class TestFlyBrowserStoreCredential:
    """Tests for FlyBrowser.store_credential()."""

    def test_store_credential_server_mode_raises(self):
        """Test store_credential raises in server mode."""
        browser = FlyBrowser(endpoint="http://localhost:8000")
        
        with pytest.raises(NotImplementedError):
            browser.store_credential("password", "secret123")


class TestFlyBrowserContextManager:
    """Tests for FlyBrowser async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        browser = FlyBrowser()
        
        with patch.object(browser, "start", new_callable=AsyncMock) as mock_start:
            with patch.object(browser, "stop", new_callable=AsyncMock) as mock_stop:
                async with browser:
                    mock_start.assert_awaited_once()
                
                mock_stop.assert_awaited_once()


class TestFlyBrowserMaskPII:
    """Tests for FlyBrowser.mask_pii()."""

    def test_mask_pii_server_mode(self):
        """Test mask_pii in server mode."""
        browser = FlyBrowser(endpoint="http://localhost:8000")
        
        with patch("flybrowser.security.pii_handler.PIIMasker") as mock_masker_class:
            mock_masker = MagicMock()
            mock_masker.mask_text = MagicMock(return_value="***@***.com")
            mock_masker_class.return_value = mock_masker
            
            result = browser.mask_pii("test@example.com")
            
            assert "***" in result or result == "***@***.com"

    def test_mask_pii_embedded_mode(self):
        """Test mask_pii in embedded mode."""
        browser = FlyBrowser()
        browser.pii_handler = MagicMock()
        browser.pii_handler.mask_for_llm = MagicMock(return_value="***@***.com")
        
        result = browser.mask_pii("test@example.com")
        
        browser.pii_handler.mask_for_llm.assert_called_once()
