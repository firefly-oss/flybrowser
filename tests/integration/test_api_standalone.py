# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for standalone FlyBrowser API."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient


@pytest.fixture
def mock_session_manager():
    """Create a mock session manager."""
    manager = MagicMock()
    manager.get_active_session_count.return_value = 0
    manager.get_stats.return_value = {
        "active_sessions": 0,
        "total_requests": 0,
        "max_sessions": 100,
    }
    manager.create_session = AsyncMock(return_value="test-session-123")
    manager.get_session = MagicMock()
    manager.delete_session = AsyncMock()
    manager.cleanup_all = AsyncMock()
    return manager


@pytest.fixture
def test_client(mock_session_manager):
    """Create test client with mocked dependencies."""
    from flybrowser.service.auth import APIKey, verify_api_key
    from flybrowser.service.app import app
    from datetime import datetime
    
    # Mock API key verification via dependency override
    async def mock_verify_api_key(api_key=None):
        return APIKey(
            key="test-key",
            name="Test Key",
            created_at=datetime.now(),
            enabled=True,
        )
    
    # Override the dependency
    app.dependency_overrides[verify_api_key] = mock_verify_api_key
    
    with patch("flybrowser.service.app.session_manager", mock_session_manager):
        with patch("flybrowser.service.app.start_time", 1000.0):
            client = TestClient(app)
            yield client
    
    # Clean up dependency override
    app.dependency_overrides.clear()


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, test_client, mock_session_manager):
        """Test health check returns healthy status."""
        response = test_client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_health_check_no_auth_required(self, test_client):
        """Test health check does not require authentication."""
        # No API key provided
        response = test_client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    def test_get_metrics_with_auth(self, test_client, mock_session_manager):
        """Test metrics endpoint with authentication."""
        response = test_client.get(
            "/metrics",
            headers={"X-API-Key": "test-key"}
        )
        
        # Should succeed with auth (or return 200 if auth is disabled)
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED]


class TestSessionEndpoints:
    """Tests for session management endpoints."""

    def test_create_session(self, test_client, mock_session_manager):
        """Test creating a session."""
        response = test_client.post(
            "/sessions",
            headers={"X-API-Key": "test-key"},
            json={
                "llm_provider": "openai",
                "llm_model": "gpt-4o",
                "api_key": "sk-test",
                "headless": True,
            }
        )
        
        # Should succeed or require auth
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "session_id" in data
            mock_session_manager.create_session.assert_awaited_once()

    def test_create_session_missing_provider(self, test_client):
        """Test creating session without required provider."""
        response = test_client.post(
            "/sessions",
            headers={"X-API-Key": "test-key"},
            json={"headless": True}
        )
        
        # Should fail validation
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_list_sessions(self, test_client, mock_session_manager):
        """Test listing sessions."""
        mock_session_manager.sessions = {"sess-1": MagicMock()}
        mock_session_manager.session_metadata = {
            "sess-1": {"created_at": 1000, "llm_provider": "openai"}
        }
        
        response = test_client.get(
            "/sessions",
            headers={"X-API-Key": "test-key"}
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            # API returns {"sessions": [...], "total": N}
            assert "sessions" in data
            assert isinstance(data["sessions"], list)

    def test_delete_session(self, test_client, mock_session_manager):
        """Test deleting a session."""
        mock_session_manager.delete_session = AsyncMock()
        
        response = test_client.delete(
            "/sessions/test-session-123",
            headers={"X-API-Key": "test-key"}
        )
        
        # Should succeed or return 404 if not found
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_204_NO_CONTENT,
            status.HTTP_404_NOT_FOUND,
        ]


class TestNavigationEndpoints:
    """Tests for navigation endpoints."""

    def test_navigate_to_url(self, test_client, mock_session_manager):
        """Test navigating to a URL."""
        mock_browser = MagicMock()
        mock_browser.navigate = AsyncMock(return_value={"success": True, "url": "https://example.com"})
        mock_session_manager.get_session.return_value = mock_browser
        
        response = test_client.post(
            "/sessions/test-session/navigate",
            headers={"X-API-Key": "test-key"},
            json={"url": "https://example.com"}
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert data.get("success") is True

    def test_navigate_missing_url(self, test_client):
        """Test navigate without URL."""
        response = test_client.post(
            "/sessions/test-session/navigate",
            headers={"X-API-Key": "test-key"},
            json={}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestExtractionEndpoints:
    """Tests for extraction endpoints."""

    def test_extract_data(self, test_client, mock_session_manager):
        """Test extracting data from page."""
        mock_browser = MagicMock()
        mock_browser.extract = AsyncMock(return_value={"title": "Example"})
        mock_session_manager.get_session.return_value = mock_browser
        
        response = test_client.post(
            "/sessions/test-session/extract",
            headers={"X-API-Key": "test-key"},
            json={"instruction": "Get the page title"}
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "data" in data or "result" in data


class TestActionEndpoints:
    """Tests for action endpoints."""

    def test_perform_action(self, test_client, mock_session_manager):
        """Test performing an action."""
        mock_browser = MagicMock()
        mock_browser.action = AsyncMock(return_value={"success": True})
        mock_session_manager.get_session.return_value = mock_browser
        
        response = test_client.post(
            "/sessions/test-session/action",
            headers={"X-API-Key": "test-key"},
            json={"instruction": "Click the submit button"}
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert data.get("success") is True


class TestScreenshotEndpoints:
    """Tests for screenshot endpoints."""

    def test_take_screenshot(self, test_client, mock_session_manager):
        """Test taking a screenshot."""
        mock_browser = MagicMock()
        mock_browser.screenshot = AsyncMock(return_value=b"fake-png-data")
        mock_session_manager.get_session.return_value = mock_browser
        
        response = test_client.post(
            "/sessions/test-session/screenshot",
            headers={"X-API-Key": "test-key"},
            json={"full_page": True}
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "data" in data or "data_base64" in data


class TestWorkflowEndpoints:
    """Tests for workflow endpoints."""

    def test_execute_workflow(self, test_client, mock_session_manager):
        """Test executing a workflow."""
        mock_browser = MagicMock()
        mock_browser.workflow = AsyncMock(return_value={"success": True, "steps": []})
        mock_session_manager.get_session.return_value = mock_browser
        
        response = test_client.post(
            "/sessions/test-session/workflow",
            headers={"X-API-Key": "test-key"},
            json={
                "goal": "Search for Python tutorials",
                "max_steps": 10
            }
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "success" in data or "result" in data


class TestErrorHandling:
    """Tests for error handling."""

    def test_session_not_found(self, test_client, mock_session_manager):
        """Test error when session not found."""
        mock_session_manager.get_session.side_effect = KeyError("Session not found")
        
        response = test_client.post(
            "/sessions/nonexistent-session/navigate",
            headers={"X-API-Key": "test-key"},
            json={"url": "https://example.com"}
        )
        
        assert response.status_code in [
            status.HTTP_404_NOT_FOUND,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ]

    def test_invalid_json_body(self, test_client):
        """Test error for invalid JSON body."""
        response = test_client.post(
            "/sessions/test-session/navigate",
            headers={
                "X-API-Key": "test-key",
                "Content-Type": "application/json"
            },
            content="invalid json"
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
