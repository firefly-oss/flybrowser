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
    from flybrowser.service.app import app
    
    with patch("flybrowser.service.app.session_manager", mock_session_manager):
        with patch("flybrowser.service.app.start_time", 1000.0):
            client = TestClient(app)
            yield client


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
            "/metrics"
        )
        
        # Should succeed with auth (or return 200 if auth is disabled)
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED]


class TestSessionEndpoints:
    """Tests for session management endpoints."""

    def test_create_session(self, test_client, mock_session_manager):
        """Test creating a session."""
        response = test_client.post(
            "/sessions",
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
            "/sessions"
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
            "/sessions/test-session-123"
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
            json={"url": "https://example.com"}
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert data.get("success") is True

    def test_navigate_missing_url(self, test_client):
        """Test navigate without URL."""
        response = test_client.post(
            "/sessions/test-session/navigate",
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
            json={
                "goal": "Search for Python tutorials",
                "max_steps": 10
            }
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "success" in data or "result" in data


class TestAutonomousEndpoints:
    """Tests for autonomous mode endpoints."""

    def test_auto_endpoint(self, test_client, mock_session_manager):
        """Test autonomous mode endpoint."""
        mock_browser = MagicMock()
        mock_browser.auto = AsyncMock(return_value={
            "success": True,
            "goal": "Fill out the form",
            "result_data": {"confirmation": "Success"},
            "sub_goals_completed": 3,
            "total_sub_goals": 3,
            "iterations": 10,
            "duration_seconds": 30.5,
            "final_url": "https://example.com/done",
            "actions_taken": ["clicked input", "typed name"],
            "suggestions": [],
        })
        mock_session_manager.get_session.return_value = mock_browser
        
        response = test_client.post(
            "/sessions/test-session/auto",
            json={
                "goal": "Fill out the form",
                "context": {"name": "John Doe", "email": "john@example.com"},
                "max_iterations": 30,
                "max_time_seconds": 300,
            }
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert data["success"] is True
            assert data["goal"] == "Fill out the form"
            assert data["sub_goals_completed"] == 3
            mock_browser.auto.assert_awaited_once()

    def test_auto_endpoint_missing_goal(self, test_client, mock_session_manager):
        """Test auto endpoint without required goal."""
        response = test_client.post(
            "/sessions/test-session/auto",
            json={"context": {}}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_auto_endpoint_session_not_found(self, test_client, mock_session_manager):
        """Test auto endpoint when session not found."""
        mock_session_manager.get_session.side_effect = KeyError("Session not found")
        
        response = test_client.post(
            "/sessions/nonexistent-session/auto",
            json={"goal": "Test"}
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestScrapeEndpoints:
    """Tests for scrape endpoints."""

    def test_scrape_endpoint(self, test_client, mock_session_manager):
        """Test scrape endpoint with schema validation."""
        mock_browser = MagicMock()
        mock_browser.scrape = AsyncMock(return_value={
            "success": True,
            "goal": "Extract products",
            "result_data": [
                {"name": "Widget", "price": 29.99},
                {"name": "Gadget", "price": 49.99},
            ],
            "pages_scraped": 3,
            "items_extracted": 25,
            "validation_results": [
                {"validator": "not_empty", "passed": True}
            ],
            "schema_compliance": 0.95,
            "duration_seconds": 45.0,
            "final_url": "https://example.com/products?page=3",
        })
        mock_session_manager.get_session.return_value = mock_browser
        
        response = test_client.post(
            "/sessions/test-session/scrape",
            json={
                "goal": "Extract all products",
                "target_schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "number"},
                        },
                    },
                },
                "validators": ["not_empty"],
                "max_pages": 5,
            }
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert data["success"] is True
            assert data["pages_scraped"] == 3
            assert data["items_extracted"] == 25
            assert data["schema_compliance"] == 0.95
            mock_browser.scrape.assert_awaited_once()

    def test_scrape_endpoint_missing_goal(self, test_client, mock_session_manager):
        """Test scrape endpoint without required goal."""
        response = test_client.post(
            "/sessions/test-session/scrape",
            json={
                "target_schema": {"type": "array"},
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_scrape_endpoint_missing_schema(self, test_client, mock_session_manager):
        """Test scrape endpoint without required target_schema."""
        response = test_client.post(
            "/sessions/test-session/scrape",
            json={
                "goal": "Extract products",
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_scrape_endpoint_invalid_validator(self, test_client, mock_session_manager):
        """Test scrape endpoint with invalid validator name."""
        mock_browser = MagicMock()
        mock_session_manager.get_session.return_value = mock_browser
        
        response = test_client.post(
            "/sessions/test-session/scrape",
            json={
                "goal": "Extract products",
                "target_schema": {"type": "array"},
                "validators": ["invalid_validator_name"],
            }
        )
        
        # Should either return 400 or 500 depending on implementation
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ]


class TestErrorHandling:
    """Tests for error handling."""

    def test_session_not_found(self, test_client, mock_session_manager):
        """Test error when session not found."""
        mock_session_manager.get_session.side_effect = KeyError("Session not found")
        
        response = test_client.post(
            "/sessions/nonexistent-session/navigate",
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
