# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SessionManager."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flybrowser.service.session_manager import SessionManager


class TestSessionManagerInit:
    """Tests for SessionManager initialization."""

    def test_default_values(self):
        """Test default initialization values."""
        with patch.object(SessionManager, "_start_cleanup_task"):
            manager = SessionManager()
            
            assert manager.max_sessions == 100
            assert manager.session_timeout == 3600
            assert manager.sessions == {}
            assert manager.session_metadata == {}
            assert manager._total_requests == 0

    def test_custom_values(self):
        """Test initialization with custom values."""
        with patch.object(SessionManager, "_start_cleanup_task"):
            manager = SessionManager(max_sessions=50, session_timeout=1800)
            
            assert manager.max_sessions == 50
            assert manager.session_timeout == 1800


class TestSessionManagerCreateSession:
    """Tests for SessionManager.create_session."""

    @pytest.mark.asyncio
    async def test_create_session(self):
        """Test creating a session."""
        with patch.object(SessionManager, "_start_cleanup_task"):
            manager = SessionManager()
        
        mock_browser = MagicMock()
        mock_browser.start = AsyncMock()
        
        with patch("flybrowser.service.session_manager.FlyBrowser", return_value=mock_browser):
            session_id = await manager.create_session(
                llm_provider="openai",
                llm_model="gpt-4o",
                headless=True
            )
            
            assert session_id is not None
            assert session_id in manager.sessions
            assert session_id in manager.session_metadata
            
            metadata = manager.session_metadata[session_id]
            assert metadata["llm_provider"] == "openai"
            assert metadata["llm_model"] == "gpt-4o"
            assert "created_at" in metadata
            assert "last_activity" in metadata
            
            mock_browser.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_session_with_custom_id(self):
        """Test creating a session with custom ID."""
        with patch.object(SessionManager, "_start_cleanup_task"):
            manager = SessionManager()
        
        mock_browser = MagicMock()
        mock_browser.start = AsyncMock()
        
        with patch("flybrowser.service.session_manager.FlyBrowser", return_value=mock_browser):
            session_id = await manager.create_session(
                llm_provider="openai",
                session_id="custom-session-123"
            )
            
            assert session_id == "custom-session-123"
            assert "custom-session-123" in manager.sessions

    @pytest.mark.asyncio
    async def test_create_session_max_reached(self):
        """Test create_session raises when max sessions reached."""
        with patch.object(SessionManager, "_start_cleanup_task"):
            manager = SessionManager(max_sessions=1)
        
        # Add a fake session to hit the limit
        manager.sessions["existing"] = MagicMock()
        
        with pytest.raises(RuntimeError, match="Maximum sessions"):
            await manager.create_session(llm_provider="openai")


class TestSessionManagerGetSession:
    """Tests for SessionManager.get_session."""

    def test_get_session(self):
        """Test getting a session."""
        with patch.object(SessionManager, "_start_cleanup_task"):
            manager = SessionManager()
        
        mock_browser = MagicMock()
        manager.sessions["sess-123"] = mock_browser
        manager.session_metadata["sess-123"] = {
            "created_at": time.time(),
            "last_activity": time.time() - 100,
        }
        
        result = manager.get_session("sess-123")
        
        assert result is mock_browser
        # Check last_activity was updated
        assert manager.session_metadata["sess-123"]["last_activity"] > time.time() - 1
        assert manager._total_requests == 1

    def test_get_session_not_found(self):
        """Test get_session raises for unknown session."""
        with patch.object(SessionManager, "_start_cleanup_task"):
            manager = SessionManager()
        
        with pytest.raises(KeyError, match="Session not found"):
            manager.get_session("unknown-session")


class TestSessionManagerDeleteSession:
    """Tests for SessionManager.delete_session."""

    @pytest.mark.asyncio
    async def test_delete_session(self):
        """Test deleting a session."""
        with patch.object(SessionManager, "_start_cleanup_task"):
            manager = SessionManager()
        
        mock_browser = MagicMock()
        mock_browser.stop = AsyncMock()
        manager.sessions["sess-123"] = mock_browser
        manager.session_metadata["sess-123"] = {"created_at": time.time()}
        
        await manager.delete_session("sess-123")
        
        assert "sess-123" not in manager.sessions
        assert "sess-123" not in manager.session_metadata
        mock_browser.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self):
        """Test delete_session raises for unknown session."""
        with patch.object(SessionManager, "_start_cleanup_task"):
            manager = SessionManager()
        
        with pytest.raises(KeyError, match="Session not found"):
            await manager.delete_session("unknown-session")

    @pytest.mark.asyncio
    async def test_delete_session_stop_error(self):
        """Test delete_session handles browser stop errors."""
        with patch.object(SessionManager, "_start_cleanup_task"):
            manager = SessionManager()
        
        mock_browser = MagicMock()
        mock_browser.stop = AsyncMock(side_effect=Exception("Stop failed"))
        manager.sessions["sess-123"] = mock_browser
        manager.session_metadata["sess-123"] = {"created_at": time.time()}
        
        # Should not raise
        await manager.delete_session("sess-123")
        
        assert "sess-123" not in manager.sessions


class TestSessionManagerCleanup:
    """Tests for SessionManager cleanup methods."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self):
        """Test cleanup of expired sessions."""
        with patch.object(SessionManager, "_start_cleanup_task"):
            manager = SessionManager(session_timeout=60)
        
        mock_browser = MagicMock()
        mock_browser.stop = AsyncMock()
        
        # Active session
        manager.sessions["active"] = mock_browser
        manager.session_metadata["active"] = {
            "created_at": time.time(),
            "last_activity": time.time(),
        }
        
        # Expired session
        manager.sessions["expired"] = mock_browser
        manager.session_metadata["expired"] = {
            "created_at": time.time() - 120,
            "last_activity": time.time() - 120,  # 2 minutes ago
        }
        
        await manager._cleanup_expired_sessions()
        
        assert "active" in manager.sessions
        assert "expired" not in manager.sessions

    @pytest.mark.asyncio
    async def test_cleanup_all(self):
        """Test cleanup_all method."""
        with patch.object(SessionManager, "_start_cleanup_task"):
            manager = SessionManager()
        
        mock_browser = MagicMock()
        mock_browser.stop = AsyncMock()
        
        manager.sessions["sess-1"] = mock_browser
        manager.sessions["sess-2"] = mock_browser
        manager.session_metadata["sess-1"] = {"created_at": time.time()}
        manager.session_metadata["sess-2"] = {"created_at": time.time()}
        
        manager._cleanup_task = MagicMock()
        
        await manager.cleanup_all()
        
        assert len(manager.sessions) == 0
        assert len(manager.session_metadata) == 0
        manager._cleanup_task.cancel.assert_called_once()


class TestSessionManagerStats:
    """Tests for SessionManager statistics methods."""

    def test_get_active_session_count(self):
        """Test get_active_session_count."""
        with patch.object(SessionManager, "_start_cleanup_task"):
            manager = SessionManager()
        
        assert manager.get_active_session_count() == 0
        
        manager.sessions["sess-1"] = MagicMock()
        manager.sessions["sess-2"] = MagicMock()
        
        assert manager.get_active_session_count() == 2

    def test_get_stats(self):
        """Test get_stats method."""
        with patch.object(SessionManager, "_start_cleanup_task"):
            manager = SessionManager(max_sessions=50, session_timeout=1800)
        
        manager.sessions["sess-1"] = MagicMock()
        manager._total_requests = 10
        
        stats = manager.get_stats()
        
        assert stats["active_sessions"] == 1
        assert stats["max_sessions"] == 50
        assert stats["total_requests"] == 10
        assert stats["session_timeout"] == 1800
