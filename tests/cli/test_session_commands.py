# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FlyBrowser session management CLI commands."""

import argparse
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSessionCreate:
    """Tests for the session create command."""

    def test_creates_session_embedded(self, capsys):
        """Test creating a session in embedded mode (no endpoint)."""
        from flybrowser.cli.session import cmd_session_create

        args = argparse.Namespace(
            provider="openai",
            model="gpt-4o",
            headless=True,
            name="test-session",
            api_key="sk-test-key",
            endpoint=None,
        )

        mock_result = {
            "session_id": "sess-abc123",
            "provider": "openai",
            "model": "gpt-4o",
            "status": "active",
        }

        with patch(
            "flybrowser.cli.session.create_session_embedded",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_create:
            result = cmd_session_create(args)

            mock_create.assert_called_once_with(
                provider="openai",
                model="gpt-4o",
                headless=True,
                name="test-session",
                api_key="sk-test-key",
            )
            assert result == 0

        captured = capsys.readouterr()
        assert "sess-abc123" in captured.out

    def test_creates_session_server(self, capsys):
        """Test creating a session via server endpoint."""
        from flybrowser.cli.session import cmd_session_create

        args = argparse.Namespace(
            provider="openai",
            model="gpt-4o",
            headless=True,
            name="test-session",
            api_key=None,
            endpoint="http://localhost:8000",
        )

        mock_result = {
            "session_id": "sess-server-123",
            "provider": "openai",
            "model": "gpt-4o",
            "status": "active",
        }

        with patch(
            "flybrowser.cli.session.create_session_server",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_create:
            result = cmd_session_create(args)

            mock_create.assert_called_once_with(
                endpoint="http://localhost:8000",
                provider="openai",
                model="gpt-4o",
                headless=True,
                name="test-session",
                api_key=None,
            )
            assert result == 0

        captured = capsys.readouterr()
        assert "sess-server-123" in captured.out


class TestSessionList:
    """Tests for the session list command."""

    def test_lists_sessions_table_format(self, capsys):
        """Test listing sessions in table format."""
        from flybrowser.cli.session import cmd_session_list

        args = argparse.Namespace(
            format="table",
            status="active",
            endpoint="http://localhost:8000",
        )

        mock_sessions = [
            {
                "session_id": "sess-001",
                "status": "active",
                "provider": "openai",
                "model": "gpt-4o",
                "created_at": "2026-01-01T00:00:00Z",
            },
            {
                "session_id": "sess-002",
                "status": "active",
                "provider": "anthropic",
                "model": "claude-3",
                "created_at": "2026-01-01T01:00:00Z",
            },
        ]

        with patch(
            "flybrowser.cli.session.list_sessions",
            return_value=mock_sessions,
        ) as mock_list:
            result = cmd_session_list(args)

            mock_list.assert_called_once_with(
                endpoint="http://localhost:8000",
                status="active",
            )
            assert result == 0

        captured = capsys.readouterr()
        assert "sess-001" in captured.out
        assert "sess-002" in captured.out

    def test_lists_sessions_json_format(self, capsys):
        """Test listing sessions in JSON format."""
        from flybrowser.cli.session import cmd_session_list

        args = argparse.Namespace(
            format="json",
            status="active",
            endpoint="http://localhost:8000",
        )

        mock_sessions = [
            {
                "session_id": "sess-001",
                "status": "active",
                "provider": "openai",
                "model": "gpt-4o",
                "created_at": "2026-01-01T00:00:00Z",
            },
        ]

        with patch(
            "flybrowser.cli.session.list_sessions",
            return_value=mock_sessions,
        ) as mock_list:
            result = cmd_session_list(args)

            mock_list.assert_called_once_with(
                endpoint="http://localhost:8000",
                status="active",
            )
            assert result == 0

        captured = capsys.readouterr()
        # Should be valid JSON
        data = json.loads(captured.out)
        assert "sessions" in data
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["session_id"] == "sess-001"


class TestSessionExec:
    """Tests for the session exec command."""

    def test_executes_command_on_session(self, capsys):
        """Test executing a command on a session."""
        from flybrowser.cli.session import cmd_session_exec

        args = argparse.Namespace(
            session_id="sess-001",
            command="navigate to https://example.com",
            endpoint="http://localhost:8000",
        )

        mock_result = {
            "success": True,
            "result": "Navigated to https://example.com",
        }

        with patch(
            "flybrowser.cli.session.exec_on_session",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_exec:
            result = cmd_session_exec(args)

            mock_exec.assert_called_once_with(
                session_id="sess-001",
                command="navigate to https://example.com",
                endpoint="http://localhost:8000",
            )
            assert result == 0

        captured = capsys.readouterr()
        assert "success" in captured.out


def _mock_run_async(return_value: Any = None):
    """Create a mock for _run_async that properly closes passed coroutines.

    This prevents 'coroutine was never awaited' warnings in tests.
    """
    def side_effect(coro):
        # Close the coroutine to prevent RuntimeWarning
        coro.close()
        return return_value

    return MagicMock(side_effect=side_effect)


class TestSessionClose:
    """Tests for the session close command."""

    def test_close_session(self, capsys):
        """Test closing a session via server endpoint."""
        from flybrowser.cli.session import cmd_session_close

        args = argparse.Namespace(
            session_id="sess-001",
            endpoint="http://localhost:8000",
        )

        with patch(
            "flybrowser.cli.session._run_async",
            _mock_run_async(return_value=True),
        ):
            result = cmd_session_close(args)

            assert result == 0

        captured = capsys.readouterr()
        assert "sess-001" in captured.out

    def test_close_embedded_session(self, capsys):
        """Test closing an embedded session."""
        from flybrowser.cli.session import cmd_session_close, _session_store

        mock_browser = AsyncMock()
        _session_store["embedded-test123"] = {
            "browser": mock_browser,
            "name": "test",
            "status": "active",
        }

        args = argparse.Namespace(
            session_id="embedded-test123",
            endpoint=None,
        )

        with patch(
            "flybrowser.cli.session._run_async",
            _mock_run_async(return_value=None),
        ):
            result = cmd_session_close(args)
            assert result == 0

        captured = capsys.readouterr()
        assert "embedded-test123" in captured.out
        assert "embedded-test123" not in _session_store


class TestSessionCloseAll:
    """Tests for the session close-all command."""

    def test_close_all_sessions(self, capsys):
        """Test closing all sessions via server endpoint."""
        from flybrowser.cli.session import cmd_session_close_all

        args = argparse.Namespace(
            force=True,
            endpoint="http://localhost:8000",
        )

        with patch(
            "flybrowser.cli.session._run_async",
            _mock_run_async(return_value={"closed": 3}),
        ):
            result = cmd_session_close_all(args)

            assert result == 0

        captured = capsys.readouterr()
        assert "3" in captured.out or "closed" in captured.out.lower()

    def test_close_all_embedded_sessions(self, capsys):
        """Test closing all embedded sessions."""
        from flybrowser.cli.session import cmd_session_close_all, _session_store

        mock_browser1 = AsyncMock()
        mock_browser2 = AsyncMock()
        _session_store["embedded-aaa"] = {
            "browser": mock_browser1,
            "name": "s1",
            "status": "active",
        }
        _session_store["embedded-bbb"] = {
            "browser": mock_browser2,
            "name": "s2",
            "status": "active",
        }

        args = argparse.Namespace(
            force=True,
            endpoint=None,
        )

        with patch(
            "flybrowser.cli.session._run_async",
            _mock_run_async(return_value=None),
        ):
            result = cmd_session_close_all(args)
            assert result == 0

        captured = capsys.readouterr()
        assert "2" in captured.out
        assert len(_session_store) == 0
