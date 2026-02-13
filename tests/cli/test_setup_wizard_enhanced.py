# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Tests for enhanced setup wizard subcommands.

Tests for the component-specific setup functions:
- setup_quick: 30-second quick start
- setup_llm: LLM provider configuration
- setup_server: Server mode configuration
- setup_observability: Tracing/metrics configuration
- setup_security: RBAC/auth configuration
"""

from unittest.mock import patch

import pytest


class TestSetupSubcommandCallables:
    """Verify that all new setup subcommand functions exist and are callable."""

    def test_setup_quick_exists(self):
        """setup_quick should be importable and callable."""
        from flybrowser.cli.setup import setup_quick

        assert callable(setup_quick)

    def test_setup_llm_exists(self):
        """setup_llm should be importable and callable."""
        from flybrowser.cli.setup import setup_llm

        assert callable(setup_llm)

    def test_setup_server_exists(self):
        """setup_server should be importable and callable."""
        from flybrowser.cli.setup import setup_server

        assert callable(setup_server)

    def test_setup_observability_exists(self):
        """setup_observability should be importable and callable."""
        from flybrowser.cli.setup import setup_observability

        assert callable(setup_observability)

    def test_setup_security_exists(self):
        """setup_security should be importable and callable."""
        from flybrowser.cli.setup import setup_security

        assert callable(setup_security)


class TestSetupQuick:
    """Tests for the quick setup subcommand."""

    def test_setup_quick_returns_dict(self):
        """setup_quick should return a dict with a 'provider' key."""
        from flybrowser.cli.setup import setup_quick

        with patch("flybrowser.cli.setup.prompt_choice", return_value="OpenAI (GPT-5.2, GPT-4o)"):
            with patch("flybrowser.cli.setup.prompt", return_value="sk-fake-key"):
                with patch("flybrowser.cli.setup.prompt_bool", return_value=True):
                    with patch("flybrowser.cli.setup.install_browsers", return_value=True):
                        with patch("flybrowser.cli.setup.verify_installation", return_value=True):
                            result = setup_quick()

        assert isinstance(result, dict)
        assert "provider" in result

    def test_setup_quick_auto_detects_env(self):
        """setup_quick should auto-detect provider from environment variables."""
        from flybrowser.cli.setup import setup_quick

        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "sk-test-key"},
            clear=False,
        ):
            with patch("flybrowser.cli.setup.prompt_bool", return_value=True):
                with patch("flybrowser.cli.setup.install_browsers", return_value=True):
                    with patch("flybrowser.cli.setup.verify_installation", return_value=True):
                        result = setup_quick()

        assert result["provider"] == "openai"

    def test_setup_quick_installs_chromium(self):
        """setup_quick should call install_browsers with chromium."""
        from flybrowser.cli.setup import setup_quick

        with patch("flybrowser.cli.setup.prompt_choice", return_value="OpenAI (GPT-5.2, GPT-4o)"):
            with patch("flybrowser.cli.setup.prompt", return_value="sk-fake-key"):
                with patch("flybrowser.cli.setup.prompt_bool", return_value=True):
                    with patch(
                        "flybrowser.cli.setup.install_browsers", return_value=True
                    ) as mock_install:
                        with patch("flybrowser.cli.setup.verify_installation", return_value=True):
                            setup_quick()

        mock_install.assert_called_once_with(["chromium"])


class TestSetupLLM:
    """Tests for the LLM setup subcommand."""

    def test_setup_llm_returns_dict(self):
        """setup_llm should return a dict with 'llm_provider' key."""
        from flybrowser.cli.setup import setup_llm

        with patch(
            "flybrowser.cli.setup.prompt_choice",
            side_effect=["OpenAI (GPT-5.2, GPT-4o)", "gpt-4o"],
        ):
            with patch("flybrowser.cli.setup.prompt", return_value="sk-fake-key"):
                with patch("flybrowser.cli.setup.prompt_bool", return_value=False):
                    result = setup_llm()

        assert isinstance(result, dict)
        assert "llm_provider" in result

    def test_setup_llm_anthropic_provider(self):
        """setup_llm should handle Anthropic provider selection."""
        from flybrowser.cli.setup import setup_llm

        with patch(
            "flybrowser.cli.setup.prompt_choice",
            side_effect=["Anthropic (Claude)", "claude-sonnet-4-5-20250929"],
        ):
            with patch("flybrowser.cli.setup.prompt", return_value="sk-ant-fake"):
                with patch("flybrowser.cli.setup.prompt_bool", return_value=False):
                    result = setup_llm()

        assert result["llm_provider"] == "anthropic"

    def test_setup_llm_ollama_provider(self):
        """setup_llm should handle Ollama (local) provider selection."""
        from flybrowser.cli.setup import setup_llm

        with patch(
            "flybrowser.cli.setup.prompt_choice",
            side_effect=["Ollama (Local)", "llama3.2"],
        ):
            with patch(
                "flybrowser.cli.setup.prompt",
                side_effect=["http://localhost:11434", "llama3.2"],
            ):
                with patch("flybrowser.cli.setup.prompt_bool", return_value=False):
                    result = setup_llm()

        assert result["llm_provider"] == "ollama"


class TestSetupServer:
    """Tests for the server setup subcommand."""

    def test_setup_server_returns_dict(self):
        """setup_server should return a dict with 'host' and 'port' keys."""
        from flybrowser.cli.setup import setup_server

        with patch(
            "flybrowser.cli.setup.prompt",
            side_effect=["0.0.0.0", "8000", "4"],
        ):
            with patch("flybrowser.cli.setup.prompt_bool", return_value=False):
                result = setup_server()

        assert isinstance(result, dict)
        assert "host" in result
        assert "port" in result

    def test_setup_server_tls_enabled(self):
        """setup_server should include TLS options when enabled."""
        from flybrowser.cli.setup import setup_server

        with patch(
            "flybrowser.cli.setup.prompt",
            side_effect=["0.0.0.0", "443", "4", "/path/cert.pem", "/path/key.pem"],
        ):
            with patch("flybrowser.cli.setup.prompt_bool", return_value=True):
                result = setup_server()

        assert result.get("tls_enabled") is True


class TestSetupObservability:
    """Tests for the observability setup subcommand."""

    def test_setup_observability_returns_dict(self):
        """setup_observability should return a dict with 'otlp_endpoint' key."""
        from flybrowser.cli.setup import setup_observability

        with patch(
            "flybrowser.cli.setup.prompt",
            side_effect=["http://localhost:4317", "9090"],
        ):
            with patch(
                "flybrowser.cli.setup.prompt_choice",
                return_value="INFO",
            ):
                with patch("flybrowser.cli.setup.prompt_bool", return_value=True):
                    result = setup_observability()

        assert isinstance(result, dict)
        assert "otlp_endpoint" in result

    def test_setup_observability_prometheus_port(self):
        """setup_observability should include prometheus_port when metrics enabled."""
        from flybrowser.cli.setup import setup_observability

        with patch(
            "flybrowser.cli.setup.prompt",
            side_effect=["http://localhost:4317", "9090"],
        ):
            with patch(
                "flybrowser.cli.setup.prompt_choice",
                return_value="DEBUG",
            ):
                with patch("flybrowser.cli.setup.prompt_bool", return_value=True):
                    result = setup_observability()

        assert "prometheus_port" in result


class TestSetupSecurity:
    """Tests for the security setup subcommand."""

    def test_setup_security_returns_dict(self):
        """setup_security should return a dict with 'rbac_enabled' key."""
        from flybrowser.cli.setup import setup_security

        with patch("flybrowser.cli.setup.prompt_bool", return_value=True):
            with patch(
                "flybrowser.cli.setup.prompt",
                return_value="my-secret-key",
            ):
                result = setup_security()

        assert isinstance(result, dict)
        assert "rbac_enabled" in result

    def test_setup_security_rbac_disabled(self):
        """setup_security should handle RBAC disabled."""
        from flybrowser.cli.setup import setup_security

        with patch("flybrowser.cli.setup.prompt_bool", return_value=False):
            result = setup_security()

        assert result["rbac_enabled"] is False

    def test_setup_security_generates_admin_token(self):
        """setup_security should generate an admin_token when RBAC is enabled."""
        from flybrowser.cli.setup import setup_security

        with patch("flybrowser.cli.setup.prompt_bool", return_value=True):
            with patch(
                "flybrowser.cli.setup.prompt",
                return_value="my-jwt-secret",
            ):
                result = setup_security()

        assert "admin_token" in result
        assert len(result["admin_token"]) > 0


class TestCmdQuick:
    """Tests for the cmd_quick CLI handler."""

    def test_cmd_quick_exists(self):
        """cmd_quick should be importable and callable."""
        from flybrowser.cli.setup import cmd_quick

        assert callable(cmd_quick)

    def test_quick_subparser_registered(self):
        """The 'quick' subcommand should be registered in the argument parser."""
        from flybrowser.cli.setup import create_parser

        parser = create_parser()
        args = parser.parse_args(["quick"])
        assert args.command == "quick"
