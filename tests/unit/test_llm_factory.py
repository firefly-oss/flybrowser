# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LLM provider factory."""

from unittest.mock import MagicMock, patch

import pytest

from flybrowser.exceptions import ConfigurationError
from flybrowser.llm.base import BaseLLMProvider
from flybrowser.llm.factory import LLMProviderFactory


class TestLLMProviderFactoryCreate:
    """Tests for LLMProviderFactory.create."""

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        with patch("flybrowser.llm.openai_provider.AsyncOpenAI"):
            provider = LLMProviderFactory.create(
                "openai",
                model="gpt-4o",
                api_key="test-key"
            )
            
            assert provider.model == "gpt-4o"
            assert provider.api_key == "test-key"

    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider."""
        with patch("flybrowser.llm.anthropic_provider.AsyncAnthropic"):
            provider = LLMProviderFactory.create(
                "anthropic",
                model="claude-3-5-sonnet-20241022",
                api_key="test-key"
            )
            
            assert provider.model == "claude-3-5-sonnet-20241022"

    def test_create_ollama_provider(self):
        """Test creating Ollama provider."""
        provider = LLMProviderFactory.create(
            "ollama",
            model="llama3.3"
        )
        
        assert provider.model == "llama3.3"

    def test_create_gemini_provider(self):
        """Test creating Gemini provider."""
        # GeminiProvider uses aiohttp, no special mocking needed for instantiation
        provider = LLMProviderFactory.create(
            "gemini",
            model="gemini-2.0-flash",
            api_key="test-key"
        )
        
        assert provider.model == "gemini-2.0-flash"
        assert provider.api_key == "test-key"

    def test_create_google_alias(self):
        """Test creating provider using 'google' alias."""
        # GeminiProvider uses aiohttp, no special mocking needed for instantiation
        provider = LLMProviderFactory.create(
            "google",
            model="gemini-2.0-flash",
            api_key="test-key"
        )
        
        assert provider.model == "gemini-2.0-flash"

    def test_create_case_insensitive(self):
        """Test provider names are case insensitive."""
        with patch("flybrowser.llm.openai_provider.AsyncOpenAI"):
            provider = LLMProviderFactory.create(
                "OpenAI",
                model="gpt-4o",
                api_key="test-key"
            )
            
            assert provider.model == "gpt-4o"

    def test_create_unsupported_provider(self):
        """Test creating unsupported provider raises error."""
        with pytest.raises(ConfigurationError, match="Unsupported LLM provider"):
            LLMProviderFactory.create("unsupported_provider")

    def test_create_with_default_model(self):
        """Test creating provider with default model."""
        with patch("flybrowser.llm.openai_provider.AsyncOpenAI"):
            provider = LLMProviderFactory.create(
                "openai",
                api_key="test-key"
            )
            
            # Should use default model
            assert provider.model is not None


class TestLLMProviderFactoryCreateFromConfig:
    """Tests for LLMProviderFactory.create_from_config."""

    def test_create_from_config(self):
        """Test creating provider from config object."""
        from flybrowser.llm.config import LLMProviderConfig, LLMProviderType
        
        config = LLMProviderConfig(
            provider_type=LLMProviderType.OPENAI,
            model="gpt-4o",
            api_key="test-key",
        )
        
        with patch("flybrowser.llm.openai_provider.AsyncOpenAI"):
            provider = LLMProviderFactory.create_from_config(config)
            
            assert provider.model == "gpt-4o"
            assert provider.api_key == "test-key"


class TestLLMProviderFactoryRegister:
    """Tests for LLMProviderFactory.register_provider."""

    def test_register_custom_provider(self):
        """Test registering a custom provider."""
        class CustomProvider(BaseLLMProvider):
            async def generate(self, prompt, **kwargs):
                pass
            
            async def generate_with_vision(self, prompt, image_data, **kwargs):
                pass
            
            async def generate_structured(self, prompt, schema, **kwargs):
                pass
        
        LLMProviderFactory.register_provider("custom", CustomProvider)
        
        assert "custom" in LLMProviderFactory._providers
        
        # Clean up
        del LLMProviderFactory._providers["custom"]

    def test_register_invalid_provider(self):
        """Test registering non-BaseLLMProvider class raises error."""
        class NotAProvider:
            pass
        
        with pytest.raises(ConfigurationError, match="must inherit from BaseLLMProvider"):
            LLMProviderFactory.register_provider("invalid", NotAProvider)


class TestLLMProviderFactoryListProviders:
    """Tests for LLMProviderFactory.list_providers."""

    def test_list_providers(self):
        """Test listing all providers."""
        providers = LLMProviderFactory.list_providers()
        
        assert "openai" in providers
        assert "anthropic" in providers
        assert "ollama" in providers
        assert "gemini" in providers
        assert "google" in providers
