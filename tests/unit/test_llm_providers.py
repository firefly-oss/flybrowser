# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LLM providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flybrowser.exceptions import LLMProviderError
from flybrowser.llm.base import (
    BaseLLMProvider,
    ImageInput,
    LLMResponse,
    ModelCapability,
    ModelInfo,
    ToolCall,
    ToolDefinition,
)


class TestImageInput:
    """Tests for ImageInput."""

    def test_from_bytes(self):
        """Test creating ImageInput from bytes."""
        data = b"test image data"
        img = ImageInput.from_bytes(data, media_type="image/jpeg")
        
        assert img.data == data
        assert img.media_type == "image/jpeg"
        assert img.source_type == "bytes"

    def test_from_base64(self):
        """Test creating ImageInput from base64."""
        data = "dGVzdCBpbWFnZSBkYXRh"
        img = ImageInput.from_base64(data, media_type="image/png")
        
        assert img.data == data
        assert img.media_type == "image/png"
        assert img.source_type == "base64"

    def test_from_url(self):
        """Test creating ImageInput from URL."""
        url = "https://example.com/image.png"
        img = ImageInput.from_url(url)
        
        assert img.data == url
        assert img.media_type == "url"
        assert img.source_type == "url"


class TestToolDefinition:
    """Tests for ToolDefinition."""

    def test_tool_definition(self):
        """Test ToolDefinition creation."""
        tool = ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters={"type": "object", "properties": {"location": {"type": "string"}}},
            required=["location"]
        )
        
        assert tool.name == "get_weather"
        assert tool.description == "Get weather for a location"
        assert "location" in tool.parameters["properties"]
        assert tool.required == ["location"]


class TestToolCall:
    """Tests for ToolCall."""

    def test_tool_call(self):
        """Test ToolCall creation."""
        call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"location": "Paris"}
        )
        
        assert call.id == "call_123"
        assert call.name == "get_weather"
        assert call.arguments["location"] == "Paris"


class TestModelInfo:
    """Tests for ModelInfo."""

    def test_model_info_defaults(self):
        """Test ModelInfo with defaults."""
        info = ModelInfo(name="test-model", provider="test")
        
        assert info.name == "test-model"
        assert info.provider == "test"
        assert info.context_window == 128000
        assert info.max_output_tokens == 4096
        assert info.supports_system_prompt is True

    def test_model_info_with_capabilities(self):
        """Test ModelInfo with capabilities."""
        info = ModelInfo(
            name="test-model",
            provider="test",
            capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.VISION]
        )
        
        assert ModelCapability.TEXT_GENERATION in info.capabilities
        assert ModelCapability.VISION in info.capabilities


class TestLLMResponse:
    """Tests for LLMResponse."""

    def test_llm_response(self):
        """Test LLMResponse creation."""
        response = LLMResponse(
            content="Hello, world!",
            model="gpt-4o",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        
        assert response.content == "Hello, world!"
        assert response.model == "gpt-4o"
        assert response.usage["total_tokens"] == 15
        assert response.cached is False

    def test_llm_response_with_tool_calls(self):
        """Test LLMResponse with tool calls."""
        response = LLMResponse(
            content="",
            model="gpt-4o",
            tool_calls=[ToolCall(id="1", name="test", arguments={"arg": "value"})],
            finish_reason="tool_calls"
        )
        
        assert len(response.tool_calls) == 1
        assert response.finish_reason == "tool_calls"


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    @pytest.mark.asyncio
    async def test_generate(self):
        """Test OpenAI generate."""
        with patch("flybrowser.llm.openai_provider.AsyncOpenAI") as mock_client:
            from flybrowser.llm.openai_provider import OpenAIProvider
            
            # Setup mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
            mock_response.model = "gpt-4o"
            mock_response.usage = MagicMock(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            )
            
            mock_client.return_value.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            
            provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
            response = await provider.generate("Hi", system_prompt="Be helpful")
            
            assert response.content == "Hello!"
            assert response.model == "gpt-4o"
            assert response.usage["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_generate_error(self):
        """Test OpenAI generate handles errors."""
        with patch("flybrowser.llm.openai_provider.AsyncOpenAI") as mock_client:
            from flybrowser.llm.openai_provider import OpenAIProvider
            
            mock_client.return_value.chat.completions.create = AsyncMock(
                side_effect=Exception("API Error")
            )
            
            provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
            
            with pytest.raises(LLMProviderError, match="OpenAI generation failed"):
                await provider.generate("Hi")

    def test_get_model_info(self):
        """Test OpenAI get_model_info."""
        with patch("flybrowser.llm.openai_provider.AsyncOpenAI"):
            from flybrowser.llm.openai_provider import OpenAIProvider
            
            provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
            info = provider.get_model_info()
            
            assert info.name == "gpt-4o"
            assert info.provider == "openai"
            assert ModelCapability.TEXT_GENERATION in info.capabilities


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    @pytest.mark.asyncio
    async def test_generate(self):
        """Test Anthropic generate."""
        with patch("flybrowser.llm.anthropic_provider.AsyncAnthropic") as mock_client:
            from flybrowser.llm.anthropic_provider import AnthropicProvider
            
            # Setup mock response
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Hello from Claude!")]
            mock_response.model = "claude-3-5-sonnet-20241022"
            mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
            
            mock_client.return_value.messages.create = AsyncMock(
                return_value=mock_response
            )
            
            provider = AnthropicProvider(
                model="claude-3-5-sonnet-20241022",
                api_key="test-key"
            )
            response = await provider.generate("Hi", system_prompt="Be helpful")
            
            assert response.content == "Hello from Claude!"
            assert response.model == "claude-3-5-sonnet-20241022"
            assert response.usage["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_generate_error(self):
        """Test Anthropic generate handles errors."""
        with patch("flybrowser.llm.anthropic_provider.AsyncAnthropic") as mock_client:
            from flybrowser.llm.anthropic_provider import AnthropicProvider
            
            mock_client.return_value.messages.create = AsyncMock(
                side_effect=Exception("API Error")
            )
            
            provider = AnthropicProvider(
                model="claude-3-5-sonnet-20241022",
                api_key="test-key"
            )
            
            with pytest.raises(LLMProviderError, match="Anthropic generation failed"):
                await provider.generate("Hi")

    def test_get_model_info(self):
        """Test Anthropic get_model_info."""
        with patch("flybrowser.llm.anthropic_provider.AsyncAnthropic"):
            from flybrowser.llm.anthropic_provider import AnthropicProvider
            
            provider = AnthropicProvider(
                model="claude-3-5-sonnet-20241022",
                api_key="test-key"
            )
            info = provider.get_model_info()
            
            assert info.name == "claude-3-5-sonnet-20241022"
            assert info.provider == "anthropic"


class TestOllamaProvider:
    """Tests for OllamaProvider."""

    @pytest.mark.asyncio
    async def test_generate(self):
        """Test Ollama generate."""
        from flybrowser.llm.ollama_provider import OllamaProvider
        
        provider = OllamaProvider(model="llama3.3")
        
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "response": "Hello from Ollama!",
                "model": "llama3.3",
                "prompt_eval_count": 10,
                "eval_count": 5,
            })
            
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session.return_value)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value.post = MagicMock(return_value=mock_response)
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            
            response = await provider.generate("Hi")
            
            assert response.content == "Hello from Ollama!"

    def test_get_model_info(self):
        """Test Ollama get_model_info."""
        from flybrowser.llm.ollama_provider import OllamaProvider
        
        provider = OllamaProvider(model="llama3.3")
        info = provider.get_model_info()
        
        assert info.name == "llama3.3"
        assert info.provider == "ollama"


class TestGeminiProvider:
    """Tests for GeminiProvider."""

    @pytest.mark.asyncio
    async def test_generate(self):
        """Test Gemini generate."""
        from flybrowser.llm.gemini_provider import GeminiProvider
        
        provider = GeminiProvider(model="gemini-2.0-flash", api_key="test-key")
        
        # Mock the aiohttp session
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Hello from Gemini!"}]
                    },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15
                }
            })
            
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            
            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(return_value=mock_context)
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            
            response = await provider.generate("Hi")
            
            assert response.content == "Hello from Gemini!"
            assert response.usage["total_tokens"] == 15

    def test_get_model_info(self):
        """Test Gemini get_model_info."""
        from flybrowser.llm.gemini_provider import GeminiProvider
        
        provider = GeminiProvider(model="gemini-2.0-flash", api_key="test-key")
        info = provider.get_model_info()
        
        assert info.name == "gemini-2.0-flash"
        assert info.provider == "gemini"


class TestBaseLLMProviderFeatures:
    """Tests for BaseLLMProvider production features."""

    def test_normalize_images_bytes(self):
        """Test normalizing bytes to ImageInput."""
        with patch("flybrowser.llm.openai_provider.AsyncOpenAI"):
            from flybrowser.llm.openai_provider import OpenAIProvider
            
            provider = OpenAIProvider(model="gpt-4o")
            images = provider._normalize_images(b"test data")
            
            assert len(images) == 1
            assert isinstance(images[0], ImageInput)

    def test_normalize_images_single(self):
        """Test normalizing single ImageInput."""
        with patch("flybrowser.llm.openai_provider.AsyncOpenAI"):
            from flybrowser.llm.openai_provider import OpenAIProvider
            
            provider = OpenAIProvider(model="gpt-4o")
            img = ImageInput.from_bytes(b"test")
            images = provider._normalize_images(img)
            
            assert len(images) == 1
            assert images[0] is img

    def test_normalize_images_list(self):
        """Test normalizing list of ImageInput."""
        with patch("flybrowser.llm.openai_provider.AsyncOpenAI"):
            from flybrowser.llm.openai_provider import OpenAIProvider
            
            provider = OpenAIProvider(model="gpt-4o")
            imgs = [ImageInput.from_bytes(b"test1"), ImageInput.from_bytes(b"test2")]
            images = provider._normalize_images(imgs)
            
            assert len(images) == 2

    def test_supports_capability(self):
        """Test supports_capability method."""
        with patch("flybrowser.llm.openai_provider.AsyncOpenAI"):
            from flybrowser.llm.openai_provider import OpenAIProvider
            
            provider = OpenAIProvider(model="gpt-4o")
            
            assert provider.supports_capability(ModelCapability.TEXT_GENERATION) is True

    def test_get_stats_no_config(self):
        """Test get_stats without config."""
        with patch("flybrowser.llm.openai_provider.AsyncOpenAI"):
            from flybrowser.llm.openai_provider import OpenAIProvider
            
            provider = OpenAIProvider(model="gpt-4o")
            stats = provider.get_stats()
            
            assert stats["model"] == "gpt-4o"
            assert "cache" not in stats  # No config, no cache

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        """Test generate_stream returns streaming chunks."""
        with patch("flybrowser.llm.openai_provider.AsyncOpenAI") as mock_client:
            from flybrowser.llm.openai_provider import OpenAIProvider
            
            # Create mock streaming chunks
            mock_chunk1 = MagicMock()
            mock_chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"))]
            mock_chunk2 = MagicMock()
            mock_chunk2.choices = [MagicMock(delta=MagicMock(content=" world!"))]
            
            # Create async iterator for streaming
            async def mock_stream():
                for chunk in [mock_chunk1, mock_chunk2]:
                    yield chunk
            
            mock_client.return_value.chat.completions.create = AsyncMock(
                return_value=mock_stream()
            )
            
            provider = OpenAIProvider(model="gpt-4o")
            
            chunks = []
            async for chunk in provider.generate_stream("Hi"):
                chunks.append(chunk)
            
            assert len(chunks) == 2
            assert chunks[0] == "Hello"
            assert chunks[1] == " world!"
