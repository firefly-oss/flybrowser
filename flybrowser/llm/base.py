# Copyright 2026 Firefly Software Solutions Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Base LLM provider interface with production features.

This module defines the abstract base class for all LLM providers in FlyBrowser.
It provides a unified interface for interacting with different LLM services
(OpenAI, Anthropic, Ollama, Google Gemini, etc.) and includes production-ready
features like caching, retry logic, rate limiting, and cost tracking.

The module also defines the LLMResponse dataclass which standardizes responses
across all providers.

Key Features:
- Unified API across all providers
- Vision/multimodal support with single and multiple images
- Streaming response support
- Tool/function calling support
- Structured output with JSON schemas
- Embeddings generation
- Model capability introspection
- Production features (caching, retry, rate limiting, cost tracking)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

from flybrowser.llm.cache import LLMCache
from flybrowser.llm.config import LLMProviderConfig
from flybrowser.llm.cost_tracker import CostTracker
from flybrowser.llm.provider_status import ProviderStatus
from flybrowser.llm.rate_limiter import RateLimiter
from flybrowser.llm.retry import RetryHandler
from flybrowser.utils.logger import logger


class ModelCapability(str, Enum):
    """Capabilities that a model may support."""

    TEXT_GENERATION = "text_generation"
    VISION = "vision"
    MULTI_IMAGE_VISION = "multi_image_vision"
    STREAMING = "streaming"
    TOOL_CALLING = "tool_calling"
    STRUCTURED_OUTPUT = "structured_output"
    EMBEDDINGS = "embeddings"
    CODE_EXECUTION = "code_execution"
    EXTENDED_THINKING = "extended_thinking"


@dataclass
class ImageInput:
    """
    Represents an image input for vision-capable models.

    Attributes:
        data: Image data as bytes or base64 string
        media_type: MIME type of the image (e.g., "image/png", "image/jpeg")
        detail: Level of detail for image analysis ("low", "high", "auto")
        source_type: Whether data is "bytes" or "base64"
    """

    data: Union[bytes, str]
    media_type: str = "image/png"
    detail: str = "auto"
    source_type: str = "bytes"

    @classmethod
    def from_bytes(cls, data: bytes, media_type: str = "image/png", detail: str = "auto") -> "ImageInput":
        """Create ImageInput from raw bytes."""
        return cls(data=data, media_type=media_type, detail=detail, source_type="bytes")

    @classmethod
    def from_base64(cls, data: str, media_type: str = "image/png", detail: str = "auto") -> "ImageInput":
        """Create ImageInput from base64 string."""
        return cls(data=data, media_type=media_type, detail=detail, source_type="base64")

    @classmethod
    def from_url(cls, url: str, detail: str = "auto") -> "ImageInput":
        """Create ImageInput from URL (for providers that support it)."""
        return cls(data=url, media_type="url", detail=detail, source_type="url")


@dataclass
class ToolDefinition:
    """
    Definition of a tool/function that can be called by the LLM.

    Attributes:
        name: Name of the tool
        description: Description of what the tool does
        parameters: JSON schema for the tool's parameters
        required: List of required parameter names
    """

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)


@dataclass
class ToolCall:
    """
    Represents a tool call made by the LLM.

    Attributes:
        id: Unique identifier for this tool call
        name: Name of the tool being called
        arguments: Arguments passed to the tool (as dict)
    """

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ModelInfo:
    """
    Information about a model's capabilities and limits.

    Attributes:
        name: Model name/identifier
        provider: Provider name
        capabilities: Set of supported capabilities
        context_window: Maximum context window size in tokens
        max_output_tokens: Maximum output tokens
        supports_system_prompt: Whether model supports system prompts
        cost_per_1k_input_tokens: Cost per 1000 input tokens (USD)
        cost_per_1k_output_tokens: Cost per 1000 output tokens (USD)
    """

    name: str
    provider: str
    capabilities: List[ModelCapability] = field(default_factory=list)
    context_window: int = 128000
    max_output_tokens: int = 4096
    supports_system_prompt: bool = True
    cost_per_1k_input_tokens: Optional[float] = None
    cost_per_1k_output_tokens: Optional[float] = None


@dataclass
class LLMResponse:
    """
    Standardized response from an LLM provider.

    This dataclass encapsulates the response from any LLM provider, providing
    a consistent interface regardless of the underlying service.

    Attributes:
        content: The generated text content from the LLM
        model: Name of the model that generated the response
        usage: Token usage statistics with keys:
            - prompt_tokens: Number of tokens in the prompt
            - completion_tokens: Number of tokens in the completion
            - total_tokens: Total tokens used
        metadata: Additional provider-specific metadata
        cached: Whether this response was served from cache
        tool_calls: List of tool calls made by the model
        finish_reason: Reason for completion (e.g., "stop", "tool_calls", "length")

    Example:
        >>> response = LLMResponse(
        ...     content="The capital of France is Paris.",
        ...     model="gpt-4o",
        ...     usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        ...     cached=False
        ... )
    """

    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None
    cached: bool = False
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers with production features.

    This class defines the interface that all LLM providers must implement.
    It also provides production-ready features like caching, retry logic,
    rate limiting, and cost tracking when configured.

    Subclasses must implement:
    - generate(): Basic text generation
    - generate_with_vision(): Generation with image input
    - generate_structured(): Generation with structured output

    Attributes:
        model: Model name/identifier
        api_key: API key for the LLM provider (e.g., OpenAI API key)
        provider_config: Full provider configuration
        extra_config: Additional provider-specific configuration
        cache: LLM response cache (if configured)
        cost_tracker: Token usage and cost tracker (if configured)
        rate_limiter: Rate limiter for API requests (if configured)
        retry_handler: Retry handler with exponential backoff (if configured)

    Example:
        Subclass implementation:

        >>> class MyLLMProvider(BaseLLMProvider):
        ...     async def generate(self, prompt, **kwargs):
        ...         # Implementation here
        ...         pass
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        config: Optional[LLMProviderConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the LLM provider with configuration.

        Args:
            model: Model name/identifier (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
            api_key: API key for the LLM provider (e.g., OpenAI API key, Anthropic API key).
                Not required for local providers like Ollama.
            config: Full provider configuration including retry, cache, rate limit settings.
                If not provided, production features will be disabled.
            **kwargs: Additional provider-specific configuration options

        Note:
            When config is provided, the following production features are enabled:
            - Response caching (LRU cache with TTL)
            - Automatic retry with exponential backoff
            - Rate limiting (requests/min and tokens/min)
            - Cost tracking and reporting
        """
        self.model = model
        self.api_key = api_key
        self.provider_config = config
        self.extra_config = kwargs

        # Initialize production features if config provided
        if config:
            self.cache = LLMCache(config.cache_config)
            self.cost_tracker = CostTracker(config.cost_tracking_config)
            self.rate_limiter = RateLimiter(config.rate_limit_config)
            self.retry_handler = RetryHandler(config.retry_config)
        else:
            # No config provided - production features disabled
            self.cache = None
            self.cost_tracker = None
            self.rate_limiter = None
            self.retry_handler = None

    async def generate_with_features(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate with production features (caching, rate limiting, retry, cost tracking).

        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            use_cache: Whether to use caching
            **kwargs: Additional parameters

        Returns:
            LLMResponse object
        """
        # Check cache first
        if use_cache and self.cache:
            cached_response = self.cache.get(
                prompt, system_prompt, self.model, temperature, **kwargs
            )
            if cached_response:
                logger.debug("Returning cached response")
                cached_response.cached = True
                return cached_response

        # Acquire rate limit
        if self.rate_limiter:
            estimated_tokens = len(prompt.split()) * 2  # Rough estimate
            await self.rate_limiter.acquire(estimated_tokens)

        try:
            # Execute with retry
            if self.retry_handler:
                response = await self.retry_handler.execute_with_retry(
                    self._generate_impl,
                    prompt,
                    system_prompt,
                    temperature,
                    max_tokens,
                    **kwargs,
                )
            else:
                response = await self._generate_impl(
                    prompt, system_prompt, temperature, max_tokens, **kwargs
                )

            # Track cost
            if self.cost_tracker and response.usage:
                self.cost_tracker.record_usage(
                    provider=self.__class__.__name__.replace("Provider", "").lower(),
                    model=self.model,
                    prompt_tokens=response.usage.get("prompt_tokens", 0),
                    completion_tokens=response.usage.get("completion_tokens", 0),
                    cached=False,
                )

            # Cache response
            if use_cache and self.cache:
                self.cache.set(
                    prompt, system_prompt, self.model, temperature, response, **kwargs
                )

            return response

        finally:
            # Release rate limit
            if self.rate_limiter:
                actual_tokens = (
                    response.usage.get("total_tokens", 0) if response.usage else 0
                )
                self.rate_limiter.release(actual_tokens)

    async def _generate_impl(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs: Any,
    ) -> LLMResponse:
        """Internal implementation that calls the abstract generate method."""
        return await self.generate(prompt, system_prompt, temperature, max_tokens, **kwargs)

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object
        """
        pass

    @abstractmethod
    async def generate_with_vision(
        self,
        prompt: str,
        image_data: Union[bytes, ImageInput, List[ImageInput]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from the LLM with vision capabilities.

        Supports single image (bytes or ImageInput) or multiple images (List[ImageInput]).

        Args:
            prompt: User prompt
            image_data: Image data as bytes, ImageInput, or list of ImageInput for multi-image
            system_prompt: System prompt for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object
        """
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate a structured response matching the provided schema.

        Args:
            prompt: User prompt
            schema: JSON schema for the expected response
            system_prompt: System prompt for context
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Structured data matching the schema
        """
        pass

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[ToolDefinition],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response with tool/function calling capabilities.

        Args:
            prompt: User prompt
            tools: List of tool definitions available to the model
            system_prompt: System prompt for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tool_choice: How to select tools ("auto", "none", or specific tool name)
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object with potential tool_calls

        Note:
            Default implementation raises NotImplementedError.
            Providers that support tool calling should override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support tool calling. "
            "Override generate_with_tools() to add support."
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Yields:
            Response chunks as strings

        Note:
            Default implementation falls back to non-streaming.
            Providers that support streaming should override this method.
        """
        # Default implementation: non-streaming fallback
        response = await self.generate(prompt, system_prompt, temperature, max_tokens, **kwargs)
        yield response.content

    async def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts to embed
            model: Optional embedding model (uses provider default if not specified)
            **kwargs: Additional parameters

        Returns:
            List of embedding vectors (one per input text)

        Note:
            Default implementation raises NotImplementedError.
            Providers that support embeddings should override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support embeddings. "
            "Override generate_embeddings() to add support."
        )

    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current model's capabilities.

        Returns:
            ModelInfo object with model capabilities and limits

        Note:
            Default implementation returns basic info.
            Providers should override to provide accurate model information.
        """
        return ModelInfo(
            name=self.model,
            provider=self.__class__.__name__.replace("Provider", "").lower(),
            capabilities=[ModelCapability.TEXT_GENERATION],
            context_window=128000,
            max_output_tokens=4096,
            supports_system_prompt=True,
        )

    def supports_capability(self, capability: ModelCapability) -> bool:
        """
        Check if the current model supports a specific capability.

        Args:
            capability: The capability to check

        Returns:
            True if the capability is supported
        """
        model_info = self.get_model_info()
        return capability in model_info.capabilities

    def get_stats(self) -> Dict[str, Any]:
        """
        Get provider statistics.

        Returns:
            Dictionary with stats including cache, cost, and rate limit info
        """
        stats = {
            "model": self.model,
            "provider": self.__class__.__name__,
            "capabilities": [c.value for c in self.get_model_info().capabilities],
        }

        if self.cache:
            stats["cache"] = self.cache.get_stats()

        if self.cost_tracker:
            stats["cost"] = self.cost_tracker.get_summary()

        if self.rate_limiter:
            stats["rate_limit"] = self.rate_limiter.get_stats()

        return stats

    @classmethod
    def check_availability(cls) -> ProviderStatus:
        """
        Check if this provider is available and properly configured.

        This method should be overridden by each provider to check:
        - API key configuration (if required)
        - Connectivity to the service (for local providers)
        - Any other provider-specific requirements

        Returns:
            ProviderStatus object with availability information

        Note:
            Default implementation returns INFO status indicating
            the provider hasn't implemented availability checking.
        """
        provider_name = cls.__name__.replace("Provider", "")
        return ProviderStatus.info(
            name=provider_name,
            message="Availability check not implemented",
        )

    def _normalize_images(
        self, image_data: Union[bytes, ImageInput, List[ImageInput]]
    ) -> List[ImageInput]:
        """
        Normalize image input to a list of ImageInput objects.

        Args:
            image_data: Image data in various formats

        Returns:
            List of ImageInput objects
        """
        if isinstance(image_data, bytes):
            return [ImageInput.from_bytes(image_data)]
        elif isinstance(image_data, ImageInput):
            return [image_data]
        elif isinstance(image_data, list):
            return image_data
        else:
            raise ValueError(f"Unsupported image_data type: {type(image_data)}")

