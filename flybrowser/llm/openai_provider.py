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
OpenAI LLM provider implementation.

This module provides the OpenAIProvider class which implements the BaseLLMProvider
interface for OpenAI's GPT models. It supports both text and vision capabilities
using OpenAI's Chat Completions API.

Supported models include (as of January 2026):
- GPT-5.2 (default): Latest flagship model for coding and agentic tasks
- GPT-5 mini: Faster, cost-efficient version of GPT-5
- GPT-5 nano: Fastest, most cost-efficient version of GPT-5
- GPT-4.1: Smartest non-reasoning model
- GPT-4o: Fast, intelligent, flexible GPT model (legacy)

The provider handles:
- Text generation with system prompts
- Vision-based generation with single and multiple images
- Structured output with JSON schemas
- Tool/function calling
- Streaming responses
- Embeddings generation
- Token usage tracking
- Error handling and retries
"""

from __future__ import annotations

import base64
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from openai import AsyncOpenAI

import os

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
from flybrowser.llm.provider_status import ProviderStatus
from flybrowser.utils.logger import logger


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider implementation using Chat Completions API.

    This provider supports all OpenAI chat models including GPT-5.2, GPT-5 mini,
    GPT-4.1, and GPT-4o. It provides text generation, vision capabilities, and
    structured output support.

    Attributes:
        client: AsyncOpenAI client instance for API calls
        model: OpenAI model name (e.g., "gpt-5.2", "gpt-5-mini", "gpt-4.1")
        api_key: OpenAI API key for authentication

    Example:
        >>> provider = OpenAIProvider(model="gpt-5.2", api_key="sk-...")
        >>> response = await provider.generate("Hello, how are you?")
        >>> print(response.content)
    """

    def __init__(self, model: str = "gpt-5.2", api_key: Optional[str] = None, **kwargs: Any) -> None:
        """
        Initialize OpenAI provider with API credentials.

        Args:
            model: OpenAI model name. Supported models (as of January 2026):
                - "gpt-5.2": Latest flagship model (default)
                - "gpt-5-mini": Faster, cost-efficient version
                - "gpt-5-nano": Fastest, most cost-efficient
                - "gpt-4.1": Smartest non-reasoning model
                - "gpt-4o": Fast, intelligent model (legacy)
            api_key: OpenAI API key (starts with "sk-"). Required for API access.
                Can also be set via OPENAI_API_KEY environment variable.
            **kwargs: Additional configuration passed to BaseLLMProvider

        Example:
            >>> provider = OpenAIProvider(
            ...     model="gpt-4o",
            ...     api_key="sk-proj-YOUR_KEY_HERE"
            ... )
        """
        super().__init__(model, api_key, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key)

    @classmethod
    def check_availability(cls) -> ProviderStatus:
        """
        Check if OpenAI provider is available.

        Checks for OPENAI_API_KEY environment variable.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            return ProviderStatus.ok(
                name="OpenAI",
                message="API key configured",
                api_key_env_var="OPENAI_API_KEY",
                requires_api_key=True,
            )
        return ProviderStatus.info(
            name="OpenAI",
            message="API key not set (optional)",
            api_key_env_var="OPENAI_API_KEY",
            requires_api_key=True,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a text response using OpenAI's Chat Completions API.

        Args:
            prompt: User prompt/message to send to the model
            system_prompt: Optional system prompt to set model behavior.
                Example: "You are a helpful assistant that extracts data from web pages."
            temperature: Sampling temperature (0.0-2.0). Higher values make output
                more random, lower values more deterministic. Default: 0.7
            max_tokens: Maximum tokens to generate. If None, uses model default.
            **kwargs: Additional parameters for OpenAI API:
                - top_p: Nucleus sampling parameter
                - frequency_penalty: Penalize frequent tokens
                - presence_penalty: Penalize tokens based on presence
                - stop: Stop sequences

        Returns:
            LLMResponse containing:
            - content: Generated text
            - model: Model used for generation
            - usage: Token usage statistics

        Raises:
            LLMProviderError: If API call fails

        Example:
            >>> response = await provider.generate(
            ...     prompt="What is the capital of France?",
            ...     system_prompt="You are a geography expert.",
            ...     temperature=0.3
            ... )
            >>> print(response.content)
            'The capital of France is Paris.'
        """
        try:
            # Build messages array with optional system prompt
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Call OpenAI Chat Completions API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            # Extract response and usage information
            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens
                    if response.usage
                    else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
            )
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise LLMProviderError(f"OpenAI generation failed: {e}") from e

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
        Generate a response with vision using OpenAI.

        Supports single image (bytes or ImageInput) or multiple images (List[ImageInput]).
        """
        try:
            # Normalize images to list of ImageInput
            images = self._normalize_images(image_data)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Build content with text and images
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

            for img in images:
                if img.source_type == "url":
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img.data, "detail": img.detail},
                    })
                else:
                    # Convert bytes to base64 if needed
                    if img.source_type == "bytes":
                        img_base64 = base64.b64encode(img.data).decode("utf-8")
                    else:
                        img_base64 = img.data

                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img.media_type};base64,{img_base64}",
                            "detail": img.detail,
                        },
                    })

            messages.append({"role": "user", "content": content})

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
                **kwargs,
            )

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=response.choices[0].finish_reason,
            )
        except Exception as e:
            logger.error(f"OpenAI vision generation error: {e}")
            raise LLMProviderError(f"OpenAI vision generation failed: {e}") from e

    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate structured output using OpenAI."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
                **kwargs,
            )

            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as e:
            logger.error(f"OpenAI structured generation error: {e}")
            raise LLMProviderError(f"OpenAI structured generation failed: {e}") from e

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
        """Generate a response with tool/function calling capabilities."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Convert ToolDefinition to OpenAI format
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": tool.parameters,
                            "required": tool.required,
                        },
                    },
                }
                for tool in tools
            ]

            api_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "tools": openai_tools,
                **kwargs,
            }

            if max_tokens:
                api_kwargs["max_tokens"] = max_tokens
            if tool_choice:
                api_kwargs["tool_choice"] = tool_choice

            response = await self.client.chat.completions.create(**api_kwargs)

            # Extract tool calls if present
            tool_calls = None
            if response.choices[0].message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                    for tc in response.choices[0].message.tool_calls
                ]

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                tool_calls=tool_calls,
                finish_reason=response.choices[0].finish_reason,
            )
        except Exception as e:
            logger.error(f"OpenAI tool calling error: {e}")
            raise LLMProviderError(f"OpenAI tool calling failed: {e}") from e

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using OpenAI."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise LLMProviderError(f"OpenAI streaming failed: {e}") from e

    async def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        try:
            if isinstance(texts, str):
                texts = [texts]

            response = await self.client.embeddings.create(
                model=model or "text-embedding-3-small",
                input=texts,
                **kwargs,
            )

            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI embeddings error: {e}")
            raise LLMProviderError(f"OpenAI embeddings failed: {e}") from e

    def get_model_info(self) -> ModelInfo:
        """Get information about the current OpenAI model."""
        # Define capabilities based on model
        capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.STREAMING,
            ModelCapability.TOOL_CALLING,
            ModelCapability.STRUCTURED_OUTPUT,
        ]

        # Vision-capable models
        vision_models = ["gpt-4o", "gpt-4-turbo", "gpt-4-vision", "gpt-5.2", "gpt-5-mini"]
        if any(vm in self.model.lower() for vm in vision_models):
            capabilities.append(ModelCapability.VISION)
            capabilities.append(ModelCapability.MULTI_IMAGE_VISION)

        return ModelInfo(
            name=self.model,
            provider="openai",
            capabilities=capabilities,
            context_window=128000,
            max_output_tokens=4096,
            supports_system_prompt=True,
            cost_per_1k_input_tokens=0.005,
            cost_per_1k_output_tokens=0.015,
        )

