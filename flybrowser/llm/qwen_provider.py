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
Qwen AI LLM provider implementation.

This module provides the QwenProvider class which implements the BaseLLMProvider
interface for Alibaba Cloud's Qwen models via the DashScope API. The DashScope API
provides an OpenAI-compatible endpoint, making integration seamless.

Supported models include (as of January 2026):
- Qwen3-235B-A22B: Latest flagship MoE model with 235B parameters (22B active)
- Qwen3-32B: High-performance dense model
- Qwen3-Max: Maximum capability model
- Qwen-Plus: Advanced model with enhanced reasoning
- Qwen-Turbo: Fast, cost-efficient model
- Qwen-VL-Max: Vision-language model for multimodal tasks
- Qwen-VL-Plus: Vision-language model (balanced)

The provider handles:
- Text generation with system prompts
- Vision-based generation with single and multiple images (Qwen-VL models)
- Structured output with JSON schemas
- Tool/function calling
- Streaming responses
- Token usage tracking
- Error handling and retries

API Documentation: https://www.alibabacloud.com/help/en/model-studio/qwen-api-reference/
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from openai import AsyncOpenAI

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


# Default DashScope API endpoints (OpenAI-compatible)
DASHSCOPE_BASE_URLS = {
    "default": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "international": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "us": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
}


class QwenProvider(BaseLLMProvider):
    """
    Qwen AI LLM provider implementation using OpenAI-compatible DashScope API.

    This provider supports all Qwen models including Qwen3, Qwen-Plus, Qwen-Turbo,
    and Qwen-VL (vision-language) models. It uses the OpenAI SDK with a custom
    base URL for seamless integration.

    Attributes:
        client: AsyncOpenAI client configured for DashScope API
        model: Qwen model name (e.g., "qwen3-235b-a22b", "qwen-plus", "qwen-vl-max")
        api_key: DashScope API key for authentication

    Example:
        >>> provider = QwenProvider(model="qwen-plus", api_key="sk-...")
        >>> response = await provider.generate("Hello, how are you?")
        >>> print(response.content)
    """

    def __init__(
        self,
        model: str = "qwen-plus",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        region: str = "default",
        **kwargs: Any,
    ) -> None:
        """
        Initialize Qwen provider with API credentials.

        Args:
            model: Qwen model name. Supported models (as of January 2026):
                Text models:
                - "qwen3-235b-a22b": Latest flagship MoE model (default for complex tasks)
                - "qwen3-32b": High-performance dense model
                - "qwen3-max": Maximum capability model
                - "qwen-plus": Advanced model with enhanced reasoning (default)
                - "qwen-turbo": Fast, cost-efficient model
                - "qwen-max": Premium text generation
                Vision-language models:
                - "qwen-vl-max": Best vision-language model
                - "qwen-vl-plus": Balanced vision-language model
            api_key: DashScope API key. Required for API access.
                Can also be set via DASHSCOPE_API_KEY or QWEN_API_KEY environment variables.
            base_url: Custom API base URL. If not provided, uses region-based default.
            region: API region. Options: "default" (China), "international", "us"
            **kwargs: Additional configuration passed to BaseLLMProvider

        Example:
            >>> provider = QwenProvider(
            ...     model="qwen-plus",
            ...     api_key="sk-YOUR_KEY_HERE"
            ... )
        """
        # Try multiple environment variables for API key
        if api_key is None:
            api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
        
        super().__init__(model, api_key, **kwargs)
        
        # Determine base URL
        if base_url:
            self._base_url = base_url
        else:
            self._base_url = DASHSCOPE_BASE_URLS.get(region, DASHSCOPE_BASE_URLS["default"])
        
        # Initialize OpenAI client with DashScope endpoint
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self._base_url,
        )
        
        # Cache for model capabilities
        self._model_capabilities_cache: Optional[List[ModelCapability]] = None
        self._capabilities_detected: bool = False

    async def initialize(self) -> None:
        """
        Initialize the provider and detect model capabilities.

        Vision support is determined by model name patterns (qwen-vl-* models).
        This is called automatically on first use.
        """
        if not self._capabilities_detected:
            self._model_capabilities_cache = self._get_basic_capabilities()
            self._capabilities_detected = True

            has_vision = ModelCapability.VISION in self._model_capabilities_cache
            if has_vision:
                logger.info(f"Vision: [INFO] ENABLED for model {self.model}")
            else:
                logger.debug(f"Vision: [INFO] DISABLED for model {self.model}")

    @classmethod
    def check_availability(cls) -> ProviderStatus:
        """
        Check if Qwen provider is available.

        Checks for DASHSCOPE_API_KEY or QWEN_API_KEY environment variable.
        """
        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
        if api_key:
            return ProviderStatus.ok(
                name="Qwen",
                message="API key configured",
                api_key_env_var="DASHSCOPE_API_KEY or QWEN_API_KEY",
                requires_api_key=True,
            )
        return ProviderStatus.info(
            name="Qwen",
            message="API key not set (optional)",
            api_key_env_var="DASHSCOPE_API_KEY or QWEN_API_KEY",
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
        Generate a text response using Qwen's Chat Completions API.

        Args:
            prompt: User prompt/message to send to the model
            system_prompt: Optional system prompt to set model behavior
            temperature: Sampling temperature (0.0-2.0). Default: 0.7
            max_tokens: Maximum tokens to generate. If None, uses model default.
            **kwargs: Additional parameters for the API:
                - top_p: Nucleus sampling parameter
                - presence_penalty: Penalize tokens based on presence
                - stop: Stop sequences

        Returns:
            LLMResponse containing generated text, model info, and usage stats

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
        await self.initialize()
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            start_time = self._log_llm_request(
                "GENERATE", self.model,
                prompt=prompt, system_prompt=system_prompt or ""
            )

            # Build API kwargs
            api_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                api_kwargs["max_tokens"] = max_tokens

            # Add any additional kwargs
            for key in ["top_p", "presence_penalty", "stop"]:
                if key in kwargs:
                    api_kwargs[key] = kwargs[key]

            response = await self.client.chat.completions.create(**api_kwargs)

            total_tokens = response.usage.total_tokens if response.usage else 0
            response_content = response.choices[0].message.content or ""
            self._log_llm_response("GENERATE", start_time, total_tokens, response_content)

            llm_response = LLMResponse(
                content=response_content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": total_tokens,
                },
            )
            
            self._track_usage(llm_response)
            return llm_response

        except Exception as e:
            logger.error(f"Qwen generation error: {e}")
            raise LLMProviderError(f"Qwen generation failed: {e}") from e

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
        Generate a response with vision using Qwen-VL models.

        Supports single image (bytes or ImageInput) or multiple images (List[ImageInput]).
        Requires a vision-capable model (qwen-vl-max, qwen-vl-plus, etc.).

        If the model does not support vision, falls back to text-only generation.

        Args:
            prompt: User prompt describing what to do with the image
            image_data: Image data as bytes, ImageInput, or list of ImageInput
            system_prompt: Optional system prompt for context
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated analysis

        Raises:
            LLMProviderError: If API call fails
        """
        await self.initialize()
        
        if not self.supports_vision():
            logger.warning(
                f"[WARN] Model {self.model} does not support vision! "
                f"Vision-capable models: qwen-vl-max, qwen-vl-plus, qwen3-vl-*. "
                f"Falling back to text-only generation."
            )
            return await self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        
        try:
            images = self._normalize_images(image_data)

            start_time = self._log_llm_request(
                "VISION", self.model, f"{len(images)} images",
                prompt=prompt, system_prompt=system_prompt or ""
            )

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

            api_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                api_kwargs["max_tokens"] = max_tokens
            else:
                api_kwargs["max_tokens"] = 8192  # Generous default for vision

            response = await self.client.chat.completions.create(**api_kwargs)

            total_tokens = response.usage.total_tokens if response.usage else 0
            response_content = response.choices[0].message.content or ""
            self._log_llm_response("VISION", start_time, total_tokens, response_content)

            llm_response = LLMResponse(
                content=response_content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": total_tokens,
                },
                finish_reason=response.choices[0].finish_reason,
            )
            
            self._track_usage(llm_response)
            return llm_response

        except Exception as e:
            logger.error(f"Qwen vision generation error: {e}")
            raise LLMProviderError(f"Qwen vision generation failed: {e}") from e

    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate structured output matching the provided schema.

        Uses JSON mode to ensure valid JSON output.

        Args:
            prompt: User prompt (should mention JSON format for best results)
            schema: JSON schema for expected response structure
            system_prompt: System prompt for context
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Parsed JSON response as dictionary

        Raises:
            LLMProviderError: If API call fails or response is not valid JSON
        """
        await self.initialize()
        
        try:
            start_time = self._log_llm_request(
                "STRUCTURED", self.model,
                prompt=prompt, system_prompt=system_prompt or ""
            )

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            effective_max_tokens = kwargs.pop("max_tokens", None) or 8192
            
            api_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": effective_max_tokens,
                "response_format": {"type": "json_object"},
            }

            response = await self.client.chat.completions.create(**api_kwargs)

            content = response.choices[0].message.content or "{}"
            total_tokens = response.usage.total_tokens if response.usage else 0
            self._log_llm_response("STRUCTURED", start_time, total_tokens, content)
            
            llm_response = LLMResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": total_tokens,
                },
            )
            self._track_usage(llm_response)
            
            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"Qwen structured response not valid JSON: {e}")
            raise LLMProviderError(f"Qwen structured response not valid JSON: {e}") from e
        except Exception as e:
            logger.error(f"Qwen structured generation error: {e}")
            raise LLMProviderError(f"Qwen structured generation failed: {e}") from e

    async def generate_structured_with_vision(
        self,
        prompt: str,
        image_data: Union[bytes, ImageInput, List[ImageInput]],
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate structured output with vision input.

        Combines vision capabilities with JSON mode for deterministic structured output.

        Args:
            prompt: User prompt (should mention JSON format)
            image_data: Image data
            schema: JSON schema for response structure
            system_prompt: System prompt for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Parsed JSON response as dictionary
        """
        await self.initialize()
        
        if not self.supports_vision():
            logger.warning(
                f"[WARN] Model {self.model} does not support vision! "
                f"Falling back to text-only structured generation."
            )
            return await self.generate_structured(
                prompt=prompt,
                schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs,
            )
        
        try:
            images = self._normalize_images(image_data)
            
            start_time = self._log_llm_request(
                "STRUCTURED_VISION", self.model,
                f"{len(images)} images",
                prompt=prompt, system_prompt=system_prompt or ""
            )
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            
            for img in images:
                if img.source_type == "url":
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img.data, "detail": img.detail},
                    })
                else:
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
            
            api_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 8192,
                "response_format": {"type": "json_object"},
            }
            
            response = await self.client.chat.completions.create(**api_kwargs)
            
            total_tokens = response.usage.total_tokens if response.usage else 0
            response_content = response.choices[0].message.content or "{}"
            self._log_llm_response("STRUCTURED_VISION", start_time, total_tokens, response_content)
            
            llm_response = LLMResponse(
                content=response_content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": total_tokens,
                },
            )
            self._track_usage(llm_response)
            
            return json.loads(response_content)
            
        except json.JSONDecodeError as e:
            logger.error(f"Qwen structured vision response not valid JSON: {e}")
            raise LLMProviderError(f"Qwen structured vision response not valid JSON: {e}") from e
        except Exception as e:
            logger.error(f"Qwen structured vision generation error: {e}")
            raise LLMProviderError(f"Qwen structured vision generation failed: {e}") from e

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
            **kwargs: Additional parameters

        Returns:
            LLMResponse with potential tool_calls

        Raises:
            LLMProviderError: If API call fails
        """
        await self.initialize()
        
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

            api_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "tools": openai_tools,
            }
            if max_tokens is not None:
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
            logger.error(f"Qwen tool calling error: {e}")
            raise LLMProviderError(f"Qwen tool calling failed: {e}") from e

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
        """
        await self.initialize()
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            api_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }
            if max_tokens is not None:
                api_kwargs["max_tokens"] = max_tokens

            stream = await self.client.chat.completions.create(**api_kwargs)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Qwen streaming error: {e}")
            raise LLMProviderError(f"Qwen streaming failed: {e}") from e

    def _get_basic_capabilities(self) -> List[ModelCapability]:
        """
        Get model capabilities based on model name patterns.

        Returns:
            List of ModelCapability enum values
        """
        model_lower = self.model.lower()
        capabilities = [ModelCapability.TEXT_GENERATION, ModelCapability.STREAMING]

        # Most Qwen models support tool calling and structured output
        if any(x in model_lower for x in ["qwen", "qwen3", "qwen-plus", "qwen-max", "qwen-turbo"]):
            capabilities.append(ModelCapability.TOOL_CALLING)
            capabilities.append(ModelCapability.STRUCTURED_OUTPUT)
        
        # Vision models: qwen-vl-*, qwen3-vl-*
        if self._vision_enabled_override is not None:
            if self._vision_enabled_override:
                capabilities.append(ModelCapability.VISION)
        elif any(x in model_lower for x in ["vl", "vision"]):
            capabilities.append(ModelCapability.VISION)
            capabilities.append(ModelCapability.MULTI_IMAGE_VISION)
        
        # Thinking/reasoning models
        if "thinking" in model_lower:
            capabilities.append(ModelCapability.EXTENDED_THINKING)

        return capabilities

    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current Qwen model.
        """
        if self._model_capabilities_cache is not None:
            capabilities = self._model_capabilities_cache
        else:
            capabilities = self._get_basic_capabilities()

        model_lower = self.model.lower()
        
        # Model-specific configurations
        if "qwen3-235b" in model_lower:
            context_window = 128000
            max_output = 16384
            input_cost = 0.004
            output_cost = 0.012
        elif "qwen3-32b" in model_lower:
            context_window = 128000
            max_output = 8192
            input_cost = 0.002
            output_cost = 0.006
        elif "qwen-max" in model_lower or "qwen3-max" in model_lower:
            context_window = 128000
            max_output = 8192
            input_cost = 0.004
            output_cost = 0.012
        elif "qwen-plus" in model_lower:
            context_window = 128000
            max_output = 8192
            input_cost = 0.002
            output_cost = 0.006
        elif "qwen-turbo" in model_lower:
            context_window = 128000
            max_output = 8192
            input_cost = 0.0005
            output_cost = 0.0015
        elif "vl" in model_lower:
            context_window = 32768
            max_output = 4096
            input_cost = 0.003
            output_cost = 0.009
        else:
            context_window = 128000
            max_output = 8192
            input_cost = 0.002
            output_cost = 0.006
        
        return ModelInfo(
            name=self.model,
            provider="qwen",
            capabilities=capabilities,
            context_window=context_window,
            max_output_tokens=max_output,
            supports_system_prompt=True,
            cost_per_1k_input_tokens=input_cost,
            cost_per_1k_output_tokens=output_cost,
        )
