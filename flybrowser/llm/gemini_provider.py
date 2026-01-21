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
Google Gemini LLM provider implementation.

This module provides the GeminiProvider class which implements the BaseLLMProvider
interface for Google's Gemini models. It supports both text and vision capabilities
using Google's Generative AI API.

Supported models include (as of January 2026):
- Gemini 2.0 Flash (default): Fast, versatile model with multimodal capabilities
- Gemini 2.0 Pro: Most capable model for complex tasks
- Gemini 1.5 Pro: Previous generation high-performance model
- Gemini 1.5 Flash: Fast, efficient model

The provider handles:
- Text generation with system prompts
- Vision-based generation with single and multiple images
- Structured output with JSON schemas
- Tool/function calling
- Streaming responses
- Token usage tracking
- Error handling and retries
"""

from __future__ import annotations

import base64
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import os

import aiohttp

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


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini LLM provider implementation using Generative AI API.

    This provider supports all Gemini models including Gemini 2.0 Flash,
    Gemini 2.0 Pro, and Gemini 1.5 series. It provides text generation, vision
    capabilities, tool calling, and structured output support.

    Attributes:
        model: Gemini model name (e.g., "gemini-2.0-flash", "gemini-2.0-pro")
        api_key: Google AI API key for authentication
        base_url: API base URL

    Example:
        >>> provider = GeminiProvider(
        ...     model="gemini-2.0-flash",
        ...     api_key="AIza..."
        ... )
        >>> response = await provider.generate("Hello, how are you?")
        >>> print(response.content)
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        **kwargs: Any,
    ) -> None:
        """
        Initialize Gemini provider with API credentials.

        Args:
            model: Gemini model name. Supported models (as of January 2026):
                - "gemini-2.0-flash": Fast, versatile model (default)
                - "gemini-2.0-pro": Most capable model
                - "gemini-1.5-pro": Previous generation pro model
                - "gemini-1.5-flash": Fast, efficient model
            api_key: Google AI API key. Required for API access.
                Can also be set via GOOGLE_API_KEY environment variable.
            base_url: API base URL. Default: Google's Generative AI API
            **kwargs: Additional configuration passed to BaseLLMProvider
        """
        super().__init__(model, api_key, **kwargs)
        self.base_url = base_url.rstrip("/")

    @classmethod
    def check_availability(cls) -> ProviderStatus:
        """
        Check if Gemini provider is available.

        Checks for GOOGLE_API_KEY or GEMINI_API_KEY environment variable.
        """
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            return ProviderStatus.ok(
                name="Gemini",
                message="API key configured",
                api_key_env_var="GOOGLE_API_KEY",
                requires_api_key=True,
            )
        return ProviderStatus.info(
            name="Gemini",
            message="API key not set (optional)",
            api_key_env_var="GOOGLE_API_KEY",
            requires_api_key=True,
        )

    async def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Make an HTTP request to the Gemini API."""
        url = f"{self.base_url}/models/{self.model}:{endpoint}?key={self.api_key}"

        headers = {"Content-Type": "application/json"}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMProviderError(
                        f"Gemini API request failed: {response.status} - {error_text}"
                    )
                return await response.json()

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a text response using Google's Generative AI API."""
        try:
            # Build request payload
            contents = []

            # Add system instruction if provided
            system_instruction = None
            if system_prompt:
                system_instruction = {"parts": [{"text": system_prompt}]}

            contents.append({"role": "user", "parts": [{"text": prompt}]})

            payload: Dict[str, Any] = {"contents": contents}

            if system_instruction:
                payload["systemInstruction"] = system_instruction

            # Add generation config
            generation_config: Dict[str, Any] = {"temperature": temperature}
            if max_tokens:
                generation_config["maxOutputTokens"] = max_tokens
            payload["generationConfig"] = generation_config

            result = await self._make_request("generateContent", payload)

            # Extract response
            content = ""
            if result.get("candidates"):
                candidate = result["candidates"][0]
                if candidate.get("content", {}).get("parts"):
                    content = candidate["content"]["parts"][0].get("text", "")

            # Extract usage
            usage_metadata = result.get("usageMetadata", {})

            return LLMResponse(
                content=content,
                model=self.model,
                usage={
                    "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                    "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
                    "total_tokens": usage_metadata.get("totalTokenCount", 0),
                },
                finish_reason=result.get("candidates", [{}])[0].get("finishReason"),
            )
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise LLMProviderError(f"Gemini generation failed: {e}") from e

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
        Generate a response with vision using Gemini.

        Supports single image (bytes or ImageInput) or multiple images (List[ImageInput]).
        """
        try:
            # Normalize images to list of ImageInput
            images = self._normalize_images(image_data)

            # Build parts with images and text
            parts: List[Dict[str, Any]] = []

            for img in images:
                if img.source_type == "bytes":
                    img_base64 = base64.b64encode(img.data).decode("utf-8")
                else:
                    img_base64 = img.data

                parts.append({
                    "inlineData": {
                        "mimeType": img.media_type,
                        "data": img_base64,
                    }
                })

            parts.append({"text": prompt})

            contents = [{"role": "user", "parts": parts}]

            payload: Dict[str, Any] = {"contents": contents}

            if system_prompt:
                payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

            generation_config: Dict[str, Any] = {"temperature": temperature}
            if max_tokens:
                generation_config["maxOutputTokens"] = max_tokens
            payload["generationConfig"] = generation_config

            result = await self._make_request("generateContent", payload)

            # Extract response
            content = ""
            if result.get("candidates"):
                candidate = result["candidates"][0]
                if candidate.get("content", {}).get("parts"):
                    content = candidate["content"]["parts"][0].get("text", "")

            usage_metadata = result.get("usageMetadata", {})

            return LLMResponse(
                content=content,
                model=self.model,
                usage={
                    "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                    "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
                    "total_tokens": usage_metadata.get("totalTokenCount", 0),
                },
                finish_reason=result.get("candidates", [{}])[0].get("finishReason"),
            )
        except Exception as e:
            logger.error(f"Gemini vision generation error: {e}")
            raise LLMProviderError(f"Gemini vision generation failed: {e}") from e

    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate structured output using Gemini."""
        try:
            # Add JSON schema instruction to prompt
            schema_instruction = (
                f"\n\nYou must respond with valid JSON matching this schema: "
                f"{json.dumps(schema)}\n\nRespond only with the JSON, no other text."
            )
            full_prompt = prompt + schema_instruction

            response = await self.generate(
                prompt=full_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs,
            )

            # Parse JSON from response
            content = response.content.strip()

            # Try to extract JSON if wrapped in markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```json"):
                    lines = lines[1:]
                if lines[-1] == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)

            return json.loads(content)
        except Exception as e:
            logger.error(f"Gemini structured generation error: {e}")
            raise LLMProviderError(f"Gemini structured generation failed: {e}") from e


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
            # Convert ToolDefinition to Gemini format
            gemini_tools = [{
                "functionDeclarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": tool.parameters,
                            "required": tool.required,
                        },
                    }
                    for tool in tools
                ]
            }]

            contents = [{"role": "user", "parts": [{"text": prompt}]}]

            payload: Dict[str, Any] = {
                "contents": contents,
                "tools": gemini_tools,
            }

            if system_prompt:
                payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

            generation_config: Dict[str, Any] = {"temperature": temperature}
            if max_tokens:
                generation_config["maxOutputTokens"] = max_tokens
            payload["generationConfig"] = generation_config

            result = await self._make_request("generateContent", payload)

            # Extract tool calls if present
            tool_calls = None
            content = ""

            if result.get("candidates"):
                candidate = result["candidates"][0]
                parts = candidate.get("content", {}).get("parts", [])

                for part in parts:
                    if "text" in part:
                        content = part["text"]
                    elif "functionCall" in part:
                        if tool_calls is None:
                            tool_calls = []
                        fc = part["functionCall"]
                        tool_calls.append(
                            ToolCall(
                                id=f"call_{len(tool_calls)}",
                                name=fc.get("name", ""),
                                arguments=fc.get("args", {}),
                            )
                        )

            usage_metadata = result.get("usageMetadata", {})

            return LLMResponse(
                content=content,
                model=self.model,
                usage={
                    "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                    "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
                    "total_tokens": usage_metadata.get("totalTokenCount", 0),
                },
                tool_calls=tool_calls,
                finish_reason=result.get("candidates", [{}])[0].get("finishReason"),
            )
        except Exception as e:
            logger.error(f"Gemini tool calling error: {e}")
            raise LLMProviderError(f"Gemini tool calling failed: {e}") from e

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using Gemini."""
        try:
            url = f"{self.base_url}/models/{self.model}:streamGenerateContent?key={self.api_key}"

            contents = [{"role": "user", "parts": [{"text": prompt}]}]

            payload: Dict[str, Any] = {"contents": contents}

            if system_prompt:
                payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

            generation_config: Dict[str, Any] = {"temperature": temperature}
            if max_tokens:
                generation_config["maxOutputTokens"] = max_tokens
            payload["generationConfig"] = generation_config

            headers = {"Content-Type": "application/json"}

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMProviderError(
                            f"Gemini streaming request failed: {response.status} - {error_text}"
                        )

                    async for line in response.content:
                        if line:
                            try:
                                # Gemini streams JSON objects
                                text = line.decode("utf-8").strip()
                                if text.startswith("["):
                                    text = text[1:]
                                if text.endswith("]"):
                                    text = text[:-1]
                                if text.startswith(","):
                                    text = text[1:]
                                if text:
                                    data = json.loads(text)
                                    if data.get("candidates"):
                                        parts = data["candidates"][0].get("content", {}).get("parts", [])
                                        for part in parts:
                                            if "text" in part:
                                                yield part["text"]
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise LLMProviderError(f"Gemini streaming failed: {e}") from e

    async def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings using Gemini."""
        try:
            if isinstance(texts, str):
                texts = [texts]

            embed_model = model or "text-embedding-004"
            url = f"{self.base_url}/models/{embed_model}:batchEmbedContents?key={self.api_key}"

            requests = [
                {"model": f"models/{embed_model}", "content": {"parts": [{"text": text}]}}
                for text in texts
            ]

            payload = {"requests": requests}
            headers = {"Content-Type": "application/json"}

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMProviderError(
                            f"Gemini embeddings request failed: {response.status} - {error_text}"
                        )

                    result = await response.json()

                    embeddings = []
                    for embedding in result.get("embeddings", []):
                        embeddings.append(embedding.get("values", []))

                    return embeddings
        except Exception as e:
            logger.error(f"Gemini embeddings error: {e}")
            raise LLMProviderError(f"Gemini embeddings failed: {e}") from e

    def get_model_info(self) -> ModelInfo:
        """Get information about the current Gemini model."""
        capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.VISION,
            ModelCapability.MULTI_IMAGE_VISION,
            ModelCapability.STREAMING,
            ModelCapability.TOOL_CALLING,
            ModelCapability.STRUCTURED_OUTPUT,
            ModelCapability.EMBEDDINGS,
        ]

        # Context window varies by model
        context_window = 1000000  # Gemini 1.5+ supports 1M tokens
        if "2.0" in self.model:
            context_window = 1000000
        elif "1.5" in self.model:
            context_window = 1000000
        else:
            context_window = 32768

        return ModelInfo(
            name=self.model,
            provider="gemini",
            capabilities=capabilities,
            context_window=context_window,
            max_output_tokens=8192,
            supports_system_prompt=True,
            cost_per_1k_input_tokens=0.00025,  # Gemini Flash pricing
            cost_per_1k_output_tokens=0.0005,
        )


