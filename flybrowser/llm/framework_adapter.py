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

"""Bridge between BaseLLMProvider (used by ElementDetector) and pydantic-ai models.

After the migration to fireflyframework-genai, the old per-provider LLM classes
(OpenAIProvider, AnthropicProvider, etc.) were removed.  BrowserAgent now talks
to the framework's FireflyAgent, which internally uses pydantic-ai.

The ElementDetector still requires a BaseLLMProvider instance for its fast-path
element detection.  This adapter satisfies that interface by delegating to
pydantic-ai directly, supporting all providers that pydantic-ai supports.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, List, Optional

from pydantic_ai import Agent as PydanticAgent

from flybrowser.llm.base import (
    BaseLLMProvider,
    LLMResponse,
    ModelCapability,
    ModelInfo,
)

logger = logging.getLogger(__name__)

# Models known to support vision
_VISION_PATTERNS = [
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-5",
    "claude-3",
    "claude-sonnet",
    "claude-opus",
    "gemini",
    "qwen-vl",
]


class FrameworkLLMAdapter(BaseLLMProvider):
    """BaseLLMProvider implementation backed by pydantic-ai.

    Parameters
    ----------
    model_str : str
        pydantic-ai model string, e.g. ``"anthropic:claude-sonnet-4-5-20250929"``
        or ``"openai:gpt-4o"``.
    api_key : str | None
        Forwarded to pydantic-ai (usually read from env var automatically).
    """

    def __init__(self, model_str: str, api_key: Optional[str] = None, **kwargs: Any) -> None:
        # Extract the short model name (after the colon) for display
        short_name = model_str.split(":", 1)[-1] if ":" in model_str else model_str
        super().__init__(model=short_name, api_key=api_key, **kwargs)
        self._model_str = model_str

    # ------------------------------------------------------------------
    # Core generation methods
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        agent = PydanticAgent(
            self._model_str,
            system_prompt=system_prompt or "",
        )
        result = await agent.run(prompt)
        text = result.output if hasattr(result, "output") else str(result.data)
        return LLMResponse(
            content=text,
            model=self.model,
            usage=self._extract_usage(result),
        )

    async def generate_with_vision(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        from pydantic_ai.messages import BinaryContent

        agent = PydanticAgent(
            self._model_str,
            system_prompt=system_prompt or "",
        )

        # Build user prompt parts: text + images
        parts: list[Any] = [prompt]
        for img in images or []:
            if isinstance(img, bytes):
                parts.append(BinaryContent(data=img, media_type="image/png"))
            elif isinstance(img, str):
                parts.append(BinaryContent(data=base64.b64decode(img), media_type="image/png"))

        result = await agent.run(parts)
        text = result.output if hasattr(result, "output") else str(result.data)
        return LLMResponse(
            content=text,
            model=self.model,
            usage=self._extract_usage(result),
        )

    async def generate_structured(
        self,
        prompt: str,
        schema: Any = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # Structured output — just add schema instruction to the prompt
        if schema:
            import json
            schema_str = json.dumps(schema, indent=2) if isinstance(schema, dict) else str(schema)
            prompt = f"{prompt}\n\nRespond with JSON matching this schema:\n```json\n{schema_str}\n```"
        return await self.generate(prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens)

    @staticmethod
    def _extract_usage(result: Any) -> dict:
        """Extract token usage dict from a pydantic-ai RunResult."""
        usage_obj = getattr(result, "usage", None) or getattr(result, "_usage", None)
        if usage_obj is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return {
            "prompt_tokens": getattr(usage_obj, "request_tokens", 0) or getattr(usage_obj, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage_obj, "response_tokens", 0) or getattr(usage_obj, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
        }

    # ------------------------------------------------------------------
    # Capability detection
    # ------------------------------------------------------------------

    def supports_vision(self) -> bool:
        model_lower = self._model_str.lower()
        return any(p in model_lower for p in _VISION_PATTERNS)

    def get_model_info(self) -> ModelInfo:
        caps = [ModelCapability.TEXT_GENERATION]
        if self.supports_vision():
            caps.append(ModelCapability.VISION)
        return ModelInfo(
            name=self.model,
            provider=self._model_str.split(":")[0] if ":" in self._model_str else "unknown",
            capabilities=caps,
            context_window=128000,
        )

    def get_usage_summary(self) -> dict:
        return {
            "prompt_tokens": self._session_prompt_tokens,
            "completion_tokens": self._session_completion_tokens,
            "total_tokens": self._session_total_tokens,
            "cost_usd": self._session_cost,
            "calls_count": self._session_calls,
            "cached_calls": self._session_cached_calls,
            "model": self.model,
        }
