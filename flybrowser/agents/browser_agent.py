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

"""BrowserAgent — main browser automation agent built on FireflyAgent."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional, Type

from fireflyframework_genai.agents import FireflyAgent
from fireflyframework_genai.agents.builtin_middleware import (
    CostGuardMiddleware,
    ExplainabilityMiddleware,
    LoggingMiddleware,
)
from fireflyframework_genai.reasoning import (
    PlanAndExecutePattern,
    ReActPattern,
    ReflexionPattern,
)
from fireflyframework_genai.reasoning.base import AbstractReasoningPattern
from fireflyframework_genai.validation import OutputReviewer

from flybrowser.agents.toolkits import create_all_toolkits
from flybrowser.agents.types import ReasoningStrategy
from flybrowser.agents.memory.browser_memory import BrowserMemoryManager
from flybrowser.agents.middleware.obstacle import ObstacleDetectionMiddleware
from flybrowser.agents.middleware.screenshot import ScreenshotOnErrorMiddleware

if TYPE_CHECKING:
    from flybrowser.core.page import PageController


@dataclass
class BrowserAgentConfig:
    model: str = "openai:gpt-4o"
    max_iterations: int = 50
    max_time: int = 1800
    budget_limit_usd: float = 5.0
    session_id: Optional[str] = None
    reasoning_strategy: ReasoningStrategy = ReasoningStrategy.REACT_STANDARD


_SYSTEM_INSTRUCTIONS = """You are a browser automation agent. You control a real web browser via tools.

RULES:
1. Always call get_page_state or extract_text first to understand the current page.
2. Use CSS selectors for precision. Prefer IDs > data attributes > class names.
3. After clicking, check if the page navigated by calling get_page_state again.
4. If an element is not found, scroll or wait before retrying.
5. For forms, fill all required fields before submitting.
6. Call 'complete' with a summary when the task is done.
7. Call 'fail' with a reason if the task is impossible after trying alternatives.
8. Never hallucinate selectors — observe the page first.
"""

_ACT_PROMPT = "Execute this browser action: {instruction}\n\n{context_section}\n\nPerform the action and call 'complete' with a summary."
_EXTRACT_PROMPT = "Extract the following data from the current page: {query}\n\nReturn the extracted data as structured text."
_OBSERVE_PROMPT = "Find elements on the page matching: {query}\n\n{context_section}\n\nDescribe each matching element with its selector, text, and position. Do NOT interact."


class BrowserAgent:
    def __init__(
        self,
        page_controller: "PageController",
        config: BrowserAgentConfig,
        search_coordinator: Optional[Any] = None,
        captcha_solver: Optional[Any] = None,
        user_input_callback: Optional[Any] = None,
    ) -> None:
        self._page = page_controller
        self._config = config
        self._toolkits = create_all_toolkits(
            page=page_controller,
            search_coordinator=search_coordinator,
            captcha_solver=captcha_solver,
            user_input_callback=user_input_callback,
        )
        self._memory = BrowserMemoryManager()
        self._middleware = [
            LoggingMiddleware(),
            CostGuardMiddleware(budget_usd=config.budget_limit_usd),
            ExplainabilityMiddleware(),
            ObstacleDetectionMiddleware(page_controller),
            ScreenshotOnErrorMiddleware(page_controller),
        ]
        self._agent = FireflyAgent(
            name="flybrowser",
            model=config.model,
            instructions=_SYSTEM_INSTRUCTIONS,
            tools=self._toolkits,
            middleware=self._middleware,
        )
        self._reasoning_strategy = config.reasoning_strategy
        self._react = self._create_reasoning_pattern(config)

    @staticmethod
    def _create_reasoning_pattern(config: BrowserAgentConfig) -> AbstractReasoningPattern:
        """Select and instantiate a reasoning pattern based on the configured strategy."""
        strategy = config.reasoning_strategy
        if strategy == ReasoningStrategy.PLAN_AND_SOLVE:
            return PlanAndExecutePattern(max_steps=config.max_iterations)
        if strategy == ReasoningStrategy.SELF_REFLECTION:
            return ReflexionPattern(max_steps=config.max_iterations)
        # REACT_STANDARD and any other strategy default to ReAct
        return ReActPattern(max_steps=config.max_iterations)

    @property
    def memory(self) -> BrowserMemoryManager:
        return self._memory

    async def act(self, instruction: str, context: Optional[dict] = None) -> dict:
        prompt = _ACT_PROMPT.format(
            instruction=instruction,
            context_section=self._format_context(context),
        )
        result = await self._agent.run(prompt)
        return self._format_result(result, instruction)

    async def extract(
        self,
        query: str,
        schema: Optional[Type] = None,
        context: Optional[dict] = None,
        max_retries: int = 3,
    ) -> dict:
        prompt = _EXTRACT_PROMPT.format(query=query)
        if schema:
            reviewer = OutputReviewer(output_type=schema, max_retries=max_retries)
            review_result = await reviewer.review(self._agent, prompt)
            return self._format_result(review_result.output, query)
        else:
            result = await self._agent.run(prompt)
            return self._format_result(result, query)

    async def observe(self, query: str, context: Optional[dict] = None) -> dict:
        prompt = _OBSERVE_PROMPT.format(
            query=query,
            context_section=self._format_context(context),
        )
        result = await self._agent.run(prompt)
        return self._format_result(result, query)

    async def run_task(self, task: str, context: Optional[dict] = None) -> dict:
        memory_ctx = self._memory.format_for_prompt()
        full_prompt = f"{task}\n\nBrowser state:\n{memory_ctx}"
        if context:
            full_prompt += f"\n\n{self._format_context(context)}"
        result = await self._agent.run_with_reasoning(
            self._react, full_prompt, timeout=self._config.max_time,
        )
        return self._format_result(result, task)

    async def agent_stream(self, task: str) -> AsyncIterator[dict]:
        async with await self._agent.run_stream(
            task, streaming_mode="incremental",
        ) as stream:
            async for token in stream.stream_tokens():
                yield {"type": "thought", "content": token, "timestamp": time.time()}

    def _format_context(self, context: Optional[dict]) -> str:
        if not context:
            return ""
        return "Context:\n" + "\n".join(f"{k}: {v}" for k, v in context.items())

    def _format_result(self, result: Any, task: str) -> dict:
        if isinstance(result, dict):
            return result
        return {"success": True, "result": str(result) if result else "", "task": task}
