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

"""Tests for framework middleware integration in BrowserAgent."""

import pytest

from fireflyframework_genai.agents.builtin_middleware import (
    CostGuardMiddleware,
    ExplainabilityMiddleware,
    LoggingMiddleware,
)

from flybrowser.agents.browser_agent import BrowserAgent, BrowserAgentConfig
from flybrowser.agents.middleware.obstacle import ObstacleDetectionMiddleware
from flybrowser.agents.middleware.screenshot import ScreenshotOnErrorMiddleware


class TestBrowserAgentFrameworkMiddleware:
    """Verify that framework middleware is wired into BrowserAgent."""

    def _make_agent(self, mock_page_controller, **config_kwargs):
        config = BrowserAgentConfig(**config_kwargs)
        return BrowserAgent(page_controller=mock_page_controller, config=config)

    def test_has_logging_middleware(self, mock_page_controller):
        agent = self._make_agent(mock_page_controller)
        types = [type(mw) for mw in agent._middleware]
        assert LoggingMiddleware in types

    def test_has_cost_guard_middleware(self, mock_page_controller):
        agent = self._make_agent(mock_page_controller, budget_limit_usd=2.0)
        cost_guards = [mw for mw in agent._middleware if isinstance(mw, CostGuardMiddleware)]
        assert len(cost_guards) == 1
        assert cost_guards[0]._budget == 2.0

    def test_has_explainability_middleware(self, mock_page_controller):
        agent = self._make_agent(mock_page_controller)
        types = [type(mw) for mw in agent._middleware]
        assert ExplainabilityMiddleware in types

    def test_custom_middleware_still_present(self, mock_page_controller):
        agent = self._make_agent(mock_page_controller)
        types = [type(mw) for mw in agent._middleware]
        assert ObstacleDetectionMiddleware in types
        assert ScreenshotOnErrorMiddleware in types

    def test_middleware_order(self, mock_page_controller):
        """Framework middleware must run before custom browser middleware."""
        agent = self._make_agent(mock_page_controller)
        types = [type(mw) for mw in agent._middleware]

        # Expected order:
        # LoggingMiddleware -> CostGuardMiddleware -> ExplainabilityMiddleware
        # -> ObstacleDetectionMiddleware -> ScreenshotOnErrorMiddleware
        assert types == [
            LoggingMiddleware,
            CostGuardMiddleware,
            ExplainabilityMiddleware,
            ObstacleDetectionMiddleware,
            ScreenshotOnErrorMiddleware,
        ]

    def test_cost_guard_uses_config_budget(self, mock_page_controller):
        """CostGuardMiddleware budget should come from BrowserAgentConfig.budget_limit_usd."""
        agent = self._make_agent(mock_page_controller, budget_limit_usd=10.0)
        cost_guards = [mw for mw in agent._middleware if isinstance(mw, CostGuardMiddleware)]
        assert len(cost_guards) == 1
        assert cost_guards[0]._budget == 10.0

    def test_default_budget_is_five(self, mock_page_controller):
        """Default config budget_limit_usd=5.0 should flow to CostGuardMiddleware."""
        agent = self._make_agent(mock_page_controller)
        cost_guards = [mw for mw in agent._middleware if isinstance(mw, CostGuardMiddleware)]
        assert len(cost_guards) == 1
        assert cost_guards[0]._budget == 5.0

    def test_middleware_count(self, mock_page_controller):
        """Total middleware count should be 5 (3 framework + 2 custom)."""
        agent = self._make_agent(mock_page_controller)
        assert len(agent._middleware) == 5
