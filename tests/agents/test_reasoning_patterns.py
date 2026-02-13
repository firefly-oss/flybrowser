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

"""Tests for configurable reasoning pattern selection in BrowserAgent."""

import pytest
from fireflyframework_genai.reasoning import (
    PlanAndExecutePattern,
    ReActPattern,
    ReflexionPattern,
)

from flybrowser.agents.browser_agent import BrowserAgent, BrowserAgentConfig
from flybrowser.agents.types import ReasoningStrategy


class TestReasoningPatternSelection:
    """Verify that BrowserAgent creates the correct reasoning pattern based on config."""

    def test_default_reasoning_is_react(self, mock_page_controller):
        """Default strategy should be REACT_STANDARD, producing a ReActPattern."""
        config = BrowserAgentConfig()
        assert config.reasoning_strategy == ReasoningStrategy.REACT_STANDARD

        agent = BrowserAgent(
            page_controller=mock_page_controller, config=config
        )
        assert isinstance(agent._react, ReActPattern)

    def test_plan_and_solve_strategy(self, mock_page_controller):
        """PLAN_AND_SOLVE strategy should produce a PlanAndExecutePattern."""
        config = BrowserAgentConfig(
            reasoning_strategy=ReasoningStrategy.PLAN_AND_SOLVE
        )
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=config
        )
        assert isinstance(agent._react, PlanAndExecutePattern)

    def test_self_reflection_strategy(self, mock_page_controller):
        """SELF_REFLECTION strategy should produce a ReflexionPattern."""
        config = BrowserAgentConfig(
            reasoning_strategy=ReasoningStrategy.SELF_REFLECTION
        )
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=config
        )
        assert isinstance(agent._react, ReflexionPattern)

    def test_config_accepts_reasoning_strategy(self):
        """BrowserAgentConfig should store the reasoning_strategy field."""
        for strategy in (
            ReasoningStrategy.REACT_STANDARD,
            ReasoningStrategy.PLAN_AND_SOLVE,
            ReasoningStrategy.SELF_REFLECTION,
        ):
            config = BrowserAgentConfig(reasoning_strategy=strategy)
            assert config.reasoning_strategy is strategy

    def test_reasoning_strategy_stored_on_agent(self, mock_page_controller):
        """BrowserAgent should expose _reasoning_strategy attribute."""
        config = BrowserAgentConfig(
            reasoning_strategy=ReasoningStrategy.PLAN_AND_SOLVE
        )
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=config
        )
        assert agent._reasoning_strategy == ReasoningStrategy.PLAN_AND_SOLVE

    def test_unknown_strategy_falls_back_to_react(self, mock_page_controller):
        """An unrecognized strategy value should fall back to ReActPattern."""
        config = BrowserAgentConfig(
            reasoning_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
        )
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=config
        )
        # Strategies without a dedicated mapping should default to ReAct
        assert isinstance(agent._react, ReActPattern)
