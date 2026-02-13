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

"""Tests for BrowserAgent creation and interface."""

import pytest
from flybrowser.agents.browser_agent import BrowserAgent, BrowserAgentConfig


class TestBrowserAgentConfig:
    def test_defaults(self):
        config = BrowserAgentConfig()
        assert config.model == "openai:gpt-4o"
        assert config.max_iterations == 50
        assert config.max_time == 1800
        assert config.budget_limit_usd == 5.0
        assert config.session_id is None

    def test_custom_model(self):
        config = BrowserAgentConfig(model="anthropic:claude-3-5-sonnet-latest")
        assert config.model == "anthropic:claude-3-5-sonnet-latest"

    def test_custom_iterations(self):
        config = BrowserAgentConfig(max_iterations=100, max_time=3600)
        assert config.max_iterations == 100
        assert config.max_time == 3600


class TestBrowserAgentCreation:
    def test_creates_with_page_controller(self, mock_page_controller):
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=BrowserAgentConfig()
        )
        assert agent is not None

    def test_has_all_toolkits(self, mock_page_controller):
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=BrowserAgentConfig()
        )
        names = {tk.name for tk in agent._toolkits}
        assert "navigation" in names
        assert "interaction" in names
        assert "extraction" in names
        assert "system" in names
        assert "search" in names
        assert "captcha" in names

    def test_has_memory(self, mock_page_controller):
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=BrowserAgentConfig()
        )
        assert agent.memory is not None

    def test_has_middleware(self, mock_page_controller):
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=BrowserAgentConfig()
        )
        assert len(agent._middleware) == 5

    def test_has_methods(self, mock_page_controller):
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=BrowserAgentConfig()
        )
        assert callable(agent.act)
        assert callable(agent.extract)
        assert callable(agent.observe)
        assert callable(agent.run_task)
        assert callable(agent.agent_stream)

    def test_format_context_empty(self, mock_page_controller):
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=BrowserAgentConfig()
        )
        assert agent._format_context(None) == ""
        assert agent._format_context({}) == ""

    def test_format_context_with_data(self, mock_page_controller):
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=BrowserAgentConfig()
        )
        result = agent._format_context({"url": "https://example.com", "step": "1"})
        assert "url: https://example.com" in result
        assert "step: 1" in result

    def test_format_result_dict_passthrough(self, mock_page_controller):
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=BrowserAgentConfig()
        )
        d = {"success": True, "data": "hello"}
        assert agent._format_result(d, "test") is d

    def test_format_result_string_wrapped(self, mock_page_controller):
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=BrowserAgentConfig()
        )
        result = agent._format_result("some output", "my task")
        assert result["success"] is True
        assert result["result"] == "some output"
        assert result["task"] == "my task"

    def test_format_result_none_wrapped(self, mock_page_controller):
        agent = BrowserAgent(
            page_controller=mock_page_controller, config=BrowserAgentConfig()
        )
        result = agent._format_result(None, "my task")
        assert result["success"] is True
        assert result["result"] == ""
