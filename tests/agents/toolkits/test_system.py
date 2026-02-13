"""Tests for SystemToolKit."""

import pytest

from flybrowser.agents.toolkits.system import create_system_toolkit


class TestSystemToolKit:
    def test_toolkit_has_four_tools(self):
        toolkit = create_system_toolkit()
        assert len(toolkit.tools) == 4

    def test_tool_names(self):
        toolkit = create_system_toolkit()
        names = {t.name for t in toolkit.tools}
        assert names == {"complete", "fail", "wait", "ask_user"}

    @pytest.mark.asyncio
    async def test_complete(self):
        toolkit = create_system_toolkit()
        tool = next(t for t in toolkit.tools if t.name == "complete")
        result = await tool.execute(summary="All done", result="42")
        assert "TASK COMPLETE: All done" in result
        assert "42" in result

    @pytest.mark.asyncio
    async def test_fail(self):
        toolkit = create_system_toolkit()
        tool = next(t for t in toolkit.tools if t.name == "fail")
        result = await tool.execute(reason="Something went wrong")
        assert "TASK FAILED: Something went wrong" in result

    @pytest.mark.asyncio
    async def test_wait_with_tiny_sleep(self):
        toolkit = create_system_toolkit()
        tool = next(t for t in toolkit.tools if t.name == "wait")
        result = await tool.execute(seconds=0.01)
        assert "Waited" in result
        assert "seconds" in result

    @pytest.mark.asyncio
    async def test_ask_user_without_callback(self):
        toolkit = create_system_toolkit()
        tool = next(t for t in toolkit.tools if t.name == "ask_user")
        result = await tool.execute(question="What next?")
        assert "AWAITING USER INPUT: What next?" in result
