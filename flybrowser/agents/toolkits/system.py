"""System ToolKit for task lifecycle and control flow.

Provides tools for completing or failing tasks, waiting, and requesting
user input, all built on the fireflyframework-genai ToolKit pattern.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, Optional

from fireflyframework_genai.tools.decorators import firefly_tool
from fireflyframework_genai.tools.toolkit import ToolKit


def create_system_toolkit(
    user_input_callback: Optional[Callable[[str], Coroutine[Any, Any, str]]] = None,
) -> ToolKit:
    """Create a system toolkit for task lifecycle management.

    The returned :class:`ToolKit` contains four tools:

    * **complete** -- mark the current task as successfully completed
    * **fail** -- mark the current task as failed
    * **wait** -- pause execution for a given number of seconds
    * **ask_user** -- request input from the user

    Parameters
    ----------
    user_input_callback:
        Optional async callback invoked when the agent asks the user a
        question.  If *None*, the tool returns a placeholder string.
    """

    @firefly_tool(
        name="complete",
        description=(
            "Mark the current task as successfully completed. Provide a "
            "summary and optional result data."
        ),
        auto_register=False,
    )
    async def complete(summary: str, result: str = "") -> str:
        msg = f"TASK COMPLETE: {summary}"
        if result:
            msg += f"\nResult: {result}"
        return msg

    @firefly_tool(
        name="fail",
        description="Mark the current task as failed with a reason.",
        auto_register=False,
    )
    async def fail(reason: str) -> str:
        return f"TASK FAILED: {reason}"

    @firefly_tool(
        name="wait",
        description=(
            "Wait for a specified number of seconds (0-30). Useful for "
            "waiting for page loads or animations to complete."
        ),
        auto_register=False,
    )
    async def wait(seconds: float = 1.0) -> str:
        capped = min(max(0, seconds), 30)
        await asyncio.sleep(capped)
        return f"Waited {capped} seconds"

    @firefly_tool(
        name="ask_user",
        description="Ask the user a question and wait for their response.",
        auto_register=False,
    )
    async def ask_user(question: str) -> str:
        if user_input_callback:
            response = await user_input_callback(question)
            return f"User response: {response}"
        return f"AWAITING USER INPUT: {question}"

    return ToolKit(
        "system",
        [complete, fail, wait, ask_user],
        description="Tools for task lifecycle management and control flow.",
    )
