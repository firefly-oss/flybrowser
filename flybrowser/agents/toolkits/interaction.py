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

"""Interaction ToolKit for browser element interaction.

Provides tools for clicking, typing, scrolling, hovering, keyboard
input, form filling, and other DOM interactions, all built on the
fireflyframework-genai ToolKit pattern.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fireflyframework_genai.tools.decorators import firefly_tool
from fireflyframework_genai.tools.toolkit import ToolKit

if TYPE_CHECKING:
    from flybrowser.core.page import PageController


def create_interaction_toolkit(page: PageController) -> ToolKit:
    """Create an interaction toolkit bound to the given *page*.

    The returned :class:`ToolKit` contains seventeen tools for
    interacting with page elements: clicking, typing, scrolling,
    hovering, keyboard input, form filling, file uploads, JavaScript
    evaluation, and more.

    Each tool is a closure over *page* so it can drive the browser
    without requiring dependency injection at call time.
    """

    @firefly_tool(
        name="click",
        description="Click an element identified by the given CSS selector.",
        auto_register=False,
    )
    async def click(selector: str) -> str:
        await page.click_and_track(selector)
        return f"Clicked element: {selector}"

    @firefly_tool(
        name="type_text",
        description=(
            "Type text into an element. Optionally clear the field first "
            "and/or press Enter after typing."
        ),
        auto_register=False,
    )
    async def type_text(
        selector: str,
        text: str,
        clear_first: bool = True,
        press_enter: bool = False,
    ) -> str:
        await page.type_and_track(
            selector, text, clear_first=clear_first, press_enter=press_enter
        )
        return f"Typed '{text}' into {selector}"

    @firefly_tool(
        name="scroll",
        description=(
            "Scroll the page. Direction can be 'up', 'down', 'top', or "
            "'bottom'. Amount is in pixels (default 500)."
        ),
        auto_register=False,
    )
    async def scroll(direction: str = "down", amount: int = 500) -> str:
        if direction == "top":
            await page.page.evaluate("window.scrollTo(0, 0)")
        elif direction == "bottom":
            await page.page.evaluate(
                "window.scrollTo(0, document.body.scrollHeight)"
            )
        elif direction == "up":
            await page.page.evaluate(f"window.scrollBy(0, -{amount})")
        else:
            # default: down
            await page.page.evaluate(f"window.scrollBy(0, {amount})")
        return f"Scrolled {direction}" + (
            f" by {amount}px" if direction in ("up", "down") else ""
        )

    @firefly_tool(
        name="hover",
        description="Hover over an element identified by the given CSS selector.",
        auto_register=False,
    )
    async def hover(selector: str) -> str:
        await page.page.locator(selector).first.hover()
        return f"Hovered over element: {selector}"

    @firefly_tool(
        name="press_key",
        description=(
            "Press a keyboard key (e.g. 'Enter', 'Tab', 'Escape', "
            "'ArrowDown')."
        ),
        auto_register=False,
    )
    async def press_key(key: str) -> str:
        await page.page.keyboard.press(key)
        return f"Pressed key: {key}"

    @firefly_tool(
        name="fill",
        description="Fill an input element with the given value (clears first).",
        auto_register=False,
    )
    async def fill(selector: str, value: str) -> str:
        await page.page.locator(selector).fill(value)
        return f"Filled {selector} with '{value}'"

    @firefly_tool(
        name="select_option",
        description=(
            "Select an option from a <select> element by its visible label. "
            "Falls back to selecting by value if the label is not found."
        ),
        auto_register=False,
    )
    async def select_option(selector: str, option: str) -> str:
        locator = page.page.locator(selector)
        try:
            await locator.select_option(label=option)
        except Exception:
            await locator.select_option(value=option)
        return f"Selected option '{option}' in {selector}"

    @firefly_tool(
        name="check_checkbox",
        description=(
            "Check or uncheck a checkbox. Set checked=True to check, "
            "checked=False to uncheck."
        ),
        auto_register=False,
    )
    async def check_checkbox(selector: str, checked: bool = True) -> str:
        locator = page.page.locator(selector)
        if checked:
            await locator.check()
            return f"Checked element: {selector}"
        else:
            await locator.uncheck()
            return f"Unchecked element: {selector}"

    @firefly_tool(
        name="focus",
        description="Focus on an element identified by the given CSS selector.",
        auto_register=False,
    )
    async def focus(selector: str) -> str:
        await page.page.locator(selector).focus()
        return f"Focused on element: {selector}"

    @firefly_tool(
        name="wait_for_selector",
        description=(
            "Wait for an element to reach the specified state. "
            "State can be 'attached', 'detached', 'visible', or 'hidden'."
        ),
        auto_register=False,
    )
    async def wait_for_selector(
        selector: str, state: str = "visible", timeout: int = 30000
    ) -> str:
        await page.page.locator(selector).wait_for(state=state, timeout=timeout)
        return f"Selector {selector} reached state '{state}'"

    @firefly_tool(
        name="double_click",
        description="Double-click an element identified by the given CSS selector.",
        auto_register=False,
    )
    async def double_click(selector: str) -> str:
        await page.page.locator(selector).dblclick()
        return f"Double-clicked element: {selector}"

    @firefly_tool(
        name="right_click",
        description="Right-click (context-click) an element identified by the given CSS selector.",
        auto_register=False,
    )
    async def right_click(selector: str) -> str:
        await page.page.locator(selector).click(button="right")
        return f"Right-clicked element: {selector}"

    @firefly_tool(
        name="drag_and_drop",
        description="Drag an element from the source selector and drop it onto the target selector.",
        auto_register=False,
    )
    async def drag_and_drop(source: str, target: str) -> str:
        await page.page.locator(source).drag_to(page.page.locator(target))
        return f"Dragged {source} to {target}"

    @firefly_tool(
        name="upload_file",
        description="Upload a file to a file input element.",
        auto_register=False,
    )
    async def upload_file(selector: str, file_path: str) -> str:
        await page.page.locator(selector).set_input_files(file_path)
        return f"Uploaded file '{file_path}' to {selector}"

    @firefly_tool(
        name="evaluate_javascript",
        description="Evaluate a JavaScript expression in the page context and return the result.",
        auto_register=False,
    )
    async def evaluate_javascript(script: str) -> str:
        import re
        cleaned = script.strip()
        # Wrap in IIFE if the script contains bare `return` statements
        # (common LLM mistake — page.evaluate expects an expression, not a function body)
        has_bare_return = bool(re.search(r'(?:^|[;\n{]\s*)return\s', cleaned))
        if has_bare_return:
            cleaned = f"(() => {{ {cleaned} }})()"
        result = await page.page.evaluate(cleaned)
        return f"JavaScript result: {json.dumps(result, default=str)}"

    @firefly_tool(
        name="get_attribute",
        description="Get the value of an attribute on an element.",
        auto_register=False,
    )
    async def get_attribute(selector: str, attribute: str) -> str:
        value = await page.page.locator(selector).get_attribute(attribute)
        return f"Attribute '{attribute}' of {selector}: {value}"

    @firefly_tool(
        name="clear_input",
        description="Clear the contents of an input or textarea element.",
        auto_register=False,
    )
    async def clear_input(selector: str) -> str:
        await page.page.locator(selector).clear()
        return f"Cleared input: {selector}"

    return ToolKit(
        "interaction",
        [
            click,
            type_text,
            scroll,
            hover,
            press_key,
            fill,
            select_option,
            check_checkbox,
            focus,
            wait_for_selector,
            double_click,
            right_click,
            drag_and_drop,
            upload_file,
            evaluate_javascript,
            get_attribute,
            clear_input,
        ],
        description="Tools for interacting with browser page elements.",
    )
