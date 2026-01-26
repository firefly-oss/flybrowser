# Copyright 2026 Firefly Software Solutions Inc.
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
Element interaction tools for browser control.

This module provides ReAct-compatible tools for interacting with
page elements including clicking, typing, scrolling, and hovering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from flybrowser.agents.types import SafetyLevel, ToolCategory, ToolResult
from flybrowser.agents.tools.base import BaseTool, ToolMetadata, ToolParameter

if TYPE_CHECKING:
    from flybrowser.core.page import PageController
    from flybrowser.core.element import ElementDetector


class ClickTool(BaseTool):
    """Click an element on the page."""
    
    def __init__(
        self,
        page_controller: PageController,
        element_detector: Optional[ElementDetector] = None,
    ) -> None:
        self._page_controller = page_controller
        self._element_detector = element_detector
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="click",
            description=(
                "Click an element on the page. Use CSS selectors for precision, or natural language "
                "with use_ai=true for dynamic pages. Supports buttons, links, checkboxes, radio buttons, "
                "and any clickable element. After clicking, the page may navigate or update dynamically."
            ),
            category=ToolCategory.INTERACTION,
            safety_level=SafetyLevel.MODERATE,
            parameters=[
                ToolParameter(
                    name="selector",
                    type="string",
                    description=(
                        "CSS selector (e.g., '#submit-btn', '.nav-link', 'button[type=submit]') "
                        "or natural language description (e.g., 'the blue login button', 'Sign In link')"
                    ),
                    required=True,
                ),
                ToolParameter(
                    name="use_ai",
                    type="boolean",
                    description=(
                        "Set true to use AI/VLM to find element by visual appearance or description. "
                        "Use when CSS selector is unknown or element is dynamically generated."
                    ),
                    required=False,
                ),
            ],
            returns_description="Returns clicked element info, navigation status, and method used",
            examples=[
                'click({"selector": "#login-button"}) - Click by ID',
                'click({"selector": ".submit-form", "use_ai": false}) - Click by class',
                'click({"selector": "Sign In", "use_ai": true}) - AI finds the Sign In button',
                'click({"selector": "button[data-action=checkout]"}) - Click by attribute',
            ],
            requires_page=True,
        )
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute click on element."""
        selector = kwargs.get("selector")
        use_ai = kwargs.get("use_ai", False)
        
        if not selector:
            return ToolResult.error_result("Selector is required")
        
        try:
            if use_ai and self._element_detector:
                element_info = await self._element_detector.find_element(selector)
                actual_selector = element_info.get("selector", selector)
                selector_type = element_info.get("selector_type", "css")
                await self._element_detector.click(
                    actual_selector, selector_type=selector_type
                )
                return ToolResult.success_result(
                    data={
                        "clicked": actual_selector,
                        "method": "ai_detection",
                        "confidence": element_info.get("confidence", 0),
                        "message": f"Clicked element: {actual_selector}",
                    },
                )
            else:
                result = await self._page_controller.click_and_track(selector)
                # Check if click actually succeeded
                if result.get("success", False):
                    return ToolResult.success_result(
                        data={
                            "clicked": selector,
                            "method": "direct_selector",
                            "navigated": result.get("navigated", False),
                            "message": f"Clicked element: {selector}",
                        },
                        metadata=result,
                    )
                else:
                    return ToolResult.error_result(
                        f"Click failed: {result.get('error', 'Unknown error')}",
                        data=result,
                    )
        except Exception as e:
            return ToolResult.error_result(f"Click failed: {str(e)}")


class TypeTool(BaseTool):
    """Type text into an input field."""
    
    def __init__(
        self,
        page_controller: PageController,
        element_detector: Optional[ElementDetector] = None,
    ) -> None:
        self._page_controller = page_controller
        self._element_detector = element_detector
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="type_text",
            description=(
                "Type text into an input field, textarea, or contenteditable element. "
                "By default clears existing content first. Use press_enter=true to submit "
                "search forms or trigger autocomplete. Works with login forms, search boxes, "
                "and any text input."
            ),
            category=ToolCategory.INTERACTION,
            safety_level=SafetyLevel.MODERATE,
            parameters=[
                ToolParameter(
                    name="selector",
                    type="string",
                    description=(
                        "CSS selector of the input field (e.g., 'input[name=email]', '#search', "
                        "'textarea.comment', '[placeholder=Search]')"
                    ),
                    required=True,
                ),
                ToolParameter(
                    name="text",
                    type="string",
                    description="The text to type into the field",
                    required=True,
                ),
                ToolParameter(
                    name="clear_first",
                    type="boolean",
                    description="Clear existing content before typing (default: true)",
                    required=False,
                    default=True,
                ),
                ToolParameter(
                    name="press_enter",
                    type="boolean",
                    description="Press Enter after typing to submit (default: false)",
                    required=False,
                    default=False,
                ),
            ],
            returns_description="Returns typed text, selector used, navigation status if Enter was pressed",
            examples=[
                'type_text({"selector": "input[name=q]", "text": "python tutorials", "press_enter": true}) - Search',
                'type_text({"selector": "#email", "text": "user@example.com"}) - Fill email field',
                'type_text({"selector": "textarea", "text": "Hello world", "clear_first": false}) - Append text',
            ],
            requires_page=True,
        )
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute typing into field.
        
        Supports both explicit parameters and context-based form filling.
        Context format: {"form_data": {"field_selector": "value", ...}}
        """
        selector = kwargs.get("selector")
        text = kwargs.get("text")
        clear_first = kwargs.get("clear_first", True)
        press_enter = kwargs.get("press_enter", False)

        # Check if text is not provided - try to get from context form_data
        if text is None and selector:
            user_context = self.get_user_context()
            form_data = user_context.get("form_data", {})
            
            # Try to match selector to a form_data key
            if selector in form_data:
                text = form_data[selector]
            else:
                # Try to match by field name or ID
                for field_key, field_value in form_data.items():
                    # Check if selector contains the field key
                    if field_key.lower() in selector.lower():
                        text = field_value
                        break
        
        if not selector:
            return ToolResult.error_result("Selector is required")
        if text is None:
            return ToolResult.error_result("Text is required (or provide context with form_data)")

        try:
            result = await self._page_controller.type_and_track(
                selector, text, clear_first=clear_first, press_enter=press_enter
            )
            # Check if typing actually succeeded
            if result.get("success", False):
                return ToolResult.success_result(
                    data={
                        "typed_text": text,
                        "selector": selector,
                        "pressed_enter": press_enter,
                        "navigated": result.get("navigated", False),
                        "message": f"Typed '{text}' into {selector}",
                    },
                    metadata=result,
                )
            else:
                return ToolResult.error_result(
                    f"Type failed: {result.get('error', 'Unknown error')}",
                    data=result,
                )
        except Exception as e:
            return ToolResult.error_result(f"Type failed: {str(e)}")


class ScrollTool(BaseTool):
    """Scroll the page in a specified direction."""

    def __init__(self, page_controller: PageController) -> None:
        self._page_controller = page_controller

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="scroll",
            description="Scroll the page up, down, to top, or to bottom.",
            category=ToolCategory.INTERACTION,
            safety_level=SafetyLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="direction",
                    type="string",
                    description="Direction to scroll: 'up', 'down', 'top' (scroll to top), or 'bottom' (scroll to bottom)",
                    required=True,
                    enum=["up", "down", "top", "bottom"],
                ),
                ToolParameter(
                    name="amount",
                    type="number",
                    description="Amount to scroll in pixels (default: 500, ignored for 'top'/'bottom')",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute page scroll with real position feedback."""
        direction = kwargs.get("direction")
        amount = kwargs.get("amount", 500)

        if not direction:
            return ToolResult.error_result("Direction is required")

        if direction not in ("up", "down", "top", "bottom"):
            return ToolResult.error_result(
                f"Invalid direction: {direction}. Must be 'up', 'down', 'top', or 'bottom'"
            )

        try:
            page = self._page_controller.page
            
            # Get scroll position BEFORE scrolling
            before_position = await page.evaluate("""
                () => ({
                    x: window.pageXOffset || document.documentElement.scrollLeft,
                    y: window.pageYOffset || document.documentElement.scrollTop,
                    maxY: document.documentElement.scrollHeight - window.innerHeight,
                    maxX: document.documentElement.scrollWidth - window.innerWidth
                })
            """)
            
            # Execute scroll based on direction
            if direction == "top":
                await page.evaluate("window.scrollTo(0, 0)")
                scroll_description = "Scrolled to top of page"
            elif direction == "bottom":
                await page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
                scroll_description = "Scrolled to bottom of page"
            elif direction == "down":
                await page.evaluate(f"window.scrollBy(0, {amount})")
                scroll_description = f"Scrolled down by {amount}px"
            elif direction == "up":
                await page.evaluate(f"window.scrollBy(0, -{amount})")
                scroll_description = f"Scrolled up by {amount}px"
            
            # Get scroll position AFTER scrolling
            after_position = await page.evaluate("""
                () => ({
                    x: window.pageXOffset || document.documentElement.scrollLeft,
                    y: window.pageYOffset || document.documentElement.scrollTop,
                    maxY: document.documentElement.scrollHeight - window.innerHeight,
                    maxX: document.documentElement.scrollWidth - window.innerWidth
                })
            """)
            
            # Calculate actual scroll delta
            actual_delta_y = after_position["y"] - before_position["y"]
            actual_delta_x = after_position["x"] - before_position["x"]
            
            # Determine if we're at boundaries
            at_top = after_position["y"] <= 0
            at_bottom = after_position["y"] >= after_position["maxY"] - 1
            
            return ToolResult.success_result(
                data={
                    "direction": direction,
                    "requested_amount": amount if direction in ("up", "down") else None,
                    "actual_scroll_y": actual_delta_y,
                    "actual_scroll_x": actual_delta_x,
                    "position_before": {"x": before_position["x"], "y": before_position["y"]},
                    "position_after": {"x": after_position["x"], "y": after_position["y"]},
                    "at_top": at_top,
                    "at_bottom": at_bottom,
                    "page_height": after_position["maxY"] + before_position.get("viewportHeight", 0),
                    "message": scroll_description,
                },
            )
        except Exception as e:
            return ToolResult.error_result(f"Scroll failed: {str(e)}")


class HoverTool(BaseTool):
    """Hover over an element on the page."""

    def __init__(self, page_controller: PageController) -> None:
        self._page_controller = page_controller

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="hover",
            description="Move the mouse to hover over an element, useful for revealing tooltips or dropdown menus.",
            category=ToolCategory.INTERACTION,
            safety_level=SafetyLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="selector",
                    type="string",
                    description="CSS selector of the element to hover over",
                    required=True,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute hover over element with real position feedback."""
        selector = kwargs.get("selector")

        if not selector:
            return ToolResult.error_result("Selector is required")

        try:
            page = self._page_controller.page
            locator = page.locator(selector)
            
            # Check if element exists and is visible
            element_count = await locator.count()
            if element_count == 0:
                return ToolResult.error_result(f"Element not found: {selector}")
            
            # Get element bounding box before hover
            bounding_box = await locator.first.bounding_box()
            
            # Perform hover
            await locator.first.hover()
            
            # Get element text/attributes for feedback
            element_text = await locator.first.text_content() or ""
            element_tag = await locator.first.evaluate("el => el.tagName.toLowerCase()")
            
            return ToolResult.success_result(
                data={
                    "selector": selector,
                    "element_tag": element_tag,
                    "element_text": element_text[:100] if element_text else "",
                    "bounding_box": bounding_box,
                    "message": f"Successfully hovered over {element_tag} element: {selector}",
                },
            )
        except Exception as e:
            return ToolResult.error_result(f"Hover failed: {str(e)}")


class PressKeyTool(BaseTool):
    """Press a keyboard key."""

    def __init__(self, page_controller: PageController) -> None:
        self._page_controller = page_controller

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="press_key",
            description="Press a keyboard key (Enter, Escape, Tab, ArrowDown, etc.).",
            category=ToolCategory.INTERACTION,
            safety_level=SafetyLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="key",
                    type="string",
                    description="The key to press (e.g., 'Enter', 'Escape', 'Tab', 'ArrowDown', 'ArrowUp')",
                    required=True,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute key press with feedback."""
        key = kwargs.get("key")

        if not key:
            return ToolResult.error_result("Key is required")

        try:
            page = self._page_controller.page
            
            # Get focused element before key press
            focused_before = await page.evaluate("""
                () => {
                    const el = document.activeElement;
                    return el ? {
                        tag: el.tagName.toLowerCase(),
                        id: el.id || null,
                        className: el.className || null,
                        type: el.type || null
                    } : null;
                }
            """)
            
            # Press the key
            await page.keyboard.press(key)
            
            # Get focused element after key press (may have changed)
            focused_after = await page.evaluate("""
                () => {
                    const el = document.activeElement;
                    return el ? {
                        tag: el.tagName.toLowerCase(),
                        id: el.id || null,
                        className: el.className || null,
                        type: el.type || null
                    } : null;
                }
            """)
            
            focus_changed = focused_before != focused_after
            
            return ToolResult.success_result(
                data={
                    "key": key,
                    "focused_element_before": focused_before,
                    "focused_element_after": focused_after,
                    "focus_changed": focus_changed,
                    "message": f"Pressed '{key}' key" + (" (focus changed)" if focus_changed else ""),
                },
            )
        except Exception as e:
            return ToolResult.error_result(f"Press key failed: {str(e)}")


class SelectOptionTool(BaseTool):
    """Select an option from a dropdown/select element using AI or CSS selector."""

    def __init__(
        self,
        page_controller: PageController,
        element_detector: Optional[ElementDetector] = None,
    ) -> None:
        self._page_controller = page_controller
        self._element_detector = element_detector

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="select_option",
            description="Select an option from a dropdown or select element. Can use CSS selector or natural language description.",
            category=ToolCategory.INTERACTION,
            safety_level=SafetyLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="selector",
                    type="string",
                    description="CSS selector or natural language description of the select element (e.g., 'country dropdown', '#country-select')",
                    required=True,
                ),
                ToolParameter(
                    name="option",
                    type="string",
                    description="The option to select - can be the visible text, value, or description",
                    required=True,
                ),
                ToolParameter(
                    name="use_ai",
                    type="boolean",
                    description="Whether to use AI/VLM to find the element and option (for natural language)",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute option selection with AI support."""
        selector = kwargs.get("selector")
        option = kwargs.get("option")
        use_ai = kwargs.get("use_ai", False)

        if not selector:
            return ToolResult.error_result("Selector is required")
        if not option:
            return ToolResult.error_result("Option to select is required")

        try:
            if use_ai and self._element_detector:
                # Use AI to find and interact with the select element
                element_info = await self._element_detector.find_element(
                    f"dropdown or select element: {selector}"
                )
                actual_selector = element_info.get("selector", selector)
                
                # Try to select the option using AI understanding
                result = await self._element_detector.select_option(
                    actual_selector, option
                )
                return ToolResult.success_result(
                    data={
                        "selector": actual_selector,
                        "selected_option": option,
                        "method": "ai_detection",
                        "confidence": element_info.get("confidence", 0),
                        "message": f"Selected '{option}' from {actual_selector}",
                    },
                )
            else:
                # Direct CSS selector approach
                page = self._page_controller.page
                locator = page.locator(selector)
                
                if await locator.count() == 0:
                    return ToolResult.error_result(f"Select element not found: {selector}")
                
                # Try selecting by label first, then value
                try:
                    selected_values = await locator.select_option(label=option)
                except Exception:
                    selected_values = await locator.select_option(value=option)
                
                selected_text = await locator.evaluate("""
                    el => {
                        const opt = el.options[el.selectedIndex];
                        return opt ? opt.text : null;
                    }
                """)
                
                return ToolResult.success_result(
                    data={
                        "selector": selector,
                        "selected_values": selected_values,
                        "selected_text": selected_text,
                        "method": "direct_selector",
                        "message": f"Selected '{selected_text}' from {selector}",
                    },
                )
        except Exception as e:
            return ToolResult.error_result(f"Select option failed: {str(e)}")


class CheckboxTool(BaseTool):
    """Check or uncheck a checkbox element using AI or CSS selector."""

    def __init__(
        self,
        page_controller: PageController,
        element_detector: Optional[ElementDetector] = None,
    ) -> None:
        self._page_controller = page_controller
        self._element_detector = element_detector

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="check_checkbox",
            description="Check or uncheck a checkbox. Can use CSS selector or natural language description.",
            category=ToolCategory.INTERACTION,
            safety_level=SafetyLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="selector",
                    type="string",
                    description="CSS selector or natural language description (e.g., 'remember me checkbox', '#terms-agree')",
                    required=True,
                ),
                ToolParameter(
                    name="checked",
                    type="boolean",
                    description="Whether to check (true) or uncheck (false) the checkbox",
                    required=True,
                ),
                ToolParameter(
                    name="use_ai",
                    type="boolean",
                    description="Whether to use AI/VLM to find the checkbox (for natural language)",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute checkbox check/uncheck with AI support."""
        selector = kwargs.get("selector")
        checked = kwargs.get("checked", True)
        use_ai = kwargs.get("use_ai", False)

        if not selector:
            return ToolResult.error_result("Selector is required")

        try:
            if use_ai and self._element_detector:
                element_info = await self._element_detector.find_element(
                    f"checkbox: {selector}"
                )
                actual_selector = element_info.get("selector", selector)
                
                if checked:
                    await self._element_detector.check(actual_selector)
                else:
                    await self._element_detector.uncheck(actual_selector)
                
                return ToolResult.success_result(
                    data={
                        "selector": actual_selector,
                        "is_checked": checked,
                        "method": "ai_detection",
                        "confidence": element_info.get("confidence", 0),
                        "message": f"{'Checked' if checked else 'Unchecked'} checkbox: {actual_selector}",
                    },
                )
            else:
                page = self._page_controller.page
                locator = page.locator(selector)
                
                if await locator.count() == 0:
                    return ToolResult.error_result(f"Checkbox not found: {selector}")
                
                was_checked = await locator.is_checked()
                
                if checked:
                    await locator.check()
                else:
                    await locator.uncheck()
                
                is_checked = await locator.is_checked()
                changed = was_checked != is_checked
                
                return ToolResult.success_result(
                    data={
                        "selector": selector,
                        "was_checked": was_checked,
                        "is_checked": is_checked,
                        "changed": changed,
                        "method": "direct_selector",
                        "message": f"{'Checked' if checked else 'Unchecked'} checkbox: {selector}" + ("" if changed else " (no change needed)"),
                    },
                )
        except Exception as e:
            return ToolResult.error_result(f"Checkbox operation failed: {str(e)}")


class FocusTool(BaseTool):
    """Focus on an element using AI or CSS selector."""

    def __init__(
        self,
        page_controller: PageController,
        element_detector: Optional[ElementDetector] = None,
    ) -> None:
        self._page_controller = page_controller
        self._element_detector = element_detector

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="focus",
            description="Focus on an element. Useful before typing or interacting. Can use CSS selector or natural language.",
            category=ToolCategory.INTERACTION,
            safety_level=SafetyLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="selector",
                    type="string",
                    description="CSS selector or natural language description (e.g., 'search box', 'email input')",
                    required=True,
                ),
                ToolParameter(
                    name="use_ai",
                    type="boolean",
                    description="Whether to use AI/VLM to find the element (for natural language)",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute focus with AI support."""
        selector = kwargs.get("selector")
        use_ai = kwargs.get("use_ai", False)

        if not selector:
            return ToolResult.error_result("Selector is required")

        try:
            if use_ai and self._element_detector:
                element_info = await self._element_detector.find_element(selector)
                actual_selector = element_info.get("selector", selector)
                
                await self._element_detector.focus(actual_selector)
                
                return ToolResult.success_result(
                    data={
                        "selector": actual_selector,
                        "method": "ai_detection",
                        "confidence": element_info.get("confidence", 0),
                        "message": f"Focused on element: {actual_selector}",
                    },
                )
            else:
                page = self._page_controller.page
                locator = page.locator(selector)
                
                if await locator.count() == 0:
                    return ToolResult.error_result(f"Element not found: {selector}")
                
                await locator.focus()
                
                element_info = await locator.evaluate("""
                    el => ({
                        tag: el.tagName.toLowerCase(),
                        type: el.type || null,
                        id: el.id || null,
                        name: el.name || null,
                        placeholder: el.placeholder || null
                    })
                """)
                
                return ToolResult.success_result(
                    data={
                        "selector": selector,
                        "element": element_info,
                        "method": "direct_selector",
                        "message": f"Focused on {element_info['tag']} element: {selector}",
                    },
                )
        except Exception as e:
            return ToolResult.error_result(f"Focus failed: {str(e)}")


class FillTool(BaseTool):
    """Fill a form field using AI or CSS selector."""

    def __init__(
        self,
        page_controller: PageController,
        element_detector: Optional[ElementDetector] = None,
    ) -> None:
        self._page_controller = page_controller
        self._element_detector = element_detector

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="fill",
            description="Fill a form field by clearing it and typing the value. Can use CSS selector or natural language.",
            category=ToolCategory.INTERACTION,
            safety_level=SafetyLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="selector",
                    type="string",
                    description="CSS selector or natural language description (e.g., 'username field', 'email input')",
                    required=True,
                ),
                ToolParameter(
                    name="value",
                    type="string",
                    description="The value to fill in the field",
                    required=True,
                ),
                ToolParameter(
                    name="use_ai",
                    type="boolean",
                    description="Whether to use AI/VLM to find the element (for natural language)",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute fill with AI support."""
        selector = kwargs.get("selector")
        value = kwargs.get("value", "")
        use_ai = kwargs.get("use_ai", False)

        if not selector:
            return ToolResult.error_result("Selector is required")

        try:
            if use_ai and self._element_detector:
                element_info = await self._element_detector.find_element(
                    f"input field: {selector}"
                )
                actual_selector = element_info.get("selector", selector)
                
                await self._element_detector.fill(actual_selector, value)
                
                return ToolResult.success_result(
                    data={
                        "selector": actual_selector,
                        "filled_value": value,
                        "method": "ai_detection",
                        "confidence": element_info.get("confidence", 0),
                        "message": f"Filled '{value}' into {actual_selector}",
                    },
                )
            else:
                page = self._page_controller.page
                locator = page.locator(selector)
                
                if await locator.count() == 0:
                    return ToolResult.error_result(f"Input field not found: {selector}")
                
                value_before = await locator.input_value()
                await locator.fill(value)
                value_after = await locator.input_value()
                
                return ToolResult.success_result(
                    data={
                        "selector": selector,
                        "value_before": value_before,
                        "value_after": value_after,
                        "filled_value": value,
                        "method": "direct_selector",
                        "message": f"Filled '{value}' into {selector}",
                    },
                )
        except Exception as e:
            return ToolResult.error_result(f"Fill failed: {str(e)}")


class WaitForSelectorTool(BaseTool):
    """Wait for an element to appear on the page."""

    def __init__(self, page_controller: PageController) -> None:
        self._page_controller = page_controller

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="wait_for_selector",
            description="Wait for an element matching the selector to appear. Useful for dynamically loaded content.",
            category=ToolCategory.INTERACTION,
            safety_level=SafetyLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="selector",
                    type="string",
                    description="CSS selector of the element to wait for",
                    required=True,
                ),
                ToolParameter(
                    name="state",
                    type="string",
                    description="State to wait for: 'attached', 'visible', 'hidden', 'detached'",
                    required=False,
                    enum=["attached", "visible", "hidden", "detached"],
                ),
                ToolParameter(
                    name="timeout",
                    type="number",
                    description="Maximum time to wait in milliseconds (default: 30000)",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute wait for selector with feedback."""
        selector = kwargs.get("selector")
        state = kwargs.get("state", "visible")
        timeout = kwargs.get("timeout", 30000)

        if not selector:
            return ToolResult.error_result("Selector is required")

        try:
            page = self._page_controller.page
            locator = page.locator(selector)
            await locator.wait_for(state=state, timeout=timeout)
            
            count = await locator.count()
            element_info = None
            
            if count > 0 and state != "detached":
                element_info = await locator.first.evaluate("""
                    el => ({
                        tag: el.tagName.toLowerCase(),
                        id: el.id || null,
                        className: el.className || null,
                        visible: !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length)
                    })
                """)
            
            return ToolResult.success_result(
                data={
                    "selector": selector,
                    "state": state,
                    "found": count > 0,
                    "count": count,
                    "element_info": element_info,
                    "message": f"Element '{selector}' is now {state}" + (f" ({count} found)" if count > 1 else ""),
                },
            )
        except Exception as e:
            return ToolResult.error_result(f"Wait for selector failed: {str(e)}")


class DoubleClickTool(BaseTool):
    """Double-click an element on the page."""

    def __init__(
        self,
        page_controller: PageController,
        element_detector: Optional[ElementDetector] = None,
    ) -> None:
        self._page_controller = page_controller
        self._element_detector = element_detector

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="double_click",
            description="Double-click an element. Useful for opening files, editing text in-place, or triggering double-click events.",
            category=ToolCategory.INTERACTION,
            safety_level=SafetyLevel.MODERATE,
            parameters=[
                ToolParameter(
                    name="selector",
                    type="string",
                    description="CSS selector or natural language description of the element to double-click",
                    required=True,
                ),
                ToolParameter(
                    name="use_ai",
                    type="boolean",
                    description="Whether to use AI/VLM to find the element",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute double-click with AI support."""
        selector = kwargs.get("selector")
        use_ai = kwargs.get("use_ai", False)

        if not selector:
            return ToolResult.error_result("Selector is required")

        try:
            if use_ai and self._element_detector:
                element_info = await self._element_detector.find_element(selector)
                actual_selector = element_info.get("selector", selector)
                await self._element_detector.dblclick(actual_selector)
                return ToolResult.success_result(
                    data={
                        "selector": actual_selector,
                        "method": "ai_detection",
                        "confidence": element_info.get("confidence", 0),
                        "message": f"Double-clicked element: {actual_selector}",
                    },
                )
            else:
                page = self._page_controller.page
                locator = page.locator(selector)
                
                if await locator.count() == 0:
                    return ToolResult.error_result(f"Element not found: {selector}")
                
                await locator.dblclick()
                
                return ToolResult.success_result(
                    data={
                        "selector": selector,
                        "method": "direct_selector",
                        "message": f"Double-clicked element: {selector}",
                    },
                )
        except Exception as e:
            return ToolResult.error_result(f"Double-click failed: {str(e)}")


class RightClickTool(BaseTool):
    """Right-click (context menu) an element on the page."""

    def __init__(
        self,
        page_controller: PageController,
        element_detector: Optional[ElementDetector] = None,
    ) -> None:
        self._page_controller = page_controller
        self._element_detector = element_detector

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="right_click",
            description="Right-click an element to open context menu. Useful for accessing context menus and additional options.",
            category=ToolCategory.INTERACTION,
            safety_level=SafetyLevel.MODERATE,
            parameters=[
                ToolParameter(
                    name="selector",
                    type="string",
                    description="CSS selector or natural language description of the element to right-click",
                    required=True,
                ),
                ToolParameter(
                    name="use_ai",
                    type="boolean",
                    description="Whether to use AI/VLM to find the element",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute right-click with AI support."""
        selector = kwargs.get("selector")
        use_ai = kwargs.get("use_ai", False)

        if not selector:
            return ToolResult.error_result("Selector is required")

        try:
            if use_ai and self._element_detector:
                element_info = await self._element_detector.find_element(selector)
                actual_selector = element_info.get("selector", selector)
                # Use page.click with button='right'
                page = self._page_controller.page
                await page.locator(actual_selector).click(button="right")
                return ToolResult.success_result(
                    data={
                        "selector": actual_selector,
                        "method": "ai_detection",
                        "confidence": element_info.get("confidence", 0),
                        "message": f"Right-clicked element: {actual_selector}",
                    },
                )
            else:
                page = self._page_controller.page
                locator = page.locator(selector)
                
                if await locator.count() == 0:
                    return ToolResult.error_result(f"Element not found: {selector}")
                
                await locator.click(button="right")
                
                return ToolResult.success_result(
                    data={
                        "selector": selector,
                        "method": "direct_selector",
                        "message": f"Right-clicked element: {selector}",
                    },
                )
        except Exception as e:
            return ToolResult.error_result(f"Right-click failed: {str(e)}")


class DragAndDropTool(BaseTool):
    """Drag an element and drop it onto another element."""

    def __init__(
        self,
        page_controller: PageController,
        element_detector: Optional[ElementDetector] = None,
    ) -> None:
        self._page_controller = page_controller
        self._element_detector = element_detector

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="drag_and_drop",
            description="Drag an element from source to target. Useful for file uploads, sortable lists, sliders, and drag-drop interfaces.",
            category=ToolCategory.INTERACTION,
            safety_level=SafetyLevel.MODERATE,
            parameters=[
                ToolParameter(
                    name="source",
                    type="string",
                    description="CSS selector or description of the element to drag",
                    required=True,
                ),
                ToolParameter(
                    name="target",
                    type="string",
                    description="CSS selector or description of the drop target",
                    required=True,
                ),
                ToolParameter(
                    name="use_ai",
                    type="boolean",
                    description="Whether to use AI/VLM to find elements",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute drag and drop operation."""
        source = kwargs.get("source")
        target = kwargs.get("target")
        use_ai = kwargs.get("use_ai", False)

        if not source or not target:
            return ToolResult.error_result("Both source and target selectors are required")

        try:
            page = self._page_controller.page
            
            if use_ai and self._element_detector:
                source_info = await self._element_detector.find_element(f"drag source: {source}")
                target_info = await self._element_detector.find_element(f"drop target: {target}")
                actual_source = source_info.get("selector", source)
                actual_target = target_info.get("selector", target)
            else:
                actual_source = source
                actual_target = target
            
            source_locator = page.locator(actual_source)
            target_locator = page.locator(actual_target)
            
            if await source_locator.count() == 0:
                return ToolResult.error_result(f"Source element not found: {actual_source}")
            if await target_locator.count() == 0:
                return ToolResult.error_result(f"Target element not found: {actual_target}")
            
            # Perform drag and drop
            await source_locator.drag_to(target_locator)
            
            return ToolResult.success_result(
                data={
                    "source": actual_source,
                    "target": actual_target,
                    "method": "ai_detection" if use_ai else "direct_selector",
                    "message": f"Dragged '{actual_source}' to '{actual_target}'",
                },
            )
        except Exception as e:
            return ToolResult.error_result(f"Drag and drop failed: {str(e)}")


class UploadFileTool(BaseTool):
    """Upload a file to a file input element."""

    def __init__(
        self,
        page_controller: PageController,
        element_detector: Optional[ElementDetector] = None,
    ) -> None:
        self._page_controller = page_controller
        self._element_detector = element_detector

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="upload_file",
            description="Upload a file to a file input element. Supports single or multiple files.",
            category=ToolCategory.INTERACTION,
            safety_level=SafetyLevel.MODERATE,
            parameters=[
                ToolParameter(
                    name="selector",
                    type="string",
                    description="CSS selector or description of the file input element",
                    required=True,
                ),
                ToolParameter(
                    name="file_path",
                    type="string",
                    description="Path to the file to upload (or comma-separated paths for multiple files)",
                    required=True,
                ),
                ToolParameter(
                    name="use_ai",
                    type="boolean",
                    description="Whether to use AI/VLM to find the file input",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute file upload.
        
        Supports both explicit file_path parameter and context-based uploads.
        Context format: {"files": [{"field": "selector", "path": "file.pdf", "mime_type": "..."}]}
        """
        selector = kwargs.get("selector")
        file_path = kwargs.get("file_path")
        use_ai = kwargs.get("use_ai", False)

        # Check if selector is not provided - try to get from context
        if not selector:
            # Try to get first file from context
            user_context = self.get_user_context()
            files_from_context = user_context.get("files", [])
            if files_from_context and len(files_from_context) > 0:
                first_file = files_from_context[0]
                selector = first_file.get("field")
                if not file_path:
                    file_path = first_file.get("path")
        
        if not selector:
            return ToolResult.error_result("Selector is required (or provide context with files array)")
        
        # If file_path still not provided, check context for matching field
        if not file_path:
            user_context = self.get_user_context()
            files_from_context = user_context.get("files", [])
            for file_info in files_from_context:
                if file_info.get("field") == selector:
                    file_path = file_info.get("path")
                    break
        
        if not file_path:
            return ToolResult.error_result("File path is required (or provide context with files array)")

        try:
            page = self._page_controller.page
            
            if use_ai and self._element_detector:
                element_info = await self._element_detector.find_element(
                    f"file input: {selector}"
                )
                actual_selector = element_info.get("selector", selector)
            else:
                actual_selector = selector
            
            locator = page.locator(actual_selector)
            
            if await locator.count() == 0:
                return ToolResult.error_result(f"File input not found: {actual_selector}")
            
            # Handle multiple files
            files = [f.strip() for f in file_path.split(",")]
            
            # Set input files
            await locator.set_input_files(files if len(files) > 1 else files[0])
            
            return ToolResult.success_result(
                data={
                    "selector": actual_selector,
                    "files_uploaded": files,
                    "file_count": len(files),
                    "method": "ai_detection" if use_ai else "direct_selector",
                    "message": f"Uploaded {len(files)} file(s) to {actual_selector}",
                },
            )
        except Exception as e:
            return ToolResult.error_result(f"File upload failed: {str(e)}")


class EvaluateJavaScriptTool(BaseTool):
    """Execute JavaScript in the page context."""

    def __init__(self, page_controller: PageController) -> None:
        self._page_controller = page_controller

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="evaluate_javascript",
            description="Execute JavaScript code in the page context. Returns the result of the expression.",
            category=ToolCategory.INTERACTION,
            safety_level=SafetyLevel.SENSITIVE,
            parameters=[
                ToolParameter(
                    name="script",
                    type="string",
                    description="JavaScript code to execute in the page context",
                    required=True,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute JavaScript in page context."""
        script = kwargs.get("script")

        if not script:
            return ToolResult.error_result("JavaScript script is required")

        try:
            page = self._page_controller.page
            result = await page.evaluate(script)
            
            return ToolResult.success_result(
                data={
                    "result": result,
                    "script_executed": script[:100] + ("..." if len(script) > 100 else ""),
                    "message": f"JavaScript executed successfully",
                },
            )
        except Exception as e:
            return ToolResult.error_result(f"JavaScript evaluation failed: {str(e)}")


class GetAttributeTool(BaseTool):
    """Get an attribute value from an element."""

    def __init__(
        self,
        page_controller: PageController,
        element_detector: Optional[ElementDetector] = None,
    ) -> None:
        self._page_controller = page_controller
        self._element_detector = element_detector

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="get_attribute",
            description="Get the value of an attribute from an element. Useful for extracting href, src, data-* attributes.",
            category=ToolCategory.EXTRACTION,
            safety_level=SafetyLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="selector",
                    type="string",
                    description="CSS selector or description of the element",
                    required=True,
                ),
                ToolParameter(
                    name="attribute",
                    type="string",
                    description="Name of the attribute to get (e.g., 'href', 'src', 'data-id')",
                    required=True,
                ),
                ToolParameter(
                    name="use_ai",
                    type="boolean",
                    description="Whether to use AI/VLM to find the element",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Get attribute value from element."""
        selector = kwargs.get("selector")
        attribute = kwargs.get("attribute")
        use_ai = kwargs.get("use_ai", False)

        if not selector:
            return ToolResult.error_result("Selector is required")
        if not attribute:
            return ToolResult.error_result("Attribute name is required")

        try:
            page = self._page_controller.page
            
            if use_ai and self._element_detector:
                element_info = await self._element_detector.find_element(selector)
                actual_selector = element_info.get("selector", selector)
            else:
                actual_selector = selector
            
            locator = page.locator(actual_selector)
            
            if await locator.count() == 0:
                return ToolResult.error_result(f"Element not found: {actual_selector}")
            
            value = await locator.get_attribute(attribute)
            
            return ToolResult.success_result(
                data={
                    "selector": actual_selector,
                    "attribute": attribute,
                    "value": value,
                    "method": "ai_detection" if use_ai else "direct_selector",
                    "message": f"Attribute '{attribute}' = '{value}'",
                },
            )
        except Exception as e:
            return ToolResult.error_result(f"Get attribute failed: {str(e)}")


class ClearInputTool(BaseTool):
    """Clear the content of an input field."""

    def __init__(
        self,
        page_controller: PageController,
        element_detector: Optional[ElementDetector] = None,
    ) -> None:
        self._page_controller = page_controller
        self._element_detector = element_detector

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="clear_input",
            description="Clear the content of an input field or textarea.",
            category=ToolCategory.INTERACTION,
            safety_level=SafetyLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="selector",
                    type="string",
                    description="CSS selector or description of the input field to clear",
                    required=True,
                ),
                ToolParameter(
                    name="use_ai",
                    type="boolean",
                    description="Whether to use AI/VLM to find the element",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Clear input field content."""
        selector = kwargs.get("selector")
        use_ai = kwargs.get("use_ai", False)

        if not selector:
            return ToolResult.error_result("Selector is required")

        try:
            page = self._page_controller.page
            
            if use_ai and self._element_detector:
                element_info = await self._element_detector.find_element(
                    f"input field: {selector}"
                )
                actual_selector = element_info.get("selector", selector)
            else:
                actual_selector = selector
            
            locator = page.locator(actual_selector)
            
            if await locator.count() == 0:
                return ToolResult.error_result(f"Input field not found: {actual_selector}")
            
            # Get value before clearing
            value_before = await locator.input_value()
            
            # Clear the input
            await locator.clear()
            
            return ToolResult.success_result(
                data={
                    "selector": actual_selector,
                    "value_cleared": value_before,
                    "method": "ai_detection" if use_ai else "direct_selector",
                    "message": f"Cleared input field: {actual_selector}",
                },
            )
        except Exception as e:
            return ToolResult.error_result(f"Clear input failed: {str(e)}")
