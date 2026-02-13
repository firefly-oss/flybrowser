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

"""Extraction ToolKit for page content extraction.

Provides tools for extracting text content, taking screenshots,
and inspecting page state, all built on the fireflyframework-genai
ToolKit pattern.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING

from fireflyframework_genai.tools.decorators import firefly_tool
from fireflyframework_genai.tools.toolkit import ToolKit

if TYPE_CHECKING:
    from flybrowser.core.page import PageController


def create_extraction_toolkit(page: PageController) -> ToolKit:
    """Create an extraction toolkit bound to the given *page*.

    The returned :class:`ToolKit` contains three tools:

    * **extract_text** -- extract text content from the page or a specific element
    * **screenshot** -- capture a screenshot of the page
    * **get_page_state** -- get a JSON summary of the current page state

    Each tool is a closure over *page* so it can drive the browser
    without requiring dependency injection at call time.
    """

    @firefly_tool(
        name="extract_text",
        description=(
            "Extract text content from the page. Without a selector, returns "
            "structured page content (title, URL, headings, navigation links, "
            "main content, visible text). With a selector, returns the text "
            "content of the matching element."
        ),
        auto_register=False,
    )
    async def extract_text(selector: str = "") -> str:
        if selector:
            locator = page.page.locator(selector)
            count = await locator.count()
            if count == 0:
                return f"No elements found for selector '{selector}'"
            if count == 1:
                text = await locator.text_content()
                return f"Text content of '{selector}': {text}"
            # Multiple matches — concatenate first 10 elements
            parts = []
            for i in range(min(count, 10)):
                text = await locator.nth(i).text_content()
                if text and text.strip():
                    parts.append(text.strip())
            suffix = f" (showing 10 of {count})" if count > 10 else ""
            return f"Text content of '{selector}' ({count} elements{suffix}):\n" + "\n---\n".join(parts)

        data = await page.page.evaluate(
            """() => {
                const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'))
                    .map(h => h.textContent.trim());
                const navLinks = Array.from(document.querySelectorAll('nav a, header a'))
                    .map(a => ({text: a.textContent.trim(), href: a.href}));
                const main = document.querySelector('main, [role="main"], article');
                const mainContent = main ? main.textContent.trim() : '';
                const visibleText = document.body.innerText || document.body.textContent || '';
                return {
                    title: document.title,
                    url: window.location.href,
                    headings: headings,
                    navLinks: navLinks,
                    mainContent: mainContent,
                    visibleText: visibleText.substring(0, 5000)
                };
            }"""
        )

        parts = [
            f"Title: {data.get('title', '')}",
            f"URL: {data.get('url', '')}",
        ]
        headings = data.get("headings", [])
        if headings:
            parts.append(f"Headings: {', '.join(headings)}")
        nav_links = data.get("navLinks", [])
        if nav_links:
            links_str = ", ".join(
                f"{l.get('text', '')} ({l.get('href', '')})" for l in nav_links
            )
            parts.append(f"Nav Links: {links_str}")
        main_content = data.get("mainContent", "")
        if main_content:
            parts.append(f"Main Content: {main_content[:2000]}")
        visible_text = data.get("visibleText", "")
        if visible_text:
            parts.append(f"Visible Text: {visible_text[:2000]}")

        return "\n".join(parts)

    @firefly_tool(
        name="screenshot",
        description=(
            "Capture a screenshot of the current page. Set full_page=True "
            "to capture the entire scrollable page."
        ),
        auto_register=False,
    )
    async def screenshot(full_page: bool = False) -> str:
        image_bytes = await page.screenshot(full_page=full_page)
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        size = len(image_bytes)
        return (
            f"Screenshot captured ({size} bytes, "
            f"{'full page' if full_page else 'viewport'}). "
            f"Base64 length: {len(encoded)}"
        )

    @firefly_tool(
        name="get_page_state",
        description=(
            "Get a rich JSON summary of the current page state including "
            "URL, title, viewport, scroll position, links, buttons, forms, "
            "and inputs."
        ),
        auto_register=False,
    )
    async def get_page_state() -> str:
        state = await page.get_rich_state()
        return json.dumps(state, indent=2, default=str)

    return ToolKit(
        "extraction",
        [extract_text, screenshot, get_page_state],
        description="Tools for extracting content and state from browser pages.",
    )
