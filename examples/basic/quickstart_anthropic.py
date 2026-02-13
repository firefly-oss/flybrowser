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

"""
Example: Quickstart with Anthropic Claude

Demonstrates core FlyBrowser operations using Anthropic as the LLM provider.
All configuration is done via environment variables — no hardcoded keys.

Prerequisites:
    pip install flybrowser
    playwright install chromium
    export ANTHROPIC_API_KEY="sk-ant-..."

Usage:
    python examples/basic/quickstart_anthropic.py
"""

import asyncio
import json
import os
import sys

from flybrowser import FlyBrowser


async def main() -> None:
    """Run core FlyBrowser operations with Anthropic Claude."""
    # Configuration via environment variables (recommended)
    # The framework reads ANTHROPIC_API_KEY automatically
    provider = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
    model = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print("=" * 60)

    async with FlyBrowser(
        llm_provider=provider,
        llm_model=model,
        headless=True,
    ) as browser:
        # ── 1. Navigation ────────────────────────────────────────
        print("\n1. Navigating to Hacker News...")
        await browser.goto("https://news.ycombinator.com")
        print("   Done.")

        # ── 2. Extraction ────────────────────────────────────────
        print("\n2. Extracting top stories...")
        result = await browser.extract(
            "Get the titles and scores of the top 3 stories on the page"
        )
        # Result is an AgentRequestResponse object
        print(f"   Success: {result.success}")
        if result.data:
            data_str = json.dumps(result.data, indent=2, default=str)
            print(f"   Data: {data_str[:500]}")
        if result.llm_usage.total_tokens > 0:
            print(f"   Tokens: {result.llm_usage.total_tokens}")

        # ── 3. Observation ───────────────────────────────────────
        print("\n3. Observing page elements...")
        elements = await browser.observe("find the 'More' link at the bottom")
        print(f"   Success: {elements.success}")
        if elements.data:
            print(f"   Elements: {str(elements.data)[:300]}")

        # ── 4. Screenshot ────────────────────────────────────────
        print("\n4. Taking screenshot...")
        screenshot = await browser.screenshot()
        if isinstance(screenshot, dict):
            has_data = "data_base64" in screenshot and len(screenshot.get("data_base64", "")) > 0
            print(f"   Screenshot captured: {has_data}")
            if has_data:
                print(f"   Size: {len(screenshot['data_base64'])} bytes (base64)")
        else:
            print(f"   Screenshot result: {type(screenshot).__name__}")

        # ── 5. Usage summary ─────────────────────────────────────
        print("\n5. Usage summary:")
        usage = browser.get_usage_summary()
        print(f"   {json.dumps(usage, indent=2, default=str)}")

    print("\n" + "=" * 60)
    print("All operations completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
