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
Human-Like Search Tool for FlyBrowser.

This tool performs web searches by simulating human behavior:
- Natural typing with delays
- Mouse movements and clicks
- Scrolling patterns
- Random timing variations

Fully guided by the autonomous planning system:
- Automatic phase creation (Navigate → Query → Extract)
- Goal tracking and completion
- Adaptive to search engine changes
- Self-correcting on failures

Usage:
    tool = SearchHumanTool(page_controller)
    result = await tool.execute(query="python tutorials", engine="google")
"""

import asyncio
import logging
import random
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BaseTool, ToolMetadata, ToolParameter, ToolResult
from .search_utils import (
    SearchResponse,
    SearchResult,
    SearchEngine,
    SearchProvider,
    normalize_query,
    clean_snippet,
    is_valid_url,
)
from .navigation import NavigateTool

if TYPE_CHECKING:
    from flybrowser.core.page import PageController

logger = logging.getLogger(__name__)


class HumanBehaviorSimulator:
    """
    Simulates human-like behavior patterns.
    
    Adds natural delays, variations, and patterns that make
    automation less detectable.
    """
    
    @staticmethod
    def typing_delay() -> float:
        """
        Calculate delay between keystrokes.
        
        Simulates typing speed of 80-120 WPM with natural variation.
        
        Returns:
            Delay in seconds
        """
        # Average typing speed: 100 WPM = ~5 chars/sec = 0.2s per char
        base_delay = 0.2
        # Add random variation (-50% to +50%)
        variation = random.uniform(-0.1, 0.1)
        return max(0.05, base_delay + variation)
    
    @staticmethod
    def word_pause() -> float:
        """
        Calculate pause between words.
        
        Returns:
            Delay in seconds
        """
        return random.uniform(0.1, 0.3)
    
    @staticmethod
    def click_delay() -> float:
        """
        Calculate delay after mouse click.
        
        Returns:
            Delay in seconds
        """
        return random.uniform(0.2, 0.5)
    
    @staticmethod
    def scroll_delay() -> float:
        """
        Calculate delay during scrolling.
        
        Returns:
            Delay in seconds
        """
        return random.uniform(0.3, 0.8)
    
    @staticmethod
    def think_delay() -> float:
        """
        Calculate delay for 'thinking' (reading page).
        
        Returns:
            Delay in seconds
        """
        return random.uniform(1.0, 2.5)


class SearchEngineAdapter:
    """
    Adapters for different search engines.
    
    Only provides URLs - element detection is done via LLM/VLM.
    """
    
    ENGINES = {
        SearchEngine.GOOGLE: {
            "url": "https://www.google.com",
        },
        SearchEngine.DUCKDUCKGO: {
            "url": "https://duckduckgo.com",
        },
        SearchEngine.BING: {
            "url": "https://www.bing.com",
        },
    }
    
    @classmethod
    def get_url(cls, engine: SearchEngine) -> str:
        """Get URL for a search engine."""
        config = cls.ENGINES.get(engine, cls.ENGINES[SearchEngine.DUCKDUCKGO])
        return config["url"]


class SearchHumanTool(BaseTool):
    """
    Human-like search tool.
    
    Performs searches by simulating human behavior patterns,
    fully guided by the autonomous planning system.
    """
    
    def __init__(self, page_controller: Optional["PageController"] = None) -> None:
        """Initialize the human-like search tool."""
        super().__init__(page_controller)
        self.behavior = HumanBehaviorSimulator()
        self._navigate_tool: Optional[NavigateTool] = None
    
    @property
    def metadata(self) -> ToolMetadata:
        """Tool metadata."""
        return ToolMetadata(
            name="search_human",
            description=(
                "FALLBACK browser-based search - Only use if 'search' tool is unavailable or fails. "
                "Slower (2-5 seconds) but works without API keys. "
                "Uses human-like browser automation (typing, mouse movements). "
                "Prefer 'search' tool for faster API-based search when available."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="engine",
                    type="string",
                    description="Search engine: 'google', 'duckduckgo', 'bing'",
                    required=False,
                    default="duckduckgo",
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results to extract (1-20)",
                    required=False,
                    default=10,
                ),
            ],
            examples=['search_human(query="python tutorials", engine="duckduckgo", max_results=10)'],
        )
    
    async def execute(
        self,
        query: str,
        engine: str = "duckduckgo",
        max_results: int = 10,
        **kwargs
    ) -> ToolResult:
        """
        Execute human-like search using LLM/VLM for element detection.
        
        Uses intelligent element detection instead of hardcoded selectors:
        1. Navigate to search engine (with obstacle handling)
        2. Find search input using LLM/VLM
        3. Enter query with human-like typing
        4. Submit search (Enter key or button click)
        5. Extract results using LLM/VLM
        
        Args:
            query: Search query
            engine: Search engine to use
            max_results: Maximum results to extract
            
        Returns:
            ToolResult with SearchResponse data
        """
        start_time = time.time()
        
        try:
            # Normalize query
            query = normalize_query(query)
            
            # Parse engine
            try:
                search_engine = SearchEngine(engine.lower())
            except ValueError:
                search_engine = SearchEngine.DUCKDUCKGO
            
            # Get engine URL
            engine_url = SearchEngineAdapter.get_url(search_engine)
            
            logger.info(f"Starting human-like search: '{query}' on {search_engine.value}")
            
            # Phase 1: Navigate to search engine (NavigateTool handles obstacles)
            await self._navigate_to_engine(engine_url)
            
            # Phase 2: Find and fill search input using LLM
            await self._enter_query_with_llm(query)
            
            # Phase 3: Submit search (press Enter - most reliable)
            await self._submit_search()
            
            # Phase 4: Wait for results page to load
            await self._wait_for_results_page()
            
            # Check for bot detection
            current_url = await self.page.get_url()
            if '/sorry/' in current_url or 'captcha' in current_url.lower():
                logger.warning(f"Bot detection detected: {current_url}")
                return await self._handle_bot_detection_retry(query, engine_url, max_results, search_engine, start_time)
            
            # Phase 5: Extract results using LLM
            results = await self._extract_results_with_llm(max_results)
            
            # Create response
            elapsed_ms = (time.time() - start_time) * 1000
            
            response = SearchResponse(
                query=query,
                results=results,
                total_results=len(results) * 10,  # Estimate
                search_time_ms=elapsed_ms,
                provider=SearchProvider.BROWSER.value,
                metadata={
                    "engine": search_engine.value,
                    "human_simulation": True,
                    "llm_detection": True,
                    "final_url": current_url,
                },
            )
            
            # Build detailed message with page context
            message = f"Found {len(results)} results via {search_engine.value}.\n"
            if results:
                message += f"Top result: {results[0].title}\n"
                message += f"Page URL: {current_url}\n"
                message += f"Status: Successfully reached results page"
            else:
                message += f"No results extracted. Page URL: {current_url}\n"
                message += "Consider using search_rank tool or trying a different engine."
            
            return ToolResult.success_result(
                data=response.to_dict(),
                message=message,
                metadata={
                    "engine": search_engine.value,
                    "final_url": current_url,
                    "has_results": len(results) > 0
                },
            )
        
        except Exception as e:
            logger.exception(f"Human search execution failed: {e}")
            return ToolResult.error_result(
                error=f"Search execution error: {str(e)}",
                error_code="EXECUTION_ERROR",
            )
    
    def _get_navigate_tool(self) -> NavigateTool:
        """Get or create NavigateTool instance for navigation with obstacle handling."""
        if self._navigate_tool is None:
            self._navigate_tool = NavigateTool(self.page)
            # Pass through injected LLM and config from ReActAgent
            self._navigate_tool.llm_provider = getattr(self, 'llm_provider', None)
            self._navigate_tool.agent_config = getattr(self, 'agent_config', None)
        return self._navigate_tool
    
    async def _navigate_to_engine(self, url: str) -> None:
        """Navigate to search engine homepage using NavigateTool.
        
        Uses NavigateTool which handles obstacle detection (cookie banners, etc.)
        through the framework's ObstacleDetector.
        """
        logger.info(f"Navigating to {url}")
        
        if not self.page:
            raise RuntimeError("Page controller not available")
        
        # Use NavigateTool for navigation - it handles obstacle detection
        navigate_tool = self._get_navigate_tool()
        result = await navigate_tool.execute(url=url)
        
        if not result.success:
            raise RuntimeError(f"Navigation failed: {result.error}")
        
        # Think delay (simulate reading page)
        await asyncio.sleep(self.behavior.think_delay())
    
    async def _enter_query_with_llm(self, query: str) -> None:
        """
        Find search input using Playwright's intelligent locators and enter query.
        
        Uses Playwright's role-based and semantic locators which are more reliable
        than CSS selectors, especially for modern sites like Google that use
        textarea elements or complex DOM structures.
        """
        logger.info(f"Finding search input and typing query: {query}")
        
        if not self.page:
            raise RuntimeError("Page controller not available")
        
        # Try multiple strategies to find the search input, prioritizing Playwright's
        # intelligent locators over CSS selectors
        search_input = None
        
        # Strategy 1: Use Playwright's get_by_role for search/combobox
        # Modern search engines use role="combobox" or role="searchbox"
        try:
            # Google uses role="combobox" for their search textarea
            combobox = self.page.page.get_by_role("combobox")
            if await combobox.count() > 0:
                search_input = combobox.first
                logger.info("Found search input via role='combobox'")
        except Exception as e:
            logger.debug(f"Role combobox not found: {e}")
        
        # Strategy 2: Try searchbox role
        if not search_input:
            try:
                searchbox = self.page.page.get_by_role("searchbox")
                if await searchbox.count() > 0:
                    search_input = searchbox.first
                    logger.info("Found search input via role='searchbox'")
            except Exception as e:
                logger.debug(f"Role searchbox not found: {e}")
        
        # Strategy 3: Use Playwright's get_by_placeholder for common search placeholders
        if not search_input:
            for placeholder in ["Search", "Buscar", "Buscar en Google", "Search Google"]:
                try:
                    by_placeholder = self.page.page.get_by_placeholder(placeholder, exact=False)
                    if await by_placeholder.count() > 0:
                        search_input = by_placeholder.first
                        logger.info(f"Found search input via placeholder='{placeholder}'")
                        break
                except Exception:
                    continue
        
        # Strategy 4: Try textarea elements (Google's modern search)
        if not search_input:
            try:
                # Google uses <textarea> for their search input
                textarea = self.page.page.locator("textarea[name='q'], textarea[title*='Search'], textarea[title*='Buscar']")
                if await textarea.count() > 0:
                    search_input = textarea.first
                    logger.info("Found search input via textarea selector")
            except Exception as e:
                logger.debug(f"Textarea selector not found: {e}")
        
        # Strategy 5: Fall back to traditional input selectors
        if not search_input:
            try:
                traditional = self.page.page.locator("input[name='q'], input[type='search'], input[type='text'][aria-label*='earch']")
                if await traditional.count() > 0:
                    search_input = traditional.first
                    logger.info("Found search input via traditional CSS selector")
            except Exception as e:
                logger.debug(f"Traditional selector not found: {e}")
        
        # Strategy 6: Use LLM with ACTUAL HTML content as last resort
        if not search_input:
            llm = getattr(self, 'llm_provider', None)
            if llm:
                try:
                    from flybrowser.agents.structured_llm import StructuredLLMWrapper
                    wrapper = StructuredLLMWrapper(llm, max_repair_attempts=2)
                    
                    # Get ACTUAL HTML of input/textarea elements - this is what the LLM needs!
                    input_elements_html = await self.page.page.evaluate("""
                        () => {
                            const elements = [];
                            // Get all potential search inputs
                            document.querySelectorAll('input, textarea').forEach((el, idx) => {
                                if (idx < 15) {  // Limit for token efficiency
                                    const rect = el.getBoundingClientRect();
                                    // Only include visible elements
                                    if (rect.width > 0 && rect.height > 0) {
                                        // Get outerHTML but limit length
                                        let html = el.outerHTML;
                                        if (html.length > 500) html = html.substring(0, 500) + '...';
                                        elements.push({
                                            html: html,
                                            tagName: el.tagName.toLowerCase(),
                                            type: el.type || null,
                                            name: el.name || null,
                                            id: el.id || null,
                                            placeholder: el.placeholder || null,
                                            ariaLabel: el.getAttribute('aria-label') || null,
                                            role: el.getAttribute('role') || null,
                                            title: el.title || null,
                                            isVisible: rect.top >= 0 && rect.top <= window.innerHeight,
                                            position: { x: Math.round(rect.x), y: Math.round(rect.y) }
                                        });
                                    }
                                }
                            });
                            return elements;
                        }
                    """)
                    
                    page_state = await self.page.get_page_state()
                    schema = {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string", "description": "CSS selector for the main search input"},
                            "reasoning": {"type": "string", "description": "Why this element is the search input"}
                        },
                        "required": ["selector"]
                    }
                    
                    # Format the actual HTML for the LLM
                    elements_text = "\n".join([
                        f"{i+1}. {el.get('tagName')} - {el.get('html', '')[:200]}"
                        for i, el in enumerate(input_elements_html)
                    ]) if input_elements_html else "No input elements found"
                    
                    prompt = f"""Find the main SEARCH input on this page.

Page URL: {page_state.get('url', 'unknown')}
Page Title: {page_state.get('title', 'unknown')}

## ACTUAL INPUT ELEMENTS ON PAGE:
{elements_text}

Analyze the ACTUAL elements above and return a CSS selector for the PRIMARY search input.
Use attributes you can see: name, id, role, aria-label, placeholder, title, etc.
DO NOT guess - use ONLY attributes that exist in the HTML above."""
                    
                    result = await wrapper.generate_structured(
                        prompt=prompt,
                        schema=schema,
                        system_prompt="You are analyzing REAL HTML. Return a CSS selector using ONLY attributes visible in the provided HTML. Never guess or hallucinate attributes.",
                        temperature=0.1,
                    )
                    selector = result.get("selector")
                    if selector:
                        search_input = self.page.page.locator(selector).first
                        logger.info(f"LLM found search input: {selector} (reasoning: {result.get('reasoning', 'N/A')})")
                except Exception as e:
                    logger.warning(f"LLM element detection failed: {e}")
        
        if not search_input:
            raise RuntimeError("Could not find search input on page")
        
        # Wait for the element to be visible and click it
        try:
            await search_input.wait_for(state="visible", timeout=10000)
            await search_input.click()
            await asyncio.sleep(self.behavior.click_delay())
        except Exception as e:
            raise RuntimeError(f"Failed to click search input: {e}")
        
        # Type the query with human-like delays
        await self._type_human_like(query)
        
        logger.info("Query typed successfully")
    
    async def _type_human_like(self, text: str) -> None:
        """Type text with human-like delays between keystrokes."""
        words = text.split()
        for i, word in enumerate(words):
            for char in word:
                delay_seconds = self.behavior.typing_delay()
                await self.page.type_text(char, delay=int(delay_seconds * 1000))
            
            if i < len(words) - 1:
                await self.page.type_text(" ", delay=int(self.behavior.word_pause() * 1000))
    
    async def _submit_search(self) -> None:
        """Submit the search by pressing Enter (most reliable cross-engine method)."""
        logger.info("Submitting search")
        await asyncio.sleep(self.behavior.click_delay())
        await self.page.press_key("Enter")
        await asyncio.sleep(self.behavior.click_delay())
    
    async def _wait_for_results_page(self) -> None:
        """Wait for search results page to load."""
        logger.info("Waiting for results page to load")
        
        # Wait for navigation/page load
        await asyncio.sleep(2.0)  # Give time for page transition
        
        # Wait for page to stabilize (URL should change from homepage)
        for _ in range(10):
            current_url = await self.page.get_url()
            # Most search engines have query params after search
            if '?' in current_url or 'search' in current_url.lower():
                break
            await asyncio.sleep(0.5)
        
        # Additional settle time
        await asyncio.sleep(self.behavior.scroll_delay())
        
        # Scroll down slightly (humans do this)
        await self.page.evaluate("window.scrollBy(0, 300)")
        await asyncio.sleep(self.behavior.scroll_delay())
    
    async def _extract_results_with_llm(self, max_results: int) -> List[SearchResult]:
        """
        Extract search results using LLM to analyze actual HTML structure.
        
        Following the pattern from page_explorer.py and page_analyzer.py:
        - Extracts actual link elements with outerHTML and href attributes
        - Provides real HTML context so LLM doesn't hallucinate
        - Gets URLs directly from href attributes, not guessing
        """
        logger.info(f"Extracting up to {max_results} results using LLM")
        
        if not self.page:
            raise RuntimeError("Page controller not available")
        
        llm = getattr(self, 'llm_provider', None)
        if not llm:
            logger.warning("LLM not available, cannot extract results")
            return []
        
        from flybrowser.agents.structured_llm import StructuredLLMWrapper
        wrapper = StructuredLLMWrapper(llm, max_repair_attempts=2)
        
        # Extract ACTUAL link elements with their HTML and href - following page_explorer.py pattern
        # This gives us real URLs instead of requiring LLM to guess
        search_results_data = await self.page.page.evaluate("""
            () => {
                const results = [];
                const origin = window.location.origin;
                
                // Get all anchor elements that could be search results
                // Search results typically are links with substantial text
                const allLinks = document.querySelectorAll('a[href]');
                
                allLinks.forEach((a, idx) => {
                    if (results.length >= 50) return;  // Limit for token efficiency
                    
                    const href = a.href || '';
                    const text = (a.innerText || a.textContent || '').trim();
                    
                    // Skip empty links, same-page anchors, and navigation links
                    if (!href || href === '#' || href.startsWith('javascript:') || !text) return;
                    if (href.includes('google.com/search') || href.includes('bing.com/search')) return;
                    if (text.length < 10 || text.length > 500) return;  // Skip very short/long text
                    
                    const rect = a.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) return;  // Skip hidden elements
                    
                    // Get parent context to help identify if it's a search result
                    const parent = a.parentElement;
                    const grandparent = parent?.parentElement;
                    
                    // Get outerHTML but limit size (following page_analyzer.py pattern)
                    let html = a.outerHTML;
                    if (html.length > 300) html = html.substring(0, 300) + '...';
                    
                    // Get surrounding text for snippet (sibling elements)
                    let snippet = '';
                    if (parent) {
                        // Look for description/snippet in nearby elements
                        const siblings = parent.querySelectorAll('span, p, div');
                        siblings.forEach(sib => {
                            const sibText = (sib.innerText || '').trim();
                            if (sibText.length > snippet.length && sibText.length < 500 && sibText !== text) {
                                snippet = sibText;
                            }
                        });
                    }
                    
                    results.push({
                        // Actual data from DOM - not hallucinated
                        title: text.substring(0, 200),
                        url: href,  // Real URL from href attribute
                        snippet: snippet.substring(0, 300),
                        html: html,  // outerHTML for LLM context
                        // Position context
                        position: {
                            top: Math.round(rect.top + window.scrollY),
                            inViewport: rect.top >= 0 && rect.top < window.innerHeight
                        },
                        // Parent context helps identify search results vs nav
                        parentTag: parent?.tagName?.toLowerCase() || '',
                        grandparentTag: grandparent?.tagName?.toLowerCase() || '',
                        parentClass: parent?.className?.substring(0, 100) || ''
                    });
                });
                
                return {
                    links: results,
                    pageInfo: {
                        url: window.location.href,
                        title: document.title,
                        totalLinks: allLinks.length
                    }
                };
            }
        """)
        
        current_url = await self.page.get_url()
        links = search_results_data.get('links', [])
        page_info = search_results_data.get('pageInfo', {})
        
        logger.info(f"Extracted {len(links)} candidate links from DOM")
        
        if not links:
            logger.warning("No links found in DOM")
            return []
        
        # Format links for LLM to analyze - give it actual HTML context
        links_text = "\n".join([
            f"{i+1}. Title: {l.get('title', '')}\n"
            f"   URL: {l.get('url', '')}\n"
            f"   Snippet: {l.get('snippet', '')[:150]}\n"
            f"   HTML: {l.get('html', '')[:150]}\n"
            f"   Parent: <{l.get('parentTag', '')} class='{l.get('parentClass', '')[:50]}'>"
            for i, l in enumerate(links[:30])  # Limit for tokens
        ])
        
        schema = {
            "type": "object",
            "properties": {
                "result_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Indices (1-based) of links that are actual search results"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of how you identified search results"
                }
            },
            "required": ["result_indices"]
        }
        
        prompt = f"""Identify which of these links are ACTUAL SEARCH RESULTS (not ads, navigation, or related searches).

Page URL: {current_url}
Page Title: {page_info.get('title', 'Search Results')}

## LINKS EXTRACTED FROM PAGE:
{links_text}

Return the indices (1-based) of links that are actual organic search results.
Look for:
- Links with descriptive titles and real destination URLs
- Links that appear in the main content area (not header/footer/sidebar)
- Links with snippets/descriptions

Exclude:
- Ads (usually marked or in special containers)
- Navigation links (Home, About, etc.)
- Related searches or suggestions
- Pagination links"""

        try:
            result = await wrapper.generate_structured(
                prompt=prompt,
                schema=schema,
                system_prompt="You are analyzing REAL extracted links from a search results page. Select only actual search results by their index.",
                temperature=0.1,
                max_tokens=1024,
            )
            
            indices = result.get("result_indices", [])
            logger.info(f"LLM identified {len(indices)} search results (reasoning: {result.get('reasoning', 'N/A')[:100]})")
            
            # Build SearchResult objects from the selected indices
            results = []
            for idx in indices[:max_results]:
                if 1 <= idx <= len(links):
                    link = links[idx - 1]  # Convert 1-based to 0-based
                    url = link.get('url', '')
                    if url and is_valid_url(url):
                        results.append(SearchResult(
                            title=link.get('title', ''),
                            url=url,  # Real URL from DOM, not hallucinated
                            snippet=clean_snippet(link.get('snippet', '')),
                            position=len(results) + 1,
                            source="llm_extraction",
                        ))
            
            logger.info(f"Built {len(results)} SearchResult objects")
            return results
            
        except Exception as e:
            logger.error(f"LLM result extraction failed: {e}")
            # Fallback: return first N links that look like results
            results = []
            for link in links[:max_results]:
                url = link.get('url', '')
                if url and is_valid_url(url) and not any(x in url.lower() for x in ['google.com', 'bing.com', 'duckduckgo.com']):
                    results.append(SearchResult(
                        title=link.get('title', ''),
                        url=url,
                        snippet=clean_snippet(link.get('snippet', '')),
                        position=len(results) + 1,
                        source="dom_extraction_fallback",
                    ))
            return results
    
    async def _handle_bot_detection_retry(self, query: str, engine_url: str, max_results: int, search_engine: SearchEngine, start_time: float) -> ToolResult:
        """
        Handle bot detection adaptively by retrying.
        
        Args:
            query: Search query
            config: Engine configuration
            max_results: Maximum results
            search_engine: Search engine being used
            start_time: Start time for elapsed calculation
            
        Returns:
            ToolResult with retry outcome
        """
        logger.info(" Being adaptive: waiting and will retry...")
        
        # Wait - sometimes Google lets you through after a pause
        await asyncio.sleep(random.uniform(2.0, 4.0))
        
        try:
            logger.info("Attempting to return to search homepage...")
            # Use NavigateTool for navigation with obstacle handling
            navigate_tool = self._get_navigate_tool()
            nav_result = await navigate_tool.execute(url=engine_url)
            if not nav_result.success:
                logger.warning(f"Navigation failed during retry: {nav_result.error}")
            await asyncio.sleep(random.uniform(1.0, 2.0))
            
            # Retry search using LLM-based methods
            logger.info("Retrying search after bot detection...")
            await self._enter_query_with_llm(query)
            await self._submit_search()
            await self._wait_for_results_page()
            
            # Extract results using LLM
            results = await self._extract_results_with_llm(max_results)
            current_url = await self.page.get_url()
            
            # Check if still blocked
            if '/sorry/' in current_url or 'captcha' in current_url.lower():
                return ToolResult.error_result(
                    error=f"Bot detection persists after retry. Try engine='duckduckgo' or engine='bing' which are less restrictive.",
                    error_code="BOT_DETECTED_PERSISTENT"
                )
            
            logger.info(" Retry after bot detection successful!")
            
            # Build successful response
            elapsed_ms = (time.time() - start_time) * 1000
            response = SearchResponse(
                query=query,
                results=results,
                total_results=len(results) * 10,
                search_time_ms=elapsed_ms,
                provider=SearchProvider.BROWSER.value,
                metadata={
                    "engine": search_engine.value,
                    "human_simulation": True,
                    "recovered_from_bot_detection": True,
                },
            )
            
            return ToolResult.success_result(
                data=response.to_dict(),
                message=f"Found {len(results)} results (recovered from bot detection)",
                metadata={"engine": search_engine.value, "bot_detection_recovered": True},
            )
        
        except Exception as retry_error:
            logger.warning(f"Retry after bot detection failed: {retry_error}")
            return ToolResult.error_result(
                error=f"Bot detection - retry failed. Try engine='duckduckgo' or engine='bing' instead.",
                error_code="BOT_DETECTED"
            )
    

class SearchHumanAdvancedTool(SearchHumanTool):
    """
    Advanced human-like search with planning integration.
    
    This version creates an explicit execution plan and uses
    the autonomous planning system for more complex searches.
    """
    
    async def execute(
        self,
        query: str,
        engine: str = "duckduckgo",
        max_results: int = 10,
        use_planning: bool = True,
        **kwargs
    ) -> ToolResult:
        """
        Execute with optional planning mode.
        
        Args:
            query: Search query
            engine: Search engine
            max_results: Max results
            use_planning: Use autonomous planning system
            
        Returns:
            ToolResult
        """
        if use_planning:
            # Let the planning system guide the search
            # The ReAct agent will create phases automatically
            logger.info("Using planning-guided search")
        
        # Execute normal human-like search
        # (planning integration happens at agent level)
        return await super().execute(query, engine, max_results, **kwargs)
