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
Parallel Page Explorer - DAG-based concurrent site exploration.

This module provides parallel exploration capabilities for multi-page
site analysis. It uses a DAG (Directed Acyclic Graph) structure to
determine which pages can be explored concurrently while respecting
dependencies (child pages can only be explored after parent is visited).

Key optimizations:
1. Parallel page navigation: Multiple pages explored concurrently
2. Pipeline mode: Start capturing next page while analyzing current page
3. Batch LLM analysis: Analyze multiple screenshots in fewer LLM calls

Expected speedup: 2-4x for typical site exploration tasks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .sitemap_graph import (
    ExplorationDAG, SitemapGraph, SitemapNode, PageStatus,
    filter_navigation_links_async, LinkType
)
from .config import ParallelExplorationConfig

if TYPE_CHECKING:
    from flybrowser.core.page import PageController
    from flybrowser.llm.base import BaseLLMProvider
    from .tools.page_explorer import PageExplorerTool
    from .page_analyzer import PageAnalyzer
    from .memory import AgentMemory

logger = logging.getLogger(__name__)


@dataclass
class PageExplorationResult:
    """Result of exploring a single page."""
    url: str
    success: bool
    title: str = ""
    summary: str = ""
    section_count: int = 0
    navigation_links: List[Dict[str, str]] = field(default_factory=list)
    page_map: Optional[Any] = None
    error: str = ""
    duration_ms: float = 0.0


@dataclass 
class ParallelExplorationStats:
    """Statistics from parallel exploration run."""
    total_pages: int = 0
    pages_explored: int = 0
    pages_failed: int = 0
    pages_skipped: int = 0
    total_duration_ms: float = 0.0
    avg_page_duration_ms: float = 0.0
    parallelism_achieved: float = 1.0  # Average concurrent pages
    pipeline_savings_ms: float = 0.0  # Time saved by pipeline mode


class ParallelPageExplorer:
    """
    Manages parallel exploration of multiple pages during site navigation.
    
    Uses a DAG-based approach to determine which pages can be explored
    concurrently. Pages are only explored after their parent pages have
    been visited (to discover their links).
    
    Supports two modes:
    1. Parallel mode: Explore multiple independent pages simultaneously
    2. Pipeline mode: Start capturing screenshots for next page while
       LLM analyzes current page (reduces LLM wait time)
    
    Usage:
        explorer = ParallelPageExplorer(
            page_controller=page,
            llm_provider=llm,
            page_explorer=page_explorer_tool,
            page_analyzer=page_analyzer,
            memory=agent_memory,
            config=parallel_config,
        )
        
        stats = await explorer.explore_site(
            sitemap_graph=sitemap,
            task="Analyze the site structure"
        )
    """
    
    def __init__(
        self,
        page_controller: "PageController",
        llm_provider: "BaseLLMProvider",
        page_explorer: "PageExplorerTool",
        page_analyzer: "PageAnalyzer",
        memory: "AgentMemory",
        config: Optional[ParallelExplorationConfig] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize parallel explorer.
        
        Args:
            page_controller: Browser page controller for navigation
            llm_provider: LLM for analysis
            page_explorer: Tool for capturing page screenshots
            page_analyzer: Analyzer for understanding page structure
            memory: Agent memory for storing PageMaps
            config: Parallel exploration configuration
            progress_callback: Optional callback for progress updates
        """
        self._page = page_controller
        self._llm = llm_provider
        self._page_explorer = page_explorer
        self._page_analyzer = page_analyzer
        self._memory = memory
        self._config = config or ParallelExplorationConfig()
        self._progress_callback = progress_callback
        
        # State tracking
        self._current_task: str = ""
        self._exploration_start: float = 0
        self._pages_in_progress: Dict[str, float] = {}  # url -> start_time
        
    def _report_progress(self, message: str) -> None:
        """Report progress if callback is set."""
        if self._progress_callback:
            self._progress_callback(message)
        logger.info(f"[ParallelExplorer] {message}")
    
    async def explore_site(
        self,
        sitemap_graph: SitemapGraph,
        task: str,
        operation_mode: Optional[Any] = None,
    ) -> ParallelExplorationStats:
        """
        Explore site using parallel/pipeline execution strategy.
        
        This is the main entry point for parallel site exploration.
        It coordinates concurrent page exploration while respecting
        the DAG structure (pages can only be explored after parents).
        
        Args:
            sitemap_graph: Initialized sitemap with homepage explored
            task: User's task description
            operation_mode: Optional operation mode for exploration scope
            
        Returns:
            Statistics about the exploration run
        """
        self._current_task = task
        self._exploration_start = time.time()
        
        stats = ParallelExplorationStats()
        dag = ExplorationDAG(sitemap_graph)
        
        self._report_progress(
            f" Starting parallel exploration (max_parallel={self._config.max_parallel_pages})"
        )
        
        # Determine which mode to use
        if self._config.enable_pipeline_mode:
            stats = await self._explore_pipeline_mode(dag, operation_mode)
        elif self._config.enable_parallel:
            stats = await self._explore_parallel_mode(dag, operation_mode)
        else:
            stats = await self._explore_sequential_mode(dag, operation_mode)
        
        # Calculate final stats
        stats.total_duration_ms = (time.time() - self._exploration_start) * 1000
        if stats.pages_explored > 0:
            stats.avg_page_duration_ms = stats.total_duration_ms / stats.pages_explored
        
        self._report_progress(
            f" Exploration complete: {stats.pages_explored} pages in "
            f"{stats.total_duration_ms/1000:.1f}s "
            f"(avg: {stats.avg_page_duration_ms/1000:.1f}s/page)"
        )
        
        return stats
    
    async def _explore_sequential_mode(
        self,
        dag: ExplorationDAG,
        operation_mode: Optional[Any] = None,
    ) -> ParallelExplorationStats:
        """
        Explore pages sequentially (fallback/baseline mode).
        
        Used when parallel execution is disabled.
        """
        stats = ParallelExplorationStats()
        stats.parallelism_achieved = 1.0
        
        while not dag.is_complete():
            # Get next page to explore
            ready = dag.get_ready_batch(max_size=1)
            if not ready:
                # No pages ready - might need to wait or we're done
                if dag.get_pending_count() == 0:
                    break
                # Pages pending but not ready (parents not visited)
                # This shouldn't happen in sequential mode
                logger.warning("[Sequential] Pages pending but none ready - breaking")
                break
            
            node = ready[0]
            dag.mark_in_progress(node.url)
            
            result = await self._explore_single_page(node.url, operation_mode)
            
            if result.success:
                dag.mark_complete(
                    node.url, 
                    success=True,
                    title=result.title,
                    summary=result.summary,
                    section_count=result.section_count
                )
                stats.pages_explored += 1
                
                # Add discovered links to graph
                await self._add_discovered_links(dag.sitemap, node.url, result)
            else:
                dag.mark_complete(node.url, success=False, error=result.error)
                stats.pages_failed += 1
            
            stats.total_pages += 1
        
        return stats
    
    async def _explore_parallel_mode(
        self,
        dag: ExplorationDAG,
        operation_mode: Optional[Any] = None,
    ) -> ParallelExplorationStats:
        """
        Sequential navigation with parallel LLM analysis.
        
        IMPORTANT: Since we only have one browser tab, navigation must be
        sequential. However, we can parallelize LLM analysis by capturing
        pages first, then analyzing multiple PageMaps concurrently.
        
        This mode captures a batch of pages sequentially, then analyzes
        them in parallel using multiple concurrent LLM calls.
        """
        stats = ParallelExplorationStats()
        
        while not dag.is_complete():
            # Get batch of ready pages (but we'll navigate sequentially)
            batch_size = min(self._config.max_parallel_pages, 3)
            ready = dag.get_ready_batch(max_size=batch_size)
            if not ready:
                if dag.get_pending_count() == 0:
                    break
                logger.warning("[Parallel] Pages pending but none ready - breaking")
                break
            
            self._report_progress(
                f"📦 Processing batch of {len(ready)} pages..."
            )
            
            # Step 1: Capture all pages sequentially (one navigation at a time)
            captured: List[tuple] = []  # [(url, page_map), ...]
            for node in ready:
                dag.mark_in_progress(node.url)
                self._report_progress(f" Capturing: {node.url}")
                
                page_map = await self._navigate_and_capture(node.url, operation_mode)
                captured.append((node.url, page_map))
            
            # Step 2: Analyze all captured pages in parallel
            analysis_tasks = []
            for url, page_map in captured:
                if page_map is not None:
                    task = self._analyze_captured_page(url, page_map)
                    analysis_tasks.append((url, task))
                else:
                    dag.mark_complete(url, success=False, error="Capture failed")
                    stats.pages_failed += 1
                    stats.total_pages += 1
            
            if analysis_tasks:
                self._report_progress(f" Analyzing {len(analysis_tasks)} pages in parallel...")
                results = await asyncio.gather(
                    *[task for _, task in analysis_tasks],
                    return_exceptions=True
                )
                
                for (url, _), result in zip(analysis_tasks, results):
                    if isinstance(result, Exception):
                        dag.mark_complete(url, success=False, error=str(result))
                        stats.pages_failed += 1
                        logger.error(f"[Parallel] Analysis failed: {url} - {result}")
                    elif result.success:
                        dag.mark_complete(
                            url, 
                            success=True,
                            title=result.title,
                            summary=result.summary,
                            section_count=result.section_count
                        )
                        stats.pages_explored += 1
                        await self._add_discovered_links(dag.sitemap, url, result)
                    else:
                        dag.mark_complete(url, success=False, error=result.error)
                        stats.pages_failed += 1
                    
                    stats.total_pages += 1
            
            # Rate limiting between batches
            if self._config.batch_delay_ms > 0:
                await asyncio.sleep(self._config.batch_delay_ms / 1000)
        
        # Parallelism is achieved in LLM analysis phase
        stats.parallelism_achieved = min(self._config.max_parallel_pages, 3)
        
        return stats
    
    async def _explore_pipeline_mode(
        self,
        dag: ExplorationDAG,
        operation_mode: Optional[Any] = None,
    ) -> ParallelExplorationStats:
        """
        Pipeline mode: overlap LLM analysis with next page navigation.
        
        IMPORTANT: Browser navigation must be SEQUENTIAL (one page at a time)
        because we only have one browser tab. However, we can parallelize
        LLM analysis while navigating to the next page.
        
        Timeline:
        Sequential: [Nav1][Cap1][LLM1][Nav2][Cap2][LLM2]...
        Pipeline:   [Nav1][Cap1]─────────[Nav2][Cap2]─────────[Nav3]...
                            └──[LLM1]────┘    └──[LLM2]────┘
        
        The speedup comes from running LLM analysis in background while
        navigating and capturing the next page.
        """
        stats = ParallelExplorationStats()
        llm_tasks_completed = 0
        
        # Queue for pages captured but not yet analyzed
        pending_analysis: List[tuple] = []  # [(url, page_map), ...]
        analysis_tasks: Dict[str, asyncio.Task] = {}  # url -> analysis task
        
        while not dag.is_complete() or pending_analysis or analysis_tasks:
            # Step 1: Check for completed analysis tasks
            completed_urls = []
            for url, task in list(analysis_tasks.items()):
                if task.done():
                    completed_urls.append(url)
                    try:
                        result = task.result()
                        if result.success:
                            dag.mark_complete(
                                url, 
                                success=True,
                                title=result.title,
                                summary=result.summary,
                                section_count=result.section_count
                            )
                            stats.pages_explored += 1
                            await self._add_discovered_links(dag.sitemap, url, result)
                        else:
                            dag.mark_complete(url, success=False, error=result.error)
                            stats.pages_failed += 1
                    except Exception as e:
                        dag.mark_complete(url, success=False, error=str(e))
                        stats.pages_failed += 1
                        logger.error(f"[Pipeline] Analysis failed: {url} - {e}")
                    stats.total_pages += 1
                    llm_tasks_completed += 1
            
            for url in completed_urls:
                del analysis_tasks[url]
            
            # Step 2: Start analysis for captured pages (if slots available)
            max_concurrent_analysis = 2  # Limit concurrent LLM calls
            while pending_analysis and len(analysis_tasks) < max_concurrent_analysis:
                url, page_map = pending_analysis.pop(0)
                if page_map is None:
                    dag.mark_complete(url, success=False, error="Capture returned no data")
                    stats.pages_failed += 1
                    stats.total_pages += 1
                    continue
                
                # Start analysis in background
                task = asyncio.create_task(self._analyze_captured_page(url, page_map))
                analysis_tasks[url] = task
            
            # Step 3: Navigate to and capture next page (SEQUENTIAL - one at a time)
            if not pending_analysis or len(pending_analysis) < 2:  # Keep queue fed
                ready = dag.get_ready_batch(max_size=1)  # Only get ONE page
                if ready:
                    node = ready[0]
                    dag.mark_in_progress(node.url)
                    
                    self._report_progress(f" Exploring: {node.url}")
                    
                    # Navigate and capture (this is synchronous to the browser)
                    page_map = await self._navigate_and_capture(node.url, operation_mode)
                    
                    # Queue for analysis (will run in background)
                    pending_analysis.append((node.url, page_map))
            
            # Brief yield to allow analysis tasks to progress
            if analysis_tasks:
                await asyncio.sleep(0.05)
            
            # Check if we're stuck with pending pages that will never be ready
            if not dag.is_complete():
                pending_count = dag.get_pending_count()
                ready = dag.get_ready_batch(max_size=1)
                
                # If we have pending pages but none are ready, and nothing is in progress
                if pending_count > 0 and not ready and not pending_analysis and not analysis_tasks:
                    logger.warning(
                        f"[Pipeline] Stuck: {pending_count} pages pending but none ready. "
                        f"Likely waiting for parent pages that failed. Breaking."
                    )
                    break
                
                # If no pages pending at all and nothing in progress, we're done
                if pending_count == 0 and not pending_analysis and not analysis_tasks:
                    break
        
        # Wait for any remaining analysis tasks
        if analysis_tasks:
            self._report_progress(f"⏳ Waiting for {len(analysis_tasks)} analysis tasks...")
            results = await asyncio.gather(*analysis_tasks.values(), return_exceptions=True)
            for url, result in zip(analysis_tasks.keys(), results):
                if isinstance(result, Exception):
                    dag.mark_complete(url, success=False, error=str(result))
                    stats.pages_failed += 1
                elif result.success:
                    dag.mark_complete(
                        url, 
                        success=True,
                        title=result.title,
                        summary=result.summary,
                        section_count=result.section_count
                    )
                    stats.pages_explored += 1
                    await self._add_discovered_links(dag.sitemap, url, result)
                else:
                    dag.mark_complete(url, success=False, error=result.error)
                    stats.pages_failed += 1
                stats.total_pages += 1
        
        # Pipeline achieves ~1.5x parallelism (LLM overlaps with navigation)
        stats.parallelism_achieved = 1.5 if llm_tasks_completed > 1 else 1.0
        
        return stats
    
    async def _explore_single_page(
        self,
        url: str,
        operation_mode: Optional[Any] = None,
    ) -> PageExplorationResult:
        """
        Explore a single page (navigate, capture, analyze).
        
        This is the core exploration logic for one page.
        """
        start_time = time.time()
        result = PageExplorationResult(url=url, success=False)
        
        try:
            self._report_progress(f" Exploring: {url}")
            
            # Navigate to page
            await self._page.navigate(url)
            
            # Wait for page to settle
            await asyncio.sleep(0.5)
            
            # Capture screenshots
            from .types import OperationMode
            mode = operation_mode or OperationMode.NAVIGATE
            explorer_result = await self._page_explorer.execute(operation_mode=mode)
            
            if not explorer_result.success:
                result.error = f"Capture failed: {explorer_result.error}"
                return result
            
            page_map = explorer_result.metadata.get("page_map")
            if not page_map:
                result.error = "No page map captured"
                return result
            
            # Analyze page
            analyzed_map = await self._page_analyzer.analyze_page_map(page_map)
            
            if not analyzed_map:
                # Use raw page_map
                analyzed_map = page_map
            
            # Store in memory
            self._memory.store_page_map(url, analyzed_map)
            
            # Extract result data
            result.success = True
            result.title = getattr(analyzed_map, 'title', '')
            result.summary = getattr(analyzed_map, 'summary', '')
            result.section_count = len(getattr(analyzed_map, 'sections', []))
            result.page_map = analyzed_map
            
            # Extract navigation links
            dom_links = getattr(analyzed_map, 'dom_navigation_links', {})
            if dom_links:
                for link in dom_links.get('all_links', []):
                    href = link.get('href', '')
                    text = link.get('text', '')
                    if href and not href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                        result.navigation_links.append({'url': href, 'text': text})
            
            result.duration_ms = (time.time() - start_time) * 1000
            
            logger.info(
                f"[ParallelExplorer] [ok] {url}: {result.section_count} sections, "
                f"{len(result.navigation_links)} links ({result.duration_ms:.0f}ms)"
            )
            
        except Exception as e:
            result.error = str(e)
            result.duration_ms = (time.time() - start_time) * 1000
            logger.error(f"[ParallelExplorer] [fail] {url}: {e}")
        
        return result
    
    async def _navigate_and_capture(
        self,
        url: str,
        operation_mode: Optional[Any] = None,
    ) -> Optional[Any]:
        """
        Navigate to page and capture screenshots (no LLM analysis).
        
        Used in pipeline mode to separate capture from analysis.
        """
        try:
            await self._page.navigate(url)
            await asyncio.sleep(0.5)
            
            from .types import OperationMode
            mode = operation_mode or OperationMode.NAVIGATE
            result = await self._page_explorer.execute(operation_mode=mode)
            
            if result.success:
                return result.metadata.get("page_map")
            return None
            
        except Exception as e:
            logger.error(f"[Pipeline] Navigate/capture failed for {url}: {e}")
            return None
    
    async def _analyze_captured_page(
        self,
        url: str,
        page_map: Any,
    ) -> PageExplorationResult:
        """
        Analyze an already-captured page (LLM analysis only).
        
        Used in pipeline mode after screenshots are captured.
        """
        start_time = time.time()
        result = PageExplorationResult(url=url, success=False)
        
        try:
            analyzed_map = await self._page_analyzer.analyze_page_map(page_map)
            
            if not analyzed_map:
                analyzed_map = page_map
            
            self._memory.store_page_map(url, analyzed_map)
            
            result.success = True
            result.title = getattr(analyzed_map, 'title', '')
            result.summary = getattr(analyzed_map, 'summary', '')
            result.section_count = len(getattr(analyzed_map, 'sections', []))
            result.page_map = analyzed_map
            
            dom_links = getattr(analyzed_map, 'dom_navigation_links', {})
            if dom_links:
                for link in dom_links.get('all_links', []):
                    href = link.get('href', '')
                    text = link.get('text', '')
                    if href and not href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                        result.navigation_links.append({'url': href, 'text': text})
            
            result.duration_ms = (time.time() - start_time) * 1000
            
        except Exception as e:
            result.error = str(e)
            result.duration_ms = (time.time() - start_time) * 1000
        
        return result
    
    async def _add_discovered_links(
        self,
        sitemap: SitemapGraph,
        parent_url: str,
        result: PageExplorationResult,
    ) -> None:
        """
        Add discovered links from exploration result to sitemap.
        
        Filters links using LLM to remove language variants and duplicates.
        """
        if not result.navigation_links:
            return
        
        # Filter links to remove language variants
        filtered_links = await filter_navigation_links_async(
            links=result.navigation_links,
            llm_provider=self._llm,
            current_url=parent_url,
            task=self._current_task
        )
        
        if filtered_links:
            added = sitemap.add_discovered_links(
                parent_url, 
                filtered_links, 
                LinkType.MAIN_NAV
            )
            if added > 0:
                logger.debug(f"[ParallelExplorer] Added {added} links from {parent_url}")
