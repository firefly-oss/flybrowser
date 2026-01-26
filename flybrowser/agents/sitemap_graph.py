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
SitemapGraph - Site Exploration State Machine

This module provides a graph-based data structure for tracking multi-page
site exploration with:
- Depth-limited traversal (Level 0 = homepage, Level 1 = main nav, Level 2 = subpages)
- Visited/discovered/pending state tracking for each page
- Parent-child navigation hierarchy
- Configurable limits enforcement
- Real-time exploration status for agent context

The SitemapGraph maintains the "source of truth" for what pages have been
discovered, which have been visited, and what remains to explore.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse, urljoin

logger = logging.getLogger(__name__)


class PageStatus(Enum):
    """Status of a page in the sitemap exploration."""
    DISCOVERED = "discovered"   # Found via link extraction, not yet visited
    QUEUED = "queued"          # Scheduled to visit (within depth/count limits)
    VISITING = "visiting"       # Currently being explored
    VISITED = "visited"         # Successfully visited and analyzed
    SKIPPED = "skipped"         # Skipped due to limits or filtering
    FAILED = "failed"           # Visit attempted but failed


class LinkType(Enum):
    """Type of navigation link."""
    MAIN_NAV = "main_nav"       # Primary navigation (header/hamburger menu)
    FOOTER = "footer"           # Footer links
    SIDEBAR = "sidebar"         # Sidebar navigation
    CONTENT = "content"         # Links within page content
    BREADCRUMB = "breadcrumb"   # Breadcrumb navigation
    PAGINATION = "pagination"   # Pagination links
    UNKNOWN = "unknown"


@dataclass
class SitemapNode:
    """
    A node in the sitemap graph representing a single page.
    
    Tracks all metadata about a discovered/visited page including
    its position in the navigation hierarchy.
    """
    url: str
    title: str = ""
    
    # Navigation hierarchy
    depth: int = 0  # 0 = homepage, 1 = main nav pages, 2 = subpages
    parent_url: Optional[str] = None
    link_type: LinkType = LinkType.UNKNOWN
    link_text: str = ""  # Text of the link that led to this page
    
    # Exploration state
    status: PageStatus = PageStatus.DISCOVERED
    discovered_at: float = field(default_factory=time.time)
    visited_at: Optional[float] = None
    visit_duration_ms: Optional[float] = None
    
    # Analysis results (populated after visit)
    page_map_stored: bool = False
    section_count: int = 0
    summary: str = ""
    child_links_found: int = 0
    
    # Error tracking
    error_message: Optional[str] = None
    retry_count: int = 0
    
    def __post_init__(self):
        """Normalize URL on creation."""
        self.url = self._normalize_url(self.url)
    
    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL for consistent comparison."""
        # Remove trailing slash, fragment, and normalize
        parsed = urlparse(url)
        # Reconstruct without fragment
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        # Remove trailing slash (except for root)
        if normalized.endswith('/') and len(parsed.path) > 1:
            normalized = normalized[:-1]
        return normalized
    
    def mark_visiting(self) -> None:
        """Mark page as currently being visited."""
        self.status = PageStatus.VISITING
    
    def mark_visited(self, duration_ms: float = 0) -> None:
        """Mark page as successfully visited."""
        self.status = PageStatus.VISITED
        self.visited_at = time.time()
        self.visit_duration_ms = duration_ms
    
    def mark_failed(self, error: str) -> None:
        """Mark page visit as failed."""
        self.status = PageStatus.FAILED
        self.error_message = error
        self.retry_count += 1
    
    def mark_skipped(self, reason: str) -> None:
        """Mark page as skipped."""
        self.status = PageStatus.SKIPPED
        self.error_message = reason
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "url": self.url,
            "title": self.title,
            "depth": self.depth,
            "parent_url": self.parent_url,
            "link_type": self.link_type.value,
            "link_text": self.link_text,
            "status": self.status.value,
            "discovered_at": self.discovered_at,
            "visited_at": self.visited_at,
            "page_map_stored": self.page_map_stored,
            "section_count": self.section_count,
            "summary": self.summary[:200] if self.summary else "",
            "error_message": self.error_message,
        }


@dataclass
class SitemapLimits:
    """
    Configuration limits for sitemap exploration.
    
    Controls how deep and wide the exploration can go.
    """
    max_depth: int = 2          # 0=homepage, 1=main nav, 2=subpages
    max_level1_pages: int = 10  # Max main navigation pages
    max_level2_pages: int = 10  # Max subpages total
    max_total_pages: int = 20   # Hard limit on total visited pages
    
    # Filtering
    same_domain_only: bool = True  # Only explore same domain
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "/login", "/logout", "/signin", "/signup", "/register",
        "/cart", "/checkout", "/account", "/admin",
        ".pdf", ".jpg", ".png", ".gif", ".zip", ".exe",
        "javascript:", "mailto:", "tel:", "#"
    ])


# Schema for LLM-based link filtering (detect duplicates like language versions)
LINK_FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "links_to_visit": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "text": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["url", "text"]
            },
            "description": "Links that should be visited (unique content)"
        },
        "links_to_skip": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "text": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["url", "text", "reason"]
            },
            "description": "Links that should be skipped (duplicates, language variants, etc.)"
        }
    },
    "required": ["links_to_visit", "links_to_skip"]
}


# Schema for LLM-based exploration intent analysis
EXPLORATION_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_site_exploration": {
            "type": "boolean",
            "description": "True if task requires visiting multiple pages of a website"
        },
        "exploration_depth": {
            "type": "string",
            "enum": ["minimal", "shallow", "balanced", "deep"],
            "description": "How deep the exploration should go based on user intent"
        },
        "max_main_pages": {
            "type": "integer",
            "description": "Recommended max main navigation pages to visit (1-15)",
            "minimum": 1,
            "maximum": 15
        },
        "max_subpages": {
            "type": "integer", 
            "description": "Recommended max subpages to visit (0-15)",
            "minimum": 0,
            "maximum": 15
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of why these limits were chosen"
        }
    },
    "required": ["is_site_exploration", "exploration_depth", "max_main_pages", "max_subpages", "reasoning"]
}


class ExplorationIntentAnalyzer:
    """
    LLM-based analyzer for determining site exploration intent and limits.
    
    Uses the LLM to understand user intent and dynamically determine
    appropriate exploration depth - no hardcoded keywords.
    
    Two-phase analysis:
    1. Quick preliminary check (before navigation) - just detect if exploration likely needed
    2. Informed analysis (after first page) - use page content for accurate limits
    """
    
    def __init__(self, llm_provider: Any = None):
        """
        Initialize with optional LLM provider.
        
        Args:
            llm_provider: LLM provider for intent analysis
        """
        self._llm = llm_provider
        self._cached_result: Optional[Dict[str, Any]] = None
        self._cached_task: str = ""
        self._cached_page_context: str = ""
    
    def set_llm_provider(self, llm_provider: Any) -> None:
        """Set LLM provider for analysis."""
        self._llm = llm_provider
    
    def clear_cache(self) -> None:
        """Clear cached analysis to force re-analysis with new context."""
        self._cached_result = None
        self._cached_task = ""
        self._cached_page_context = ""
    
    async def analyze_intent(
        self, 
        task: str, 
        page_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze task intent for site exploration.
        
        Args:
            task: User's task description
            page_context: Optional dict with page info for informed decision:
                - url: Current page URL
                - title: Page title
                - nav_links: List of main navigation links found
                - sections: List of page sections
                - summary: Page summary if available
            
        Returns:
            Dict with exploration intent analysis
        """
        # Generate context key for caching
        context_key = str(page_context) if page_context else ""
        
        # Return cached result if same task AND same context
        if (task == self._cached_task and 
            context_key == self._cached_page_context and 
            self._cached_result):
            return self._cached_result
        
        if not self._llm:
            logger.warning("[ExplorationIntent] No LLM provider - using balanced defaults")
            return self._get_default_analysis()
        
        try:
            from flybrowser.agents.structured_llm import StructuredLLMWrapper
            
            structured_llm = StructuredLLMWrapper(self._llm)
            
            # Build context-aware system prompt
            system_prompt = """You are an expert at analyzing user tasks for web browser automation.

Your job is to determine if a task requires exploring multiple pages of a website,
and if so, how deep and wide that exploration should be.

Consider:
- Does the user want to visit just ONE page or MULTIPLE pages?
- Is the user looking for something SPECIFIC or wanting a COMPREHENSIVE overview?
- Should we visit main navigation pages only, or also go into subpages?
- How many pages would reasonably satisfy the user's request without wasting time?

Be efficient - don't recommend exploring more than necessary.
Most tasks need fewer pages than you might think.

You MUST respond with a valid JSON object."""

            # Build user prompt with optional page context
            user_prompt = f"""Analyze this task and determine the appropriate site exploration strategy:

Task: {task}
"""
            
            # Add page context if available (for informed decision-making)
            if page_context:
                user_prompt += f"""
## Current Page Context (use this to make an INFORMED decision)
"""
                if page_context.get('url'):
                    user_prompt += f"URL: {page_context['url']}\n"
                if page_context.get('title'):
                    user_prompt += f"Title: {page_context['title']}\n"
                if page_context.get('summary'):
                    user_prompt += f"Summary: {page_context['summary'][:300]}\n"
                
                nav_links = page_context.get('nav_links', [])
                if nav_links:
                    user_prompt += f"\nMain Navigation Links ({len(nav_links)} found):\n"
                    for link in nav_links[:15]:  # Show up to 15 links
                        text = link.get('text', 'Unknown')[:50]
                        href = link.get('href', link.get('url', ''))[:80]
                        user_prompt += f"  - {text}: {href}\n"
                
                sections = page_context.get('sections', [])
                if sections:
                    user_prompt += f"\nPage Sections ({len(sections)} found):\n"
                    for section in sections[:10]:
                        name = section.get('name', section.get('title', 'Unknown'))[:50]
                        user_prompt += f"  - {name}\n"
                
                user_prompt += """
Based on the ACTUAL site structure above, determine the appropriate exploration depth.
"""
            
            user_prompt += """
Determine and respond with EXACTLY these fields:
- "is_site_exploration": true/false - Does task require visiting multiple pages?
- "exploration_depth": "minimal"/"shallow"/"balanced"/"deep"
  - minimal: Just find specific info (1-5 pages max)
  - shallow: Quick overview of main pages (5-8 pages)
  - balanced: Good coverage without excess (8-12 pages)
  - deep: Comprehensive exploration requested (12-20 pages)
- "max_main_pages": integer (1-15) - How many main navigation pages to visit
- "max_subpages": integer (0-15) - How many subpages to visit
- "reasoning": string - Brief explanation of your choice"""

            result = await structured_llm.generate_structured(
                prompt=user_prompt,
                schema=EXPLORATION_INTENT_SCHEMA,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=500,
            )
            
            if result and result.get("is_site_exploration") is not None:
                self._cached_result = result
                self._cached_task = task
                self._cached_page_context = context_key
                
                context_note = " (with page context)" if page_context else " (preliminary)"
                logger.info(
                    f"[ExplorationIntent] LLM analysis{context_note}: {result.get('exploration_depth')} "
                    f"(main={result.get('max_main_pages')}, sub={result.get('max_subpages')}) - "
                    f"{result.get('reasoning', '')[:100]}"
                )
                return result
            else:
                logger.warning(f"[ExplorationIntent] LLM analysis returned invalid result: {result}")
                return self._get_default_analysis()
                
        except Exception as e:
            logger.warning(f"[ExplorationIntent] Error in LLM analysis: {e}")
            return self._get_default_analysis()
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Return balanced default analysis."""
        return {
            "is_site_exploration": True,
            "exploration_depth": "balanced",
            "max_main_pages": 8,
            "max_subpages": 5,
            "reasoning": "Default balanced exploration (no LLM analysis available)"
        }
    
    def analysis_to_limits(self, analysis: Dict[str, Any]) -> SitemapLimits:
        """
        Convert LLM analysis to SitemapLimits configuration.
        
        Args:
            analysis: Result from analyze_intent()
            
        Returns:
            SitemapLimits configured based on analysis
        """
        depth = analysis.get("exploration_depth", "balanced")
        max_main = analysis.get("max_main_pages", 8)
        max_sub = analysis.get("max_subpages", 5)
        
        # Map depth to max_depth value
        depth_map = {
            "minimal": 1,
            "shallow": 1,
            "balanced": 2,
            "deep": 2
        }
        
        return SitemapLimits(
            max_depth=depth_map.get(depth, 2),
            max_level1_pages=min(max_main, 15),
            max_level2_pages=min(max_sub, 15),
            max_total_pages=min(max_main + max_sub + 1, 25)  # +1 for homepage
        )


class LinkFilterAnalyzer:
    """
    LLM-based analyzer for filtering navigation links.
    
    Intelligently identifies and filters out:
    - Language variants (same content in different languages)
    - Duplicate pages with different URLs
    - Administrative/utility pages
    - Any other redundant links
    
    Uses LLM to understand link semantics - NO hardcoded patterns.
    Has memory to avoid re-analyzing the same links.
    Uses PromptManager for templated prompts (link_filter template).
    """
    
    def __init__(self, llm_provider: Any = None, prompt_manager: Any = None):
        self._llm = llm_provider
        self._prompt_manager = prompt_manager
        # Memory: track URLs we've already decided on
        self._decided_urls: Dict[str, bool] = {}  # url -> should_visit
        self._skip_reasons: Dict[str, str] = {}   # url -> reason for skipping
    
    def set_llm_provider(self, llm_provider: Any) -> None:
        """Set LLM provider for analysis."""
        self._llm = llm_provider
    
    def set_prompt_manager(self, prompt_manager: Any) -> None:
        """Set PromptManager for templated prompts."""
        self._prompt_manager = prompt_manager
    
    def clear_memory(self) -> None:
        """Clear filter memory for a new exploration session."""
        self._decided_urls.clear()
        self._skip_reasons.clear()
    
    def is_already_decided(self, url: str) -> Optional[bool]:
        """Check if we already have a decision for this URL."""
        normalized = url.rstrip('/')
        return self._decided_urls.get(normalized)
    
    def remember_decision(self, url: str, should_visit: bool, reason: str = "") -> None:
        """Remember a decision about a URL."""
        normalized = url.rstrip('/')
        self._decided_urls[normalized] = should_visit
        if not should_visit and reason:
            self._skip_reasons[normalized] = reason
    
    def _is_external_domain(self, url: str, base_url: Optional[str]) -> bool:
        """
        Check if a URL is on a different domain than the base URL.
        
        This is a simple domain comparison - NOT pattern matching.
        External domains are filtered before LLM to save costs.
        
        Args:
            url: URL to check
            base_url: Base URL to compare against
            
        Returns:
            True if external domain, False if same domain or relative
        """
        if not base_url or not url:
            return False
        
        # Handle relative URLs - they're always same domain
        if not url.startswith(('http://', 'https://')):
            return False
        
        try:
            base_parsed = urlparse(base_url)
            url_parsed = urlparse(url)
            
            # Compare domains
            base_domain = base_parsed.netloc.lower()
            url_domain = url_parsed.netloc.lower()
            
            return base_domain != url_domain
        except Exception:
            return False
    
    async def filter_links(
        self,
        links: List[Dict[str, str]],
        current_url: Optional[str] = None,
        task: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Use LLM to filter navigation links and identify duplicates.
        
        Has memory - won't re-analyze links we've already decided on.
        Pre-filters external domains before LLM (simple domain check).
        Language variants and homepage duplicates are detected dynamically by LLM.
        
        Args:
            links: List of link dicts with 'url', 'text' keys
            current_url: URL of the current page (for context)
            task: User's task for context
            
        Returns:
            Dict with 'to_visit' and 'to_skip' lists
        """
        if not links:
            return {'to_visit': [], 'to_skip': []}
        
        # Pre-filter: Remove external domains BEFORE LLM (simple domain check, not pattern matching)
        # This saves LLM calls and prevents external suggestions
        pre_filtered_links = []
        pre_filtered_skip = []
        
        for link in links:
            url = link.get('url', link.get('href', ''))
            if not url:
                continue
            
            # Filter: External domains only (LLM handles language variants dynamically)
            if self._is_external_domain(url, current_url):
                pre_filtered_skip.append({
                    **link,
                    'reason': 'External domain - outside target site scope'
                })
                self.remember_decision(url, False, 'External domain')
                continue
            
            pre_filtered_links.append(link)
        
        if pre_filtered_skip:
            logger.debug(
                f"[LinkFilter] Pre-filtered {len(pre_filtered_skip)} external domain links"
            )
        
        # If all links were pre-filtered, return early
        if not pre_filtered_links:
            return {'to_visit': [], 'to_skip': pre_filtered_skip}
        
        # Separate links we've already decided on vs new ones
        already_decided_visit = []
        already_decided_skip = []
        new_links = []
        
        for link in pre_filtered_links:
            url = link.get('url', link.get('href', ''))
            if not url:
                continue
            
            decision = self.is_already_decided(url)
            if decision is True:
                already_decided_visit.append(link)
            elif decision is False:
                reason = self._skip_reasons.get(url.rstrip('/'), 'Previously filtered')
                already_decided_skip.append({**link, 'reason': reason})
            else:
                new_links.append(link)
        
        # Combine pre-filtered skips with already decided skips
        all_skipped = pre_filtered_skip + already_decided_skip
        
        # If all links already decided, return from memory
        if not new_links:
            logger.debug(f"[LinkFilter] All {len(pre_filtered_links)} links already in memory")
            return {'to_visit': already_decided_visit, 'to_skip': all_skipped}
        
        # If no LLM, accept all new links
        if not self._llm:
            for link in new_links:
                url = link.get('url', link.get('href', ''))
                self.remember_decision(url, True)
            return {'to_visit': already_decided_visit + new_links, 'to_skip': already_decided_skip}
        
        try:
            from flybrowser.agents.structured_llm import StructuredLLMWrapper
            
            structured_llm = StructuredLLMWrapper(self._llm)
            
            # Format links for prompt
            links_text = "\n".join([
                f"  - {link.get('text', 'Unknown')}: {link.get('url', link.get('href', ''))}"
                for link in new_links[:25]  # Limit to 25 new links
            ])
            
            # Try to get prompts from PromptManager (templated approach)
            system_prompt = None
            user_prompt = None
            
            if self._prompt_manager:
                try:
                    prompts = self._prompt_manager.get_prompt(
                        "link_filter",
                        link_count=len(new_links),
                        links_text=links_text,
                        current_url=current_url or "",
                        task=task or ""
                    )
                    system_prompt = prompts.get("system", "")
                    user_prompt = prompts.get("user", "")
                    logger.debug("[LinkFilter] Using templated prompt from PromptManager")
                except Exception as e:
                    logger.debug(f"[LinkFilter] PromptManager fallback: {e}")
            
            # Fallback to inline prompts if PromptManager not available or failed
            if not system_prompt or not user_prompt:
                system_prompt = self._get_fallback_system_prompt()
                user_prompt = self._get_fallback_user_prompt(links_text, len(new_links), current_url, task)
            
            result = await structured_llm.generate_structured(
                prompt=user_prompt,
                schema=LINK_FILTER_SCHEMA,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=1500,
            )
            
            if result and 'links_to_visit' in result:
                llm_to_visit = result.get('links_to_visit', [])
                llm_to_skip = result.get('links_to_skip', [])
                
                # Remember decisions for future
                for link in llm_to_visit:
                    url = link.get('url', link.get('href', ''))
                    if url:
                        self.remember_decision(url, True)
                
                for link in llm_to_skip:
                    url = link.get('url', link.get('href', ''))
                    reason = link.get('reason', 'Filtered by LLM')
                    if url:
                        self.remember_decision(url, False, reason)
                
                # Log filtering results
                if llm_to_skip:
                    skip_summary = [f"{s.get('text', '?')[:20]}: {s.get('reason', '?')[:30]}" for s in llm_to_skip[:3]]
                    logger.info(f"[LinkFilter] Filtered {len(llm_to_skip)} links: {', '.join(skip_summary)}")
                
                total_visit = already_decided_visit + llm_to_visit
                total_skip = pre_filtered_skip + already_decided_skip + llm_to_skip
                
                logger.info(
                    f"[LinkFilter] Result: {len(total_visit)} to visit, {len(total_skip)} skipped "
                    f"({len(pre_filtered_skip)} pre-filtered, {len(already_decided_visit)} from memory, {len(llm_to_skip)} by LLM)"
                )
                
                return {'to_visit': total_visit, 'to_skip': total_skip}
            else:
                logger.warning(f"[LinkFilter] LLM returned invalid result, keeping all new links")
                for link in new_links:
                    url = link.get('url', link.get('href', ''))
                    self.remember_decision(url, True)
                return {'to_visit': already_decided_visit + new_links, 'to_skip': pre_filtered_skip + already_decided_skip}
                
        except Exception as e:
            logger.warning(f"[LinkFilter] Error in LLM analysis: {e}, keeping all new links")
            for link in new_links:
                url = link.get('url', link.get('href', ''))
                self.remember_decision(url, True)
            return {'to_visit': already_decided_visit + new_links, 'to_skip': pre_filtered_skip + already_decided_skip}
    
    def _get_fallback_system_prompt(self) -> str:
        """Return fallback system prompt when PromptManager is not available."""
        return """You are an expert web analyst that identifies unique, valuable navigation links.

Your task: From a list of links, identify which ones lead to UNIQUE content pages.

FILTER OUT (mark as skip):
1. Language variants - same page in different languages (e.g., /es/, /fr/, /en-US/, lang=de)
2. Homepage duplicates - links that go back to the main page (/, /home, /index)
3. Utility pages - login, cart, account, search, mailto, tel
4. Anchor links - same-page anchors (#section)
5. Obvious duplicates - same content with different URL structure

KEEP (mark as visit):
1. Distinct content pages (products, services, about, contact, pricing, features)
2. Category/section pages that lead to real content
3. Blog posts, articles, documentation pages

IMPORTANT: 
- Analyze link TEXT and URL together to understand what each link represents
- When you see multiple language variants (same text pattern with /es/, /fr/, etc.), keep only ONE
- The homepage is ALREADY being analyzed - filter out any link that returns to it

Return valid JSON with links_to_visit and links_to_skip arrays."""
    
    def _get_fallback_user_prompt(
        self, 
        links_text: str, 
        link_count: int, 
        current_url: Optional[str], 
        task: Optional[str]
    ) -> str:
        """Return fallback user prompt when PromptManager is not available."""
        url_context = f"\nCurrent page: {current_url}" if current_url else ""
        task_context = f"\nUser's task: {task}" if task else ""
        
        return f"""Analyze these {link_count} navigation links and categorize them:{url_context}{task_context}

Links to analyze:
{links_text}

Return JSON with:
- links_to_visit: Array of {{"url": "...", "text": "...", "reason": "why it's valuable"}}
- links_to_skip: Array of {{"url": "...", "text": "...", "reason": "why filtered"}}"""


# Global analyzer instances (LLM set later)
_exploration_analyzer = ExplorationIntentAnalyzer()
_link_filter_analyzer = LinkFilterAnalyzer()


async def analyze_exploration_intent_async(
    task: str, 
    llm_provider: Any = None,
    page_context: Optional[Dict[str, Any]] = None
) -> SitemapLimits:
    """
    Analyze user task using LLM to determine exploration limits.
    
    Args:
        task: User's task description
        llm_provider: LLM provider for analysis
        page_context: Optional page info for informed decision:
            - url: Current page URL
            - title: Page title  
            - nav_links: List of navigation links
            - sections: List of page sections
            - summary: Page summary
        
    Returns:
        SitemapLimits configured based on LLM analysis
    """
    if llm_provider:
        _exploration_analyzer.set_llm_provider(llm_provider)
    
    # Clear cache if we have new page context to force re-analysis
    if page_context:
        _exploration_analyzer.clear_cache()
    
    analysis = await _exploration_analyzer.analyze_intent(task, page_context)
    limits = _exploration_analyzer.analysis_to_limits(analysis)
    
    context_note = " (with page context)" if page_context else " (preliminary)"
    logger.info(
        f"[SitemapGraph] LLM intent analysis{context_note} → "
        f"depth={limits.max_depth}, L1={limits.max_level1_pages}, "
        f"L2={limits.max_level2_pages}, total={limits.max_total_pages}"
    )
    
    return limits


async def is_site_exploration_task_async(
    task: str, 
    llm_provider: Any = None,
    page_context: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Use LLM to determine if task requires multi-page exploration.
    
    Args:
        task: User's task description
        llm_provider: LLM provider for analysis
        page_context: Optional page context for informed decision
        
    Returns:
        True if task involves exploring multiple pages
    """
    if llm_provider:
        _exploration_analyzer.set_llm_provider(llm_provider)
    
    analysis = await _exploration_analyzer.analyze_intent(task, page_context)
    return analysis.get("is_site_exploration", False)


def reset_link_filter_memory() -> None:
    """
    Clear the link filter's memory for a new exploration session.
    
    Call this at the start of a new site exploration to ensure
    fresh link analysis.
    """
    _link_filter_analyzer.clear_memory()
    logger.debug("[LinkFilter] Memory cleared for new exploration session")


async def filter_navigation_links_async(
    links: List[Dict[str, str]],
    llm_provider: Any = None,
    current_url: Optional[str] = None,
    task: Optional[str] = None,
    prompt_manager: Any = None
) -> List[Dict[str, str]]:
    """
    Use LLM to filter navigation links, removing language variants and duplicates.
    
    Fully LLM-driven - NO hardcoded patterns. The LLM analyzes URLs to detect
    language variants, duplicates, and other redundant links.
    
    Has memory - previously analyzed URLs are cached to avoid re-processing.
    Call reset_link_filter_memory() at the start of new exploration sessions.
    
    Args:
        links: List of link dicts with 'url'/'href', 'text' keys
        llm_provider: LLM provider for analysis
        current_url: URL of current page (for context)
        task: User's task for context
        prompt_manager: Optional PromptManager for templated prompts
        
    Returns:
        Filtered list of links (unique content only)
    """
    if llm_provider:
        _link_filter_analyzer.set_llm_provider(llm_provider)
    if prompt_manager:
        _link_filter_analyzer.set_prompt_manager(prompt_manager)
    
    # Pass current_url directly - let LLM figure out language from context
    result = await _link_filter_analyzer.filter_links(
        links=links,
        current_url=current_url,
        task=task
    )
    
    return result.get('to_visit', links)


class SitemapGraph:
    """
    Graph-based sitemap for tracking site exploration.
    
    Maintains complete state of multi-page exploration including:
    - All discovered pages with their hierarchy
    - Visit status (discovered/visiting/visited/skipped/failed)
    - Depth tracking with configurable limits
    - Real-time statistics for agent context
    
    Usage:
        graph = SitemapGraph(homepage_url, limits=SitemapLimits(max_depth=2))
        graph.mark_visited(homepage_url)
        graph.add_discovered_links(homepage_url, links, depth=1)
        
        # Get next page to visit
        next_page = graph.get_next_pending()
        
        # Get status for agent prompt
        status = graph.format_exploration_status()
    """
    
    def __init__(
        self, 
        homepage_url: str,
        limits: Optional[SitemapLimits] = None
    ) -> None:
        """
        Initialize sitemap graph with homepage.
        
        Args:
            homepage_url: The starting URL (Level 0)
            limits: Exploration limits configuration
        """
        self.limits = limits or SitemapLimits()
        self._base_domain = urlparse(homepage_url).netloc
        
        # Core graph storage: URL -> SitemapNode
        self._nodes: Dict[str, SitemapNode] = {}
        
        # Indexes for fast lookup
        self._by_depth: Dict[int, Set[str]] = {0: set(), 1: set(), 2: set()}
        self._by_status: Dict[PageStatus, Set[str]] = {s: set() for s in PageStatus}
        self._children: Dict[str, Set[str]] = {}  # parent_url -> set of child urls
        
        # Initialize with homepage
        homepage = SitemapNode(
            url=homepage_url,
            title="Homepage",
            depth=0,
            link_type=LinkType.MAIN_NAV,
            status=PageStatus.QUEUED
        )
        self._add_node(homepage)
        
        logger.info(
            f"[SitemapGraph] Initialized with homepage: {homepage_url}, "
            f"limits: depth={self.limits.max_depth}, "
            f"L1={self.limits.max_level1_pages}, "
            f"L2={self.limits.max_level2_pages}, "
            f"total={self.limits.max_total_pages}"
        )
    
    def _add_node(self, node: SitemapNode) -> bool:
        """
        Add a node to the graph with index updates.
        
        Returns:
            True if added, False if URL already exists
        """
        if node.url in self._nodes:
            return False
        
        self._nodes[node.url] = node
        
        # Update indexes
        if node.depth in self._by_depth:
            self._by_depth[node.depth].add(node.url)
        self._by_status[node.status].add(node.url)
        
        # Update children index
        if node.parent_url:
            if node.parent_url not in self._children:
                self._children[node.parent_url] = set()
            self._children[node.parent_url].add(node.url)
        
        return True
    
    def _update_status(self, url: str, new_status: PageStatus) -> None:
        """Update node status and indexes."""
        if url not in self._nodes:
            return
        
        node = self._nodes[url]
        old_status = node.status
        
        # Update indexes
        self._by_status[old_status].discard(url)
        self._by_status[new_status].add(url)
        
        node.status = new_status
    
    def _should_exclude(self, url: str) -> bool:
        """Check if URL should be excluded based on patterns."""
        url_lower = url.lower()
        for pattern in self.limits.exclude_patterns:
            if pattern in url_lower:
                return True
        return False
    
    def _is_same_domain(self, url: str) -> bool:
        """Check if URL is on the same domain."""
        parsed = urlparse(url)
        return parsed.netloc == self._base_domain or parsed.netloc == ""
    
    def get_node(self, url: str) -> Optional[SitemapNode]:
        """Get node by URL."""
        normalized = SitemapNode._normalize_url(url)
        return self._nodes.get(normalized)
    
    def has_url(self, url: str) -> bool:
        """Check if URL is in the graph."""
        normalized = SitemapNode._normalize_url(url)
        return normalized in self._nodes
    
    def add_discovered_links(
        self,
        parent_url: str,
        links: List[Dict[str, str]],
        link_type: LinkType = LinkType.UNKNOWN
    ) -> int:
        """
        Add newly discovered links from a visited page.
        
        Args:
            parent_url: URL of the page where links were found
            links: List of link dicts with 'url', 'text' keys
            link_type: Type of navigation these links came from
            
        Returns:
            Number of new links added (not duplicates or excluded)
        """
        parent_node = self.get_node(parent_url)
        if not parent_node:
            logger.warning(f"[SitemapGraph] Parent not found: {parent_url}")
            return 0
        
        child_depth = parent_node.depth + 1
        added_count = 0
        
        for link in links:
            url = link.get("url", link.get("href", ""))
            text = link.get("text", link.get("label", ""))
            
            if not url:
                continue
            
            # Resolve relative URLs
            if not url.startswith(("http://", "https://")):
                url = urljoin(parent_url, url)
            
            # Normalize
            url = SitemapNode._normalize_url(url)
            
            # Skip if already known
            if self.has_url(url):
                continue
            
            # Apply filters
            if self._should_exclude(url):
                continue
            
            if self.limits.same_domain_only and not self._is_same_domain(url):
                continue
            
            # Check depth limit
            if child_depth > self.limits.max_depth:
                continue
            
            # Check level-specific limits
            if not self._can_add_at_depth(child_depth):
                continue
            
            # Check total limit
            if self.total_count >= self.limits.max_total_pages:
                logger.info(f"[SitemapGraph] Total page limit reached ({self.limits.max_total_pages})")
                break
            
            # Create and add node
            node = SitemapNode(
                url=url,
                title=text or url,
                depth=child_depth,
                parent_url=parent_url,
                link_type=link_type,
                link_text=text,
                status=PageStatus.QUEUED
            )
            
            if self._add_node(node):
                added_count += 1
        
        # Update parent's child count
        parent_node.child_links_found = len(self._children.get(parent_url, []))
        
        logger.info(
            f"[SitemapGraph] Added {added_count} links from {parent_url} "
            f"(depth {child_depth}, type: {link_type.value})"
        )
        
        return added_count
    
    def _can_add_at_depth(self, depth: int) -> bool:
        """Check if we can add more pages at given depth."""
        if depth == 1:
            current = len(self._by_depth.get(1, set()))
            return current < self.limits.max_level1_pages
        elif depth == 2:
            current = len(self._by_depth.get(2, set()))
            return current < self.limits.max_level2_pages
        return depth <= self.limits.max_depth
    
    def mark_visiting(self, url: str) -> bool:
        """
        Mark a page as currently being visited.
        
        Returns:
            True if status updated, False if URL not found
        """
        node = self.get_node(url)
        if not node:
            return False
        
        self._update_status(url, PageStatus.VISITING)
        node.mark_visiting()
        return True
    
    def mark_visited(
        self,
        url: str,
        title: str = "",
        summary: str = "",
        section_count: int = 0,
        page_map_stored: bool = True,
        duration_ms: float = 0
    ) -> bool:
        """
        Mark a page as successfully visited with analysis results.
        
        Args:
            url: Page URL
            title: Page title from analysis
            summary: Page summary from PageMap
            section_count: Number of sections identified
            page_map_stored: Whether PageMap was stored in memory
            duration_ms: Time spent analyzing page
            
        Returns:
            True if status updated, False if URL not found
        """
        node = self.get_node(url)
        if not node:
            # Auto-add if not in graph (e.g., direct navigation)
            node = SitemapNode(url=url, title=title, depth=0)
            self._add_node(node)
        
        self._update_status(node.url, PageStatus.VISITED)
        node.mark_visited(duration_ms)
        
        if title:
            node.title = title
        node.summary = summary
        node.section_count = section_count
        node.page_map_stored = page_map_stored
        
        logger.debug(
            f"[SitemapGraph] Marked visited: {url} "
            f"(depth={node.depth}, sections={section_count})"
        )
        
        return True
    
    def mark_failed(self, url: str, error: str) -> bool:
        """Mark a page visit as failed."""
        node = self.get_node(url)
        if not node:
            return False
        
        self._update_status(url, PageStatus.FAILED)
        node.mark_failed(error)
        return True
    
    def mark_skipped(self, url: str, reason: str) -> bool:
        """Mark a page as skipped."""
        node = self.get_node(url)
        if not node:
            return False
        
        self._update_status(url, PageStatus.SKIPPED)
        node.mark_skipped(reason)
        return True
    
    def get_next_pending(self, depth: Optional[int] = None) -> Optional[SitemapNode]:
        """
        Get next page to visit (breadth-first by depth).
        
        Args:
            depth: Optional depth filter (None = any depth, prioritize lower)
            
        Returns:
            Next SitemapNode to visit, or None if exploration complete
        """
        # Check total visited limit
        if self.visited_count >= self.limits.max_total_pages:
            return None
        
        # Prioritize by depth (breadth-first)
        for d in range(self.limits.max_depth + 1):
            if depth is not None and d != depth:
                continue
            
            for url in self._by_depth.get(d, set()):
                node = self._nodes[url]
                if node.status == PageStatus.QUEUED:
                    return node
        
        return None
    
    def get_pending_at_depth(self, depth: int) -> List[SitemapNode]:
        """Get all pending pages at a specific depth."""
        result = []
        for url in self._by_depth.get(depth, set()):
            node = self._nodes[url]
            if node.status == PageStatus.QUEUED:
                result.append(node)
        return result
    
    def get_visited_at_depth(self, depth: int) -> List[SitemapNode]:
        """Get all visited pages at a specific depth."""
        result = []
        for url in self._by_depth.get(depth, set()):
            node = self._nodes[url]
            if node.status == PageStatus.VISITED:
                result.append(node)
        return result
    
    # Statistics properties
    @property
    def total_count(self) -> int:
        """Total pages in graph (all statuses)."""
        return len(self._nodes)
    
    @property
    def visited_count(self) -> int:
        """Count of visited pages."""
        return len(self._by_status[PageStatus.VISITED])
    
    @property
    def pending_count(self) -> int:
        """Count of pages waiting to visit."""
        return len(self._by_status[PageStatus.QUEUED])
    
    @property
    def failed_count(self) -> int:
        """Count of failed page visits."""
        return len(self._by_status[PageStatus.FAILED])
    
    @property
    def skipped_count(self) -> int:
        """Count of skipped pages."""
        return len(self._by_status[PageStatus.SKIPPED])
    
    def get_depth_stats(self) -> Dict[int, Dict[str, int]]:
        """Get statistics broken down by depth level."""
        stats = {}
        for depth in range(self.limits.max_depth + 1):
            urls = self._by_depth.get(depth, set())
            visited = sum(1 for u in urls if self._nodes[u].status == PageStatus.VISITED)
            pending = sum(1 for u in urls if self._nodes[u].status == PageStatus.QUEUED)
            stats[depth] = {
                "total": len(urls),
                "visited": visited,
                "pending": pending,
                "limit": (
                    1 if depth == 0 else
                    self.limits.max_level1_pages if depth == 1 else
                    self.limits.max_level2_pages
                )
            }
        return stats
    
    def is_exploration_complete(self) -> bool:
        """
        Check if exploration is complete.
        
        Complete when:
        - No pending pages remain, OR
        - Total visit limit reached
        """
        if self.visited_count >= self.limits.max_total_pages:
            return True
        return self.pending_count == 0
    
    def format_exploration_status(self) -> str:
        """
        Format current exploration status for agent context.
        
        This is the key method that provides real-time tracking
        information to the agent's prompt.
        
        Returns:
            Formatted string showing exploration progress
        """
        lines = ["##  Site Exploration Status (Navigation Tracking)"]
        lines.append("_Note: This tracks VISITED pages. Check 'Extracted Data' section for text extraction status._")
        
        # Overall progress
        lines.append(f"\n**Progress**: {self.visited_count}/{self.total_count} pages visited")
        
        if self.pending_count > 0:
            lines.append(f"**Remaining**: {self.pending_count} pages to visit")
        
        # Per-depth breakdown
        depth_stats = self.get_depth_stats()
        lines.append("\n**By Level**:")
        
        level_names = {0: "Homepage", 1: "Main Pages", 2: "Subpages"}
        for depth, stats in depth_stats.items():
            if stats["total"] > 0:
                name = level_names.get(depth, f"Level {depth}")
                status = "[ok]" if stats["visited"] == stats["total"] else "[partial]"
                lines.append(
                    f"  {status} {name}: {stats['visited']}/{stats['total']} "
                    f"(limit: {stats['limit']})"
                )
        
        # List pending pages (so agent knows what's left)
        pending = [n for n in self._nodes.values() if n.status == PageStatus.QUEUED]
        if pending:
            lines.append("\n**Pending Pages**:")
            for node in pending[:10]:  # Limit display
                lines.append(f"  - [{node.link_text or node.title}]({node.url}) (L{node.depth})")
            if len(pending) > 10:
                lines.append(f"  ... and {len(pending) - 10} more")
        
        # Completion status
        if self.is_exploration_complete():
            lines.append("\n **Exploration Complete**")
        elif self.visited_count >= self.limits.max_total_pages:
            lines.append(f"\n **Limit Reached** ({self.limits.max_total_pages} pages max)")
        
        return "\n".join(lines)
    
    def format_visited_summary(self) -> str:
        """
        Format summary of all visited pages.
        
        Used for final aggregation and reporting.
        """
        visited = [n for n in self._nodes.values() if n.status == PageStatus.VISITED]
        if not visited:
            return "No pages visited yet."
        
        lines = [f"## Visited Pages ({len(visited)} total)\n"]
        
        # Group by depth
        by_depth: Dict[int, List[SitemapNode]] = {}
        for node in visited:
            by_depth.setdefault(node.depth, []).append(node)
        
        level_names = {0: "Homepage", 1: "Main Pages", 2: "Subpages"}
        for depth in sorted(by_depth.keys()):
            nodes = by_depth[depth]
            lines.append(f"### {level_names.get(depth, f'Level {depth}')} ({len(nodes)})")
            
            for node in nodes:
                lines.append(f"- **{node.title}**")
                lines.append(f"  URL: {node.url}")
                if node.summary:
                    summary = node.summary[:200] + "..." if len(node.summary) > 200 else node.summary
                    lines.append(f"  Summary: {summary}")
                lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "base_domain": self._base_domain,
            "limits": {
                "max_depth": self.limits.max_depth,
                "max_level1_pages": self.limits.max_level1_pages,
                "max_level2_pages": self.limits.max_level2_pages,
                "max_total_pages": self.limits.max_total_pages,
            },
            "stats": {
                "total": self.total_count,
                "visited": self.visited_count,
                "pending": self.pending_count,
                "failed": self.failed_count,
                "skipped": self.skipped_count,
            },
            "depth_stats": self.get_depth_stats(),
            "nodes": {url: node.to_dict() for url, node in self._nodes.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SitemapGraph:
        """Deserialize graph from dictionary."""
        limits_data = data.get("limits", {})
        limits = SitemapLimits(
            max_depth=limits_data.get("max_depth", 2),
            max_level1_pages=limits_data.get("max_level1_pages", 10),
            max_level2_pages=limits_data.get("max_level2_pages", 10),
            max_total_pages=limits_data.get("max_total_pages", 20),
        )
        
        # Find homepage to initialize
        homepage_url = None
        for url, node_data in data.get("nodes", {}).items():
            if node_data.get("depth") == 0:
                homepage_url = url
                break
        
        if not homepage_url:
            raise ValueError("No homepage found in serialized data")
        
        graph = cls(homepage_url, limits=limits)
        
        # Restore all nodes
        for url, node_data in data.get("nodes", {}).items():
            if url == homepage_url:
                # Update existing homepage node
                node = graph._nodes[url]
            else:
                node = SitemapNode(
                    url=url,
                    title=node_data.get("title", ""),
                    depth=node_data.get("depth", 1),
                    parent_url=node_data.get("parent_url"),
                    link_type=LinkType(node_data.get("link_type", "unknown")),
                    link_text=node_data.get("link_text", ""),
                )
                graph._add_node(node)
            
            # Restore status
            status = PageStatus(node_data.get("status", "discovered"))
            graph._update_status(node.url, status)
            node.status = status
            node.summary = node_data.get("summary", "")
            node.section_count = node_data.get("section_count", 0)
            node.page_map_stored = node_data.get("page_map_stored", False)
        
        return graph
    
    # ===== PARALLEL EXPLORATION SUPPORT =====
    
    def get_ready_for_parallel(self, max_count: int = 4) -> List[SitemapNode]:
        """
        Get pages ready for parallel exploration.
        
        Returns pages that:
        - Are in QUEUED status
        - Have their parent already VISITED (or have no parent)
        - Are not currently being visited
        
        This enables DAG-based parallel execution where independent
        pages at the same level can be processed concurrently.
        
        Args:
            max_count: Maximum number of pages to return
            
        Returns:
            List of SitemapNodes ready for parallel processing
        """
        ready = []
        
        # Check total visited limit
        remaining_capacity = self.limits.max_total_pages - self.visited_count
        max_count = min(max_count, remaining_capacity)
        
        if max_count <= 0:
            return []
        
        # Get all queued pages, prioritize by depth (breadth-first)
        for depth in range(self.limits.max_depth + 1):
            for url in self._by_depth.get(depth, set()):
                if len(ready) >= max_count:
                    break
                    
                node = self._nodes[url]
                
                # Must be queued (not visiting, visited, etc.)
                if node.status != PageStatus.QUEUED:
                    continue
                
                # Parent must be visited (or no parent = homepage)
                if node.parent_url:
                    parent = self._nodes.get(node.parent_url)
                    if not parent or parent.status != PageStatus.VISITED:
                        continue
                
                ready.append(node)
            
            if len(ready) >= max_count:
                break
        
        return ready
    
    def get_all_pending_urls(self) -> List[str]:
        """Get all pending URLs in breadth-first order."""
        urls = []
        for depth in range(self.limits.max_depth + 1):
            for url in self._by_depth.get(depth, set()):
                node = self._nodes[url]
                if node.status == PageStatus.QUEUED:
                    urls.append(url)
        return urls
    
    def get_in_progress_count(self) -> int:
        """Get count of pages currently being visited."""
        return len(self._by_status[PageStatus.VISITING])
    
    def get_parallelism_factor(self) -> int:
        """
        Calculate how many pages can be processed in parallel right now.
        
        Returns:
            Number of pages that are ready for parallel processing
        """
        return len(self.get_ready_for_parallel(max_count=10))


class ExplorationDAG:
    """
    DAG (Directed Acyclic Graph) wrapper for parallel site exploration.
    
    Provides a clean interface for managing parallel page exploration
    with proper dependency tracking. Pages can only be explored after
    their parent page has been visited.
    
    Usage:
        dag = ExplorationDAG(sitemap_graph)
        
        # Get pages ready for parallel processing
        ready = dag.get_ready_batch(max_size=3)
        
        # Mark pages as in progress
        for node in ready:
            dag.mark_in_progress(node.url)
        
        # After processing, mark complete
        dag.mark_complete(url, success=True)
    """
    
    def __init__(self, sitemap: SitemapGraph):
        """
        Initialize DAG from existing SitemapGraph.
        
        Args:
            sitemap: The SitemapGraph to wrap
        """
        self._sitemap = sitemap
        self._in_progress: Set[str] = set()  # URLs currently being processed
        self._completed: Set[str] = set()  # URLs successfully completed
        self._failed: Set[str] = set()  # URLs that failed
    
    @property
    def sitemap(self) -> SitemapGraph:
        """Access the underlying SitemapGraph."""
        return self._sitemap
    
    def get_ready_batch(self, max_size: int = 3) -> List[SitemapNode]:
        """
        Get a batch of pages ready for parallel processing.
        
        Respects dependencies: only returns pages whose parents
        have been completed.
        
        Args:
            max_size: Maximum batch size
            
        Returns:
            List of SitemapNodes ready for processing
        """
        ready = []
        
        # Get candidates from sitemap
        candidates = self._sitemap.get_ready_for_parallel(max_count=max_size * 2)
        
        for node in candidates:
            if len(ready) >= max_size:
                break
            
            # Skip if already in progress or completed
            if node.url in self._in_progress or node.url in self._completed:
                continue
            
            ready.append(node)
        
        return ready
    
    def mark_in_progress(self, url: str) -> bool:
        """
        Mark a URL as currently being processed.
        
        Args:
            url: The URL to mark
            
        Returns:
            True if successfully marked, False if already in progress
        """
        if url in self._in_progress:
            return False
        
        self._in_progress.add(url)
        self._sitemap.mark_visiting(url)
        return True
    
    def mark_complete(
        self, 
        url: str, 
        success: bool = True, 
        error: str = "",
        title: str = "",
        summary: str = "",
        section_count: int = 0,
    ) -> None:
        """
        Mark a URL as completed (success or failure).
        
        Args:
            url: The URL that was processed
            success: Whether processing succeeded
            error: Error message if failed
            title: Page title (for successful completion)
            summary: Page summary (for successful completion)
            section_count: Number of sections found (for successful completion)
        """
        self._in_progress.discard(url)
        
        if success:
            self._completed.add(url)
            # Update sitemap status to VISITED so child pages can be explored
            self._sitemap.mark_visited(
                url,
                title=title,
                summary=summary,
                section_count=section_count,
                page_map_stored=True
            )
        else:
            self._failed.add(url)
            self._sitemap.mark_failed(url, error)
    
    def is_complete(self) -> bool:
        """
        Check if all exploration is complete.
        
        Returns:
            True if no more pages to process
        """
        return (
            self._sitemap.is_exploration_complete() and
            len(self._in_progress) == 0
        )
    
    def get_progress(self) -> Dict[str, int]:
        """
        Get current progress statistics.
        
        Returns:
            Dict with progress stats
        """
        return {
            "completed": len(self._completed),
            "in_progress": len(self._in_progress),
            "failed": len(self._failed),
            "pending": self._sitemap.pending_count,
            "total": self._sitemap.total_count,
            "parallelism": self._sitemap.get_parallelism_factor(),
        }
    
    def get_pending_count(self) -> int:
        """Get number of pages still waiting to be processed."""
        return self._sitemap.pending_count
    
    def get_in_progress_urls(self) -> List[str]:
        """Get list of URLs currently being processed."""
        return list(self._in_progress)
    
    def reset(self) -> None:
        """Reset the DAG state (but not the underlying sitemap)."""
        self._in_progress.clear()
        self._completed.clear()
        self._failed.clear()
