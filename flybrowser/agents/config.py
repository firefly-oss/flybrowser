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
Centralized Configuration System for FlyBrowser Agents.

This module provides a comprehensive configuration system for all agent components.
Supports YAML/JSON loading, dynamic token calculation, and environment variable overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

from flybrowser.agents.types import ReasoningStrategy, SafetyLevel, OperationMode


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a given text.
    Uses approximation: 1 token ≈ 4 characters for English text.
    """
    return max(int(len(text) / 4), 1)


def calculate_max_tokens_for_response(
    system_prompt_tokens: int,
    user_prompt_tokens: int,
    context_tokens: int = 0,
    safety_margin: float = 1.3
) -> int:
    """
    Dynamically calculate appropriate max_tokens for a response.
    
    Args:
        system_prompt_tokens: Tokens in system prompt
        user_prompt_tokens: Tokens in user prompt  
        context_tokens: Additional context tokens (memory, tools, etc.)
        safety_margin: Multiplier for safety (default 1.3 = 30% buffer)
    
    Returns:
        Recommended max_tokens for response
    """
    # Input tokens
    input_tokens = system_prompt_tokens + user_prompt_tokens + context_tokens
    
    # Response should be proportional to input complexity
    # Simple tasks: ~0.5x input, Complex tasks: ~2x input
    base_response = int(input_tokens * 0.75)
    
    # Apply safety margin and clamp to reasonable bounds
    recommended = int(base_response * safety_margin)
    return max(2048, min(recommended, 16384))


def estimate_vision_tokens(image_size_bytes: int) -> int:
    """
    Estimate tokens consumed by an image in vision models.
    
    Vision models process images as token sequences. Token count depends on:
    - Image resolution (higher resolution = more tokens)
    - Detail level (high/low detail setting)
    - Model architecture (varies by provider)
    
    OpenAI GPT-4V pricing:
    - Low detail: 85 tokens (fixed)
    - High detail: 85 base + 170 tokens per 512x512 tile
    - Max ~1700 tokens for large images
    
    Args:
        image_size_bytes: Size of image in bytes
    
    Returns:
        Estimated token count for image
    """
    # Rough estimation: larger images = more tokens
    # Typical screenshot: 500KB-2MB
    # Using conservative estimates:
    # - Small (<500KB): ~400 tokens
    # - Medium (500KB-1.5MB): ~800 tokens  
    # - Large (>1.5MB): ~1200 tokens
    
    size_kb = image_size_bytes / 1024
    
    if size_kb < 500:
        return 400
    elif size_kb < 1500:
        return 800
    else:
        return 1200


def calculate_max_tokens_for_vision_response(
    system_prompt: str,
    user_prompt: str,
    image_size_bytes: int,
    context_tokens: int = 0,
    safety_margin: float = 1.5  # Higher margin for vision (50%)
) -> int:
    """
    Dynamically calculate max_tokens for vision-enabled responses.
    
    Vision responses need more tokens because:
    - Model describes what it sees in image (300-800 tokens)
    - Thought process (100-200 tokens)
    - Action JSON (100-300 tokens)
    - Total: 500-1300+ tokens typical
    
    Args:
        system_prompt: System prompt text
        user_prompt: User prompt text
        image_size_bytes: Size of image in bytes
        context_tokens: Additional context tokens
        safety_margin: Multiplier for safety (default 1.5 = 50% buffer for vision)
    
    Returns:
        Recommended max_tokens for vision response
    """
    # Estimate text tokens
    system_tokens = estimate_tokens(system_prompt)
    user_tokens = estimate_tokens(user_prompt)
    
    # Estimate image tokens
    image_tokens = estimate_vision_tokens(image_size_bytes)
    
    # Total input
    input_tokens = system_tokens + user_tokens + image_tokens + context_tokens
    
    # Vision responses typically need:
    # - Base response proportional to complexity: 0.75x input
    # - Extra for image description: +500 tokens minimum
    base_response = int(input_tokens * 0.75) + 500
    
    # Apply safety margin
    recommended = int(base_response * safety_margin)
    
    # Clamp to reasonable bounds (higher ceiling for vision)
    return max(2048, min(recommended, 16384))


# Operation mode is now determined by SDK method calls, not keyword detection
# See sdk.py for mode assignment:
#   browser.navigate() → OperationMode.NAVIGATE
#   browser.act()      → OperationMode.EXECUTE  
#   browser.extract()  → OperationMode.SCRAPE
#   browser.auto()     → OperationMode.AUTO


@dataclass
class LLMConfig:
    """LLM-specific configuration for various operations."""
    
    # Core reasoning (ReAct loop)
    # Temperature 0.4: Balance between creativity and consistency for action selection
    # Generous token limits to avoid truncation in complex scenarios
    reasoning_temperature: float = 0.4
    reasoning_max_tokens: int = 4096  # Increased from 1536 for complex reasoning chains
    reasoning_vision_max_tokens: int = 6144  # Increased from 3072 for detailed visual analysis
    
    # Planning (task decomposition)
    # Temperature 0.1: Very deterministic for consistent plan structure
    # Higher token limit for detailed multi-phase plans
    planning_temperature: float = 0.1
    planning_max_tokens: int = 4096  # Increased from 2048 for complex plans
    plan_adaptation_temperature: float = 0.2  # Slightly higher for creative recovery
    
    # Tool operations
    # Temperature 0.05: Maximum consistency for obstacle JSON structure
    obstacle_detection_temperature: float = 0.05
    obstacle_detection_max_tokens: int = 2048  # Increased from 1024 for multiple obstacles
    
    # Temperature 0.1: Objective ranking requires deterministic scoring
    search_ranking_temperature: float = 0.1
    search_ranking_max_tokens: int = 3072  # Increased from 1536 for detailed ranking
    
    # Page analysis (HTML understanding)
    # Temperature 0.15: Low for element extraction accuracy
    page_analysis_temperature: float = 0.15
    page_analysis_max_tokens: int = 4096  # Increased from 2048 for large element lists
    
    # Thinking/reflection
    # Temperature 0.5: Creative problem solving while maintaining coherence
    thinking_temperature: float = 0.5
    thinking_max_tokens: int = 2048  # Increased from 1024 for deeper reflections
    
    # Dynamic token calculation - ENABLED BY DEFAULT
    # Automatically calculates optimal max_tokens based on prompt complexity and image size
    # Prevents truncation while optimizing cost (smaller prompts = fewer tokens)
    enable_dynamic_tokens: bool = True  # Recommended: Adapts to actual needs
    token_safety_margin: float = 1.3  # 30% buffer (1.5 for vision)
    
    # NOTE: Structured output is ALWAYS enabled and mandatory (not configurable).
    # Uses JSON mode for deterministic responses across all LLM providers.
    # This eliminates parsing errors and ensures consistent action/thought extraction.
    
    # Schema repair - when LLM returns malformed JSON, attempt to repair it
    # The repair mechanism asks the LLM to fix its output using the original context
    max_repair_attempts: int = 2  # Maximum repair attempts before failing
    repair_temperature: float = 0.1  # Low temperature for consistent repairs
    
    # Token budget limits for conversation management
    # These help prevent exceeding API rate limits (e.g., OpenAI 30K TPM orgs)
    max_request_tokens: int = 25000  # Hard limit per single LLM request (input tokens)
    max_context_tokens: int = 20000  # Soft limit for context/prompt content
    context_cleanup_threshold: float = 0.15  # Cleanup when free budget drops below this %


@dataclass
class ObstacleDetectorConfig:
    """Configuration for obstacle detection system."""
    
    # Detection behavior
    enabled: bool = True
    aggressive_mode: bool = False  # Sequential tries, not parallel
    max_strategies_per_obstacle: int = 3  # Reduced: Try top 3 most likely strategies
    max_obstacles_to_handle: int = 3  # Handle up to 3 simultaneous obstacles
    
    # LLM settings (uses config.llm.obstacle_detection_* values)
    # These are kept for backward compatibility with direct ObstacleDetector instantiation
    temperature: float = 0.05  # Near-zero for deterministic JSON
    max_tokens: int = 1024  # Compact obstacle response
    
    # Timeouts (milliseconds) - Optimized for modern SPAs
    element_visibility_timeout_ms: int = 3000  # Longer for lazy-loaded modals
    element_click_timeout_ms: int = 2000  # Standard click timeout
    page_settle_delay_ms: int = 500  # Wait for modal animations
    
    # Confidence thresholds
    min_confidence_threshold: float = 0.75  # High confidence to avoid false positives
    
    # HTML extraction - Balanced for token efficiency
    max_html_size_chars: int = 10000  # ~2500 tokens max
    max_visible_text_chars: int = 1500  # Context for verification
    max_overlay_elements: int = 10  # Capture more for complex pages


@dataclass
class PageAnalysisConfig:
    """Configuration for page analysis system."""
    
    # Feature toggles
    enable_llm_html_analysis: bool = True  # Fallback for heuristic failures
    enable_vlm_visual_analysis: bool = False  # Expensive: opt-in only
    prefer_cached_results: bool = True  # Essential for performance
    log_llm_prompts: bool = False  # Production default
    log_analysis_details: bool = False  # Verbose logging off
    
    # Heuristic thresholds - When to trigger LLM fallback
    min_elements_for_heuristic_success: int = 3  # Lower threshold: try LLM earlier
    min_confidence_for_llm_success: float = 0.70  # Acceptable LLM confidence
    
    # LLM settings (uses config.llm.page_analysis_* values)
    llm_model_name: Optional[str] = None  # Use default model
    llm_temperature: float = 0.15  # Very low for element accuracy
    llm_max_tokens: int = 2048  # ~20-30 elements with metadata
    llm_timeout_seconds: float = 15.0  # Generous for complex pages
    
    # VLM settings - Vision adds significant context
    vlm_model_name: Optional[str] = None
    vlm_temperature: float = 0.15  # Match LLM temperature
    vlm_max_tokens: int = 1536  # VLM sees layout, needs less text
    vlm_timeout_seconds: float = 20.0  # Image processing overhead
    
    # Cost controls - Reasonable limits
    max_cost_per_page_usd: float = 0.02  # ~100 pages per $2
    
    # HTML extraction - Optimized for token limits
    max_html_chars: int = 16000  # ~4000 tokens (header+nav focus)
    max_main_content_chars: int = 1000  # Sample of main content
    
    # Cache settings - Balance freshness vs performance
    cache_ttl_seconds: int = 300  # 5 minutes (pages change frequently)
    max_cache_entries: int = 50  # Reasonable memory footprint


@dataclass
class SearchProviderConfig:
    """Configuration for the search provider abstraction layer."""
    
    # Provider selection
    default_provider: str = "serper"  # "serper", "google", "bing", or "auto"
    fallback_providers: List[str] = field(default_factory=lambda: ["google", "bing"])
    prefer_cost_effective: bool = True  # Prefer cheaper providers
    
    # Caching
    cache_ttl_seconds: int = 300  # 5 minutes
    max_cache_entries: int = 100
    enable_caching: bool = True
    
    # Ranking
    enable_ranking: bool = True
    ranking_weights: Dict[str, float] = field(default_factory=lambda: {
        "bm25": 0.35,  # Keyword relevance
        "freshness": 0.20,  # Recency
        "domain_authority": 0.15,  # Source quality
        "position": 0.30,  # Original search engine ranking
    })
    
    # Search options defaults
    default_max_results: int = 10
    default_safe_search: str = "moderate"  # "off", "moderate", "strict"
    
    # Rate limiting
    rate_limit_rpm: int = 100  # Requests per minute
    
    # Failover
    max_retries: int = 2  # Max fallback provider attempts
    retry_on_rate_limit: bool = True


@dataclass
class SearchToolConfig:
    """Configuration for search tools (human-like and API)."""
    
    # Human-like search behavior - Balanced for speed vs detection
    human_search_enabled: bool = True
    typing_speed_wpm_min: int = 120  # Fast but human-like (avg human: 40 WPM)
    typing_speed_wpm_max: int = 180  # Upper bound for speed
    think_delay_min_ms: int = 200  # Brief pause between actions
    think_delay_max_ms: int = 1000  # Max delay for realism
    mouse_movement_steps: int = 10  # Minimal smooth movement
    
    # Search ranking (uses config.llm.search_ranking_* values)
    ranking_temperature: float = 0.1  # Objective scoring
    ranking_max_tokens: int = 1536  # Ranked list + brief analysis
    max_results_to_analyze: int = 10  # Top 10 sufficient for most queries
    default_top_n_results: int = 3  # Return top 3 by default
    
    # Obstacle handling - Critical for search engines
    handle_obstacles: bool = True  # Cookie banners are common
    obstacle_check_after_navigation: bool = True  # Verify page loaded


@dataclass
class ElementInteractionConfig:
    """Configuration for element detection and interaction."""
    
    # Timeouts (milliseconds) - Balanced for reliability
    default_timeout_ms: int = 30000  # 30s: Standard Playwright default
    navigation_timeout_ms: int = 60000  # 60s: Full page loads
    element_wait_timeout_ms: int = 10000  # 10s: Dynamic content
    
    # Retry behavior - Resilience for flaky elements
    max_retry_attempts: int = 2  # Try twice: initial + 1 retry
    retry_delay_ms: int = 1000  # 1s: Allow page to stabilize
    
    # Human-like behavior - Minimal delays for speed
    enable_human_like_delays: bool = False  # Speed over stealth
    typing_delay_ms_min: int = 10  # Minimal per-character delay
    typing_delay_ms_max: int = 30  # Fast typing
    click_delay_ms: int = 100  # Brief delay after click
    scroll_smooth_duration_ms: int = 200  # Quick scroll


@dataclass
class MemoryConfig:
    """Configuration for agent memory system.
    
    IMPORTANT: These limits are designed for browser automation agents that need to:
    - Store full page content extractions (can be 10K+ tokens)
    - Track multiple search results with URLs
    - Maintain navigation state with many links
    - Handle multi-step complex tasks with rich context
    
    The defaults are set high to support comprehensive web automation.
    Adjust down if you're hitting rate limits or cost concerns.
    """
    
    # Memory limits - High capacity for complex automation
    max_entries: int = 200  # Total memory cap (support long sessions)
    max_short_term_entries: int = 100  # Recent context (many steps in automation)
    max_working_memory_entries: int = 50  # Active task focus (multiple extractions)
    
    # Context window budget for memory formatting
    # For 128K context models like GPT-4.1, we can use more context
    # This controls how much memory context is included in prompts
    context_window_budget: int = 64000  # Large budget for rich context (half of 128K)
    
    # Extraction data limits - Generous for full page content
    # Browser automation often needs complete page data
    max_extraction_budget_percent: float = 0.40  # 40% of budget for extractions
    max_extraction_tokens: int = 16000  # Higher cap for full page extractions
    max_single_extraction_chars: int = 32000  # ~8K tokens per extraction
    
    # Intelligent compression settings - Use LLM to compress large content
    # This preserves important information instead of truncating
    enable_intelligent_compression: bool = True  # Use LLM to summarize large extractions
    compression_threshold_tokens: int = 4000  # Compress extractions larger than this
    history_compression_threshold: int = 15  # Compress when history exceeds N messages
    max_compressed_summary_tokens: int = 2000  # Max tokens for compressed summary (handles complex schemas)
    
    # Retrieval - Balance relevance vs context size
    relevance_threshold: float = 0.6  # Higher bar for inclusion
    max_results_per_query: int = 10  # Top 10 most relevant
    
    # Expiration (seconds) - Align with typical session lengths
    short_term_ttl_seconds: int = 3600  # 1 hour (active session)
    working_memory_ttl_seconds: int = 7200  # 2 hours (extended task)
    episodic_ttl_seconds: int = 86400  # 24 hours (single day)


@dataclass
class ParallelExplorationConfig:
    """Configuration for parallel/pipelined site exploration."""
    
    # Feature toggles
    enable_parallel: bool = True  # Enable parallel page exploration
    enable_pipeline_mode: bool = True  # Pipeline: capture screenshots while analyzing
    
    # Concurrency limits
    max_parallel_pages: int = 3  # Max concurrent page analyses (LLM calls)
    max_parallel_screenshots: int = 5  # Max pages to pre-capture screenshots for
    max_pending_analysis: int = 4  # Max pages queued for LLM analysis
    
    # Batching (experimental - analyze multiple pages in one LLM call)
    enable_batch_analysis: bool = False  # Batch multiple pages into single LLM call
    max_pages_per_batch: int = 3  # Max pages to analyze in one batch
    
    # Timeouts and safety
    parallel_timeout_seconds: float = 180.0  # Total timeout for parallel exploration
    per_page_timeout_seconds: float = 60.0  # Timeout per individual page
    
    # Rate limiting (to avoid LLM provider throttling)
    min_delay_between_llm_calls_ms: int = 100  # Minimum delay between LLM requests
    max_llm_requests_per_minute: int = 30  # Rate limit for LLM calls
    batch_delay_ms: int = 200  # Delay between batches of pages
    
    # Error handling
    continue_on_page_error: bool = True  # Continue exploring if one page fails
    max_consecutive_failures: int = 3  # Stop if N pages fail in a row


@dataclass
class PageExplorationConfig:
    """Configuration for systematic page exploration and understanding."""
    
    # Feature toggle
    enabled: bool = True  # Enable automatic page exploration
    
    # Sitemap exploration limits - Control multi-page exploration depth
    max_exploration_depth: int = 2  # 0=homepage, 1=main nav pages, 2=subpages
    max_level1_pages: int = 10  # Max main navigation pages to visit
    max_level2_pages: int = 10  # Max subpages to visit (total across all Level 1)
    max_total_pages: int = 20  # Hard limit on total pages explored
    
    # Scroll behavior - Optimized for comprehensive coverage
    scroll_step_px: int = 800  # Pixels per scroll step (typical viewport height)
    scroll_delay_ms: int = 500  # Wait after scroll for content to load
    max_scroll_steps: int = 20  # Maximum scroll steps per page (16,000px total)
    overlap_px: int = 100  # Overlap between screenshots for context continuity
    
    # Screenshot capture
    capture_full_page: bool = False  # Capture viewport only (faster, less memory)
    screenshot_quality: int = 80  # JPEG quality (1-100, 80 = good balance)
    max_screenshots_per_page: int = 10  # Limit to control cost/memory
    
    # Analysis behavior
    enable_multi_screenshot_analysis: bool = True  # Send all screenshots to LLM at once
    enable_incremental_analysis: bool = False  # Analyze each screenshot separately
    min_screenshots_for_analysis: int = 2  # Minimum screenshots before triggering analysis
    
    # LLM settings for page analysis
    analysis_temperature: float = 0.2  # Low temperature for structured extraction
    analysis_max_tokens: int = 4096  # Large: multiple screenshots + comprehensive understanding
    
    # Memory integration
    store_page_maps: bool = True  # Store PageMaps in agent memory
    page_map_ttl_seconds: int = 3600  # 1 hour cache for page structures
    
    # Trigger conditions - When to auto-explore
    auto_explore_on_navigation: bool = False  # Don't explore every navigation (opt-in)
    auto_explore_keywords: list = field(
        default_factory=lambda: [
            "explore", "navigate whole", "entire site", "all sections",
            "complete overview", "full page", "scroll through"
        ]
    )  # Keywords that trigger automatic exploration
    
    # Performance limits
    max_page_height_px: int = 50000  # Don't explore pages taller than 50k pixels
    timeout_seconds: float = 60.0  # Maximum time for full page exploration


@dataclass
class SafetyConfig:
    """Configuration for safety and circuit breaker mechanisms."""
    
    # Circuit breaker limits - Generous but bounded
    max_iterations: int = 50  # Typical: 10-30 steps, Max: 50 for complex flows
    max_time_seconds: float = 300.0  # 5 minutes: Reasonable timeout
    max_llm_calls: int = 100  # ~2 calls per iteration average
    max_consecutive_failures: int = 3  # Stop after 3 failed attempts
    
    # Stagnation detection - Prevent infinite loops
    stagnation_window: int = 5  # Check last 5 steps for progress
    enable_stagnation_detection: bool = True  # Critical safety feature
    
    # Loop detection - Detect repeated actions/states to break loops
    enable_loop_detection: bool = True  # Detect and break action loops
    max_repeated_actions: int = 3  # Break if same action repeated N times consecutively
    max_same_tool_calls: int = 5  # Break if same tool called N times in a row
    action_history_size: int = 10  # Number of recent actions to track for patterns
    state_hash_window: int = 5  # Compare state hashes over this window
    
    # Parse failure limits - Prevent LLM output loops
    max_consecutive_parse_failures: int = 5  # Stop after N parse failures in a row
    
    # Approval settings - Human-in-the-loop for risky actions
    require_approval_for_dangerous: bool = True  # Always require approval
    approval_timeout_seconds: float = 60.0  # 1 minute for response
    require_approval_for: list = field(
        default_factory=lambda: [SafetyLevel.SENSITIVE, SafetyLevel.DANGEROUS]
    )


@dataclass
class AgentConfig:
    """
    Comprehensive configuration for FlyBrowser ReAct agents.
    
    Aggregates all component-specific configurations and supports
    YAML/JSON loading and environment variable overrides.
    """
    
    # Sub-configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    obstacle_detector: ObstacleDetectorConfig = field(default_factory=ObstacleDetectorConfig)
    page_analysis: PageAnalysisConfig = field(default_factory=PageAnalysisConfig)
    page_exploration: PageExplorationConfig = field(default_factory=PageExplorationConfig)
    parallel_exploration: ParallelExplorationConfig = field(default_factory=ParallelExplorationConfig)
    search_tools: SearchToolConfig = field(default_factory=SearchToolConfig)
    search_providers: SearchProviderConfig = field(default_factory=SearchProviderConfig)
    element_interaction: ElementInteractionConfig = field(default_factory=ElementInteractionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    
    # Top-level execution settings - Align with safety config
    max_iterations: int = 50  # Default iteration limit per task
    timeout_seconds: float = 300.0  # 5 minute total timeout
    step_timeout_seconds: float = 60.0  # 1 minute per step
    
    # Reasoning settings - Optimize for browser automation
    reasoning_strategy: ReasoningStrategy = ReasoningStrategy.REACT_STANDARD
    enable_self_reflection: bool = True  # Learn from failures
    min_confidence_threshold: float = 0.5  # Balanced: not too cautious
    tot_failure_history_size: int = 3  # Last 3 failures for context
    enable_fast_path_optimization: bool = True  # Skip LLM for simple tasks
    
    # Logging and observability
    log_thoughts: bool = True
    log_actions: bool = True
    trace_execution: bool = False
    
    # Flat property aliases for commonly accessed hierarchical values
    # These provide backward compatibility and convenience for code that expects flat access
    @property
    def temperature(self) -> float:
        """Alias for llm.reasoning_temperature."""
        return self.llm.reasoning_temperature
    
    @property
    def max_tokens(self) -> int:
        """Alias for llm.reasoning_max_tokens."""
        return self.llm.reasoning_max_tokens
    
    @property
    def planning_temperature(self) -> float:
        """Alias for llm.planning_temperature."""
        return self.llm.planning_temperature
    
    @property
    def planning_max_tokens(self) -> int:
        """Alias for llm.planning_max_tokens."""
        return self.llm.planning_max_tokens
    
    @property
    def plan_adaptation_temperature(self) -> float:
        """Alias for llm.plan_adaptation_temperature."""
        return self.llm.plan_adaptation_temperature
    
    @property
    def max_consecutive_failures(self) -> int:
        """Alias for safety.max_consecutive_failures."""
        return self.safety.max_consecutive_failures
    
    @property
    def require_approval_for_dangerous(self) -> bool:
        """Alias for safety.require_approval_for_dangerous."""
        return self.safety.require_approval_for_dangerous
    
    @property
    def stagnation_window(self) -> int:
        """Alias for safety.stagnation_window."""
        return self.safety.stagnation_window
    
    @property
    def max_llm_calls(self) -> int:
        """Alias for safety.max_llm_calls."""
        return self.safety.max_llm_calls
    
    @property
    def max_time_seconds(self) -> float:
        """Alias for safety.max_time_seconds."""
        return self.safety.max_time_seconds
    
    @property
    def aggressive_mode(self) -> bool:
        """Alias for obstacle_detector.aggressive_mode."""
        return self.obstacle_detector.aggressive_mode
    
    @property
    def max_strategies_per_obstacle(self) -> int:
        """Alias for obstacle_detector.max_strategies_per_obstacle."""
        return self.obstacle_detector.max_strategies_per_obstacle
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create config from dictionary."""
        config = cls()
        
        # Top-level fields
        for key in ["max_iterations", "timeout_seconds", "step_timeout_seconds",
                    "min_confidence_threshold", "tot_failure_history_size",
                    "enable_self_reflection", "enable_fast_path_optimization",
                    "log_thoughts", "log_actions", "trace_execution"]:
            if key in data:
                setattr(config, key, data[key])
        
        if "reasoning_strategy" in data:
            config.reasoning_strategy = ReasoningStrategy(data["reasoning_strategy"])
        
        # Sub-configs
        if "llm" in data:
            config.llm = LLMConfig(**data["llm"])
        if "obstacle_detector" in data:
            config.obstacle_detector = ObstacleDetectorConfig(**data["obstacle_detector"])
        if "page_analysis" in data:
            config.page_analysis = PageAnalysisConfig(**data["page_analysis"])
        if "page_exploration" in data:
            config.page_exploration = PageExplorationConfig(**data["page_exploration"])
        if "search_tools" in data:
            config.search_tools = SearchToolConfig(**data["search_tools"])
        if "search_providers" in data:
            config.search_providers = SearchProviderConfig(**data["search_providers"])
        if "element_interaction" in data:
            config.element_interaction = ElementInteractionConfig(**data["element_interaction"])
        if "memory" in data:
            config.memory = MemoryConfig(**data["memory"])
        if "safety" in data:
            config.safety = SafetyConfig(**data["safety"])
        
        return config
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "AgentConfig":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            AgentConfig instance
            
        Example YAML:
            max_iterations: 100
            llm:
              reasoning_temperature: 0.8
              planning_temperature: 0.3
            obstacle_detector:
              enabled: true
              aggressive_mode: false
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML support. "
                "Install it with: pip install pyyaml"
            )
        
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data or {})
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "AgentConfig":
        """Load configuration from JSON file."""
        import json
        
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def save_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required. Install with: pip install pyyaml")
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def save_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        import json
        
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def apply_env_overrides(self) -> None:
        """
        Apply environment variable overrides.
        
        Environment variables follow pattern: FLYBROWSER_<SECTION>_<KEY>
        Examples:
            FLYBROWSER_LLM_REASONING_TEMPERATURE=0.9
            FLYBROWSER_SAFETY_MAX_ITERATIONS=100
            FLYBROWSER_OBSTACLE_DETECTOR_ENABLED=false
        """
        prefix = "FLYBROWSER_"
        
        # Map environment variables to config paths
        for env_var, value in os.environ.items():
            if not env_var.startswith(prefix):
                continue
            
            # Parse env var: FLYBROWSER_LLM_REASONING_TEMPERATURE -> llm.reasoning_temperature
            parts = env_var[len(prefix):].lower().split('_')
            
            if len(parts) < 2:
                continue
            
            section = parts[0]
            key = '_'.join(parts[1:])
            
            # Apply to config
            if section == "llm" and hasattr(self.llm, key):
                setattr(self.llm, key, self._parse_env_value(value, getattr(self.llm, key)))
            elif section == "obstacle" and len(parts) > 2:
                # obstacle_detector -> obstacle + detector_<key>
                full_key = '_'.join(parts[1:])
                if hasattr(self.obstacle_detector, full_key):
                    setattr(self.obstacle_detector, full_key, self._parse_env_value(value, getattr(self.obstacle_detector, full_key)))
            elif section == "page" and len(parts) > 2:
                # page_analysis -> page + analysis_<key>
                full_key = '_'.join(parts[1:])
                if hasattr(self.page_analysis, full_key):
                    setattr(self.page_analysis, full_key, self._parse_env_value(value, getattr(self.page_analysis, full_key)))
            elif section == "safety" and hasattr(self.safety, key):
                setattr(self.safety, key, self._parse_env_value(value, getattr(self.safety, key)))
            elif section == "memory" and hasattr(self.memory, key):
                setattr(self.memory, key, self._parse_env_value(value, getattr(self.memory, key)))
            elif hasattr(self, key):
                # Top-level config
                setattr(self, key, self._parse_env_value(value, getattr(self, key)))
    
    @staticmethod
    def _parse_env_value(value: str, current_value: Any) -> Any:
        """Parse environment variable value based on current type."""
        if isinstance(current_value, bool):
            return value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(current_value, int):
            return int(value)
        elif isinstance(current_value, float):
            return float(value)
        else:
            return value


# Singleton instance for global configuration
_global_config: Optional[AgentConfig] = None


def get_global_config() -> AgentConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = AgentConfig()
        _global_config.apply_env_overrides()
    return _global_config


def set_global_config(config: AgentConfig) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config


def reset_global_config() -> None:
    """Reset global configuration to defaults."""
    global _global_config
    _global_config = AgentConfig()
    _global_config.apply_env_overrides()
