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
Unified FlyBrowser SDK.

This module provides a single, unified FlyBrowser class that works transparently
in all deployment modes:

1. **Embedded Mode** (no endpoint): Runs browser locally in the same process
2. **Server Mode** (with endpoint): Connects to a FlyBrowser server (standalone or cluster)

The SDK automatically handles:
- Cluster discovery and failover (when connecting to a cluster)
- Session affinity and routing
- Connection pooling and retry logic
- All features work identically in both modes

Example - Embedded Mode (local browser):
    >>> async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
    ...     await browser.goto("https://example.com")
    ...     data = await browser.extract("Get the title")

Example - Server Mode (standalone or cluster):
    >>> async with FlyBrowser(endpoint="http://localhost:8000") as browser:
    ...     await browser.goto("https://example.com")
    ...     data = await browser.extract("Get the title")

The same code works regardless of whether the endpoint points to a standalone
server or a multi-node cluster. The SDK handles all complexity internally.
"""

from __future__ import annotations

import base64
from typing import Any, Callable, Dict, List, Optional, Union

from flybrowser.utils.logger import logger
from flybrowser.utils.execution_logger import configure_execution_logger, LogVerbosity
from flybrowser.agents.scope_validator import get_scope_validator


class FlyBrowser:
    """
    Unified FlyBrowser SDK for LLM-powered browser automation.

    This class provides a single, unified interface that works transparently in
    all deployment modes:

    1. **Embedded Mode** (no endpoint): Runs browser locally in the same process
    2. **Server Mode** (with endpoint): Connects to a FlyBrowser server

    The SDK automatically handles cluster discovery, failover, and session routing
    when connecting to a cluster. Developers write the same code regardless of
    deployment mode.

    Example - Embedded Mode:
        >>> async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
        ...     await browser.goto("https://example.com")
        ...     data = await browser.extract("Get the title")

    Example - Server Mode (standalone or cluster):
        >>> async with FlyBrowser(endpoint="http://localhost:8000") as browser:
        ...     await browser.goto("https://example.com")
        ...     data = await browser.extract("Get the title")

    The endpoint URL is the ONLY thing that changes between modes. All features
    (screenshots, recordings, extractions, PII masking) work identically.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        vision_enabled: Optional[bool] = None,
        model_config: Optional[Dict[str, Any]] = None,
        headless: bool = True,
        browser_type: str = "chromium",
        recording_enabled: bool = False,
        pii_masking_enabled: bool = True,
        timeout: float = 30.0,
        pretty_logs: bool = True,
        speed_preset: str = "balanced",
        log_verbosity: str = "normal",
        agent_config: Optional["AgentConfig"] = None,
        config_file: Optional[str] = None,
        search_provider: Optional[str] = None,
        search_api_key: Optional[str] = None,
        stealth_config: Optional["StealthConfig"] = None,
        observability_config: Optional["ObservabilityConfig"] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize FlyBrowser.

        Args:
            endpoint: Server endpoint URL. If provided, connects to a FlyBrowser
                server (standalone or cluster). If None, runs browser locally.
                Examples:
                - None: Embedded mode (local browser)
                - "http://localhost:8000": Standalone server
                - "http://cluster.example.com:8000": Cluster endpoint
            llm_provider: LLM provider name (openai, anthropic, ollama, gemini)
            llm_model: LLM model name. If not specified, uses provider default.
            api_key: API key for the LLM provider
            vision_enabled: Override auto-detected vision capability (optional)
                - None: Use auto-detected capabilities (default)
                - True: Force vision enabled (useful if discovery fails to detect it)
                - False: Force vision disabled (useful for testing text-only fallback)
            model_config: Manual model configuration (optional)
                Example: {"context_window": 128000, "max_output_tokens": 4096}
                Additional configuration to pass to the provider
            headless: Run browser in headless mode (default: True)
            browser_type: Browser type (chromium, firefox, webkit)
            recording_enabled: Enable session recording (default: False)
            pii_masking_enabled: Enable PII masking (default: True)
            timeout: Request timeout in seconds for server mode (default: 30.0)
            pretty_logs: Use human-readable colored logs instead of JSON (default: True)
            speed_preset: Performance preset - "fast", "balanced", or "thorough" (default: "balanced")
                - "fast": Optimized for speed, shorter timeouts, fewer retries
                - "balanced": Good balance of speed and reliability
                - "thorough": More thorough, longer timeouts for complex pages
            log_verbosity: Unified logging verbosity level (default: "normal")
                Controls execution logs, Python log level, and LLM logging:
                - "silent": Errors only, no LLM logging
                - "minimal": Errors + warnings, no LLM logging
                - "normal": Standard info logs, no LLM logging
                - "verbose": Detailed execution + basic LLM timing
                - "debug": Full technical details + detailed LLM prompts/responses
            agent_config: AgentConfig instance for ReAct framework configuration.
                If provided, this takes precedence over config_file.
                Example: agent_config=AgentConfig(max_iterations=100)
            config_file: Path to YAML or JSON config file for agent configuration.
                Example: config_file="config.yaml"
                If both agent_config and config_file are None, uses defaults.
            search_provider: Preferred search provider for API-based search.
                Options: "serper", "google", "bing", "auto" (default: auto)
                - "serper": Serper.dev API (fast, affordable, recommended)
                - "google": Google Custom Search API
                - "bing": Bing Web Search API
                - "auto": Automatically select best available provider
                Note: Requires corresponding API key environment variable:
                - SERPER_API_KEY for Serper.dev
                - GOOGLE_CUSTOM_SEARCH_API_KEY + GOOGLE_CUSTOM_SEARCH_CX for Google
                - BING_SEARCH_API_KEY for Bing
            search_api_key: API key for the search provider. If not provided,
                falls back to environment variables.
            stealth_config: Optional StealthConfig for advanced stealth capabilities.
                Enables fingerprint generation, managed CAPTCHA solving, and
                intelligent proxy network. Example:
                >>> from flybrowser.stealth import StealthConfig
                >>> stealth = StealthConfig(
                ...     fingerprint_enabled=True,
                ...     captcha_enabled=True,
                ...     captcha_provider="2captcha",
                ...     proxy_enabled=True,
                ... )
            observability_config: Optional ObservabilityConfig for session debugging.
                Enables command logging, source capture, and live view streaming.
                Example:
                >>> from flybrowser.observability import ObservabilityConfig
                >>> obs = ObservabilityConfig(
                ...     enable_command_logging=True,
                ...     enable_live_view=True,
                ...     live_view_port=8765,
                ... )
            **kwargs: Additional configuration options

        Example - Embedded Mode:
            >>> browser = FlyBrowser(
            ...     llm_provider="openai",
            ...     llm_model="gpt-4o",
            ...     api_key="sk-...",
            ... )

        Example - Anthropic Claude:
            >>> browser = FlyBrowser(
            ...     llm_provider="anthropic",
            ...     llm_model="claude-3-5-sonnet-20241022",
            ...     api_key="sk-ant-...",
            ... )

        Example - Local Ollama:
            >>> browser = FlyBrowser(
            ...     llm_provider="ollama",
            ...     llm_model="qwen3:8b",
            ... )
        """
        self._endpoint = endpoint
        self._mode = "server" if endpoint else "embedded"
        self._started = False

        # Configure performance settings based on speed preset
        from flybrowser.core.performance import (
            PerformanceConfig,
            SpeedPreset,
            set_performance_config,
        )
        
        preset_map = {
            "fast": SpeedPreset.FAST,
            "balanced": SpeedPreset.BALANCED,
            "thorough": SpeedPreset.THOROUGH,
        }
        preset = preset_map.get(speed_preset.lower(), SpeedPreset.BALANCED)
        set_performance_config(PerformanceConfig.from_preset(preset))

        # Store configuration for both modes
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        self._api_key = api_key
        self._vision_enabled = vision_enabled
        self._model_config = model_config
        self._headless = headless
        self._browser_type = browser_type
        self._recording_enabled = recording_enabled
        self._pii_masking_enabled = pii_masking_enabled
        self._timeout = timeout
        self._speed_preset = speed_preset
        self._kwargs = kwargs
        
        # Store search configuration
        self._search_provider = search_provider
        self._search_api_key = search_api_key
        
        # Load agent configuration
        from flybrowser.agents.config import AgentConfig, SearchProviderConfig
        
        if agent_config is not None:
            self._agent_config = agent_config
            logger.info("Using provided AgentConfig")
        elif config_file is not None:
            # Load from YAML or JSON file
            from pathlib import Path
            config_path = Path(config_file)
            if config_path.suffix in ('.yaml', '.yml'):
                self._agent_config = AgentConfig.from_yaml(config_file)
            elif config_path.suffix == '.json':
                self._agent_config = AgentConfig.from_json(config_file)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}. Use .yaml, .yml, or .json")
            logger.info(f"Loaded agent configuration from {config_file}")
        else:
            # Use defaults with environment variable overrides
            self._agent_config = AgentConfig()
            self._agent_config.apply_env_overrides()
            logger.info("Using default AgentConfig with environment overrides")
        
        # Apply search provider configuration
        if search_provider:
            self._agent_config.search_providers.default_provider = search_provider
            logger.info(f"Search provider set to: {search_provider}")
        
        # Set up search API key in environment if provided
        if search_api_key:
            import os
            # Determine which env var to set based on provider
            provider = (search_provider or "serper").lower()
            if provider == "serper":
                os.environ["SERPER_API_KEY"] = search_api_key
            elif provider == "google":
                os.environ["GOOGLE_CUSTOM_SEARCH_API_KEY"] = search_api_key
            elif provider == "bing":
                os.environ["BING_SEARCH_API_KEY"] = search_api_key
            logger.info(f"Search API key configured for {provider}")
        
        # Store stealth configuration
        self._stealth_config = stealth_config
        if stealth_config:
            logger.info(f"Stealth config: fingerprint={stealth_config.fingerprint_enabled}, "
                       f"captcha={stealth_config.captcha_enabled}, proxy={stealth_config.proxy_enabled}")
        
        # Store observability configuration
        self._observability_config = observability_config
        if observability_config:
            logger.info(f"Observability config: logging={observability_config.enable_command_logging}, "
                       f"capture={observability_config.enable_source_capture}, "
                       f"live_view={observability_config.enable_live_view}")

        # Unified logging configuration based on log_verbosity
        # Maps verbosity to: (execution_verbosity, python_log_level, llm_logging_level)
        verbosity_config = {
            "silent": (LogVerbosity.SILENT, "ERROR", 0),
            "minimal": (LogVerbosity.MINIMAL, "WARNING", 0),
            "normal": (LogVerbosity.NORMAL, "INFO", 0),
            "verbose": (LogVerbosity.VERBOSE, "INFO", 1),
            "debug": (LogVerbosity.DEBUG, "DEBUG", 2),
        }
        exec_verbosity, log_level, llm_logging_level = verbosity_config.get(
            log_verbosity.lower(), (LogVerbosity.NORMAL, "INFO", 0)
        )
        
        # Store derived llm_logging for use in _start_embedded_mode
        self._llm_logging = llm_logging_level
        
        # Configure Python logger (format + level)
        from flybrowser.utils.logger import configure_logging, LogFormat
        log_format = LogFormat.HUMAN if pretty_logs else LogFormat.JSON
        configure_logging(level=log_level, log_format=log_format)
        
        # Configure execution logger verbosity
        configure_execution_logger(verbosity=exec_verbosity)

        # Mode-specific components (initialized in start())
        self._client = None  # HTTP client for server mode
        self._session_id = None  # Session ID for server mode

        # Embedded mode components
        self.llm = None
        self.browser_manager = None
        self.page_controller = None
        self.element_detector = None
        self.pii_handler = None
        self._screenshot_capture = None
        self._recording_manager = None
        self._streaming_manager = None
        self._local_stream_server = None
        self._local_stream_port = None
        self._active_stream_id = None
        
        # Stealth components
        self._fingerprint = None
        self._captcha_solver = None
        self._proxy_network = None
        
        # Observability components
        self._observability_manager = None
        self._command_logger = None
        self._source_capture = None
        self._live_view_server = None
        
        # ReAct framework (native implementation)
        self.react_agent = None

        logger.info(f"Initializing FlyBrowser in {self._mode} mode")

    async def start(self) -> None:
        """
        Start the browser session.

        In embedded mode: Launches a local Playwright browser.
        In server mode: Creates a session on the FlyBrowser server.

        This method is called automatically when using the async context manager.

        Raises:
            RuntimeError: If already started or connection fails
        """
        if self._started:
            logger.warning("FlyBrowser already started")
            return

        if self._mode == "server":
            await self._start_server_mode()
        else:
            await self._start_embedded_mode()

        self._started = True
        logger.info(f"FlyBrowser started in {self._mode} mode")

    async def _start_server_mode(self) -> None:
        """Start in server mode - connect to FlyBrowser server."""
        from flybrowser.client import FlyBrowserClient

        self._client = FlyBrowserClient(
            endpoint=self._endpoint,
            timeout=self._timeout,
        )
        await self._client.start()

        # Create a session on the server
        session_data = await self._client.create_session(
            llm_provider=self._llm_provider,
            llm_model=self._llm_model,
            api_key=self._api_key,
            headless=self._headless,
        )
        self._session_id = session_data.get("session_id")

        if not self._session_id:
            raise RuntimeError("Failed to create session on server")

        logger.info(f"Created server session: {self._session_id}")

    async def _start_embedded_mode(self) -> None:
        """Start in embedded mode - run browser locally."""
        # Import components only when needed (embedded mode)
        from flybrowser.core.browser import BrowserManager
        from flybrowser.core.element import ElementDetector
        from flybrowser.core.page import PageController
        from flybrowser.core.recording import (
            RecordingConfig,
            RecordingManager,
            ScreenshotCapture,
        )
        from flybrowser.llm.factory import LLMProviderFactory
        from flybrowser.security.pii_handler import PIIConfig, PIIHandler
        from flybrowser.core.performance import get_performance_config

        # Initialize LLM provider with simplified factory
        logger.info("[SDK] Initializing LLM provider")
        
        # Build kwargs for create()
        create_kwargs = {
            "vision_enabled": self._vision_enabled,
            "llm_logging": self._llm_logging,
        }
        
        # Add model_config if provided
        if self._model_config:
            create_kwargs.update(self._model_config)
        
        # Create provider (always use simple create, no smart selection)
        self.llm = LLMProviderFactory.create(
            provider=self._llm_provider,
            model=self._llm_model or "default",
            api_key=self._api_key,
            **create_kwargs
        )
        
        # Trigger provider initialization if available
        if hasattr(self.llm, 'initialize'):
            await self.llm.initialize()
        
        # Log LLM configuration with discovered capabilities
        model_info = self.llm.get_model_info()
        
        # Build selection info string
        selection_info = ""
        if not self._llm_model:
            selection_info = " (provider default)"
        
        logger.info(f"LLM Provider: {self._llm_provider}")
        logger.info(f"Model: {model_info.name}{selection_info}")
        logger.info(f"Context Window: {model_info.context_window:,} tokens")
        logger.info(f"Max Output Tokens: {model_info.max_output_tokens:,} tokens")
        
        # Log all capabilities detected for this model
        from flybrowser.llm.base import ModelCapability
        capability_names = [cap.value for cap in model_info.capabilities]
        logger.info(f"Model Capabilities: {', '.join(capability_names)}")
        
        # Check and log vision capability - this is CRITICAL for proper operation
        self._vision_enabled = self.llm.vision_enabled
        if self._vision_enabled:
            logger.info(f"Vision: [OK] ENABLED (model supports vision)")
        else:
            logger.warning(
                f"Vision: [WARN] DISABLED (model {model_info.name} does not support vision). "
                f"All vision features will use text-only fallback."
            )
        
        # Log cost information if available
        if (model_info.cost_per_1k_input_tokens is not None and 
            model_info.cost_per_1k_input_tokens > 0):
            logger.info(
                f"Cost: ${model_info.cost_per_1k_input_tokens:.4f}/1K input, "
                f"${model_info.cost_per_1k_output_tokens:.4f}/1K output"
            )
        else:
            logger.info("Cost: Free (local model)")
        
        # Log performance configuration
        perf_config = get_performance_config()
        logger.info(f"Performance Preset: {self._speed_preset}")
        logger.info(
            f"Timeouts: navigation={perf_config.navigation_timeout_ms}ms, "
            f"action={perf_config.action_timeout_ms}ms, "
            f"element={perf_config.element_timeout_ms}ms"
        )
        logger.info(f"Wait Strategy: {perf_config.wait_strategy.value}")
        logger.info(f"Max Retries: {perf_config.max_retries}")

        # Initialize stealth components if configured
        fingerprint_profile = None
        if self._stealth_config and self._stealth_config.fingerprint_enabled:
            try:
                from flybrowser.stealth.fingerprint import FingerprintGenerator
                generator = FingerprintGenerator()
                fingerprint_profile = generator.generate()
                self._fingerprint = fingerprint_profile
                logger.info(f"[STEALTH] Generated fingerprint: {fingerprint_profile.os_type.value}/{fingerprint_profile.browser_type.value}")
            except ImportError:
                logger.warning("[STEALTH] Fingerprint module not available")
        
        # Initialize CAPTCHA solver if configured
        if self._stealth_config and self._stealth_config.captcha_enabled:
            try:
                from flybrowser.stealth.captcha import CaptchaSolver, CaptchaProviderConfig
                config = CaptchaProviderConfig(
                    provider=self._stealth_config.captcha_provider or "2captcha",
                    api_key=self._stealth_config.captcha_api_key,
                )
                self._captcha_solver = CaptchaSolver(config)
                logger.info(f"[STEALTH] CAPTCHA solver initialized: {config.provider}")
            except ImportError:
                logger.warning("[STEALTH] CAPTCHA solver module not available")
        
        # Initialize proxy network if configured
        # Note: This is the advanced ProxyNetwork from stealth package, which provides:
        # - Intelligent target-aware proxy selection
        # - Fingerprint-proxy-geolocation consistency  
        # - Multiple residential proxy provider support
        # For simpler use cases, use proxy_rotator parameter instead.
        proxy_for_browser = None
        if self._stealth_config and self._stealth_config.proxy_enabled:
            try:
                from flybrowser.stealth.proxy import ProxyNetwork, ProxyNetworkConfig
                
                # Get proxy config from stealth config
                proxy_config = self._stealth_config.get_proxy_config()
                
                self._proxy_network = ProxyNetwork(
                    config=proxy_config,
                    fingerprint=fingerprint_profile,  # For consistency
                )
                
                # Get initial proxy for browser context
                initial_proxy = await self._proxy_network.get_proxy()
                if initial_proxy:
                    proxy_for_browser = initial_proxy.to_playwright_format()
                    logger.info(f"[STEALTH] Using proxy: {initial_proxy.host}:{initial_proxy.port} ({initial_proxy.country})")
                
                logger.info("[STEALTH] Proxy network initialized")
            except ImportError:
                logger.warning("[STEALTH] Proxy network module not available")
            except Exception as e:
                logger.warning(f"[STEALTH] Failed to initialize proxy network: {e}")
        
        # Initialize browser manager with stealth mode enabled
        # Pass proxy_for_browser if we got one from ProxyNetwork
        logger.info(f"Browser: {self._browser_type} (headless={self._headless}, stealth=True)")
        
        # Build browser manager kwargs
        browser_kwargs = {
            "headless": self._headless,
            "browser_type": self._browser_type,
            "stealth": True,  # Enable anti-detection by default
            "fingerprint": fingerprint_profile,  # Use generated fingerprint if available
        }
        
        # Add proxy if we got one from ProxyNetwork
        if proxy_for_browser:
            browser_kwargs["proxy"] = proxy_for_browser
        
        self.browser_manager = BrowserManager(**browser_kwargs)
        await self.browser_manager.start()
        
        # Initialize observability components if configured
        if self._observability_config:
            try:
                from flybrowser.observability import ObservabilityManager
                self._observability_manager = ObservabilityManager(self._observability_config)
                
                # Attach to page for live view if enabled
                if self._observability_config.enable_live_view:
                    await self._observability_manager.attach(self.browser_manager.page)
                    live_url = self._observability_manager.get_live_view_url()
                    if live_url:
                        logger.info(f"[OBSERVABILITY] Live View available at: {live_url}")
                
                # Store component references
                self._command_logger = self._observability_manager.command_logger
                self._source_capture = self._observability_manager.source_capture
                self._live_view_server = self._observability_manager.live_view
                
                logger.info("[OBSERVABILITY] Observability layer initialized")
            except ImportError as e:
                logger.warning(f"[OBSERVABILITY] Observability modules not available: {e}")

        # Initialize PII handler first (needed by agents)
        pii_config = PIIConfig(enabled=self._pii_masking_enabled)
        self.pii_handler = PIIHandler(pii_config)
        logger.info(f"PII Masking: {'enabled' if self._pii_masking_enabled else 'disabled'}")

        # Initialize components
        self.page_controller = PageController(self.browser_manager.page)
        self.element_detector = ElementDetector(self.browser_manager.page, self.llm)

        # Initialize recording components if enabled
        if self._recording_enabled:
            recording_config = RecordingConfig(enabled=True)
            self._screenshot_capture = ScreenshotCapture(recording_config)
            self._recording_manager = RecordingManager(recording_config)
            logger.info("Recording: enabled")
        else:
            logger.info("Recording: disabled")
        
        # Log LLM logging status
        logger.info(f"LLM Logging: {'enabled' if self._llm_logging else 'disabled'}")
        
        # Initialize ReAct framework (now the native implementation)
        await self._initialize_react_framework()
        
        # Wire stealth components to ReAct agent after initialization
        if self._captcha_solver and self.react_agent:
            self.react_agent.captcha_solver = self._captcha_solver
            logger.info("[STEALTH] CAPTCHA solver wired to ReAct agent")

    async def _initialize_react_framework(self) -> None:
        """Initialize the new ReAct framework."""
        from flybrowser.agents.sdk_integration import ReActBrowserAgent
        from flybrowser.agents import ExecutionMode
        
        logger.info("Initializing ReAct framework...")
        
        # Create ReAct agent with autonomous mode and configuration
        self.react_agent = ReActBrowserAgent(
            page_controller=self.page_controller,
            element_detector=self.element_detector,
            llm_provider=self.llm,
            agent_config=self._agent_config,
            execution_mode=ExecutionMode.AUTONOMOUS,
            enable_memory=True,
        )
        
        # Initialize the agent (registers tools, creates orchestrator)
        await self.react_agent.initialize()
        
        logger.info(
            f"ReAct framework initialized: "
            f"{len(self.react_agent.get_tool_list())} tools registered"
        )
    
    def enable_llm_logging(self, enabled: bool = True, level: int = 1) -> None:
        """
        Enable or disable LLM request/response logging at runtime.

        Args:
            enabled: Whether to enable LLM logging (default: True)
            level: Logging level when enabled (default: 1)
                - 0: Disabled (same as enabled=False)
                - 1: Basic (shows request/response timing)
                - 2: Detailed (shows prompts and LLM responses/plans)

        Example:
            >>> browser.enable_llm_logging(True)  # Basic logging
            >>> browser.enable_llm_logging(True, level=2)  # Detailed logging
            >>> browser.enable_llm_logging(False)  # Disable logging
        """
        self._llm_logging = level if enabled else 0
        if self.llm:
            self.llm.enable_llm_logging(enabled, level)
    
    def configure_search(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_ranking: Optional[bool] = None,
        ranking_weights: Optional[Dict[str, float]] = None,
        cache_ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Configure search settings at runtime.
        
        This allows you to change search provider settings after initialization.
        Changes take effect immediately for subsequent search operations.
        
        Args:
            provider: Search provider to use. Options:
                - "serper": Serper.dev API (recommended, fast, affordable)
                - "google": Google Custom Search API
                - "bing": Bing Web Search API
                - "auto": Automatically select best available
            api_key: API key for the search provider. Sets the appropriate
                environment variable based on provider.
            enable_ranking: Enable intelligent result ranking (default: True)
            ranking_weights: Custom ranking weights. Dictionary with keys:
                - "bm25": Keyword relevance (default: 0.35)
                - "freshness": Recency (default: 0.20)
                - "domain_authority": Source quality (default: 0.15)
                - "position": Original search engine ranking (default: 0.30)
            cache_ttl_seconds: Cache TTL in seconds (default: 300)
        
        Example:
            >>> # Switch to Serper.dev provider
            >>> browser.configure_search(provider="serper", api_key="your-key")
            
            >>> # Prioritize fresh results for news searches
            >>> browser.configure_search(ranking_weights={
            ...     "freshness": 0.45,
            ...     "bm25": 0.25,
            ...     "domain_authority": 0.15,
            ...     "position": 0.15,
            ... })
        """
        import os
        
        if provider:
            self._search_provider = provider
            self._agent_config.search_providers.default_provider = provider
            logger.info(f"Search provider changed to: {provider}")
        
        if api_key:
            self._search_api_key = api_key
            # Set environment variable based on provider
            prov = (provider or self._search_provider or "serper").lower()
            if prov == "serper":
                os.environ["SERPER_API_KEY"] = api_key
            elif prov == "google":
                os.environ["GOOGLE_CUSTOM_SEARCH_API_KEY"] = api_key
            elif prov == "bing":
                os.environ["BING_SEARCH_API_KEY"] = api_key
            logger.info(f"Search API key configured for {prov}")
        
        if enable_ranking is not None:
            self._agent_config.search_providers.enable_ranking = enable_ranking
            logger.info(f"Search ranking {'enabled' if enable_ranking else 'disabled'}")
        
        if ranking_weights:
            self._agent_config.search_providers.ranking_weights.update(ranking_weights)
            logger.info(f"Search ranking weights updated: {ranking_weights}")
        
        if cache_ttl_seconds is not None:
            self._agent_config.search_providers.cache_ttl_seconds = cache_ttl_seconds
            logger.info(f"Search cache TTL set to: {cache_ttl_seconds}s")
    
    async def search(
        self,
        query: str,
        search_type: str = "auto",
        max_results: int = 10,
        ranking: str = "auto",
        return_metadata: bool = True,
    ) -> Union[Dict[str, Any], "AgentRequestResponse"]:
        """
        Perform a web search using the configured search provider.
        
        This is a convenience method that directly invokes the search tool
        without going through the full ReAct reasoning loop. It's faster
        for simple search queries.
        
        Args:
            query: Search query or natural language instruction.
                Examples:
                - "Python tutorials" (direct query)
                - "Find images of sunset" (instruction, auto-detects image search)
                - "Latest news about AI" (instruction, auto-detects news search)
            search_type: Type of search to perform. Options:
                - "auto": Automatically detect based on query (default)
                - "web": Standard web search
                - "images": Image search
                - "news": News search
                - "videos": Video search
                - "places": Local/places search
                - "shopping": Shopping/product search
            max_results: Maximum number of results (1-50, default: 10)
            ranking: Result ranking preference. Options:
                - "auto": Automatically select based on query (default)
                - "balanced": Balanced ranking (default weights)
                - "relevance": Prioritize keyword relevance
                - "freshness": Prioritize recent results
                - "authority": Prioritize authoritative sources
            return_metadata: Return AgentRequestResponse with metadata (default: True)
        
        Returns:
            Search results with ranked items. Format:
            {
                "query": str,
                "search_type": str,
                "results": [
                    {
                        "title": str,
                        "url": str,
                        "snippet": str,
                        "relevance_score": float,
                        "domain": str,
                    },
                    ...
                ],
                "result_count": int,
                "provider_used": str,
                "answer_box": dict | None,
                "knowledge_graph": dict | None,
                "related_searches": list[str],
            }
        
        Example:
            >>> # Simple web search
            >>> results = await browser.search("Python tutorials")
            >>> for r in results.data["results"]:
            ...     print(f"{r['title']}: {r['url']}")
            
            >>> # Image search
            >>> images = await browser.search("sunset photos", search_type="images")
            
            >>> # News search with freshness ranking
            >>> news = await browser.search("AI news", search_type="news", ranking="freshness")
            
            >>> # Auto-detect search type from query
            >>> results = await browser.search("Find the latest news about Python 4.0")
        """
        from flybrowser.agents.response import create_response
        
        self._ensure_started()
        
        if self._mode == "server":
            # Server mode: use search endpoint if available
            try:
                response = await self._client._request(
                    "POST",
                    f"/sessions/{self._session_id}/search",
                    json={
                        "query": query,
                        "search_type": search_type,
                        "max_results": max_results,
                        "ranking": ranking,
                    },
                )
                result = response or {}
                if return_metadata:
                    return create_response(
                        success=result.get("success", len(result.get("results", [])) > 0),
                        data=result,
                        error=result.get("error"),
                        operation="search",
                        query=query,
                    )
                return result
            except Exception as e:
                logger.warning(f"Server search failed, falling back to agent: {e}")
                # Fall through to agent-based search
        
        # Embedded mode: Use SearchAgentTool directly for faster execution
        try:
            from flybrowser.agents.tools.search import SearchAgentTool
            
            # Get or create search tool
            search_tool = SearchAgentTool.create(
                llm_provider=self.llm,
                config=self._agent_config.search_providers,
            )
            
            # Execute search
            result = await search_tool.execute(
                query=query,
                search_type=search_type,
                max_results=max_results,
                ranking=ranking,
            )
            
            if return_metadata:
                return create_response(
                    success=result.success,
                    data=result.data,
                    error=result.error,
                    operation="search",
                    query=query,
                    metadata=result.metadata,
                )
            
            return {
                "success": result.success,
                "data": result.data,
                "error": result.error,
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            if return_metadata:
                return create_response(
                    success=False,
                    data=None,
                    error=str(e),
                    operation="search",
                    query=query,
                )
            return {"success": False, "data": None, "error": str(e)}

    async def stop(self) -> None:
        """
        Stop the browser session and cleanup resources.

        In embedded mode: Closes the browser and cleans up Playwright.
        In server mode: Closes the session on the server.
        """
        if not self._started:
            logger.warning("FlyBrowser not started")
            return

        logger.info("Stopping FlyBrowser")

        if self._mode == "server":
            if self._session_id and self._client:
                try:
                    await self._client.close_session(self._session_id)
                except Exception as e:
                    logger.warning(f"Error closing session: {e}")
            if self._client:
                await self._client.stop()
        else:
            # Stop observability components first
            if self._observability_manager:
                try:
                    await self._observability_manager.detach()
                    logger.info("[OBSERVABILITY] Observability manager stopped")
                except Exception as e:
                    logger.warning(f"Error stopping observability: {e}")
            
            # Export observability data if configured
            if (self._observability_config and 
                self._observability_config.output_dir and
                (self._command_logger or self._source_capture)):
                try:
                    self._observability_manager.export_all(self._observability_config.output_dir)
                    logger.info(f"[OBSERVABILITY] Data exported to {self._observability_config.output_dir}")
                except Exception as e:
                    logger.warning(f"Error exporting observability data: {e}")
            
            # Stop active stream if any
            if self._active_stream_id:
                try:
                    await self._stop_embedded_stream()
                except Exception as e:
                    logger.warning(f"Error stopping stream: {e}")
            
            # Stop local stream server
            if self._local_stream_server:
                try:
                    await self._local_stream_server.cleanup()
                    self._local_stream_server = None
                except Exception as e:
                    logger.warning(f"Error stopping stream server: {e}")
            
            if self.browser_manager:
                await self.browser_manager.stop()

        self._started = False
        logger.info("FlyBrowser stopped successfully")

    async def goto(self, url: str, wait_until: str = "domcontentloaded") -> None:
        """
        Navigate to a URL.

        Args:
            url: URL to navigate to (must include protocol, e.g., https://)
            wait_until: When to consider navigation succeeded.

        Example:
            >>> await browser.goto("https://example.com")
        """
        self._ensure_started()

        if self._mode == "server":
            await self._client.navigate(self._session_id, url)
        else:
            await self.page_controller.goto(url, wait_until=wait_until)

    async def navigate(
        self,
        instruction: str,
        context: Optional[Dict[str, Any]] = None,
        use_vision: bool = True,
    ) -> Dict[str, Any]:
        """
        Navigate using natural language instructions.

        Uses the ReAct framework for intelligent navigation with:
        - Natural language understanding ("go to the login page")
        - Smart waiting for page loads
        - Automatic retry on navigation failures
        - Link and button detection
        - Conditional navigation based on context

        Args:
            instruction: Natural language navigation instruction
            context: Additional context for navigation (optional):
                - conditions: Conditions that must be met for navigation
                - preferences: User preferences for navigation path
                - constraints: Navigation constraints
            use_vision: Use vision for element detection (default: True)

        Returns:
            NavigationResult with details

        Example:
            >>> await browser.navigate("go to the login page")
            >>> await browser.navigate("click the 'Products' menu item")
            >>> 
            >>> # With context
            >>> await browser.navigate(
            ...     "Navigate to search results",
            ...     context={"preferences": {"sort_by": "price", "category": "electronics"}}
            ... )
        """
        self._ensure_started()

        if self._mode == "server":
            return await self._client.navigate(self._session_id, instruction)

        # Embedded mode: Use ReAct framework with NAVIGATE mode
        from flybrowser.agents.types import OperationMode
        result_dict = await self.react_agent.execute_task(
            instruction,
            max_iterations=10,
            operation_mode=OperationMode.NAVIGATE,
        )
        
        # Get current page state
        current_url = await self.page_controller.get_url()
        current_title = await self.page_controller.get_title()
        
        return {
            "success": result_dict.get("success", False),
            "url": current_url,
            "title": current_title,
            "navigation_type": "react",
            "error": result_dict.get("error"),
        }

    async def extract(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        use_vision: bool = False,
        schema: Optional[Dict[str, Any]] = None,
        return_metadata: bool = True,
        max_iterations: int = 15,
    ) -> Union[Dict[str, Any], "AgentRequestResponse"]:
        """
        Extract data from the current page using natural language.
        
        Uses the ReAct framework for:
        - Intelligent data extraction with reasoning
        - Automatic obstacle detection and handling
        - Extraction verification for data quality
        - Schema-guided structured extraction
        - Context-aware filtering and processing

        Args:
            query: Natural language query describing what to extract.
            context: Additional context to inform extraction (optional):
                - filters: Criteria to filter extracted data
                - preferences: User preferences for data format/selection
                - constraints: Any constraints on extraction
            use_vision: Use vision-based extraction (default: False).
                Set to True for visually complex pages where text extraction
                might miss important layout information.
            schema: Optional JSON schema for structured extraction.
                When provided, the agent will return data matching this schema.
            return_metadata: Return AgentRequestResponse with full metadata
                including LLM usage and timing (default: True).
            max_iterations: Maximum iterations for extraction (default: 15).
                Extraction tasks are typically simpler than full agent tasks.

        Returns:
            AgentRequestResponse with data and metadata (default), or
            raw dictionary if return_metadata=False.
            
        Raises:
            ValueError: If query is not a valid browser automation task

        Example:
            >>> result = await browser.extract("What is the page title?")
            >>> print(result)  # Shows just the data
            >>> 
            >>> # With schema for structured data
            >>> result = await browser.extract(
            ...     "Get the top 5 stories",
            ...     schema={"type": "array", "items": {"type": "object"}}
            ... )
            >>> 
            >>> # With context for filtering
            >>> result = await browser.extract(
            ...     "Get product listings",
            ...     context={"filters": {"price_max": 100, "category": "electronics"}}
            ... )
        """
        from flybrowser.agents.response import AgentRequestResponse, create_response
        
        self._ensure_started()
        
        # Validate that this is a browser automation task
        # SDK methods skip browser keyword check since extract() already defines the operation
        validator = get_scope_validator()
        is_valid, error = validator.validate_task(query, skip_browser_keyword_check=True)
        if not is_valid:
            raise ValueError(f"Invalid extraction query: {error}")

        if self._mode == "server":
            result = await self._client.extract(self._session_id, query, schema, context=context or {})
            if return_metadata:
                return create_response(
                    success=result.get("success", True),
                    data=result.get("data", result),
                    operation="extract",
                    query=query,
                    llm_usage=result.get("llm_usage"),
                    page_metrics=result.get("page_metrics"),
                    timing=result.get("timing"),
                    metadata=result,
                )
            return result.get("data", result)
        
        # Embedded mode: Use ReAct framework with SCRAPE mode
        # This is optimized for data extraction tasks
        from flybrowser.agents.types import OperationMode
        
        logger.info(f"[extract] Extracting: {query[:100]}...")
        
        # Build extraction task - the agent will understand this is an extraction task
        # and use appropriate tools (extract_text, screenshot for vision, etc.)
        task = query
        if schema:
            import json
            task = f"{query}\n\nReturn data matching this schema: {json.dumps(schema)}"
        
        # Determine vision override based on use_vision parameter
        # For extraction, vision can help understand page layout and find data
        # None = let agent decide based on model capabilities and operation mode
        # True = force vision (useful for visually complex pages)
        # False = text-only extraction (faster, cheaper)
        vision_override = True if use_vision else None
        
        # Temporarily adjust max_iterations for extraction (typically simpler than full agent)
        original_max_iterations = self.react_agent.config.max_iterations
        if max_iterations != 15:  # User provided custom value
            self.react_agent.config.max_iterations = max_iterations
        
        # Reset LLM session usage before execution to track only this operation
        if hasattr(self.react_agent.llm, 'reset_session_usage'):
            self.react_agent.llm.reset_session_usage()
        
        # CRITICAL: Store current page context and user context in agent memory BEFORE execution
        # This lets the agent know what page it's on (e.g., after browser.goto())
        try:
            current_url = await self.page_controller.get_url()
            current_title = await self.page_controller.get_title()
            if current_url:
                self.react_agent.memory.working.set_scratch("current_url", current_url)
                self.react_agent.memory.working.set_scratch("current_title", current_title or "")
                logger.debug(f"[extract] Page context: {current_url} - {current_title}")
            
            # Store user-provided context for tool access
            if context:
                self.react_agent.memory.working.set_scratch("user_context", context)
                logger.debug(f"[extract] User context: {list(context.keys())}")
        except Exception as e:
            logger.debug(f"[extract] Could not get page context: {e}")
        
        try:
            # Call execute() with SCRAPE mode - optimized for data extraction
            agent_result = await self.react_agent.execute(
                task,
                operation_mode=OperationMode.SCRAPE,
                vision_override=vision_override,
            )
            result_dict = agent_result.to_dict()
            # Add max_iterations to metadata for proper display
            result_dict["max_iterations"] = self.react_agent.config.max_iterations
            # Add LLM usage statistics from the provider (tracked for this operation)
            if hasattr(self.react_agent.llm, 'get_session_usage'):
                result_dict["llm_usage"] = self.react_agent.llm.get_session_usage()
            
            if return_metadata:
                return create_response(
                    success=result_dict.get("success", False),
                    data=result_dict.get("result"),
                    error=result_dict.get("error"),
                    operation="extract",
                    query=query,
                    llm_usage=result_dict.get("llm_usage"),
                    metadata=result_dict,
                )
            
            # Backward compatibility: return data directly if successful
            if result_dict.get("success"):
                return result_dict.get("result")
            return {"success": False, "error": result_dict.get("error")}
            
        except Exception as e:
            logger.error(f"[extract] Failed: {e}")
            if return_metadata:
                return create_response(
                    success=False,
                    data=None,
                    error=str(e),
                    operation="extract",
                    query=query,
                )
            return {"success": False, "error": str(e)}
        finally:
            # Restore original config
            self.react_agent.config.max_iterations = original_max_iterations

    async def act(
        self,
        instruction: str,
        context: Optional[Dict[str, Any]] = None,
        use_vision: bool = True,
        return_metadata: bool = True,
        max_iterations: int = 10,
    ) -> Union[Dict[str, Any], "AgentRequestResponse"]:
        """
        Perform an action on the page based on natural language instruction.
        
        Uses the ReAct framework for:
        - Intelligent action execution with reasoning
        - Automatic obstacle detection and handling (cookie banners, modals)
        - Action verification to ensure success
        - Multi-step action planning
        - Automatic retry on failure
        - PII-safe credential handling
        - Form filling and file uploads via context

        Args:
            instruction: Natural language instruction (e.g., "click the login button")
            context: Additional context for the action (optional):
                - form_data: Dict of field_name -> value for form filling
                - files: List of {"field": str, "path": str, "mime_type": str} for uploads
                - constraints: Any other contextual information
            use_vision: Use vision for element detection (default: True).
                Vision helps accurately locate elements on the page.
            return_metadata: Return AgentRequestResponse with full metadata
                including LLM usage and timing (default: True).
            max_iterations: Maximum iterations for action (default: 10).
                Actions are typically quick, single-step operations.

        Returns:
            AgentRequestResponse with data and metadata (default), or
            raw dict if return_metadata=False.
            
        Raises:
            ValueError: If instruction is not a valid browser automation task

        Example:
            >>> result = await browser.act("click the login button")
            >>> await browser.act("type 'hello' in the search box")
            >>> 
            >>> # With form data context
            >>> await browser.act(
            ...     "Fill and submit the login form",
            ...     context={"form_data": {"email": "user@example.com", "password": "***"}}
            ... )
            >>> 
            >>> # With file upload
            >>> await browser.act(
            ...     "Upload resume file",
            ...     context={"files": [{"field": "resume", "path": "resume.pdf"}]}
            ... )
        """
        from flybrowser.agents.response import create_response
        
        self._ensure_started()
        
        # Validate that this is a browser automation task
        # SDK methods skip browser keyword check since act() already defines the operation
        validator = get_scope_validator()
        is_valid, error = validator.validate_task(instruction, skip_browser_keyword_check=True)
        if not is_valid:
            raise ValueError(f"Invalid action instruction: {error}")

        if self._mode == "server":
            result = await self._client.action(self._session_id, instruction, context=context or {})
            if return_metadata:
                return create_response(
                    success=result.get("success", False),
                    data=result,
                    error=result.get("error"),
                    operation="act",
                    query=instruction,
                    llm_usage=result.get("llm_usage"),
                    page_metrics=result.get("page_metrics"),
                    timing=result.get("timing"),
                    metadata=result,
                )
            return result

        # Embedded mode: Use ReAct framework with EXECUTE mode
        # This is optimized for fast, targeted actions (click, type, etc.)
        from flybrowser.agents.types import OperationMode
        
        logger.info(f"[act] Executing action: {instruction[:100]}...")
        
        # Determine vision override based on use_vision parameter
        # For act(), vision helps with element detection and verification
        # True (default) = use vision to find elements accurately
        # False = text-only mode (faster but may miss visual elements)
        vision_override = use_vision if use_vision is not None else None
        
        # Temporarily adjust max_iterations for actions (typically quick operations)
        original_max_iterations = self.react_agent.config.max_iterations
        if max_iterations != 10:  # User provided custom value
            self.react_agent.config.max_iterations = max_iterations
        
        # Reset LLM session usage before execution to track only this operation
        if hasattr(self.react_agent.llm, 'reset_session_usage'):
            self.react_agent.llm.reset_session_usage()
        
        # CRITICAL: Store current page context and user context in agent memory BEFORE execution
        try:
            current_url = await self.page_controller.get_url()
            current_title = await self.page_controller.get_title()
            if current_url:
                self.react_agent.memory.working.set_scratch("current_url", current_url)
                self.react_agent.memory.working.set_scratch("current_title", current_title or "")
                logger.debug(f"[act] Page context: {current_url} - {current_title}")
            
            # Store user-provided context for tool access
            if context:
                self.react_agent.memory.working.set_scratch("user_context", context)
                logger.debug(f"[act] User context: {list(context.keys())}")
        except Exception as e:
            logger.debug(f"[act] Could not get page context: {e}")
        
        try:
            # Call execute() with EXECUTE mode - optimized for actions
            agent_result = await self.react_agent.execute(
                instruction,
                operation_mode=OperationMode.EXECUTE,
                vision_override=vision_override,
            )
            result_dict = agent_result.to_dict()
            # Add max_iterations to metadata for proper display
            result_dict["max_iterations"] = self.react_agent.config.max_iterations
            # Add LLM usage statistics from the provider (tracked for this operation)
            if hasattr(self.react_agent.llm, 'get_session_usage'):
                result_dict["llm_usage"] = self.react_agent.llm.get_session_usage()
            
            if return_metadata:
                return create_response(
                    success=result_dict.get("success", False),
                    data=result_dict.get("result"),
                    error=result_dict.get("error"),
                    operation="act",
                    query=instruction,
                    llm_usage=result_dict.get("llm_usage"),
                    metadata=result_dict,
                )
            
            return {
                "success": result_dict.get("success", False),
                "data": result_dict.get("result"),
                "error": result_dict.get("error"),
            }
            
        except Exception as e:
            logger.error(f"[act] Failed: {e}")
            if return_metadata:
                return create_response(
                    success=False,
                    data=None,
                    error=str(e),
                    operation="act",
                    query=instruction,
                )
            return {"success": False, "data": None, "error": str(e)}
        finally:
            # Restore original config
            self.react_agent.config.max_iterations = original_max_iterations

    async def execute_task(
        self,
        task: str,
    ) -> Dict[str, Any]:
        """
        Execute a complex task using the ReAct framework with reasoning and action.

        This is a powerful way to automate browser tasks. It uses:
        - Chain-of-thought reasoning to plan multi-step actions
        - Step-by-step execution with verification after each step
        - Intelligent retry with plan adaptation on failures
        - Automatic obstacle handling

        Args:
            task: Natural language description of what to accomplish

        Returns:
            ExecutionResult dict containing:
            - success: Whether the task completed successfully
            - result: Any extracted data or result (for extraction tasks)
            - error: Error details if failed
            - steps: List of steps executed
            - total_iterations: Number of iterations used
            - execution_time_ms: Total execution time
            
        Raises:
            ValueError: If task is not a valid browser automation task

        Example:
            >>> # Navigate and extract
            >>> result = await browser.execute_task("Go to google.com and search for 'python'")
            >>> 
            >>> # Extract data with verification
            >>> result = await browser.execute_task(
            ...     "Extract the titles and scores of the top 10 stories from hackernews"
            ... )
            >>> if result["success"]:
            ...     print(result["result"])
        """
        self._ensure_started()
        
        # Validate that this is a browser automation task
        # SDK methods skip browser keyword check since execute_task() already defines the operation
        validator = get_scope_validator()
        is_valid, error = validator.validate_task(task, skip_browser_keyword_check=True)
        if not is_valid:
            raise ValueError(f"Invalid task: {error}")

        if self._mode == "server":
            # Server mode: use orchestrator API endpoint if available
            try:
                response = await self._client._request(
                    "POST",
                    f"/sessions/{self._session_id}/execute",
                    json={"task": task},
                )
                return response or {}
            except Exception:
                # Fallback to action if execute endpoint not available
                return await self._client.action(self._session_id, task)

        # Embedded mode: Use ReAct framework with AUTO mode (uses planning)
        from flybrowser.agents.types import OperationMode
        
        # Call execute() - uses agent's configured max_iterations with AUTO mode
        agent_result = await self.react_agent.execute(
            task,
            operation_mode=OperationMode.AUTO,
        )
        return agent_result.to_dict()
    
    async def agent(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 50,
        max_time_seconds: float = 1800.0,
        return_metadata: bool = True,
    ) -> Union[Dict[str, Any], "AgentRequestResponse"]:
        """
        Execute autonomous multi-step tasks with intelligent reasoning.
        
        This is the primary method for complex, multi-step browser automation.
        The agent automatically selects the optimal reasoning strategy based
        on task complexity and adapts dynamically during execution.
        
        Capabilities:
        - Complex multi-step tasks ("Search X and summarize")
        - Multi-page navigation and data collection
        - Form filling with validation
        - Intelligent navigation and interaction
        - Automatic obstacle handling (cookie banners, modals)
        - Dynamic strategy adaptation based on failures
        - Multi-tool orchestration (32 browser tools)
        - Memory-based context retention
        
        Args:
            task: High-level task description in natural language.
            context: Optional context dict (e.g., {"budget": 1000}).
            max_iterations: Maximum execution iterations (default: 50).
            max_time_seconds: Maximum execution time in seconds (default: 1800).
            return_metadata: Return AgentRequestResponse with full metadata.
        
        Returns:
            AgentRequestResponse with comprehensive execution metadata.
            
        Example:
            >>> result = await browser.agent("Search for flights to Tokyo and extract the cheapest option")
            >>> print(result.data)  # Extracted flight info
            >>> 
            >>> # With context
            >>> result = await browser.agent(
            ...     "Book a hotel room",
            ...     context={"check_in": "2024-03-01", "nights": 3}
            ... )
        """
        from flybrowser.agents.response import AgentRequestResponse, create_response
        
        self._ensure_started()
        
        # Validate that this is a browser automation task
        # SDK methods skip browser keyword check since agent() already defines the operation
        validator = get_scope_validator()
        is_valid, error = validator.validate_task(task, skip_browser_keyword_check=True)
        if not is_valid:
            raise ValueError(f"Invalid task: {error}")
        
        if self._mode == "server":
            # Server mode: use agent endpoint
            try:
                response = await self._client._request(
                    "POST",
                    f"/sessions/{self._session_id}/agent",
                    json={
                        "task": task,
                        "context": context,
                        "max_iterations": max_iterations,
                        "max_time_seconds": max_time_seconds,
                    },
                )
                result = response or {}
                if return_metadata:
                    return create_response(
                        success=result.get("success", False),
                        data=result.get("result_data"),
                        error=result.get("error_message"),
                        operation="agent",
                        query=task,
                        llm_usage=result.get("cost_tracking"),
                        execution=result.get("execution"),
                        history=result.get("execution_history"),
                        metadata=result,
                    )
                return result
            except Exception:
                # Fallback to execute_task if agent endpoint not available
                return await self.execute_task(task)
        
        # Embedded mode: Use ReAct framework (native implementation)
        logger.info("[ReAct] Using ReAct framework for autonomous execution")
        
        # Temporarily update agent config for this execution if user provided custom limits
        original_max_iterations = self.react_agent.config.max_iterations
        original_timeout = self.react_agent.config.timeout_seconds
        
        if max_iterations != 50:  # User provided custom value
            self.react_agent.config.max_iterations = max_iterations
        if max_time_seconds != 1800.0:  # User provided custom value
            self.react_agent.config.timeout_seconds = max_time_seconds
        
        # Reset LLM session usage before execution to track only this operation
        if hasattr(self.react_agent.llm, 'reset_session_usage'):
            self.react_agent.llm.reset_session_usage()
        
        # CRITICAL: Store current page context and user context in agent memory BEFORE execution
        try:
            current_url = await self.page_controller.get_url()
            current_title = await self.page_controller.get_title()
            if current_url:
                self.react_agent.memory.working.set_scratch("current_url", current_url)
                self.react_agent.memory.working.set_scratch("current_title", current_title or "")
                logger.debug(f"[agent] Page context: {current_url} - {current_title}")
            
            # Store user-provided context for tool access
            if context:
                self.react_agent.memory.working.set_scratch("user_context", context)
                logger.debug(f"[agent] User context: {list(context.keys())}")
        except Exception as e:
            logger.debug(f"[agent] Could not get page context: {e}")
        
        try:
            # Call execute() - uses agent's configured max_iterations and timeout with AUTO mode
            from flybrowser.agents.types import OperationMode

            agent_result = await self.react_agent.execute(
                task,
                operation_mode=OperationMode.AUTO,
            )
            result_dict = agent_result.to_dict()
            # Add max_iterations to metadata for proper display
            result_dict["max_iterations"] = self.react_agent.config.max_iterations
            # Add LLM usage statistics from the provider (tracked for this operation)
            if hasattr(self.react_agent.llm, 'get_session_usage'):
                result_dict["llm_usage"] = self.react_agent.llm.get_session_usage()
        finally:
            # Restore original config
            self.react_agent.config.max_iterations = original_max_iterations
            self.react_agent.config.timeout_seconds = original_timeout
        
        # Load completion page showing task summary AFTER all execution is done
        # This only happens after the entire agent query is solved (success or failure)
        if self.browser_manager:
            try:
                # Extract completion data safely using helper function
                completion_data = self._extract_completion_data(
                    result_dict=result_dict,
                    task=task,
                    session_id=self._session_id,
                )
                
                await self.browser_manager._load_completion_page(**completion_data)
            except Exception as e:
                # Don't fail the overall operation if completion page fails
                # But log at warning level so users can see if something went wrong
                logger.warning(f"[COMPLETION] Could not load completion page: {e}")
                import traceback
                logger.debug(f"[COMPLETION] Traceback: {traceback.format_exc()}")
        
        if return_metadata:
            # Convert dict to AgentRequestResponse
            return create_response(
                success=result_dict.get("success", False),
                data=result_dict.get("result"),
                error=result_dict.get("error"),
                operation="agent",
                query=task,
                llm_usage=result_dict.get("llm_usage"),
                metadata=result_dict,
            )
        return result_dict
    
    async def observe(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        return_selectors: bool = True,
        return_metadata: bool = True,
        max_iterations: int = 10,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]], "AgentRequestResponse"]:
        """
        Observe and identify elements on the current page.
        
        Analyzes the page to find elements matching a natural language description.
        Returns selectors, element info, and actionable suggestions.
        
        Use this to:
        - Find elements before acting on them
        - Understand page structure
        - Get reliable selectors for automation
        - Verify elements exist before interaction
        
        Args:
            query: Natural language description of what to find.
                Example: "find the search bar", "locate all product cards"
            context: Additional context for element search (optional):
                - filters: Criteria to filter found elements
                - constraints: Constraints on element selection
                - region: Specific region of page to search
            return_selectors: Include CSS selectors in response (default: True).
            return_metadata: Return AgentRequestResponse with full metadata.
            max_iterations: Maximum iterations for observation (default: 10).
        
        Returns:
            List of observed elements with selectors and descriptions.
            
        Example:
            >>> # Find a specific element
            >>> elements = await browser.observe("find the login button")
            >>> print(elements[0]["selector"])  # '#login-btn'
            >>> 
            >>> # Use with act() for reliable automation
            >>> login_btn = await browser.observe("find the login button")
            >>> if login_btn:
            ...     await browser.act(f"click {login_btn[0]['selector']}")
            >>> 
            >>> # With context for filtering
            >>> elements = await browser.observe(
            ...     "find product cards",
            ...     context={"filters": {"visible_only": True, "region": "main"}}
            ... )
        """
        from flybrowser.agents.response import create_response
        
        self._ensure_started()
        
        if self._mode == "server":
            try:
                response = await self._client._request(
                    "POST",
                    f"/sessions/{self._session_id}/observe",
                    json={"query": query, "return_selectors": return_selectors, "context": context or {}},
                )
                result = response or {}
                if return_metadata:
                    return create_response(
                        success=result.get("success", True),
                        data=result.get("elements", result),
                        operation="observe",
                        query=query,
                        metadata=result,
                    )
                return result.get("elements", result)
            except Exception:
                pass  # Fall through to embedded implementation
        
        # Embedded mode: Use ReAct framework with RESEARCH mode for intelligent element finding
        # This mode is optimized for discovering and understanding page elements
        from flybrowser.agents.types import OperationMode
        
        logger.info(f"[observe] Finding elements: {query[:100]}...")
        
        try:
            # First, try element detector for fast results (if available)
            if self.element_detector:
                try:
                    element_info = await self.element_detector.find_element(query)
                    
                    if element_info and element_info.get("selector"):
                        page_state = await self.page_controller.get_page_state()
                        elements = [{
                            "description": query,
                            "selector": element_info.get("selector"),
                            "selector_type": element_info.get("selector_type", "css"),
                            "confidence": element_info.get("confidence", 0.8),
                            "tag": element_info.get("tag"),
                            "text": element_info.get("text", "")[:100],
                            "attributes": element_info.get("attributes", {}),
                            "visible": element_info.get("visible", True),
                            "actionable": element_info.get("actionable", True),
                        }]
                        
                        logger.info(f"[observe] Element detector found {len(elements)} element(s)")
                        
                        if return_metadata:
                            return create_response(
                                success=True,
                                data=elements,
                                operation="observe",
                                query=query,
                                metadata={"page_url": page_state.get("url"), "elements_found": len(elements), "method": "element_detector"},
                            )
                        return elements
                except Exception as e:
                    logger.debug(f"[observe] Element detector failed, falling back to agent: {e}")
            
            # Fallback: Use ReAct agent with RESEARCH mode to intelligently find elements
            # The agent will use vision and page analysis to locate elements
            task = f"Find and identify all elements matching: {query}. For each element found, provide its CSS selector, a description, tag name, visible text, and whether it can be interacted with (clicked, typed into, etc.)."
            
            # Temporarily adjust max_iterations for observation
            original_max_iterations = self.react_agent.config.max_iterations
            if max_iterations != 10:  # User provided custom value
                self.react_agent.config.max_iterations = max_iterations
            
            # Reset LLM session usage before execution to track only this operation
            if hasattr(self.react_agent.llm, 'reset_session_usage'):
                self.react_agent.llm.reset_session_usage()
            
            # CRITICAL: Store current page context and user context in agent memory BEFORE execution
            try:
                current_url = await self.page_controller.get_url()
                current_title = await self.page_controller.get_title()
                if current_url:
                    self.react_agent.memory.working.set_scratch("current_url", current_url)
                    self.react_agent.memory.working.set_scratch("current_title", current_title or "")
                    logger.debug(f"[observe] Page context: {current_url} - {current_title}")
                
                # Store user-provided context for tool access
                if context:
                    self.react_agent.memory.working.set_scratch("user_context", context)
                    logger.debug(f"[observe] User context: {list(context.keys())}")
            except Exception as e:
                logger.debug(f"[observe] Could not get page context: {e}")
            
            try:
                agent_result = await self.react_agent.execute(
                    task,
                    operation_mode=OperationMode.RESEARCH,  # RESEARCH mode for discovery
                    vision_override=True,  # Always use vision for observe to see the page
                )
                result_dict = agent_result.to_dict()
                # Add max_iterations to metadata for proper display
                result_dict["max_iterations"] = self.react_agent.config.max_iterations
                # Add LLM usage statistics from the provider (tracked for this operation)
                if hasattr(self.react_agent.llm, 'get_session_usage'):
                    result_dict["llm_usage"] = self.react_agent.llm.get_session_usage()
                
                # Parse agent result into element list format
                elements = result_dict.get("result", [])
                if not isinstance(elements, list):
                    elements = [elements] if elements else []
                
                logger.info(f"[observe] Agent found {len(elements)} element(s)")
                
                if return_metadata:
                    return create_response(
                        success=result_dict.get("success", False) and len(elements) > 0,
                        data=elements,
                        operation="observe",
                        query=query,
                        llm_usage=result_dict.get("llm_usage"),
                        metadata={**result_dict, "method": "react_agent"},
                    )
                return elements
            finally:
                # Restore original config
                self.react_agent.config.max_iterations = original_max_iterations
            
        except Exception as e:
            logger.error(f"[observe] Failed: {e}")
            if return_metadata:
                return create_response(
                    success=False,
                    data=[],
                    error=str(e),
                    operation="observe",
                    query=query,
                )
            return []

    # ==================== Batch Operations ====================
    
    async def batch_execute(
        self,
        tasks: List[str],
        parallel: bool = False,
        stop_on_failure: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tasks in batch.
        
        Args:
            tasks: List of task descriptions to execute
            parallel: Whether to execute tasks in parallel (default: False)
            stop_on_failure: Whether to stop on first failure (default: False)
            
        Returns:
            List of ExecutionResult dicts, one per task
            
        Example:
            >>> results = await browser.batch_execute([
            ...     "Navigate to page A and extract title",
            ...     "Navigate to page B and extract title",
            ... ], parallel=True)
        """
        self._ensure_started()
        
        if self._mode == "server":
            # Server mode: sequential execution via API
            results = []
            for task in tasks:
                result = await self.execute_task(task)
                results.append(result)
                if stop_on_failure and not result.get("success"):
                    break
            return results
        
        # Embedded mode: Use ReAct framework with AUTO mode
        from flybrowser.agents.types import OperationMode

        if parallel:
            # Execute tasks in parallel using asyncio.gather
            import asyncio

            async def execute_with_error_handling(task: str) -> Dict[str, Any]:
                try:
                    return await self.react_agent.execute_task(
                        task,
                        max_iterations=25,
                        operation_mode=OperationMode.AUTO,
                    )
                except Exception as e:
                    return {"success": False, "error": str(e), "result": None}

            results = await asyncio.gather(*[execute_with_error_handling(task) for task in tasks])
            return list(results)
        else:
            # Sequential execution
            results = []
            for task in tasks:
                result = await self.react_agent.execute_task(
                    task,
                    max_iterations=25,
                    operation_mode=OperationMode.AUTO,
                )
                results.append(result)
                if stop_on_failure and not result.get("success"):
                    break
            return results

    async def screenshot(self, full_page: bool = False, mask_pii: bool = True) -> Dict[str, Any]:
        """
        Take a screenshot of the current page.

        Args:
            full_page: Capture the full scrollable page (default: False)
            mask_pii: Apply PII masking (default: True)

        Returns:
            Dictionary with screenshot data including base64-encoded image.

        Example:
            >>> screenshot = await browser.screenshot(full_page=True)
            >>> print(screenshot["data_base64"][:50])
        """
        self._ensure_started()

        if self._mode == "server":
            return await self._client.screenshot(self._session_id, full_page)
        else:
            from flybrowser.core.recording import RecordingConfig, ScreenshotCapture

            if not self._screenshot_capture:
                config = RecordingConfig(enabled=True)
                self._screenshot_capture = ScreenshotCapture(config)

            screenshot = await self._screenshot_capture.take(
                self.browser_manager.page,
                full_page=full_page,
                save_to_file=False,
            )

            return {
                "screenshot_id": screenshot.id,
                "format": screenshot.format.value,
                "width": screenshot.width,
                "height": screenshot.height,
                "data_base64": screenshot.to_base64(),
                "url": screenshot.url,
                "timestamp": screenshot.timestamp,
            }

    async def get_screenshot_base64(self, full_page: bool = False) -> str:
        """Take a screenshot and return as base64 string."""
        result = await self.screenshot(full_page=full_page)
        return result.get("data_base64", "")

    async def start_recording(self) -> Dict[str, Any]:
        """
        Start recording the browser session.

        Returns:
            Dictionary with recording_id.

        Example:
            >>> await browser.start_recording()
            >>> await browser.goto("https://example.com")
            >>> recording = await browser.stop_recording()
        """
        self._ensure_started()

        if self._mode == "server":
            # Server mode: call recording API
            response = await self._client._request(
                "POST",
                f"/sessions/{self._session_id}/recording/start",
                json={"video_enabled": True},
            )
            return response or {}
        else:
            from flybrowser.core.recording import RecordingConfig, RecordingManager

            if not self._recording_manager:
                config = RecordingConfig(enabled=True)
                self._recording_manager = RecordingManager(config)

            await self._recording_manager.start_session(self.browser_manager.browser)
            logger.info("Recording started")
            return {"recording_id": "local", "status": "started"}

    async def stop_recording(self) -> Dict[str, Any]:
        """
        Stop recording and return recording data.

        Returns:
            Dictionary with recording info including screenshots and video.
        """
        self._ensure_started()

        if self._mode == "server":
            response = await self._client._request(
                "POST",
                f"/sessions/{self._session_id}/recording/stop",
            )
            return response or {}
        else:
            if not self._recording_manager:
                return {"session_id": None, "screenshots": [], "video": None}

            result = await self._recording_manager.stop_session()
            logger.info(f"Recording stopped: {result.get('session_id')}")
            return result

    # Streaming methods
    async def start_stream(
        self,
        protocol: str = "hls",
        quality: str = "high",
        codec: str = "h264",
        width: Optional[int] = None,
        height: Optional[int] = None,
        frame_rate: Optional[int] = None,
        rtmp_url: Optional[str] = None,
        rtmp_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start a live stream of the browser session.

        Args:
            protocol: Streaming protocol (hls, dash, rtmp)
            quality: Quality profile - one of:
                - low_bandwidth: 500kbps, for slow connections
                - medium: 1.5Mbps, balanced
                - high: 3Mbps, high quality (default)
                - ultra_high: 6Mbps, maximum quality
                - local_high: 12Mbps, optimized for localhost/LAN
                - local_4k: 25Mbps, 4K quality for localhost
                - studio: 50Mbps, near-lossless production
            codec: Video codec (h264, h265, vp9)
            width: Video width in pixels (default: 1920 for most profiles)
            height: Video height in pixels (default: 1080 for most profiles)
            frame_rate: Frames per second (default: 30)
            rtmp_url: RTMP destination URL (for RTMP protocol)
            rtmp_key: RTMP stream key

        Returns:
            Dictionary with stream URLs and stream_id

        Example:
            >>> # Standard 1080p streaming
            >>> stream = await browser.start_stream(protocol="hls", quality="high")
            >>> print(stream["hls_url"])
            
            >>> # 4K streaming for localhost
            >>> stream = await browser.start_stream(
            ...     quality="local_4k",
            ...     width=3840,
            ...     height=2160
            ... )
            >>> await browser.stop_stream()
        """
        self._ensure_started()

        if self._mode == "server":
            response = await self._client._request(
                "POST",
                f"/sessions/{self._session_id}/stream/start",
                json={
                    "protocol": protocol,
                    "quality": quality,
                    "codec": codec,
                    "width": width,
                    "height": height,
                    "frame_rate": frame_rate,
                    "rtmp_url": rtmp_url,
                    "rtmp_key": rtmp_key,
                },
            )
            return response or {}
        else:
            # Embedded mode: Start local streaming
            return await self._start_embedded_stream(
                protocol, quality, codec, width, height, frame_rate, rtmp_url, rtmp_key
            )

    async def stop_stream(self) -> Dict[str, Any]:
        """
        Stop the active stream.

        Returns:
            Dictionary with stream statistics
        """
        self._ensure_started()

        if self._mode == "server":
            response = await self._client._request(
                "POST",
                f"/sessions/{self._session_id}/stream/stop",
            )
            return response or {}
        else:
            # Embedded mode: Stop local streaming
            return await self._stop_embedded_stream()

    async def get_stream_status(self) -> Dict[str, Any]:
        """
        Get status and metrics of the active stream.

        Returns:
            Dictionary with stream status, health, metrics, and URLs
        """
        self._ensure_started()

        if self._mode == "server":
            response = await self._client._request(
                "GET",
                f"/sessions/{self._session_id}/stream/status",
            )
            return response or {}
        else:
            # Embedded mode: Get local stream status
            return await self._get_embedded_stream_status()

    # Recording management methods
    async def list_recordings(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available recordings.

        Args:
            session_id: Filter by session ID (optional)

        Returns:
            List of recording dictionaries

        Example:
            >>> recordings = await browser.list_recordings()
            >>> for rec in recordings:
            ...     print(f"{rec['recording_id']}: {rec['duration_seconds']}s")
        """
        if self._mode == "server":
            response = await self._client._request(
                "GET",
                "/recordings",
                params={"session_id": session_id} if session_id else {},
            )
            return response.get("recordings", []) if response else []
        else:
            raise NotImplementedError("Recording management is only available in server mode")

    async def download_recording(
        self,
        recording_id: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Download a recording.

        Args:
            recording_id: Recording identifier
            output_path: Local path to save file (if None, returns download URL)

        Returns:
            Dictionary with download information or path

        Example:
            >>> info = await browser.download_recording("rec_123", "recording.mp4")
            >>> print(f"Downloaded to {info['file_path']}")
        """
        if self._mode == "server":
            # Get download URL
            response = await self._client._request(
                "GET",
                f"/recordings/{recording_id}/download",
            )
            
            if not response:
                raise ValueError(f"Recording not found: {recording_id}")
            
            if output_path:
                # Download file
                import aiohttp
                download_url = response.get("download_url")
                if download_url:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(download_url) as resp:
                            if resp.status == 200:
                                with open(output_path, "wb") as f:
                                    f.write(await resp.read())
                                return {"file_path": output_path, "success": True}
                    raise Exception("Download failed")
            
            return response
        else:
            raise NotImplementedError("Recording management is only available in server mode")

    async def delete_recording(self, recording_id: str) -> bool:
        """
        Delete a recording.

        Args:
            recording_id: Recording identifier

        Returns:
            True if successful

        Example:
            >>> await browser.delete_recording("rec_123")
        """
        if self._mode == "server":
            response = await self._client._request(
                "DELETE",
                f"/recordings/{recording_id}",
            )
            return response.get("success", False) if response else False
        else:
            raise NotImplementedError("Recording management is only available in server mode")

    # PII handling methods
    def store_credential(self, name: str, value: str, pii_type: str = "password") -> str:
        """
        Store a credential securely for later use.

        Args:
            name: Name/identifier for the credential
            value: The sensitive value to store
            pii_type: Type of PII (password, email, phone, etc.)

        Returns:
            Credential ID for later retrieval

        Example:
            >>> pwd_id = browser.store_credential("login_password", "secret123")
            >>> await browser.secure_fill("#password", pwd_id)
        """
        if self._mode == "server":
            # Server mode: credentials stored on server
            raise NotImplementedError("Credential storage in server mode - use secure_fill API directly")
        else:
            from flybrowser.security.pii_handler import PIIType
            pii_type_enum = PIIType(pii_type)
            return self.pii_handler.store_credential(name, value, pii_type_enum)

    async def secure_fill(self, selector: str, credential_id: str, clear_first: bool = True) -> bool:
        """
        Securely fill a form field with a stored credential.

        Args:
            selector: CSS selector for the input field
            credential_id: ID of the stored credential
            clear_first: Clear the field before filling (default: True)

        Returns:
            True if successful
        """
        self._ensure_started()

        if self._mode == "server":
            response = await self._client._request(
                "POST",
                f"/sessions/{self._session_id}/secure-fill",
                json={"selector": selector, "credential_id": credential_id, "clear_first": clear_first},
            )
            return response.get("success", False) if response else False
        else:
            return await self.pii_handler.secure_fill(
                self.browser_manager.page,
                selector,
                credential_id,
                clear_first,
            )

    def mask_pii(self, text: str) -> str:
        """
        Mask PII in text for safe logging or display.

        Args:
            text: Text that may contain PII

        Returns:
            Text with PII masked
        """
        if self._mode == "server":
            # For server mode, use local masker
            from flybrowser.security.pii_handler import PIIMasker
            masker = PIIMasker()
            return masker.mask_text(text)
        else:
            return self.pii_handler.mask_for_llm(text)
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get accumulated LLM usage statistics for this session.
        
        This returns the total tokens, cost, and API calls made across
        all operations in this browser session.
        
        Returns:
            Dictionary with session usage statistics:
            - prompt_tokens: Total input tokens
            - completion_tokens: Total output tokens
            - total_tokens: Total tokens used
            - cost_usd: Total cost in USD
            - calls_count: Number of API calls
            - cached_calls: Number of cached responses
            - model: Primary model used
        
        Example:
            >>> async with FlyBrowser(...) as browser:
            ...     await browser.extract("Get title")
            ...     await browser.act("Click button")
            ...     usage = browser.get_usage_summary()
            ...     print(f"Total cost: ${usage['cost_usd']:.4f}")
            ...     print(f"Total tokens: {usage['total_tokens']:,}")
        """
        if self._mode == "server":
            # Server mode: we don't have direct access to LLM usage
            # Return empty summary
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "calls_count": 0,
                "cached_calls": 0,
                "model": "",
                "note": "Usage tracking not available in server mode",
            }
        
        # Embedded mode: get from LLM provider
        if self.llm:
            return self.llm.get_session_usage()
        
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "calls_count": 0,
            "cached_calls": 0,
            "model": "",
        }

    # Embedded streaming implementation methods
    async def _start_embedded_stream(
        self,
        protocol: str,
        quality: str,
        codec: str,
        width: Optional[int],
        height: Optional[int],
        frame_rate: Optional[int],
        rtmp_url: Optional[str],
        rtmp_key: Optional[str],
    ) -> Dict[str, Any]:
        """Start streaming in embedded mode with local HTTP server."""
        import asyncio
        import uuid
        from aiohttp import web
        from pathlib import Path
        import tempfile
        from flybrowser.core.ffmpeg_recorder import VideoCodec, FFmpegRecorder, QualityProfile
        from flybrowser.service.streaming import StreamingManager, StreamingProtocol
        
        logger.info(
            f"[EMBEDDED_STREAM] Starting embedded stream\n"
            f"  Protocol: {protocol}\n"
            f"  Quality: {quality}\n"
            f"  Codec: {codec}\n"
            f"  Resolution: {width}x{height} (custom: {width is not None or height is not None})\n"
            f"  Frame Rate: {frame_rate or 'default'}"
        )
        
        if self._active_stream_id:
            logger.error(f"[EMBEDDED_STREAM] Stream already active: {self._active_stream_id}")
            raise RuntimeError("Stream already active. Stop current stream first.")
        
        # Create temporary output directory for streams FIRST
        stream_dir = Path(tempfile.gettempdir()) / "flybrowser_streams"
        stream_dir.mkdir(exist_ok=True)
        
        # Convert protocol string to enum
        protocol_map = {
            "hls": StreamingProtocol.HLS,
            "dash": StreamingProtocol.DASH,
            "rtmp": StreamingProtocol.RTMP,
        }
        stream_protocol = protocol_map.get(protocol.lower(), StreamingProtocol.HLS)
        
        # Convert codec string to enum
        codec_map = {
            "h264": VideoCodec.H264,
            "h265": VideoCodec.H265,
            "vp9": VideoCodec.VP9,
        }
        video_codec = codec_map.get(codec.lower(), VideoCodec.H264)
        
        # Start local HTTP server to get the port (only for HLS/DASH)
        if not self._local_stream_server and stream_protocol in [StreamingProtocol.HLS, StreamingProtocol.DASH]:
            await self._start_local_stream_server(stream_dir)
        
        # Initialize streaming manager with correct base URL
        if not self._streaming_manager:
            base_url = f"http://localhost:{self._local_stream_port}" if self._local_stream_port else None
            self._streaming_manager = StreamingManager(
                output_dir=str(stream_dir),
                base_url=base_url,
                max_concurrent_streams=1
            )
        
        # Generate unique stream ID
        stream_id = f"stream_{uuid.uuid4().hex[:8]}"
        self._active_stream_id = stream_id
        
        # Start streaming session
        try:
            # Create stream configuration
            from flybrowser.service.streaming import StreamConfig
            from flybrowser.core.ffmpeg_recorder import QualityProfile
            
            # Map quality string to QualityProfile enum (including new profiles)
            quality_map = {
                "ultra_low_latency": QualityProfile.ULTRA_LOW_LATENCY,
                "low_bandwidth": QualityProfile.LOW_BANDWIDTH,
                "medium": QualityProfile.MEDIUM,
                "high": QualityProfile.HIGH,
                "ultra_high": QualityProfile.ULTRA_HIGH,
                "lossless": QualityProfile.LOSSLESS,
                "local_high": QualityProfile.LOCAL_HIGH,
                "local_4k": QualityProfile.LOCAL_4K,
                "studio": QualityProfile.STUDIO,
            }
            quality_profile = quality_map.get(quality.lower(), QualityProfile.HIGH)
            
            # Build config with optional resolution overrides
            config_kwargs = {
                "protocol": stream_protocol,
                "quality_profile": quality_profile,
                "codec": video_codec,
                "rtmp_url": rtmp_url,
                "rtmp_key": rtmp_key,
            }
            
            # Apply resolution if specified
            if width is not None:
                config_kwargs["width"] = width
            if height is not None:
                config_kwargs["height"] = height
            if frame_rate is not None:
                config_kwargs["frame_rate"] = frame_rate
            
            config = StreamConfig(**config_kwargs)
            
            # Validate prerequisites
            if not self.browser_manager:
                logger.error("[EMBEDDED_STREAM] Browser not initialized")
                raise RuntimeError("Browser not initialized. Call start() first.")
            
            # Get page from browser_manager
            try:
                page = self.browser_manager.page
                logger.info(f"[EMBEDDED_STREAM] Got page from browser_manager")
            except Exception as e:
                logger.error(f"[EMBEDDED_STREAM] Page not available: {e}")
                raise RuntimeError(f"Page not available: {e}")
            
            # Detect browser type for logging
            try:
                browser = page.context.browser
                browser_type = browser.browser_type.name if browser else "unknown"
                logger.info(f"[EMBEDDED_STREAM] Browser type: {browser_type}")
            except Exception:
                logger.info(f"[EMBEDDED_STREAM] Browser type: unknown")
            
            # Create and start stream
            logger.info(f"[EMBEDDED_STREAM] Creating stream via StreamingManager...")
            stream_info = await self._streaming_manager.create_stream(
                session_id=self._session_id or "embedded",
                page=page,
                config=config,
            )
            
            # Log stream directory for debugging
            stream = self._streaming_manager._streams.get(stream_info.stream_id)
            if stream:
                logger.info(
                    f"[EMBEDDED_STREAM] Stream files location:\n"
                    f"  Directory: {stream.stream_dir}\n"
                    f"  Playlist: {stream.stream_dir / 'playlist.m3u8'}"
                )
            
            # Return stream information with correct local URLs
            result = {
                "stream_id": stream_info.stream_id,
                "session_id": stream_info.session_id,
                "protocol": protocol,
                "quality": quality,
                "codec": codec,
                "status": stream_info.state.value if hasattr(stream_info.state, 'value') else str(stream_info.state),
                "hls_url": stream_info.hls_url,
                "dash_url": stream_info.dash_url,
                "rtmp_url": stream_info.rtmp_url,
                "player_url": stream_info.player_url,
                "stream_url": stream_info.hls_url or stream_info.dash_url or stream_info.rtmp_url,
                "local_server_port": self._local_stream_port,  # For debugging
            }
            
            logger.info(
                f"[EMBEDDED_STREAM] Stream started successfully\n"
                f"  Stream ID: {stream_info.stream_id}\n"
                f"  HLS URL: {stream_info.hls_url}\n"
                f"  Player URL: {stream_info.player_url}\n"
                f"  Local Server Port: {self._local_stream_port}"
            )
            return result
            
        except Exception as e:
            self._active_stream_id = None
            logger.error(f"[EMBEDDED_STREAM] Failed to start stream: {e}")
            import traceback
            logger.error(f"[EMBEDDED_STREAM] Traceback: {traceback.format_exc()}")
            raise
    
    async def _stop_embedded_stream(self) -> Dict[str, Any]:
        """Stop the active embedded stream."""
        if not self._active_stream_id:
            return {"success": False, "error": "No active stream"}
        
        try:
            # Get stream info from streaming manager (find by session_id)
            stream_info = None
            for stream_id, stream in self._streaming_manager._streams.items():
                if stream.session_id == (self._session_id or "embedded"):
                    stream_info = await self._streaming_manager.stop_stream(stream_id)
                    break
            
            stream_id = self._active_stream_id
            self._active_stream_id = None
            
            logger.info(f"Embedded stream stopped: {stream_id}")
            return {
                "stream_id": stream_id,
                "success": True,
                "info": stream_info.to_dict() if stream_info else None,
            }
        except Exception as e:
            logger.error(f"Failed to stop embedded stream: {e}")
            return {"success": False, "error": str(e)}
    
    async def _get_embedded_stream_status(self) -> Dict[str, Any]:
        """Get status of the active embedded stream."""
        if not self._active_stream_id:
            return {"active": False, "error": "No active stream"}
        
        try:
            # Find stream by session_id
            stream_info = None
            for stream_id, stream in self._streaming_manager._streams.items():
                if stream.session_id == (self._session_id or "embedded"):
                    stream_info = await self._streaming_manager.get_stream(stream_id)
                    break
            
            if not stream_info:
                return {"active": False, "error": "Stream not found"}
            
            return {
                "stream_id": stream_info.stream_id,
                "active": True,
                "status": stream_info.to_dict(),
            }
        except Exception as e:
            return {"active": False, "error": str(e)}
    
    async def _start_local_stream_server(self, stream_dir: Path) -> None:
        """Start local HTTP server for serving stream files."""
        from aiohttp import web
        import asyncio
        import time
        
        # Client connection tracking
        # Key: client identifier (IP:port or session), Value: {first_seen, last_seen, requests}
        active_clients: Dict[str, Dict[str, Any]] = {}
        client_timeout = 10.0  # Seconds before considering a client disconnected
        
        def get_client_id(request) -> str:
            """Get unique client identifier from request."""
            peername = request.transport.get_extra_info('peername') if request.transport else None
            if peername:
                return f"{peername[0]}:{peername[1]}"
            # Fallback to remote address header
            return request.headers.get('X-Forwarded-For', request.remote or 'unknown')
        
        def cleanup_stale_clients():
            """Remove clients that haven't made requests recently."""
            now = time.time()
            stale = [cid for cid, info in active_clients.items() 
                     if now - info['last_seen'] > client_timeout]
            for cid in stale:
                info = active_clients.pop(cid)
                logger.info(f"Stream client disconnected: {cid} (was connected for {now - info['first_seen']:.1f}s, {info['requests']} requests)")
        
        # Client tracking middleware (no per-request logging)
        @web.middleware
        async def track_clients(request, handler):
            client_id = get_client_id(request)
            now = time.time()
            
            # Cleanup stale clients periodically
            cleanup_stale_clients()
            
            # Track new client connection
            if client_id not in active_clients:
                active_clients[client_id] = {
                    'first_seen': now,
                    'last_seen': now,
                    'requests': 0
                }
                logger.info(f"Stream client connected: {client_id}")
            
            # Update client activity
            active_clients[client_id]['last_seen'] = now
            active_clients[client_id]['requests'] += 1
            
            try:
                return await handler(request)
            except web.HTTPException:
                raise
            except Exception as e:
                logger.error(f"Stream handler error for {client_id}: {e}")
                raise
        
        # Create app with client tracking middleware
        app = web.Application(middlewares=[track_clients])
        
        # Serve embedded web player (must be registered BEFORE the generic filename route)
        async def serve_player(request):
            stream_id_req = request.match_info['stream_id']
            
            # Get stream info for protocol/URLs and quality
            quality_label = "720p"  # Default
            if self._streaming_manager:
                stream_info = await self._streaming_manager.get_stream(stream_id_req)
                if stream_info:
                    protocol = stream_info.protocol.value if hasattr(stream_info.protocol, 'value') else str(stream_info.protocol)
                    hls_url = stream_info.hls_url or ''
                    dash_url = stream_info.dash_url or ''
                    # Get quality info from stream config
                    config = stream_info.config
                    profile_name = config.quality_profile.value if hasattr(config.quality_profile, 'value') else str(config.quality_profile)
                    quality_label = f"{profile_name.upper()} ({config.width}x{config.height}@{config.frame_rate}fps)"
                else:
                    protocol = 'hls'
                    hls_url = f"http://localhost:{self._local_stream_port}/streams/{stream_id_req}/playlist.m3u8"
                    dash_url = ''
            else:
                protocol = 'hls'
                hls_url = f"http://localhost:{self._local_stream_port}/streams/{stream_id_req}/playlist.m3u8"
                dash_url = ''
            
            # Render HTML using Jinja2 template with inline assets (embedded mode)
            from flybrowser.service.template_renderer import render_player_html
            html = render_player_html(
                stream_id=stream_id_req,
                protocol=protocol,
                hls_url=hls_url,
                dash_url=dash_url,
                quality=quality_label,
                inline_assets=True,
            )
            return web.Response(text=html, content_type='text/html')
        
        # Register player route FIRST (more specific)
        app.router.add_get('/streams/{stream_id}/player', serve_player)
        
        # Serve stream files (generic route - must be after player route)
        async def serve_stream_file(request):
            stream_id = request.match_info['stream_id']
            filename = request.match_info['filename']
            file_path = stream_dir / stream_id / filename
            
            if not file_path.exists():
                # Only log 404s at debug level - playlist retries are normal during startup
                logger.debug(f"Stream file not found: {file_path}")
                return web.Response(status=404)
            
            # Determine content type based on file extension
            if filename.endswith('.m3u8'):
                content_type = 'application/vnd.apple.mpegurl'
            elif filename.endswith('.ts'):
                content_type = 'video/mp2t'
            elif filename.endswith('.mpd'):
                content_type = 'application/dash+xml'
            else:
                content_type = 'application/octet-stream'
            
            # Read file content and return with proper headers
            with open(file_path, 'rb') as f:
                content = f.read()
            
            return web.Response(
                body=content,
                content_type=content_type,
                headers={
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                }
            )
        
        # Register filename route AFTER player route (less specific)
        app.router.add_get('/streams/{stream_id}/{filename}', serve_stream_file)
        
        # Find available port
        import socket
        sock = socket.socket()
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
        
        self._local_stream_port = port
        
        # Start server in background
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', port)
        await site.start()
        
        self._local_stream_server = runner
        logger.info(f"Local stream server started on http://localhost:{port}")
        logger.info(f"Serving streams from: {stream_dir}")
    
    async def _capture_frames_for_stream(self, stream_id: str) -> None:
        """Capture browser frames and send to stream.
        
        Note: Frame capture is handled by FFmpegRecorder in the StreamingSession.
        This method is kept for compatibility but is no longer needed.
        """
        # FFmpegRecorder handles frame capture automatically
        pass

    # Utility methods
    @property
    def mode(self) -> str:
        """Get the current mode (embedded or server)."""
        return self._mode

    @property
    def session_id(self) -> Optional[str]:
        """Get the session ID (server mode only)."""
        return self._session_id

    @property
    def endpoint(self) -> Optional[str]:
        """Get the server endpoint (server mode only)."""
        return self._endpoint

    def _extract_completion_data(
        self,
        result_dict: Dict[str, Any],
        task: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract and transform agent result data for the completion page.
        
        This method safely extracts tools_used, reasoning_steps, and other metadata
        from the agent result, handling both ReActStep objects and dict representations.
        All values are validated and transformed to ensure the template can render
        without errors.
        
        Args:
            result_dict: The raw result dictionary from agent execution
            task: The task description that was executed
            session_id: Optional session ID for metadata
            
        Returns:
            Dictionary of validated parameters for _load_completion_page()
        """
        # Safely extract tools_used from steps
        tools_used: List[Dict[str, Any]] = []
        reasoning_steps: List[Dict[str, Any]] = []
        
        steps_data = result_dict.get("steps") or []
        
        for step in steps_data:
            if step is None:
                continue
                
            try:
                # Handle ReActStep objects (have 'action' attribute)
                if hasattr(step, 'action') and step.action is not None:
                    # Extract action info
                    action_obj = step.action
                    tool_name = getattr(action_obj, 'tool_name', None) or 'unknown'
                    
                    # Determine success: check observation.success if available
                    success = True
                    if hasattr(step, 'observation') and step.observation is not None:
                        success = getattr(step.observation, 'success', True)
                    
                    tools_used.append({
                        "name": str(tool_name),
                        "duration_ms": int(getattr(step, 'duration_ms', 0) or 0),
                        "success": bool(success),
                    })
                    
                    # Extract thought if available
                    if hasattr(step, 'thought') and step.thought is not None:
                        thought_obj = step.thought
                        thought_content = getattr(thought_obj, 'content', None)
                        if thought_content is None:
                            thought_content = str(thought_obj)
                        
                        reasoning_steps.append({
                            "thought": str(thought_content),
                            "action": str(tool_name),
                        })
                
                # Handle dict representations
                elif isinstance(step, dict):
                    action = step.get("action")
                    if action and isinstance(action, dict):
                        tool_name = action.get("tool_name") or "unknown"
                        
                        # Determine success from observation if available
                        observation_success = step.get("observation_success")
                        success = observation_success if observation_success is not None else True
                        
                        tools_used.append({
                            "name": str(tool_name),
                            "duration_ms": int(step.get("duration_ms", 0) or 0),
                            "success": bool(success),
                        })
                        
                        # Extract thought if available
                        thought = step.get("thought")
                        if thought:
                            reasoning_steps.append({
                                "thought": str(thought),
                                "action": str(tool_name),
                            })
                            
            except Exception as e:
                # Log but don't fail - skip malformed steps
                logger.debug(f"[COMPLETION] Skipped malformed step: {e}")
                continue
        
        # Safely extract LLM usage - ensure all fields have defaults
        raw_llm_usage = result_dict.get("llm_usage")
        llm_usage: Optional[Dict[str, Any]] = None
        
        if raw_llm_usage and isinstance(raw_llm_usage, dict):
            llm_usage = {
                "model": str(raw_llm_usage.get("model") or "unknown"),
                "provider": str(raw_llm_usage.get("provider") or "unknown"),
                "prompt_tokens": int(raw_llm_usage.get("prompt_tokens") or 0),
                "completion_tokens": int(raw_llm_usage.get("completion_tokens") or 0),
                "total_tokens": int(raw_llm_usage.get("total_tokens") or 0),
                "cost_usd": float(raw_llm_usage.get("cost_usd") or 0.0),
                "request_count": int(raw_llm_usage.get("calls_count") or raw_llm_usage.get("request_count") or 1),
                "avg_latency_ms": float(raw_llm_usage.get("avg_latency_ms") or 0.0),
            }
        
        # Build metadata with safe defaults
        metadata = {
            "session_id": str(session_id or "N/A"),
            "reasoning_strategy": str(result_dict.get("final_state", "completed")),
            "stop_reason": "completed" if result_dict.get("success") else "error",
        }
        
        # Build the complete parameter dict for _load_completion_page
        return {
            "success": bool(result_dict.get("success", False)),
            "task": str(task),
            "duration_ms": float(result_dict.get("execution_time_ms", 0) or 0),
            "iterations": int(result_dict.get("total_iterations", 0) or 0),
            "result_data": result_dict.get("result"),
            "error_message": result_dict.get("error"),
            "max_iterations": result_dict.get("max_iterations"),
            "llm_usage": llm_usage,
            "tools_used": tools_used,  # Always a list (may be empty)
            "reasoning_steps": reasoning_steps,  # Always a list (may be empty)
            "metadata": metadata,
            "error_traceback": result_dict.get("error_traceback"),
        }

    def _ensure_started(self) -> None:
        """Ensure FlyBrowser has been started."""
        if not self._started:
            raise RuntimeError("FlyBrowser not started. Call start() first or use async context manager.")

    async def __aenter__(self) -> "FlyBrowser":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()
