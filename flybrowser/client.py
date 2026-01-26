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
FlyBrowser HTTP Client SDK with Cluster Support.

This module provides a client SDK for connecting to FlyBrowser servers,
with built-in support for:
- Cluster mode with automatic leader discovery
- Transparent failover and retry logic
- Connection pooling
- Circuit breaker pattern
- Session affinity

The client works identically for standalone and cluster deployments.

Example:
    >>> from flybrowser.client import FlyBrowserClient
    >>> 
    >>> async with FlyBrowserClient("http://localhost:8000") as client:
    ...     session = await client.create_session()
    ...     await client.navigate(session.session_id, "https://example.com")
    ...     data = await client.extract(session.session_id, "Get the title")
    ...     await client.close_session(session.session_id)
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin

import aiohttp

from flybrowser.utils.logger import logger


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance.
    
    Prevents cascading failures by stopping requests to failing endpoints.
    """
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    half_open_calls: int = 0
    
    def record_success(self) -> None:
        """Record a successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.half_open_calls = 0
    
    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker opened due to failures")
    
    def can_execute(self) -> bool:
        """Check if a call can be executed."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False
        
        return False


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 10.0
    exponential_base: float = 2.0
    jitter: bool = True
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a retry attempt."""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        if self.jitter:
            delay *= (0.5 + random.random())
        return delay


@dataclass
class ClusterNode:
    """Information about a cluster node."""
    address: str
    is_leader: bool = False
    is_healthy: bool = True
    last_check: float = field(default_factory=time.time)
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)


class FlyBrowserClient:
    """HTTP client for FlyBrowser with cluster support.
    
    Provides transparent access to FlyBrowser servers in both standalone
    and cluster modes. Handles:
    - Automatic leader discovery in cluster mode
    - Transparent failover when nodes fail
    - Connection pooling for efficiency
    - Circuit breaker for fault tolerance
    - Retry logic with exponential backoff
    
    Example:
        >>> async with FlyBrowserClient("http://localhost:8000") as client:
        ...     # Create a session
        ...     session = await client.create_session()
        ...     
        ...     # Navigate
        ...     await client.navigate(session["session_id"], "https://example.com")
        ...     
        ...     # Extract data
        ...     data = await client.extract(session["session_id"], "Get the title")
        ...     
        ...     # Close session
        ...     await client.close_session(session["session_id"])
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        retry_config: Optional[RetryConfig] = None,
        auto_discover: bool = True,
    ) -> None:
        """Initialize the client.
        
        Args:
            endpoint: Server endpoint URL (e.g., "http://localhost:8000")
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            retry_config: Configuration for retry logic
            auto_discover: Whether to auto-discover cluster nodes
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        self.auto_discover = auto_discover
        
        # Cluster state
        self._nodes: Dict[str, ClusterNode] = {}
        self._leader_address: Optional[str] = None
        self._session_routes: Dict[str, str] = {}  # session_id -> node_address

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None
        self._started = False

    # ==================== Lifecycle ====================

    async def start(self) -> None:
        """Start the client and discover cluster topology."""
        if self._started:
            return

        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=20,
            keepalive_timeout=30,
        )

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        )

        self._started = True

        # Add initial endpoint as a node
        self._nodes[self.endpoint] = ClusterNode(address=self.endpoint)

        # Discover cluster topology
        if self.auto_discover:
            await self._discover_cluster()

    async def stop(self) -> None:
        """Stop the client and close connections."""
        if not self._started:
            return

        self._started = False

        if self._session:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "FlyBrowserClient":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

    # ==================== Cluster Discovery ====================

    async def _discover_cluster(self) -> None:
        """Discover cluster nodes and leader."""
        try:
            # Get cluster status from initial endpoint
            response = await self._request("GET", "/cluster/status", follow_redirects=False)

            if response:
                # Update leader info
                if response.get("is_leader"):
                    self._leader_address = self.endpoint
                elif response.get("leader_id"):
                    # Try to get leader address from nodes
                    nodes_response = await self._request("GET", "/cluster/nodes")
                    if nodes_response and "nodes" in nodes_response:
                        for node in nodes_response["nodes"]:
                            addr = f"http://{node['api_address']}"
                            self._nodes[addr] = ClusterNode(
                                address=addr,
                                is_leader=(node["node_id"] == response.get("leader_id")),
                            )
                            if node["node_id"] == response.get("leader_id"):
                                self._leader_address = addr
        except Exception as e:
            logger.debug(f"Cluster discovery failed (standalone mode?): {e}")
            # Assume standalone mode
            self._leader_address = self.endpoint

    async def _refresh_leader(self) -> None:
        """Refresh leader information."""
        await self._discover_cluster()

    def _get_leader_url(self) -> str:
        """Get the current leader's URL."""
        return self._leader_address or self.endpoint

    def _get_session_url(self, session_id: str) -> str:
        """Get the URL for a specific session."""
        if session_id in self._session_routes:
            return self._session_routes[session_id]
        return self._get_leader_url()

    # ==================== HTTP Methods ====================

    async def _request(
        self,
        method: str,
        path: str,
        base_url: Optional[str] = None,
        json: Optional[Dict[str, Any]] = None,
        follow_redirects: bool = True,
        retry: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Make an HTTP request with retry and failover logic."""
        if not self._session:
            raise RuntimeError("Client not started")

        url = urljoin(base_url or self._get_leader_url(), path)
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                async with self._session.request(
                    method,
                    url,
                    json=json,
                    headers=headers,
                    allow_redirects=False,
                ) as response:
                    # Handle redirects (leader changed)
                    if response.status in (307, 308) and follow_redirects:
                        redirect_url = response.headers.get("Location")
                        if redirect_url:
                            # Update leader and retry
                            self._leader_address = redirect_url.rsplit("/", 1)[0]
                            url = redirect_url
                            continue

                    if response.status == 200:
                        return await response.json()
                    elif response.status == 503:
                        # Service unavailable, retry
                        if retry and attempt < self.retry_config.max_retries:
                            await asyncio.sleep(self.retry_config.get_delay(attempt))
                            await self._refresh_leader()
                            continue
                    elif response.status >= 400:
                        error = await response.json()
                        raise Exception(f"API error: {error}")

            except aiohttp.ClientError as e:
                if retry and attempt < self.retry_config.max_retries:
                    await asyncio.sleep(self.retry_config.get_delay(attempt))
                    await self._refresh_leader()
                    continue
                raise

        return None

    # ==================== Session API ====================

    async def create_session(
        self,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        headless: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a new browser session.

        Args:
            llm_provider: LLM provider name
            llm_model: LLM model name
            api_key: LLM API key
            headless: Whether to run headless
            **kwargs: Additional session options

        Returns:
            Session info including session_id and node_address
        """
        data = {
            "llm_provider": llm_provider,
            "headless": headless,
            **kwargs,
        }
        if llm_model:
            data["llm_model"] = llm_model
        if api_key:
            data["api_key"] = api_key

        response = await self._request("POST", "/sessions", json=data)

        if response:
            # Store session route for affinity
            session_id = response.get("session_id")
            node_address = response.get("node_address")
            if session_id and node_address:
                self._session_routes[session_id] = f"http://{node_address}"

        return response or {}

    async def close_session(self, session_id: str) -> bool:
        """Close a browser session.

        Args:
            session_id: Session to close

        Returns:
            True if closed successfully
        """
        response = await self._request(
            "DELETE",
            f"/sessions/{session_id}",
            base_url=self._get_session_url(session_id),
        )

        # Remove from routes
        self._session_routes.pop(session_id, None)

        return response is not None

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session information.

        Args:
            session_id: Session ID

        Returns:
            Session info
        """
        response = await self._request(
            "GET",
            f"/sessions/{session_id}",
            base_url=self._get_session_url(session_id),
        )
        return response or {}

    # ==================== Navigation API ====================

    async def navigate(self, session_id: str, url: str) -> Dict[str, Any]:
        """Navigate to a URL.

        Args:
            session_id: Session ID
            url: URL to navigate to

        Returns:
            Navigation result
        """
        return await self._request(
            "POST",
            f"/sessions/{session_id}/navigate",
            base_url=self._get_session_url(session_id),
            json={"url": url},
        ) or {}

    async def extract(
        self,
        session_id: str,
        instruction: str,
        schema: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract data from the current page.

        Args:
            session_id: Session ID
            instruction: Natural language extraction instruction
            schema: Optional JSON schema for structured extraction
            context: Additional context to inform extraction

        Returns:
            Extracted data
        """
        data = {"instruction": instruction, "context": context or {}}
        if schema:
            data["schema"] = schema

        return await self._request(
            "POST",
            f"/sessions/{session_id}/extract",
            base_url=self._get_session_url(session_id),
            json=data,
        ) or {}

    async def action(
        self,
        session_id: str,
        instruction: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform an action on the page.

        Args:
            session_id: Session ID
            instruction: Natural language action instruction
            context: Additional context for the action (form_data, files, etc.)

        Returns:
            Action result
        """
        return await self._request(
            "POST",
            f"/sessions/{session_id}/action",
            base_url=self._get_session_url(session_id),
            json={"instruction": instruction, "context": context or {}},
        ) or {}

    async def screenshot(
        self,
        session_id: str,
        full_page: bool = False,
    ) -> Dict[str, Any]:
        """Take a screenshot.

        Args:
            session_id: Session ID
            full_page: Whether to capture full page

        Returns:
            Screenshot data
        """
        return await self._request(
            "POST",
            f"/sessions/{session_id}/screenshot",
            base_url=self._get_session_url(session_id),
            json={"full_page": full_page},
        ) or {}

    # ==================== Workflow API ====================

    async def run_workflow(
        self,
        session_id: str,
        workflow: Dict[str, Any],
        variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a workflow.

        Args:
            session_id: Session ID
            workflow: Workflow definition
            variables: Variables for the workflow

        Returns:
            Workflow execution result
        """
        return await self._request(
            "POST",
            f"/sessions/{session_id}/workflow",
            base_url=self._get_session_url(session_id),
            json={"workflow": workflow, "variables": variables or {}},
        ) or {}

    async def monitor(
        self,
        session_id: str,
        condition: str,
        timeout: float = 30.0,
        poll_interval: float = 0.5,
    ) -> Dict[str, Any]:
        """Monitor for a condition.

        Args:
            session_id: Session ID
            condition: Condition to monitor for
            timeout: Maximum wait time
            poll_interval: Time between checks

        Returns:
            Monitoring result
        """
        return await self._request(
            "POST",
            f"/sessions/{session_id}/monitor",
            base_url=self._get_session_url(session_id),
            json={"condition": condition, "timeout": timeout, "poll_interval": poll_interval},
        ) or {}

    async def secure_fill(
        self,
        session_id: str,
        selector: str,
        credential_id: str,
        clear_first: bool = True,
    ) -> Dict[str, Any]:
        """Securely fill a form field.

        Args:
            session_id: Session ID
            selector: CSS selector for the field
            credential_id: Credential ID to use
            clear_first: Whether to clear field first

        Returns:
            Fill result
        """
        return await self._request(
            "POST",
            f"/sessions/{session_id}/secure-fill",
            base_url=self._get_session_url(session_id),
            json={"selector": selector, "credential_id": credential_id, "clear_first": clear_first},
        ) or {}

    async def store_credential(
        self,
        session_id: str,
        name: str,
        value: str,
        pii_type: str = "password",
    ) -> Dict[str, Any]:
        """Store a credential securely.

        Args:
            session_id: Session ID
            name: Credential name/identifier
            value: Credential value
            pii_type: Type of PII (password, email, etc.)

        Returns:
            Credential info including credential_id
        """
        return await self._request(
            "POST",
            f"/sessions/{session_id}/credentials",
            base_url=self._get_session_url(session_id),
            json={"name": name, "value": value, "pii_type": pii_type},
        ) or {}

    async def navigate_nl(
        self,
        session_id: str,
        instruction: str,
        use_vision: bool = True,
    ) -> Dict[str, Any]:
        """Navigate using natural language.

        Args:
            session_id: Session ID
            instruction: Natural language navigation instruction
            use_vision: Whether to use vision for element detection

        Returns:
            Navigation result
        """
        return await self._request(
            "POST",
            f"/sessions/{session_id}/navigate-nl",
            base_url=self._get_session_url(session_id),
            json={"instruction": instruction, "use_vision": use_vision},
        ) or {}

    async def mask_pii(self, text: str) -> Dict[str, Any]:
        """Mask PII in text.

        Args:
            text: Text that may contain PII

        Returns:
            Masked text result
        """
        return await self._request(
            "POST",
            "/pii/mask",
            json={"text": text},
        ) or {}

    # ==================== Agent Mode API (Primary Interface) ====================

    async def agent(
        self,
        session_id: str,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 50,
        max_time_seconds: float = 1800.0,
    ) -> Dict[str, Any]:
        """Execute a task using the intelligent agent.

        This is the primary and recommended method for complex browser automation.
        The agent automatically selects the optimal reasoning strategy and adapts
        dynamically during execution.

        Args:
            session_id: Session ID
            task: Natural language description of what to accomplish
            context: User-provided context (preferences, constraints, form data)
            max_iterations: Maximum action iterations (default: 50)
            max_time_seconds: Maximum execution time (default: 1800)

        Returns:
            Agent execution result including:
            - success: Whether task was completed
            - result_data: Extracted data or confirmations
            - iterations: Number of iterations executed
            - duration_seconds: Execution time
            - execution_history: Summary of actions taken
        """
        return await self._request(
            "POST",
            f"/sessions/{session_id}/agent",
            base_url=self._get_session_url(session_id),
            json={
                "task": task,
                "context": context or {},
                "max_iterations": max_iterations,
                "max_time_seconds": max_time_seconds,
            },
            timeout=max_time_seconds + 60,
        ) or {}

    async def observe(
        self,
        session_id: str,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        return_selectors: bool = True,
    ) -> Dict[str, Any]:
        """Observe and identify elements on the page.

        Analyzes the page to find elements matching a natural language description.

        Args:
            session_id: Session ID
            query: Natural language description of what to find
            context: Additional context for element search
            return_selectors: Include CSS selectors in response (default: True)

        Returns:
            Observe result including:
            - success: Whether elements were found
            - elements: List of found elements with selectors
            - page_url: Current page URL
        """
        return await self._request(
            "POST",
            f"/sessions/{session_id}/observe",
            base_url=self._get_session_url(session_id),
            json={"query": query, "return_selectors": return_selectors, "context": context or {}},
        ) or {}

    # ==================== Cluster API ====================

    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status.

        Returns:
            Cluster status info
        """
        return await self._request("GET", "/cluster/status") or {}

    async def get_cluster_nodes(self) -> List[Dict[str, Any]]:
        """Get list of cluster nodes.

        Returns:
            List of node info
        """
        response = await self._request("GET", "/cluster/nodes")
        return response.get("nodes", []) if response else []

    async def health_check(self) -> bool:
        """Check if the server is healthy.

        Returns:
            True if healthy
        """
        try:
            response = await self._request("GET", "/health", retry=False)
            return response is not None and response.get("status") == "healthy"
        except Exception:
            return False

