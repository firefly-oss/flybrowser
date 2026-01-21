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
Rate limiting for LLM requests.

This module provides rate limiting functionality to prevent exceeding API rate limits
and to control costs. It implements a token bucket algorithm with support for:
- Requests per minute limiting
- Tokens per minute limiting
- Concurrent request limiting

Rate limiting helps:
- Avoid API rate limit errors (429)
- Control API costs
- Prevent service overload
- Ensure fair resource usage

The rate limiter tracks requests and token usage over a sliding window and
automatically delays requests when limits would be exceeded.

Example:
    >>> from flybrowser.llm.config import RateLimitConfig
    >>> config = RateLimitConfig(
    ...     requests_per_minute=60,
    ...     tokens_per_minute=90000,
    ...     concurrent_requests=10
    ... )
    >>> limiter = RateLimiter(config)
    >>>
    >>> # Acquire permission before making request
    >>> await limiter.acquire(estimated_tokens=1000)
    >>> try:
    ...     response = await client.generate("prompt")
    ... finally:
    ...     limiter.release(actual_tokens=response.usage["total_tokens"])
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Deque, Optional

from flybrowser.llm.config import RateLimitConfig
from flybrowser.utils.logger import logger


class RateLimiter:
    """
    Token bucket rate limiter for LLM requests.

    This class implements rate limiting using a token bucket algorithm with
    sliding window tracking. It enforces limits on:
    1. Requests per minute
    2. Tokens per minute
    3. Concurrent requests

    Attributes:
        config: Rate limit configuration
        request_timestamps: Deque of recent request timestamps
        token_usage: Deque of recent token usage (timestamp, tokens)
        active_requests: Number of currently active requests
        semaphore: Asyncio semaphore for concurrent request limiting

    Example:
        >>> limiter = RateLimiter(RateLimitConfig(
        ...     requests_per_minute=60,
        ...     tokens_per_minute=90000
        ... ))
        >>>
        >>> # Use as context manager
        >>> async with limiter:
        ...     response = await client.generate("prompt")
        >>>
        >>> # Or manually
        >>> await limiter.acquire(estimated_tokens=1000)
        >>> try:
        ...     response = await client.generate("prompt")
        ... finally:
        ...     limiter.release(actual_tokens=1500)
    """

    def __init__(self, config: RateLimitConfig) -> None:
        """
        Initialize rate limiter with configuration.

        Args:
            config: Rate limit configuration containing:
                - requests_per_minute: Maximum requests per minute (optional)
                - tokens_per_minute: Maximum tokens per minute (optional)
                - concurrent_requests: Maximum concurrent requests (default: 10)

        Example:
            >>> config = RateLimitConfig(
            ...     requests_per_minute=60,  # 1 request per second
            ...     tokens_per_minute=90000,  # 90k tokens per minute
            ...     concurrent_requests=10    # Max 10 concurrent
            ... )
            >>> limiter = RateLimiter(config)
        """
        self.config = config

        # Request rate limiting (sliding window)
        self.request_timestamps: Deque[float] = deque()

        # Token rate limiting (sliding window)
        self.token_usage: Deque[tuple[float, int]] = deque()

        # Concurrent request limiting
        self.active_requests = 0
        self.semaphore = asyncio.Semaphore(config.concurrent_requests)

    async def acquire(self, estimated_tokens: Optional[int] = None) -> None:
        """
        Acquire permission to make a request.

        Args:
            estimated_tokens: Estimated number of tokens for the request

        Raises:
            asyncio.TimeoutError: If rate limit cannot be satisfied
        """
        # Wait for concurrent request slot
        await self.semaphore.acquire()
        self.active_requests += 1

        try:
            # Check request rate limit
            if self.config.requests_per_minute:
                await self._check_request_rate()

            # Check token rate limit
            if self.config.tokens_per_minute and estimated_tokens:
                await self._check_token_rate(estimated_tokens)

        except Exception:
            # Release semaphore if rate limit check fails
            self.active_requests -= 1
            self.semaphore.release()
            raise

    def release(self, actual_tokens: Optional[int] = None) -> None:
        """
        Release a request slot.

        Args:
            actual_tokens: Actual number of tokens used
        """
        self.active_requests -= 1
        self.semaphore.release()

        # Record actual token usage if provided
        if actual_tokens and self.config.tokens_per_minute:
            self.token_usage.append((time.time(), actual_tokens))

    async def _check_request_rate(self) -> None:
        """Check and enforce request rate limit."""
        now = time.time()
        cutoff = now - 60  # 1 minute ago

        # Remove old timestamps
        while self.request_timestamps and self.request_timestamps[0] < cutoff:
            self.request_timestamps.popleft()

        # Check if we're at the limit
        if len(self.request_timestamps) >= self.config.requests_per_minute:
            # Calculate wait time
            oldest = self.request_timestamps[0]
            wait_time = 60 - (now - oldest)
            
            if wait_time > 0:
                logger.warning(
                    f"Request rate limit reached. Waiting {wait_time:.2f}s..."
                )
                await asyncio.sleep(wait_time)
                # Recursively check again
                await self._check_request_rate()
                return

        # Record this request
        self.request_timestamps.append(now)

    async def _check_token_rate(self, tokens: int) -> None:
        """
        Check and enforce token rate limit.

        Args:
            tokens: Number of tokens for this request
        """
        now = time.time()
        cutoff = now - 60  # 1 minute ago

        # Remove old usage records
        while self.token_usage and self.token_usage[0][0] < cutoff:
            self.token_usage.popleft()

        # Calculate current token usage
        current_tokens = sum(t for _, t in self.token_usage)

        # Check if adding this request would exceed limit
        if current_tokens + tokens > self.config.tokens_per_minute:
            # Calculate wait time
            if self.token_usage:
                oldest_time = self.token_usage[0][0]
                wait_time = 60 - (now - oldest_time)
                
                if wait_time > 0:
                    logger.warning(
                        f"Token rate limit reached ({current_tokens}/{self.config.tokens_per_minute}). "
                        f"Waiting {wait_time:.2f}s..."
                    )
                    await asyncio.sleep(wait_time)
                    # Recursively check again
                    await self._check_token_rate(tokens)
                    return

    def get_stats(self) -> dict:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary with rate limiter stats
        """
        now = time.time()
        cutoff = now - 60

        # Clean up old data
        while self.request_timestamps and self.request_timestamps[0] < cutoff:
            self.request_timestamps.popleft()
        while self.token_usage and self.token_usage[0][0] < cutoff:
            self.token_usage.popleft()

        current_tokens = sum(t for _, t in self.token_usage)

        return {
            "active_requests": self.active_requests,
            "max_concurrent": self.config.concurrent_requests,
            "requests_last_minute": len(self.request_timestamps),
            "requests_per_minute_limit": self.config.requests_per_minute,
            "tokens_last_minute": current_tokens,
            "tokens_per_minute_limit": self.config.tokens_per_minute,
        }

