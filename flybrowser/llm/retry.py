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
Retry logic for LLM requests with exponential backoff.

This module provides robust retry handling for LLM API requests. It implements
exponential backoff with jitter to handle transient failures gracefully.

Features:
- Exponential backoff (delays increase exponentially)
- Jitter (randomization to avoid thundering herd)
- Configurable retry attempts and delays
- Smart error detection (retries only on transient errors)
- Detailed logging of retry attempts

Common retryable errors:
- Rate limit errors (429)
- Timeout errors
- Connection errors
- Server errors (5xx)

Non-retryable errors (fail immediately):
- Authentication errors (401, 403)
- Invalid request errors (400)
- Not found errors (404)

Example:
    >>> from flybrowser.llm.config import RetryConfig
    >>> config = RetryConfig(
    ...     max_retries=3,
    ...     initial_delay=1.0,
    ...     max_delay=60.0,
    ...     exponential_base=2.0,
    ...     jitter=True
    ... )
    >>> handler = RetryHandler(config)
    >>>
    >>> async def make_request():
    ...     # Your API call here
    ...     pass
    >>>
    >>> result = await handler.execute_with_retry(make_request)
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Callable, Optional, Type, TypeVar

from flybrowser.exceptions import LLMProviderError
from flybrowser.llm.config import RetryConfig
from flybrowser.utils.logger import logger

T = TypeVar("T")


class RetryHandler:
    """
    Handles retry logic with exponential backoff and jitter.

    This class implements a robust retry mechanism for LLM API requests.
    It uses exponential backoff to gradually increase delays between retries
    and adds jitter to prevent thundering herd problems.

    Attributes:
        config: Retry configuration settings

    Example:
        >>> handler = RetryHandler(RetryConfig(max_retries=3))
        >>>
        >>> async def api_call():
        ...     response = await client.generate("prompt")
        ...     return response
        >>>
        >>> # Automatically retries on transient failures
        >>> result = await handler.execute_with_retry(api_call)
    """

    def __init__(self, config: RetryConfig) -> None:
        """
        Initialize retry handler with configuration.

        Args:
            config: Retry configuration containing:
                - max_retries: Maximum number of retry attempts
                - initial_delay: Initial delay in seconds (default: 1.0)
                - max_delay: Maximum delay in seconds (default: 60.0)
                - exponential_base: Base for exponential backoff (default: 2.0)
                - jitter: Whether to add randomization to delays (default: True)

        Example:
            >>> config = RetryConfig(
            ...     max_retries=3,
            ...     initial_delay=1.0,
            ...     max_delay=60.0,
            ...     exponential_base=2.0,
            ...     jitter=True
            ... )
            >>> handler = RetryHandler(config)
        """
        self.config = config

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a retry attempt.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = min(
            self.config.initial_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay,
        )

        # Add jitter if enabled
        if self.config.jitter:
            delay = delay * (0.5 + random.random())

        return delay

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.

        Args:
            error: Exception that occurred

        Returns:
            True if error is retryable
        """
        # Check for common retryable errors
        error_str = str(error).lower()
        
        retryable_patterns = [
            "rate limit",
            "timeout",
            "connection",
            "server error",
            "503",
            "502",
            "500",
            "429",
            "overloaded",
        ]

        return any(pattern in error_str for pattern in retryable_patterns)

    async def execute_with_retry(
        self,
        func: Callable[..., T],
        *args: Any,
        retryable_exceptions: Optional[tuple[Type[Exception], ...]] = None,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            retryable_exceptions: Tuple of exception types to retry on
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            Exception: If all retries are exhausted
        """
        if retryable_exceptions is None:
            retryable_exceptions = (LLMProviderError, asyncio.TimeoutError)

        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"Request succeeded after {attempt} retries")
                
                return result

            except retryable_exceptions as e:
                last_exception = e

                # Check if we should retry
                if attempt >= self.config.max_retries:
                    logger.error(
                        f"Max retries ({self.config.max_retries}) exceeded. "
                        f"Last error: {e}"
                    )
                    break

                if not self._is_retryable_error(e):
                    logger.warning(f"Non-retryable error encountered: {e}")
                    raise

                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)

            except Exception as e:
                # Non-retryable exception
                logger.error(f"Non-retryable exception: {e}")
                raise

        # All retries exhausted
        raise LLMProviderError(
            f"Request failed after {self.config.max_retries} retries. "
            f"Last error: {last_exception}"
        ) from last_exception

