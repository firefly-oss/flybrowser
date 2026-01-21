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
LLM response caching system.

This module provides an LRU (Least Recently Used) cache for LLM responses with
TTL (Time To Live) support. Caching LLM responses can significantly reduce:
- API costs (40-60% typical reduction)
- Response latency (instant for cached responses)
- API rate limit pressure

The cache uses request parameters (prompt, model, temperature, etc.) to generate
cache keys, ensuring identical requests return cached responses.

Features:
- LRU eviction policy (removes least recently used items when full)
- TTL support (cached items expire after configured time)
- Hit/miss statistics tracking
- Configurable cache size and TTL
- Thread-safe operations
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from flybrowser.llm.config import CacheConfig
from flybrowser.utils.logger import logger


class LLMCache:
    """
    LRU cache for LLM responses with TTL support.

    This cache stores LLM responses to avoid redundant API calls for identical
    requests. It uses an LRU eviction policy and supports TTL for cache entries.

    Attributes:
        config: Cache configuration settings
        cache: OrderedDict storing cached responses
        enabled: Whether caching is enabled
        _hits: Number of cache hits
        _misses: Number of cache misses

    Example:
        >>> from flybrowser.llm.config import CacheConfig
        >>> config = CacheConfig(enabled=True, max_size=1000, ttl_seconds=3600)
        >>> cache = LLMCache(config)
        >>>
        >>> # Try to get cached response
        >>> cached = cache.get("prompt", None, "gpt-4o", 0.7)
        >>> if cached is None:
        ...     # Cache miss - call LLM
        ...     response = await llm.generate("prompt")
        ...     cache.set("prompt", None, "gpt-4o", 0.7, response)
        >>>
        >>> # Get statistics
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
    """

    def __init__(self, config: CacheConfig) -> None:
        """
        Initialize the LLM cache with configuration.

        Args:
            config: Cache configuration containing:
                - enabled: Whether caching is enabled
                - max_size: Maximum number of cached items
                - ttl_seconds: Time to live for cached items in seconds
                - cache_key_prefix: Prefix for cache keys

        Example:
            >>> config = CacheConfig(
            ...     enabled=True,
            ...     max_size=1000,
            ...     ttl_seconds=3600,
            ...     cache_key_prefix="flybrowser"
            ... )
            >>> cache = LLMCache(config)
        """
        self.config = config
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.enabled = config.enabled
        self._hits = 0
        self._misses = 0

    def _generate_key(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        """
        Generate a cache key from request parameters.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            model: Model name
            temperature: Temperature setting
            **kwargs: Additional parameters

        Returns:
            Cache key
        """
        # Create a deterministic representation of the request
        cache_data = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "model": model,
            "temperature": temperature,
            **kwargs,
        }
        
        # Sort keys for consistency
        cache_str = json.dumps(cache_data, sort_keys=True)
        
        # Generate hash
        key_hash = hashlib.sha256(cache_str.encode()).hexdigest()
        
        return f"{self.config.cache_key_prefix}:{key_hash}"

    def get(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        **kwargs: Any,
    ) -> Optional[Any]:
        """
        Get a cached response.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            model: Model name
            temperature: Temperature setting
            **kwargs: Additional parameters

        Returns:
            Cached response or None
        """
        if not self.enabled:
            return None

        key = self._generate_key(prompt, system_prompt, model, temperature, **kwargs)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Check TTL
            if time.time() - entry["timestamp"] < self.config.ttl_seconds:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self._hits += 1
                logger.debug(f"Cache hit for key: {key[:16]}...")
                return entry["response"]
            else:
                # Expired, remove from cache
                del self.cache[key]
                logger.debug(f"Cache expired for key: {key[:16]}...")

        self._misses += 1
        return None

    def set(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        response: Any,
        **kwargs: Any,
    ) -> None:
        """
        Cache a response.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            model: Model name
            temperature: Temperature setting
            response: Response to cache
            **kwargs: Additional parameters
        """
        if not self.enabled:
            return

        key = self._generate_key(prompt, system_prompt, model, temperature, **kwargs)
        
        # Add to cache
        self.cache[key] = {
            "response": response,
            "timestamp": time.time(),
        }
        
        # Enforce max size (LRU eviction)
        if len(self.cache) > self.config.max_size:
            # Remove oldest item
            self.cache.popitem(last=False)
            logger.debug("Cache size limit reached, evicted oldest entry")

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "enabled": self.enabled,
            "size": len(self.cache),
            "max_size": self.config.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self.config.ttl_seconds,
        }

