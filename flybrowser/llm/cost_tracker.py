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
Cost tracking for LLM usage.

This module provides cost tracking functionality for LLM API usage. It tracks
token usage and calculates costs based on current pricing for different providers
and models.

Features:
- Automatic cost calculation based on token usage
- Support for multiple providers (OpenAI, Anthropic, etc.)
- Per-request and aggregate cost tracking
- Usage statistics and reporting
- Budget alerts and warnings
- Export to CSV/JSON for analysis

The pricing table is updated as of January 2026 and should be periodically
reviewed for accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from flybrowser.llm.config import CostTrackingConfig, LLMProviderType
from flybrowser.utils.logger import logger


# Pricing per 1M tokens (as of January 2026)
# Source: Official provider pricing pages
# Note: Prices may change - update periodically
PRICING_TABLE = {
    LLMProviderType.OPENAI: {
        # GPT-5 series (latest)
        "gpt-5.2": {"input": 5.00, "output": 20.00},
        "gpt-5-mini": {"input": 1.00, "output": 4.00},
        "gpt-5-nano": {"input": 0.25, "output": 1.00},
        "gpt-5": {"input": 4.00, "output": 16.00},
        # GPT-4 series (legacy but still available)
        "gpt-4.1": {"input": 3.00, "output": 12.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    },
    LLMProviderType.ANTHROPIC: {
        # Claude 4.5 series (latest)
        "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
        "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00},
        # Claude 3.5 series (legacy)
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    },
    # Local models have zero cost
    LLMProviderType.OLLAMA: {},
    LLMProviderType.LM_STUDIO: {},
    LLMProviderType.LOCAL_AI: {},
    LLMProviderType.VLLM: {},
}


@dataclass
class UsageRecord:
    """
    Record of a single LLM API usage.

    This dataclass stores detailed information about each LLM request
    for cost tracking and analysis purposes.

    Attributes:
        timestamp: When the request was made
        provider: LLM provider name (e.g., "openai", "anthropic")
        model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used (prompt + completion)
        cost: Calculated cost in USD for this request
        cached: Whether this response was served from cache (zero cost)
        metadata: Additional metadata about the request

    Example:
        >>> record = UsageRecord(
        ...     timestamp=datetime.now(),
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     prompt_tokens=100,
        ...     completion_tokens=50,
        ...     total_tokens=150,
        ...     cost=0.000625,
        ...     cached=False
        ... )
    """

    timestamp: datetime
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    cached: bool = False
    metadata: Dict = field(default_factory=dict)


class CostTracker:
    """
    Tracks LLM usage and calculates costs.

    This class monitors all LLM API calls and calculates costs based on
    token usage and current pricing. It provides statistics, alerts, and
    export capabilities for cost analysis.

    Attributes:
        config: Cost tracking configuration
        records: List of all usage records
        _total_cost: Cumulative cost across all requests
        _total_tokens: Cumulative tokens across all requests
        _total_requests: Total number of requests tracked

    Example:
        >>> from flybrowser.llm.config import CostTrackingConfig, LLMProviderType
        >>> config = CostTrackingConfig(
        ...     enabled=True,
        ...     budget_limit_usd=10.0,
        ...     alert_threshold_usd=8.0
        ... )
        >>> tracker = CostTracker(config)
        >>>
        >>> # Track a request
        >>> cost = tracker.calculate_cost(
        ...     LLMProviderType.OPENAI,
        ...     "gpt-4o",
        ...     prompt_tokens=100,
        ...     completion_tokens=50
        ... )
        >>> tracker.track_usage(
        ...     LLMProviderType.OPENAI,
        ...     "gpt-4o",
        ...     prompt_tokens=100,
        ...     completion_tokens=50
        ... )
        >>>
        >>> # Get summary
        >>> summary = tracker.get_summary()
        >>> print(f"Total cost: ${summary['total_cost']:.4f}")
    """

    def __init__(self, config: CostTrackingConfig) -> None:
        """
        Initialize the cost tracker with configuration.

        Args:
            config: Cost tracking configuration containing:
                - enabled: Whether cost tracking is enabled
                - budget_limit_usd: Maximum budget in USD (optional)
                - alert_threshold_usd: Alert when cost exceeds this (optional)
                - track_by_session: Whether to track costs per session

        Example:
            >>> config = CostTrackingConfig(
            ...     enabled=True,
            ...     budget_limit_usd=100.0,
            ...     alert_threshold_usd=80.0
            ... )
            >>> tracker = CostTracker(config)
        """
        self.config = config
        self.records: List[UsageRecord] = []
        self._total_cost = 0.0
        self._total_tokens = 0
        self._total_requests = 0

    def calculate_cost(
        self,
        provider: LLMProviderType,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """
        Calculate cost for a request.

        Args:
            provider: Provider type
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Cost in USD
        """
        if provider not in PRICING_TABLE:
            return 0.0

        model_pricing = PRICING_TABLE[provider].get(model)
        if not model_pricing:
            logger.warning(f"No pricing data for {provider}/{model}, assuming zero cost")
            return 0.0

        input_cost = (prompt_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * model_pricing["output"]

        return input_cost + output_cost

    def record_usage(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached: bool = False,
        metadata: Optional[Dict] = None,
    ) -> UsageRecord:
        """
        Record LLM usage.

        Args:
            provider: Provider name
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            cached: Whether response was cached
            metadata: Additional metadata

        Returns:
            Usage record
        """
        if not self.config.enabled:
            return None

        total_tokens = prompt_tokens + completion_tokens

        # Calculate cost
        try:
            provider_type = LLMProviderType(provider.lower())
            cost = self.calculate_cost(provider_type, model, prompt_tokens, completion_tokens)
        except ValueError:
            cost = 0.0

        # Create record
        record = UsageRecord(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            cached=cached,
            metadata=metadata or {},
        )

        # Store record
        self.records.append(record)

        # Update totals
        if not cached:  # Don't count cached responses in totals
            self._total_cost += cost
            self._total_tokens += total_tokens
            self._total_requests += 1

        # Log if enabled
        if self.config.log_costs and cost > 0:
            logger.info(
                f"LLM usage: {model} - {total_tokens} tokens - ${cost:.6f} "
                f"(cached: {cached})"
            )

        return record

    def get_summary(self) -> Dict:
        """
        Get usage summary.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_cost": round(self._total_cost, 6),
            "cached_requests": sum(1 for r in self.records if r.cached),
            "records_count": len(self.records),
        }

    def get_breakdown_by_model(self) -> Dict[str, Dict]:
        """
        Get cost breakdown by model.

        Returns:
            Dictionary with per-model statistics
        """
        breakdown = {}
        
        for record in self.records:
            key = f"{record.provider}/{record.model}"
            if key not in breakdown:
                breakdown[key] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost": 0.0,
                }
            
            if not record.cached:
                breakdown[key]["requests"] += 1
                breakdown[key]["tokens"] += record.total_tokens
                breakdown[key]["cost"] += record.cost

        return breakdown

    def reset(self) -> None:
        """Reset all tracking data."""
        self.records.clear()
        self._total_cost = 0.0
        self._total_tokens = 0
        self._total_requests = 0
        logger.info("Cost tracker reset")

