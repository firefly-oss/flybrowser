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

"""LLM provider configuration management."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class LLMProviderType(str, Enum):
    """Supported LLM provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GEMINI = "gemini"
    GOOGLE = "google"  # Alias for gemini
    QWEN = "qwen"
    DASHSCOPE = "dashscope"  # Alias for qwen
    LM_STUDIO = "lm_studio"
    LOCAL_AI = "localai"
    VLLM = "vllm"


class RetryConfig(BaseModel):
    """Retry configuration for LLM requests."""

    max_retries: int = Field(default=3, ge=0, le=10)
    initial_delay: float = Field(default=1.0, ge=0.1, le=60.0)
    max_delay: float = Field(default=60.0, ge=1.0, le=300.0)
    exponential_base: float = Field(default=2.0, ge=1.0, le=10.0)
    jitter: bool = Field(default=True)


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    requests_per_minute: Optional[int] = Field(default=None, ge=1)
    tokens_per_minute: Optional[int] = Field(default=None, ge=1)
    concurrent_requests: int = Field(default=10, ge=1, le=100)


class CacheConfig(BaseModel):
    """Cache configuration for LLM responses."""

    enabled: bool = Field(default=True)
    ttl_seconds: int = Field(default=3600, ge=60)
    max_size: int = Field(default=1000, ge=10)
    cache_key_prefix: str = Field(default="flybrowser:llm")


class CostTrackingConfig(BaseModel):
    """Cost tracking configuration."""

    enabled: bool = Field(default=True)
    track_tokens: bool = Field(default=True)
    track_requests: bool = Field(default=True)
    log_costs: bool = Field(default=True)


class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    provider_type: LLMProviderType
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = Field(default=60.0, ge=1.0, le=300.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    
    # Advanced configurations
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    rate_limit_config: RateLimitConfig = Field(default_factory=RateLimitConfig)
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    cost_tracking_config: CostTrackingConfig = Field(default_factory=CostTrackingConfig)
    
    # Provider-specific options
    extra_options: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: Optional[str], info: Any) -> Optional[str]:
        """Validate base URL for local providers."""
        provider_type = info.data.get("provider_type")
        if provider_type in [
            LLMProviderType.OLLAMA,
            LLMProviderType.LM_STUDIO,
            LLMProviderType.LOCAL_AI,
            LLMProviderType.VLLM,
        ]:
            if not v:
                # Set defaults for local providers
                defaults = {
                    LLMProviderType.OLLAMA: "http://localhost:11434",
                    LLMProviderType.LM_STUDIO: "http://localhost:1234",
                    LLMProviderType.LOCAL_AI: "http://localhost:8080",
                    LLMProviderType.VLLM: "http://localhost:8000",
                }
                return defaults.get(provider_type)
        return v


class MultiProviderConfig(BaseModel):
    """Configuration for multiple LLM providers with fallback."""

    primary_provider: LLMProviderConfig
    fallback_providers: List[LLMProviderConfig] = Field(default_factory=list)
    auto_fallback: bool = Field(default=True)
    fallback_on_errors: List[str] = Field(
        default_factory=lambda: ["rate_limit", "timeout", "server_error"]
    )


# Default configurations for common providers (as of January 2026)
DEFAULT_CONFIGS = {
    LLMProviderType.OPENAI: {
        "model": "gpt-5.2",  # Latest flagship model
        "max_tokens": 8192,  # Generous default to avoid truncation
    },
    LLMProviderType.ANTHROPIC: {
        "model": "claude-sonnet-4-5-20250929",  # Latest Sonnet model
        "max_tokens": 8192,  # Generous default to avoid truncation
    },
    LLMProviderType.OLLAMA: {
        "model": "qwen3:8b",  # Latest Qwen3 model
        "base_url": "http://localhost:11434",
        "max_tokens": 8192,  # Generous default to avoid truncation
    },
    LLMProviderType.GEMINI: {
        "model": "gemini-2.0-flash",  # Latest Gemini Flash model
        "max_tokens": 8192,
    },
    LLMProviderType.GOOGLE: {
        "model": "gemini-2.0-flash",  # Alias for gemini
        "max_tokens": 8192,
    },
    LLMProviderType.QWEN: {
        "model": "qwen-plus",  # Advanced model with enhanced reasoning
        "max_tokens": 8192,
    },
    LLMProviderType.DASHSCOPE: {
        "model": "qwen-plus",  # Alias for qwen
        "max_tokens": 8192,
    },
}

