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

"""Factory for creating LLM provider instances."""

from typing import Any, Dict, List, Optional

from flybrowser.exceptions import ConfigurationError
from flybrowser.llm.anthropic_provider import AnthropicProvider
from flybrowser.llm.base import BaseLLMProvider
from flybrowser.llm.config import DEFAULT_CONFIGS, LLMProviderConfig, LLMProviderType
from flybrowser.llm.gemini_provider import GeminiProvider
from flybrowser.llm.ollama_provider import OllamaProvider
from flybrowser.llm.openai_provider import OpenAIProvider
from flybrowser.llm.provider_status import ProviderStatus


class LLMProviderFactory:
    """Factory for creating LLM provider instances with full configuration support."""

    _providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        "gemini": GeminiProvider,
        "google": GeminiProvider,  # Alias for gemini
        # Additional providers can be registered dynamically
    }

    @classmethod
    def create(
        cls,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[LLMProviderConfig] = None,
        **kwargs: Any,
    ) -> BaseLLMProvider:
        """
        Create an LLM provider instance.

        Args:
            provider: Provider name (openai, anthropic, ollama, etc.)
            model: Model name (optional, uses provider default if not specified)
            api_key: API key for the provider
            config: Full provider configuration (optional)
            **kwargs: Additional provider-specific configuration

        Returns:
            BaseLLMProvider instance

        Raises:
            ConfigurationError: If provider is not supported
        """
        provider_lower = provider.lower()

        if provider_lower not in cls._providers:
            raise ConfigurationError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: {', '.join(cls._providers.keys())}"
            )

        provider_class = cls._providers[provider_lower]

        # Use default model if not specified
        if not model:
            try:
                provider_type = LLMProviderType(provider_lower)
                model = DEFAULT_CONFIGS.get(provider_type, {}).get("model", "default")
            except ValueError:
                model = "default"

        # Create provider instance
        return provider_class(model=model, api_key=api_key, config=config, **kwargs)

    @classmethod
    def create_from_config(cls, config: LLMProviderConfig) -> BaseLLMProvider:
        """
        Create a provider from a configuration object.

        Args:
            config: LLM provider configuration

        Returns:
            BaseLLMProvider instance
        """
        return cls.create(
            provider=config.provider_type.value,
            model=config.model,
            api_key=config.api_key,
            config=config,
            base_url=config.base_url,
            **config.extra_options,
        )

    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """
        Register a custom LLM provider.

        Args:
            name: Provider name
            provider_class: Provider class (must inherit from BaseLLMProvider)
        """
        if not issubclass(provider_class, BaseLLMProvider):
            raise ConfigurationError(
                f"Provider class must inherit from BaseLLMProvider, got {provider_class}"
            )
        cls._providers[name.lower()] = provider_class

    @classmethod
    def list_providers(cls) -> list[str]:
        """
        List all registered providers.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())

    @classmethod
    def get_all_provider_statuses(cls) -> Dict[str, ProviderStatus]:
        """
        Get availability status for all registered providers.

        This method dynamically checks each registered provider's availability
        by calling their check_availability() method.

        Returns:
            Dictionary mapping provider names to their ProviderStatus
        """
        statuses = {}
        seen_classes = set()  # Avoid duplicate checks for aliases (e.g., google -> gemini)

        for name, provider_class in cls._providers.items():
            # Skip aliases - only check each provider class once
            if provider_class in seen_classes:
                continue
            seen_classes.add(provider_class)

            try:
                status = provider_class.check_availability()
                statuses[name] = status
            except Exception as e:
                # If check fails, report it as a warning
                statuses[name] = ProviderStatus.warn(
                    name=name.title(),
                    message=f"Error checking availability: {e}",
                )

        return statuses

    @classmethod
    def get_provider_status(cls, provider: str) -> Optional[ProviderStatus]:
        """
        Get availability status for a specific provider.

        Args:
            provider: Provider name

        Returns:
            ProviderStatus for the provider, or None if not registered
        """
        provider_lower = provider.lower()
        if provider_lower not in cls._providers:
            return None

        provider_class = cls._providers[provider_lower]
        try:
            return provider_class.check_availability()
        except Exception as e:
            return ProviderStatus.warn(
                name=provider.title(),
                message=f"Error checking availability: {e}",
            )

