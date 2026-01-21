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

"""Provider status information for dynamic discovery."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProviderStatusLevel(str, Enum):
    """Status level for a provider."""
    
    OK = "ok"           # Provider is fully available
    INFO = "info"       # Provider not configured but optional
    WARN = "warn"       # Provider has issues but may work
    ERROR = "error"     # Provider is unavailable


@dataclass
class ProviderStatus:
    """
    Status information for an LLM provider.
    
    This dataclass encapsulates the availability status of an LLM provider,
    including whether it's configured, any connectivity status, and helpful
    messages for the user.
    
    Attributes:
        name: Display name of the provider (e.g., "OpenAI", "Anthropic")
        available: Whether the provider is ready to use
        level: Status level (OK, INFO, WARN, ERROR)
        message: Status message to display to the user
        requires_api_key: Whether this provider requires an API key
        api_key_env_var: Environment variable name for the API key
        api_key_configured: Whether the API key is set
        base_url: Base URL for the provider (for local providers)
        connectivity_checked: Whether connectivity was verified
        connectivity_ok: Whether connectivity check passed
        extra_info: Additional provider-specific information
    """
    
    name: str
    available: bool
    level: ProviderStatusLevel
    message: str
    requires_api_key: bool = True
    api_key_env_var: Optional[str] = None
    api_key_configured: bool = False
    base_url: Optional[str] = None
    connectivity_checked: bool = False
    connectivity_ok: bool = False
    extra_info: dict = field(default_factory=dict)
    
    @classmethod
    def ok(
        cls,
        name: str,
        message: str,
        api_key_env_var: Optional[str] = None,
        **kwargs,
    ) -> "ProviderStatus":
        """Create an OK status."""
        return cls(
            name=name,
            available=True,
            level=ProviderStatusLevel.OK,
            message=message,
            api_key_env_var=api_key_env_var,
            api_key_configured=True,
            **kwargs,
        )
    
    @classmethod
    def info(
        cls,
        name: str,
        message: str,
        api_key_env_var: Optional[str] = None,
        **kwargs,
    ) -> "ProviderStatus":
        """Create an INFO status (not configured but optional)."""
        return cls(
            name=name,
            available=False,
            level=ProviderStatusLevel.INFO,
            message=message,
            api_key_env_var=api_key_env_var,
            **kwargs,
        )
    
    @classmethod
    def warn(
        cls,
        name: str,
        message: str,
        api_key_env_var: Optional[str] = None,
        **kwargs,
    ) -> "ProviderStatus":
        """Create a WARN status."""
        return cls(
            name=name,
            available=False,
            level=ProviderStatusLevel.WARN,
            message=message,
            api_key_env_var=api_key_env_var,
            **kwargs,
        )
    
    @classmethod
    def error(
        cls,
        name: str,
        message: str,
        api_key_env_var: Optional[str] = None,
        **kwargs,
    ) -> "ProviderStatus":
        """Create an ERROR status."""
        return cls(
            name=name,
            available=False,
            level=ProviderStatusLevel.ERROR,
            message=message,
            api_key_env_var=api_key_env_var,
            **kwargs,
        )
    
    def get_status_icon(self) -> str:
        """Get the status icon for CLI display."""
        icons = {
            ProviderStatusLevel.OK: "[OK]",
            ProviderStatusLevel.INFO: "[INFO]",
            ProviderStatusLevel.WARN: "[WARN]",
            ProviderStatusLevel.ERROR: "[FAIL]",
        }
        return icons.get(self.level, "[?]")
    
    def format_line(self) -> str:
        """Format a single line for CLI display."""
        return f"{self.get_status_icon()} {self.name}: {self.message}"
