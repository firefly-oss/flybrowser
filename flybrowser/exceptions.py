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

"""Custom exceptions for FlyBrowser."""


class FlyBrowserError(Exception):
    """Base exception for all FlyBrowser errors."""

    pass


class BrowserError(FlyBrowserError):
    """Exception raised for browser-related errors."""

    pass


class PageError(FlyBrowserError):
    """Exception raised for page-related errors."""

    pass


class ElementNotFoundError(FlyBrowserError):
    """Exception raised when an element cannot be found."""

    pass


class ActionError(FlyBrowserError):
    """Exception raised when an action fails to execute."""

    pass


class LLMError(FlyBrowserError):
    """Exception raised for LLM-related errors."""

    pass


class LLMProviderError(LLMError):
    """Exception raised when LLM provider fails."""

    pass


class ExtractionError(FlyBrowserError):
    """Exception raised when data extraction fails."""

    pass


class NavigationError(FlyBrowserError):
    """Exception raised when navigation fails."""

    pass


class TimeoutError(FlyBrowserError):
    """Exception raised when an operation times out."""

    pass


class ConfigurationError(FlyBrowserError):
    """Exception raised for configuration errors."""

    pass

