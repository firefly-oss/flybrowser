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

"""Custom exceptions for FlyBrowser.

This module defines the exception hierarchy used throughout FlyBrowser.
All exceptions inherit from FlyBrowserError for easy catching and handling.

Exception Hierarchy:
    FlyBrowserError (base)
    ├── BrowserError - Browser instance errors
    ├── PageError - Page loading and interaction errors
    ├── ElementNotFoundError - Element location failures
    ├── ActionError - Action execution failures
    ├── LLMError - LLM provider errors
    │   └── LLMProviderError - Specific provider failures
    ├── ExtractionError - Data extraction failures
    ├── NavigationError - Navigation failures
    ├── TimeoutError - Operation timeout errors
    └── ConfigurationError - Configuration errors

Example:
    try:
        await browser.act("click the button")
    except ElementNotFoundError:
        # Handle missing element
        pass
    except ActionError:
        # Handle action failure
        pass
    except FlyBrowserError:
        # Catch all FlyBrowser errors
        pass
"""


class FlyBrowserError(Exception):
    """Base exception for all FlyBrowser errors.
    
    All custom exceptions in FlyBrowser inherit from this class,
    allowing users to catch all FlyBrowser-specific errors with
    a single except clause.
    
    Attributes:
        message: Error message describing what went wrong
        
    Example:
        try:
            await browser.extract("data")
        except FlyBrowserError as e:
            logger.error(f"FlyBrowser error: {e}")
    """
    pass


class BrowserError(FlyBrowserError):
    """Exception raised for browser-related errors.
    
    Raised when browser initialization, configuration, or lifecycle
    operations fail. This includes issues with browser process
    management, browser context creation, or Playwright errors.
    
    Examples:
        - Browser failed to launch
        - Browser process crashed
        - Invalid browser configuration
    """
    pass


class PageError(FlyBrowserError):
    """Exception raised for page-related errors.
    
    Raised when page operations fail, including page load failures,
    JavaScript execution errors, or page state issues.
    
    Examples:
        - Page failed to load
        - Page became unresponsive
        - Invalid page state
    """
    pass


class ElementNotFoundError(FlyBrowserError):
    """Exception raised when an element cannot be found.
    
    Raised when the LLM-based element detection or traditional
    selector-based element location fails to find the requested
    element on the page.
    
    Examples:
        - Element with specified selector doesn't exist
        - LLM couldn't identify the described element
        - Element exists but is not visible/interactive
    """
    pass


class ActionError(FlyBrowserError):
    """Exception raised when an action fails to execute.
    
    Raised when browser actions (click, type, scroll, etc.) fail
    to execute successfully, even after retries.
    
    Examples:
        - Element is not clickable
        - Input field is disabled
        - Action blocked by page scripts
    """
    pass


class LLMError(FlyBrowserError):
    """Exception raised for LLM-related errors.
    
    Base exception for all LLM-related failures. This includes
    both provider-specific errors and general LLM interaction issues.
    
    Examples:
        - LLM API rate limit exceeded
        - Invalid LLM response format
        - LLM request timeout
    """
    pass


class LLMProviderError(LLMError):
    """Exception raised when LLM provider fails.
    
    Raised for provider-specific failures from OpenAI, Anthropic,
    Gemini, or local LLM providers.
    
    Examples:
        - Invalid API key
        - Provider service unavailable
        - Provider-specific error codes
    """
    pass


class ExtractionError(FlyBrowserError):
    """Exception raised when data extraction fails.
    
    Raised when the LLM fails to extract requested data from
    the page, or when extracted data doesn't match the expected
    schema.
    
    Examples:
        - Data not found on page
        - Extracted data doesn't match schema
        - Page structure incompatible with extraction
    """
    pass


class NavigationError(FlyBrowserError):
    """Exception raised when navigation fails.
    
    Raised when page navigation operations fail, including
    URL navigation, link following, and history navigation.
    
    Examples:
        - URL is invalid or unreachable
        - Navigation timeout
        - Network error during navigation
    """
    pass


class TimeoutError(FlyBrowserError):
    """Exception raised when an operation times out.
    
    Raised when any FlyBrowser operation exceeds its configured
    timeout period. This can occur for page loads, element waits,
    actions, or LLM requests.
    
    Examples:
        - Page load timeout
        - Element wait timeout
        - LLM response timeout
    """
    pass


class ConfigurationError(FlyBrowserError):
    """Exception raised for configuration errors.
    
    Raised when FlyBrowser is configured incorrectly or when
    required configuration is missing.
    
    Examples:
        - Missing required API keys
        - Invalid configuration values
        - Incompatible configuration combinations
    """
    pass


class ToolError(FlyBrowserError):
    """Exception raised for tool execution errors.
    
    Base exception for all tool-related failures in the ReAct framework.
    Raised when a tool fails to execute or encounters an unexpected error.
    
    Examples:
        - Tool parameter validation failure
        - Tool execution timeout
        - Tool-specific operation failure
    """
    pass


class SearchError(ToolError):
    """Exception raised when search operations fail.
    
    Raised when web search operations fail, including API search,
    human-like search, or search ranking failures.
    
    Examples:
        - Search API unavailable
        - Search query parsing failure
        - No search results found
    """
    pass


class ValidationError(FlyBrowserError):
    """Exception raised for validation errors.
    
    Raised when input validation fails, including parameter validation,
    schema validation, or data validation.
    
    Examples:
        - Invalid tool parameters
        - Data doesn't match schema
        - Invalid selector format
    """
    pass

