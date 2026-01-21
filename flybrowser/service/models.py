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
Pydantic models for FlyBrowser REST API requests and responses.

This module defines all request and response models for the FlyBrowser REST API.
Using Pydantic models provides:
- Automatic request validation
- Type safety
- API documentation generation
- Serialization/deserialization
- Clear API contracts

All models include:
- Field descriptions for API documentation
- Validation constraints
- Default values where appropriate
- Examples in docstrings

Example:
    >>> from flybrowser.service.models import SessionCreateRequest
    >>> request = SessionCreateRequest(
    ...     llm_provider="openai",
    ...     llm_model="gpt-4o",
    ...     api_key="sk-...",
    ...     headless=True
    ... )
    >>> print(request.model_dump_json())
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl


class BrowserType(str, Enum):
    """
    Supported browser types for automation.

    Attributes:
        CHROMIUM: Chromium-based browser (default, most compatible)
        FIREFOX: Mozilla Firefox
        WEBKIT: WebKit (Safari engine)
    """

    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


class SessionCreateRequest(BaseModel):
    """
    Request model for creating a new browser session.

    A session represents an isolated browser instance with its own
    LLM configuration and state.

    Attributes:
        llm_provider: LLM provider name (e.g., "openai", "anthropic", "ollama")
        llm_model: LLM model name (optional, uses provider default if not specified)
        api_key: API key for the LLM provider (e.g., OpenAI API key)
        headless: Whether to run browser in headless mode (no visible window)
        browser_type: Type of browser to use
        timeout: Default timeout for operations in seconds

    Example:
        >>> request = SessionCreateRequest(
        ...     llm_provider="openai",
        ...     llm_model="gpt-5.2",  # Latest OpenAI flagship model
        ...     api_key="sk-proj-...",
        ...     headless=True,
        ...     browser_type=BrowserType.CHROMIUM,
        ...     timeout=60
        ... )
    """

    llm_provider: str = Field(..., description="LLM provider (openai, anthropic, ollama)")
    llm_model: Optional[str] = Field(None, description="LLM model name")
    api_key: Optional[str] = Field(None, description="API key for LLM provider")
    headless: bool = Field(True, description="Run browser in headless mode")
    browser_type: BrowserType = Field(BrowserType.CHROMIUM, description="Browser type")
    timeout: int = Field(60, ge=10, le=300, description="Default timeout in seconds")


class SessionResponse(BaseModel):
    """Response containing session information."""

    session_id: str = Field(..., description="Unique session identifier")
    status: str = Field(..., description="Session status")
    created_at: str = Field(..., description="Session creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class NavigateRequest(BaseModel):
    """Request to navigate to a URL."""

    url: HttpUrl = Field(..., description="URL to navigate to")
    wait_until: str = Field("domcontentloaded", description="Wait condition")
    timeout: Optional[int] = Field(None, description="Navigation timeout in milliseconds")


class NavigateResponse(BaseModel):
    """Response from navigation."""

    success: bool = Field(..., description="Whether navigation succeeded")
    url: str = Field(..., description="Final URL after navigation")
    title: str = Field(..., description="Page title")
    duration_ms: int = Field(..., description="Navigation duration in milliseconds")


class ExtractRequest(BaseModel):
    """Request to extract data from page."""

    query: str = Field(..., description="Natural language extraction query")
    use_vision: bool = Field(False, description="Use vision-based extraction")
    schema: Optional[Dict[str, Any]] = Field(None, description="JSON schema for structured output")


class ExtractResponse(BaseModel):
    """Response from data extraction."""

    success: bool = Field(..., description="Whether extraction succeeded")
    data: Dict[str, Any] = Field(..., description="Extracted data")
    confidence: Optional[float] = Field(None, description="Confidence score")
    cached: bool = Field(False, description="Whether result was cached")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ActionRequest(BaseModel):
    """Request to perform an action."""

    instruction: str = Field(..., description="Natural language action instruction")
    use_vision: bool = Field(True, description="Use vision for element detection")
    wait_after: int = Field(1000, ge=0, description="Wait time after action in milliseconds")


class ActionResponse(BaseModel):
    """Response from action execution."""

    success: bool = Field(..., description="Whether action succeeded")
    action_type: str = Field(..., description="Type of action performed")
    element_found: bool = Field(..., description="Whether target element was found")
    duration_ms: int = Field(..., description="Action duration in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    active_sessions: int = Field(..., description="Number of active sessions")
    system_info: Dict[str, Any] = Field(default_factory=dict, description="System information")


class MetricsResponse(BaseModel):
    """Metrics response."""

    total_requests: int = Field(..., description="Total requests processed")
    active_sessions: int = Field(..., description="Active sessions")
    cache_stats: Dict[str, Any] = Field(default_factory=dict, description="Cache statistics")
    cost_stats: Dict[str, Any] = Field(default_factory=dict, description="Cost statistics")
    rate_limit_stats: Dict[str, Any] = Field(default_factory=dict, description="Rate limit stats")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")


# Screenshot models
class ScreenshotRequest(BaseModel):
    """Request to capture a screenshot."""

    full_page: bool = Field(False, description="Capture full scrollable page")
    format: str = Field("png", description="Image format (png, jpeg, webp)")
    quality: int = Field(80, ge=1, le=100, description="Image quality for JPEG/WebP")
    mask_pii: bool = Field(True, description="Apply PII masking to screenshot")


class ScreenshotResponse(BaseModel):
    """Response containing screenshot data."""

    success: bool = Field(..., description="Whether screenshot was captured")
    screenshot_id: str = Field(..., description="Unique screenshot identifier")
    format: str = Field(..., description="Image format")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    data_base64: str = Field(..., description="Base64-encoded image data")
    url: str = Field(..., description="Page URL when screenshot was taken")
    timestamp: float = Field(..., description="Capture timestamp")


# Recording models
class RecordingStartRequest(BaseModel):
    """Request to start recording a session."""

    video_enabled: bool = Field(True, description="Enable video recording")
    screenshot_interval: Optional[float] = Field(
        None, description="Interval for periodic screenshots (seconds)"
    )
    auto_screenshot_on_navigation: bool = Field(
        True, description="Capture screenshot on each navigation"
    )


class RecordingStartResponse(BaseModel):
    """Response when recording starts."""

    success: bool = Field(..., description="Whether recording started")
    recording_id: str = Field(..., description="Recording session ID")
    video_enabled: bool = Field(..., description="Whether video is being recorded")


class RecordingStopResponse(BaseModel):
    """Response when recording stops."""

    success: bool = Field(..., description="Whether recording stopped successfully")
    recording_id: str = Field(..., description="Recording session ID")
    duration_seconds: float = Field(..., description="Recording duration")
    screenshot_count: int = Field(..., description="Number of screenshots captured")
    video_path: Optional[str] = Field(None, description="Path to video file")
    video_size_bytes: Optional[int] = Field(None, description="Video file size")


# PII handling models
class PIITypeEnum(str, Enum):
    """Types of PII data."""

    PASSWORD = "password"
    USERNAME = "username"
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    CVV = "cvv"
    API_KEY = "api_key"
    TOKEN = "token"
    CUSTOM = "custom"


class StoreCredentialRequest(BaseModel):
    """Request to store a credential securely."""

    name: str = Field(..., description="Credential name/identifier")
    value: str = Field(..., description="Credential value (will be encrypted)")
    pii_type: PIITypeEnum = Field(PIITypeEnum.PASSWORD, description="Type of PII")


class StoreCredentialResponse(BaseModel):
    """Response after storing a credential."""

    success: bool = Field(..., description="Whether credential was stored")
    credential_id: str = Field(..., description="ID for retrieving the credential")
    name: str = Field(..., description="Credential name")
    pii_type: str = Field(..., description="Type of PII")


class SecureFillRequest(BaseModel):
    """Request to securely fill a form field."""

    selector: str = Field(..., description="CSS selector for the input field")
    credential_id: str = Field(..., description="ID of the stored credential")
    clear_first: bool = Field(True, description="Clear field before filling")


class SecureFillResponse(BaseModel):
    """Response after secure fill operation."""

    success: bool = Field(..., description="Whether fill succeeded")
    selector: str = Field(..., description="Selector that was filled")


class MaskPIIRequest(BaseModel):
    """Request to mask PII in text."""

    text: str = Field(..., description="Text that may contain PII")


class MaskPIIResponse(BaseModel):
    """Response with masked text."""

    original_length: int = Field(..., description="Original text length")
    masked_text: str = Field(..., description="Text with PII masked")
    pii_detected: bool = Field(..., description="Whether PII was detected and masked")


# Workflow models
class WorkflowRequest(BaseModel):
    """Request to execute a workflow."""

    workflow: Dict[str, Any] = Field(..., description="Workflow definition (steps, conditions, etc.)")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Variables for the workflow")


class WorkflowResponse(BaseModel):
    """Response from workflow execution."""

    success: bool = Field(..., description="Whether workflow completed successfully")
    steps_completed: int = Field(..., description="Number of steps completed")
    total_steps: int = Field(..., description="Total number of steps in workflow")
    error: Optional[str] = Field(None, description="Error message if failed")
    step_results: List[Dict[str, Any]] = Field(default_factory=list, description="Results from each step")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Final variable state")


# Monitor models
class MonitorRequest(BaseModel):
    """Request to monitor for a condition."""

    condition: str = Field(..., description="Natural language condition to wait for")
    timeout: float = Field(30.0, ge=1.0, le=300.0, description="Maximum wait time in seconds")
    poll_interval: float = Field(0.5, ge=0.1, le=10.0, description="Time between checks in seconds")


class MonitorResponse(BaseModel):
    """Response from monitoring operation."""

    success: bool = Field(..., description="Whether monitoring completed without error")
    condition_met: bool = Field(..., description="Whether the condition was met")
    elapsed_time: float = Field(..., description="Time elapsed in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")


# Natural language navigation models
class NavigateNLRequest(BaseModel):
    """Request for natural language navigation."""

    instruction: str = Field(..., description="Natural language navigation instruction")
    use_vision: bool = Field(True, description="Use vision for element detection")


class NavigateNLResponse(BaseModel):
    """Response from natural language navigation."""

    success: bool = Field(..., description="Whether navigation succeeded")
    url: Optional[str] = Field(None, description="Final URL after navigation")
    title: Optional[str] = Field(None, description="Page title")
    navigation_type: Optional[str] = Field(None, description="Type of navigation performed")
    error: Optional[str] = Field(None, description="Error message if failed")

