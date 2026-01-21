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
FastAPI application for FlyBrowser REST API service.

This module provides a production-ready REST API for FlyBrowser, enabling
browser automation through HTTP requests. The API is built with FastAPI
and includes:

- Session management (create, use, delete sessions)
- Navigation endpoints
- Data extraction endpoints
- Action execution endpoints
- Health checks and metrics
- API key authentication
- CORS support
- Comprehensive error handling

The service is designed for:
- Multi-tenant usage with session isolation
- Horizontal scaling (stateless design)
- Production deployment (Docker, Kubernetes)
- Monitoring and observability

Example Usage:
    Start the service:
    ```bash
    uvicorn flybrowser.service.app:app --host 0.0.0.0 --port 8000
    ```

    Create a session:
    ```bash
    curl -X POST http://localhost:8000/sessions \\
      -H "X-API-Key: your-api-key" \\
      -H "Content-Type: application/json" \\
      -d '{
        "llm_provider": "openai",
        "llm_model": "gpt-4o",
        "api_key": "sk-...",
        "headless": true
      }'
    ```

    Navigate:
    ```bash
    curl -X POST http://localhost:8000/sessions/{session_id}/navigate \\
      -H "X-API-Key: your-api-key" \\
      -H "Content-Type: application/json" \\
      -d '{"url": "https://example.com"}'
    ```
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from flybrowser import __version__
from flybrowser.service.auth import APIKey, verify_api_key
from flybrowser.service.models import (
    ActionRequest,
    ActionResponse,
    ErrorResponse,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    MaskPIIRequest,
    MaskPIIResponse,
    MetricsResponse,
    MonitorRequest,
    MonitorResponse,
    NavigateNLRequest,
    NavigateNLResponse,
    NavigateRequest,
    NavigateResponse,
    RecordingStartRequest,
    RecordingStartResponse,
    RecordingStopResponse,
    ScreenshotRequest,
    ScreenshotResponse,
    SecureFillRequest,
    SecureFillResponse,
    SessionCreateRequest,
    SessionResponse,
    StoreCredentialRequest,
    StoreCredentialResponse,
    WorkflowRequest,
    WorkflowResponse,
)
from flybrowser.service.session_manager import SessionManager
from flybrowser.utils.logger import logger

# Global state
session_manager: SessionManager = None
start_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for application startup and shutdown.

    This function handles:
    - Startup: Initialize session manager and global state
    - Shutdown: Cleanup all active sessions and resources

    Args:
        app: FastAPI application instance

    Yields:
        None during application runtime
    """
    global session_manager, start_time

    # Startup
    logger.info("Starting FlyBrowser service...")
    session_manager = SessionManager()
    start_time = time.time()
    logger.info("FlyBrowser service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down FlyBrowser service...")
    await session_manager.cleanup_all()
    logger.info("FlyBrowser service shut down")


# API Documentation
API_DESCRIPTION = """
# FlyBrowser API

Browser automation and web scraping powered by LLM agents.

## Overview

FlyBrowser provides a powerful API for browser automation with built-in support for:

- **Session Management**: Create and manage browser sessions
- **Navigation**: Navigate to URLs, click elements, fill forms
- **Screenshots**: Capture full-page or element screenshots
- **Recording**: Record browser sessions as video
- **PII Masking**: Automatically mask sensitive data in screenshots and recordings
- **LLM Integration**: Use AI agents to automate complex tasks

## Authentication

API requests require authentication via API key:

```
X-API-Key: your-api-key
```

## Quick Start

1. Create a session: `POST /sessions`
2. Navigate to a URL: `POST /sessions/{session_id}/navigate`
3. Take a screenshot: `POST /sessions/{session_id}/screenshot`
4. Close the session: `DELETE /sessions/{session_id}`

## Deployment Modes

- **Standalone**: Single node deployment (default)
- **Cluster**: Multi-node deployment for horizontal scaling

## Resources

- [GitHub Repository](https://github.com/firefly-oss/flybrowsers)
- [Documentation](https://flybrowser.dev/docs)
- [Support](mailto:support@flybrowser.dev)
"""

# Create FastAPI app
app = FastAPI(
    title="FlyBrowser API",
    description=API_DESCRIPTION,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "FlyBrowser Support",
        "url": "https://flybrowser.dev",
        "email": "support@flybrowser.dev",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    openapi_tags=[
        {
            "name": "Health",
            "description": "Health check and service status endpoints",
        },
        {
            "name": "Sessions",
            "description": "Browser session management - create, list, and delete sessions",
        },
        {
            "name": "Navigation",
            "description": "Browser navigation and interaction - navigate, click, type, etc.",
        },
        {
            "name": "Screenshots",
            "description": "Screenshot capture with optional PII masking",
        },
        {
            "name": "Recording",
            "description": "Video recording of browser sessions",
        },
        {
            "name": "Cluster",
            "description": "Cluster management endpoints (coordinator mode only)",
        },
    ],
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            details={"exception": str(exc)},
        ).model_dump(),
    )


# Health and metrics endpoints (no auth required)
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Check if the service is healthy and get basic status information.",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "version": "1.26.1",
                        "uptime_seconds": 3600.5,
                        "active_sessions": 5,
                        "system_info": {"sessions": 5}
                    }
                }
            }
        }
    },
)
async def health_check():
    """
    Health check endpoint.

    Returns the current health status of the service including:
    - Service version
    - Uptime in seconds
    - Number of active browser sessions
    - System information

    This endpoint does not require authentication.
    """
    uptime = time.time() - start_time

    return HealthResponse(
        status="healthy",
        version=__version__,
        uptime_seconds=uptime,
        active_sessions=session_manager.get_active_session_count(),
        system_info={
            "sessions": session_manager.get_active_session_count(),
        },
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Health"],
    summary="Get service metrics",
    description="Get detailed service metrics including request counts, cache stats, and rate limits.",
)
async def get_metrics(api_key: APIKey = Depends(verify_api_key)):
    """
    Get service metrics.

    Returns detailed metrics about the service including:
    - Total request count
    - Active session count
    - Cache hit/miss statistics
    - Cost tracking statistics
    - Rate limit statistics

    Requires API key authentication.
    """
    stats = session_manager.get_stats()

    return MetricsResponse(
        total_requests=stats.get("total_requests", 0),
        active_sessions=stats.get("active_sessions", 0),
        cache_stats=stats.get("cache_stats", {}),
        cost_stats=stats.get("cost_stats", {}),
        rate_limit_stats=stats.get("rate_limit_stats", {}),
    )


# Session management endpoints
@app.post(
    "/sessions",
    response_model=SessionResponse,
    tags=["Sessions"],
    summary="Create a new browser session",
    description="Create a new browser session with optional LLM integration for AI-powered automation.",
)
async def create_session(
    request: SessionCreateRequest,
    api_key: APIKey = Depends(verify_api_key),
):
    """
    Create a new browser session.

    Creates a new browser instance that can be used for automation tasks.
    The session can optionally be configured with an LLM provider for
    AI-powered automation capabilities.

    **Parameters:**
    - **llm_provider**: LLM provider to use (openai, anthropic, etc.)
    - **llm_model**: Specific model to use (e.g., gpt-4, claude-3)
    - **api_key**: API key for the LLM provider
    - **headless**: Run browser in headless mode (default: true)
    - **browser_type**: Browser type (chromium, firefox, webkit)

    **Returns:**
    - Session ID for subsequent API calls
    - Session status and creation timestamp
    """
    try:
        session_id = await session_manager.create_session(
            llm_provider=request.llm_provider,
            llm_model=request.llm_model,
            api_key=request.api_key,
            headless=request.headless,
            browser_type=request.browser_type.value,
        )

        return SessionResponse(
            session_id=session_id,
            status="active",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            metadata={"browser_type": request.browser_type.value},
        )
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}",
        )


@app.get(
    "/sessions",
    tags=["Sessions"],
    summary="List all browser sessions",
    description="Get a list of all active browser sessions.",
)
async def list_sessions(
    api_key: APIKey = Depends(verify_api_key),
):
    """
    List all active browser sessions.

    Returns a list of all sessions managed by this server instance.
    """
    sessions = []
    for session_id, metadata in session_manager.session_metadata.items():
        sessions.append({
            "session_id": session_id,
            "status": "active",
            "created_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(metadata.get("created_at", 0))
            ),
            "last_activity": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(metadata.get("last_activity", 0))
            ),
            "llm_provider": metadata.get("llm_provider"),
            "browser_type": metadata.get("browser_type"),
        })
    
    return {
        "sessions": sessions,
        "total": len(sessions),
    }


@app.get(
    "/sessions/{session_id}",
    tags=["Sessions"],
    summary="Get browser session info",
    description="Get information about a specific browser session.",
)
async def get_session(
    session_id: str,
    api_key: APIKey = Depends(verify_api_key),
):
    """
    Get session information.

    Returns detailed information about a specific session.
    """
    if session_id not in session_manager.session_metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    
    metadata = session_manager.session_metadata[session_id]
    return {
        "session_id": session_id,
        "status": "active",
        "created_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(metadata.get("created_at", 0))
        ),
        "last_activity": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(metadata.get("last_activity", 0))
        ),
        "llm_provider": metadata.get("llm_provider"),
        "llm_model": metadata.get("llm_model"),
        "browser_type": metadata.get("browser_type"),
    }


@app.delete(
    "/sessions/{session_id}",
    tags=["Sessions"],
    summary="Delete a browser session",
    description="Close and delete a browser session, releasing all associated resources.",
)
async def delete_session(
    session_id: str,
    api_key: APIKey = Depends(verify_api_key),
):
    """
    Delete a browser session.

    Closes the browser instance and releases all resources associated
    with the session. Any ongoing operations will be cancelled.

    **Parameters:**
    - **session_id**: The ID of the session to delete

    **Returns:**
    - Confirmation of deletion
    """
    try:
        await session_manager.delete_session(session_id)
        return {"status": "deleted", "session_id": session_id}
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )


# Browser automation endpoints
@app.post(
    "/sessions/{session_id}/navigate",
    response_model=NavigateResponse,
    tags=["Navigation"],
    summary="Navigate to a URL",
    description="Navigate the browser to a specified URL and wait for the page to load.",
)
async def navigate(
    session_id: str,
    request: NavigateRequest,
    api_key: APIKey = Depends(verify_api_key),
):
    """
    Navigate to a URL.

    Navigates the browser to the specified URL and waits for the page
    to reach the specified load state.

    **Parameters:**
    - **url**: The URL to navigate to
    - **wait_until**: When to consider navigation complete (load, domcontentloaded, networkidle)

    **Returns:**
    - Final URL (may differ from requested due to redirects)
    - Page title
    - Navigation duration in milliseconds
    """
    start = time.time()

    try:
        browser = session_manager.get_session(session_id)
        await browser.goto(str(request.url), wait_until=request.wait_until)

        # Get page info
        page_controller = browser.page_controller
        title = await page_controller.get_title()
        url = await page_controller.get_url()

        duration_ms = int((time.time() - start) * 1000)

        return NavigateResponse(
            success=True,
            url=url,
            title=title,
            duration_ms=duration_ms,
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    except Exception as e:
        logger.error(f"Navigation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Navigation failed: {str(e)}",
        )


@app.post("/sessions/{session_id}/extract", response_model=ExtractResponse, tags=["Automation"])
async def extract_data(
    session_id: str,
    request: ExtractRequest,
    api_key: APIKey = Depends(verify_api_key),
):
    """Extract data from the current page."""
    try:
        browser = session_manager.get_session(session_id)

        result = await browser.extract(
            query=request.query,
            use_vision=request.use_vision,
            schema=request.schema,
        )

        return ExtractResponse(
            success=True,
            data=result,
            cached=result.get("_cached", False),
            metadata={},
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {str(e)}",
        )


@app.post("/sessions/{session_id}/act", response_model=ActionResponse, tags=["Automation"])
async def perform_action(
    session_id: str,
    request: ActionRequest,
    api_key: APIKey = Depends(verify_api_key),
):
    """Perform an action on the page."""
    start = time.time()

    try:
        browser = session_manager.get_session(session_id)

        await browser.act(
            instruction=request.instruction,
            use_vision=request.use_vision,
        )

        duration_ms = int((time.time() - start) * 1000)

        return ActionResponse(
            success=True,
            action_type="act",
            element_found=True,
            duration_ms=duration_ms,
            metadata={},
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    except Exception as e:
        logger.error(f"Action failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Action failed: {str(e)}",
        )


# Natural language navigation endpoint
@app.post(
    "/sessions/{session_id}/navigate-nl",
    response_model=NavigateNLResponse,
    tags=["Navigation"],
    summary="Navigate using natural language",
    description="Navigate using natural language instructions (e.g., 'go to the login page').",
)
async def navigate_natural_language(
    session_id: str,
    request: NavigateNLRequest,
    api_key: APIKey = Depends(verify_api_key),
):
    """Navigate using natural language instructions."""
    try:
        browser = session_manager.get_session(session_id)
        result = await browser.navigate(
            instruction=request.instruction,
            use_vision=request.use_vision,
        )

        return NavigateNLResponse(
            success=result.get("success", False),
            url=result.get("url"),
            title=result.get("title"),
            navigation_type=result.get("navigation_type"),
            error=result.get("error"),
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    except Exception as e:
        logger.error(f"Natural language navigation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Navigation failed: {str(e)}",
        )


# Workflow endpoint
@app.post(
    "/sessions/{session_id}/workflow",
    response_model=WorkflowResponse,
    tags=["Automation"],
    summary="Execute a workflow",
    description="Execute a multi-step workflow with state management and error recovery.",
)
async def execute_workflow(
    session_id: str,
    request: WorkflowRequest,
    api_key: APIKey = Depends(verify_api_key),
):
    """Execute a multi-step workflow."""
    try:
        browser = session_manager.get_session(session_id)
        result = await browser.run_workflow(
            workflow_definition=request.workflow,
            variables=request.variables,
        )

        return WorkflowResponse(
            success=result.get("success", False),
            steps_completed=result.get("steps_completed", 0),
            total_steps=result.get("total_steps", 0),
            error=result.get("error"),
            step_results=result.get("step_results", []),
            variables=result.get("variables", {}),
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow failed: {str(e)}",
        )


# Monitor endpoint
@app.post(
    "/sessions/{session_id}/monitor",
    response_model=MonitorResponse,
    tags=["Automation"],
    summary="Monitor for a condition",
    description="Monitor the page for a condition to be met using natural language.",
)
async def monitor_condition(
    session_id: str,
    request: MonitorRequest,
    api_key: APIKey = Depends(verify_api_key),
):
    """Monitor for a condition to be met."""
    try:
        browser = session_manager.get_session(session_id)
        result = await browser.monitor(
            condition=request.condition,
            timeout=request.timeout,
            poll_interval=request.poll_interval,
        )

        return MonitorResponse(
            success=result.get("success", False),
            condition_met=result.get("condition_met", False),
            elapsed_time=result.get("elapsed_time", 0.0),
            error=result.get("error"),
            details=result.get("details", {}),
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    except Exception as e:
        logger.error(f"Monitor failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Monitor failed: {str(e)}",
        )


# Screenshot endpoints
@app.post(
    "/sessions/{session_id}/screenshot",
    response_model=ScreenshotResponse,
    tags=["Screenshots"],
    summary="Capture a screenshot",
    description="Capture a screenshot of the current page with optional PII masking.",
)
async def capture_screenshot(
    session_id: str,
    request: ScreenshotRequest,
    api_key: APIKey = Depends(verify_api_key),
):
    """
    Capture a screenshot of the current page.

    Supports full-page screenshots and multiple image formats.
    PII masking can be applied to protect sensitive information.
    """
    try:
        browser = session_manager.get_session(session_id)
        screenshot = await browser.screenshot(
            full_page=request.full_page,
            save_to_file=False,
            mask_pii=request.mask_pii,
        )

        return ScreenshotResponse(
            success=True,
            screenshot_id=screenshot.id,
            format=screenshot.format.value,
            width=screenshot.width,
            height=screenshot.height,
            data_base64=screenshot.to_base64(),
            url=screenshot.url,
            timestamp=screenshot.timestamp,
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    except Exception as e:
        logger.error(f"Screenshot failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Screenshot failed: {str(e)}",
        )


# Recording endpoints
@app.post(
    "/sessions/{session_id}/recording/start",
    response_model=RecordingStartResponse,
    tags=["Recording"],
    summary="Start recording",
    description="Start recording the browser session (video and/or screenshots).",
)
async def start_recording(
    session_id: str,
    request: RecordingStartRequest,
    api_key: APIKey = Depends(verify_api_key),
):
    """Start recording the browser session."""
    try:
        browser = session_manager.get_session(session_id)
        await browser.start_recording()

        return RecordingStartResponse(
            success=True,
            recording_id=str(uuid.uuid4()),
            video_enabled=request.video_enabled,
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    except Exception as e:
        logger.error(f"Start recording failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Start recording failed: {str(e)}",
        )


@app.post(
    "/sessions/{session_id}/recording/stop",
    response_model=RecordingStopResponse,
    tags=["Recording"],
    summary="Stop recording",
    description="Stop recording and return recording data.",
)
async def stop_recording(
    session_id: str,
    api_key: APIKey = Depends(verify_api_key),
):
    """Stop recording and return recording data."""
    try:
        browser = session_manager.get_session(session_id)
        result = await browser.stop_recording()

        video_info = result.get("video") or {}

        return RecordingStopResponse(
            success=True,
            recording_id=result.get("session_id", ""),
            duration_seconds=video_info.get("duration_seconds", 0.0),
            screenshot_count=len(result.get("screenshots", [])),
            video_path=video_info.get("file_path"),
            video_size_bytes=video_info.get("size_bytes"),
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    except Exception as e:
        logger.error(f"Stop recording failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stop recording failed: {str(e)}",
        )


# PII handling endpoints
@app.post(
    "/sessions/{session_id}/credentials",
    response_model=StoreCredentialResponse,
    tags=["PII"],
    summary="Store a credential",
    description="Store a credential securely for later use in form filling.",
)
async def store_credential(
    session_id: str,
    request: StoreCredentialRequest,
    api_key: APIKey = Depends(verify_api_key),
):
    """Store a credential securely."""
    try:
        browser = session_manager.get_session(session_id)
        from flybrowser.security.pii_handler import PIIType

        pii_type = PIIType(request.pii_type.value)
        credential_id = browser.store_credential(
            name=request.name,
            value=request.value,
            pii_type=pii_type,
        )

        return StoreCredentialResponse(
            success=True,
            credential_id=credential_id,
            name=request.name,
            pii_type=request.pii_type.value,
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    except Exception as e:
        logger.error(f"Store credential failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Store credential failed: {str(e)}",
        )


@app.post(
    "/sessions/{session_id}/secure-fill",
    response_model=SecureFillResponse,
    tags=["PII"],
    summary="Securely fill a form field",
    description="Fill a form field with a stored credential without exposing the value.",
)
async def secure_fill(
    session_id: str,
    request: SecureFillRequest,
    api_key: APIKey = Depends(verify_api_key),
):
    """Securely fill a form field with a stored credential."""
    try:
        browser = session_manager.get_session(session_id)
        success = await browser.secure_fill(
            selector=request.selector,
            credential_id=request.credential_id,
            clear_first=request.clear_first,
        )

        return SecureFillResponse(
            success=success,
            selector=request.selector,
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    except Exception as e:
        logger.error(f"Secure fill failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Secure fill failed: {str(e)}",
        )


@app.post(
    "/pii/mask",
    response_model=MaskPIIResponse,
    tags=["PII"],
    summary="Mask PII in text",
    description="Mask personally identifiable information in text.",
)
async def mask_pii(
    request: MaskPIIRequest,
    api_key: APIKey = Depends(verify_api_key),
):
    """Mask PII in text."""
    from flybrowser.security.pii_handler import PIIMasker

    masker = PIIMasker()
    masked_text = masker.mask_text(request.text)

    return MaskPIIResponse(
        original_length=len(request.text),
        masked_text=masked_text,
        pii_detected=masked_text != request.text,
    )


# Cluster endpoints (for cluster mode deployment)
@app.post("/cluster/register", tags=["Cluster"])
async def cluster_register(message: Dict):
    """Register a worker node with the cluster (coordinator only)."""
    # This endpoint is used by worker nodes to register with the coordinator
    # In standalone mode, this returns a not-implemented response
    return {"status": "standalone_mode", "message": "Cluster mode not enabled"}


@app.post("/cluster/unregister", tags=["Cluster"])
async def cluster_unregister(message: Dict):
    """Unregister a worker node from the cluster (coordinator only)."""
    return {"status": "standalone_mode", "message": "Cluster mode not enabled"}


@app.post("/cluster/heartbeat", tags=["Cluster"])
async def cluster_heartbeat(message: Dict):
    """Handle heartbeat from a worker node (coordinator only)."""
    return {"status": "standalone_mode", "message": "Cluster mode not enabled"}


@app.get("/cluster/status", tags=["Cluster"])
async def cluster_status():
    """Get cluster status."""
    return {
        "mode": "standalone",
        "node_count": 1,
        "total_capacity": session_manager.max_sessions if session_manager else 100,
        "active_sessions": session_manager.get_active_session_count() if session_manager else 0,
    }
