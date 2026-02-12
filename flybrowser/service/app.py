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
      -H "Content-Type: application/json" \\
      -d '{"url": "https://example.com"}'
    ```
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from flybrowser import __version__
from flybrowser.service.template_renderer import render_player_html, render_blank_html, get_static_dir
from flybrowser.service.models import (
    ActionRequest,
    ActionResponse,
    AgentRequest,
    AgentResponse,
    ErrorResponse,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    LLMUsageResponse,
    MaskPIIRequest,
    MaskPIIResponse,
    MetricsResponse,
    MonitorRequest,
    MonitorResponse,
    NavigateNLRequest,
    NavigateNLResponse,
    NavigateRequest,
    NavigateResponse,
    ObserveRequest,
    ObserveResponse,
    PageMetricsResponse,
    RecordingDownloadResponse,
    RecordingListResponse,
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
    StreamStartRequest,
    StreamStartResponse,
    StreamStatusResponse,
    StreamStopResponse,
    TimingResponse,
    WorkflowRequest,
    WorkflowResponse,
)
from flybrowser.service.session_manager import SessionManager
from flybrowser.utils.logger import logger

# Global state
session_manager: SessionManager = None
streaming_manager = None
recording_storage = None
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
    global session_manager, streaming_manager, recording_storage, start_time

    # Startup
    logger.info("Starting FlyBrowser service...")
    session_manager = SessionManager()
    
    # Initialize streaming manager
    from flybrowser.service.streaming import StreamingManager
    from flybrowser.service.config import get_config, create_storage_backend
    
    config = get_config()
    if config.streaming_enabled:
        streaming_manager = StreamingManager(
            output_dir=config.recording_output_dir + "/streams",
            base_url=config.streaming_base_url or f"http://{config.host}:{config.port}",
            max_concurrent_streams=10,
        )
        await streaming_manager.start()
        logger.info("Streaming manager initialized")
    
    # Initialize recording storage
    if config.recording_enabled:
        recording_storage = create_storage_backend(config)
        logger.info(f"Recording storage initialized: {config.recording_storage}")
    
    start_time = time.time()
    logger.info("FlyBrowser service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down FlyBrowser service...")
    if streaming_manager:
        await streaming_manager.stop()
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

## Quick Start

1. Create a session: `POST /sessions`
2. Navigate to a URL: `POST /sessions/{session_id}/navigate`
3. Take a screenshot: `POST /sessions/{session_id}/screenshot`
4. Close the session: `DELETE /sessions/{session_id}`

## Deployment Modes

- **Standalone**: Single node deployment (default)
- **Cluster**: Multi-node deployment for horizontal scaling

## Resources

- [GitHub Repository](https://github.com/fireflyresearch/flybrowsers)
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
            "name": "Automation",
            "description": "High-level automation endpoints - autonomous mode (`auto`) for complex goal execution and schema-validated web scraping (`scrape`) with pagination and validators",
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

# Mount static files for player CSS/JS
app.mount("/static", StaticFiles(directory=str(get_static_dir())), name="static")


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
async def get_metrics():
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
):
    """Extract data from the current page."""
    try:
        browser = session_manager.get_session(session_id)

        # Use return_metadata=True to get full response with metrics
        response = await browser.extract(
            query=request.query,
            context=request.context,
            use_vision=request.use_vision,
            schema=request.schema,
            return_metadata=True,
        )

        # Build response with metrics
        return ExtractResponse(
            success=response.success,
            data=response.data if response.success else {},
            cached=response.metadata.get("cached", False),
            metadata=response.metadata,
            llm_usage=LLMUsageResponse(
                prompt_tokens=response.llm_usage.prompt_tokens,
                completion_tokens=response.llm_usage.completion_tokens,
                total_tokens=response.llm_usage.total_tokens,
                cost_usd=response.llm_usage.cost_usd,
                model=response.llm_usage.model,
                calls_count=response.llm_usage.calls_count,
                cached_calls=response.llm_usage.cached_calls,
            ),
            page_metrics=PageMetricsResponse(
                url=response.page_metrics.url,
                title=response.page_metrics.title,
                html_size_bytes=response.page_metrics.html_size_bytes,
                html_size_kb=response.page_metrics.html_size_kb,
                element_count=response.page_metrics.element_count,
                interactive_element_count=response.page_metrics.interactive_element_count,
                obstacles_detected=response.page_metrics.obstacles_detected,
                obstacles_dismissed=response.page_metrics.obstacles_dismissed,
            ),
            timing=TimingResponse(
                total_ms=response.timing.total_ms,
                phases=response.timing.phases,
                started_at=response.timing.started_at,
                ended_at=response.timing.ended_at,
            ),
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
):
    """Perform an action on the page."""
    try:
        browser = session_manager.get_session(session_id)

        # Use return_metadata=True to get full response with metrics
        response = await browser.act(
            instruction=request.instruction,
            context=request.context,
            use_vision=request.use_vision,
            return_metadata=True,
        )

        return ActionResponse(
            success=response.success,
            action_type="act",
            element_found=response.success,
            duration_ms=int(response.timing.total_ms),
            metadata=response.metadata,
            llm_usage=LLMUsageResponse(
                prompt_tokens=response.llm_usage.prompt_tokens,
                completion_tokens=response.llm_usage.completion_tokens,
                total_tokens=response.llm_usage.total_tokens,
                cost_usd=response.llm_usage.cost_usd,
                model=response.llm_usage.model,
                calls_count=response.llm_usage.calls_count,
                cached_calls=response.llm_usage.cached_calls,
            ),
            page_metrics=PageMetricsResponse(
                url=response.page_metrics.url,
                title=response.page_metrics.title,
                html_size_bytes=response.page_metrics.html_size_bytes,
                html_size_kb=response.page_metrics.html_size_kb,
                element_count=response.page_metrics.element_count,
                interactive_element_count=response.page_metrics.interactive_element_count,
                obstacles_detected=response.page_metrics.obstacles_detected,
                obstacles_dismissed=response.page_metrics.obstacles_dismissed,
            ),
            timing=TimingResponse(
                total_ms=response.timing.total_ms,
                phases=response.timing.phases,
                started_at=response.timing.started_at,
                ended_at=response.timing.ended_at,
            ),
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
):
    """Navigate using natural language instructions."""
    try:
        browser = session_manager.get_session(session_id)
        result = await browser.navigate(
            instruction=request.instruction,
            context=request.context,
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


# Agent endpoint (primary interface)
@app.post(
    "/sessions/{session_id}/agent",
    response_model=AgentResponse,
    tags=["Automation"],
    summary="Execute agent task (recommended)",
    description="Execute a complex task using the intelligent agent. "
                "This is the recommended endpoint for multi-step browser automation.",
)
async def execute_agent(
    session_id: str,
    request: AgentRequest,
):
    """
    Execute a task using the intelligent agent.
    
    This is the primary and recommended endpoint for complex browser automation.
    The agent automatically selects the optimal reasoning strategy and adapts
    dynamically during execution.
    
    **Features:**
    - Automatic strategy selection based on task complexity
    - Multi-tool orchestration (16+ browser tools)
    - Automatic obstacle handling (cookie banners, modals, popups)
    - Memory-based context retention
    - Dynamic strategy adaptation
    
    **Parameters:**
    - **task**: Natural language description of what to accomplish
    - **context**: User-provided context (preferences, constraints, form data)
    - **max_iterations**: Maximum actions before stopping (default: 50)
    - **max_time_seconds**: Maximum execution time (default: 1800 / 30 min)
    
    **Returns:**
    - Success status and any extracted data
    - Execution metrics (iterations, duration)
    - LLM usage statistics
    - Execution history
    """
    try:
        browser = session_manager.get_session(session_id)
        
        # Execute agent task using the agent() method
        result = await browser.agent(
            task=request.task,
            context=request.context,
            max_iterations=request.max_iterations,
            max_time_seconds=request.max_time_seconds,
            return_metadata=False,  # Get raw dict for API response
        )
        
        # Handle both dict and AgentRequestResponse
        if hasattr(result, 'to_dict'):
            result = result.to_dict()
        elif hasattr(result, 'success'):
            result = {
                "success": result.success,
                "result": result.data,
                "error": result.error,
            }
        
        return AgentResponse(
            success=result.get("success", False),
            task=request.task,
            result_data=result.get("result") or result.get("result_data"),
            iterations=result.get("total_iterations", 0) or result.get("iterations", 0),
            duration_seconds=result.get("execution_time_ms", 0) / 1000 if result.get("execution_time_ms") else result.get("duration_seconds", 0.0),
            final_url=result.get("final_url", ""),
            error_message=result.get("error"),
            execution_history=result.get("steps", []) or result.get("execution_history", []),
            llm_usage=None,  # TODO: Extract from result if available
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(e)}",
        )


# Observe endpoint
@app.post(
    "/sessions/{session_id}/observe",
    response_model=ObserveResponse,
    tags=["Automation"],
    summary="Observe and find elements",
    description="Analyze the page to find elements matching a natural language description.",
)
async def observe_elements(
    session_id: str,
    request: ObserveRequest,
):
    """
    Observe and identify elements on the current page.
    
    Analyzes the page to find elements matching a natural language description.
    Returns selectors, element info, and actionable suggestions.
    
    **Use cases:**
    - Find elements before acting on them
    - Understand page structure
    - Get reliable selectors for automation
    - Verify elements exist before interaction
    
    **Parameters:**
    - **query**: Natural language description of what to find
    - **return_selectors**: Include CSS selectors in response (default: true)
    
    **Returns:**
    - List of found elements with selectors and descriptions
    - Confidence scores
    - Actionability information
    """
    try:
        browser = session_manager.get_session(session_id)
        
        # Execute observe using the observe() method
        result = await browser.observe(
            query=request.query,
            context=request.context,
            return_selectors=request.return_selectors,
            return_metadata=False,  # Get raw data for API response
        )
        
        # Handle both dict/list and AgentRequestResponse
        elements = []
        page_url = None
        success = True
        error = None
        
        if isinstance(result, list):
            elements = result
            success = len(result) > 0
        elif hasattr(result, 'data'):
            elements = result.data if isinstance(result.data, list) else [result.data] if result.data else []
            success = result.success
            error = result.error
        elif isinstance(result, dict):
            elements = result.get("elements", [])
            page_url = result.get("page_url")
            success = result.get("success", len(elements) > 0)
            error = result.get("error")
        
        return ObserveResponse(
            success=success,
            elements=elements,
            page_url=page_url,
            error=error,
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    except Exception as e:
        logger.error(f"Observe failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Observe failed: {str(e)}",
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


# Streaming endpoints
@app.post(
    "/sessions/{session_id}/stream/start",
    response_model=StreamStartResponse,
    tags=["Streaming"],
    summary="Start streaming",
    description="Start a live stream of the browser session.",
)
async def start_stream(
    session_id: str,
    request: StreamStartRequest,
):
    """Start streaming the browser session."""
    if not streaming_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Streaming not enabled",
        )
    
    try:
        browser = session_manager.get_session(session_id)
        page = browser.browser_manager.page
        
        # Build stream configuration
        from flybrowser.service.streaming import StreamConfig
        from flybrowser.core.ffmpeg_recorder import StreamingProtocol, QualityProfile, VideoCodec
        
        config = StreamConfig(
            protocol=StreamingProtocol(request.protocol),
            quality_profile=QualityProfile(request.quality),
            codec=VideoCodec(request.codec),
            width=request.width or 1920,
            height=request.height or 1080,
            frame_rate=request.frame_rate or 30,
            rtmp_url=request.rtmp_url,
            rtmp_key=request.rtmp_key,
            max_viewers=request.max_viewers,
        )
        
        # Start stream
        stream_info = await streaming_manager.create_stream(session_id, page, config)
        
        return StreamStartResponse(
            success=True,
            stream_id=stream_info.stream_id,
            hls_url=stream_info.hls_url,
            dash_url=stream_info.dash_url,
            rtmp_url=stream_info.rtmp_url,
            websocket_url=stream_info.websocket_url,
            player_url=stream_info.player_url,
        )
        
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Start stream failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Start stream failed: {str(e)}",
        )


@app.get(
    "/sessions/{session_id}/stream/status",
    response_model=StreamStatusResponse,
    tags=["Streaming"],
    summary="Get stream status",
    description="Get current status and metrics of the stream.",
)
async def get_stream_status(
    session_id: str,
):
    """Get stream status for a session."""
    if not streaming_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Streaming not enabled",
        )
    
    try:
        streams = await streaming_manager.list_streams(session_id=session_id)
        if not streams:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active stream for session {session_id}",
            )
        
        stream_info = streams[0]
        return StreamStatusResponse(
            stream_id=stream_info.stream_id,
            session_id=stream_info.session_id,
            state=stream_info.state.value,
            health=stream_info.health.value,
            protocol=stream_info.protocol.value,
            started_at=stream_info.started_at,
            uptime_seconds=time.time() - stream_info.started_at,
            viewer_count=len(stream_info.viewer_ids),
            metrics=stream_info.metrics.to_dict(),
            urls={
                "hls": stream_info.hls_url,
                "dash": stream_info.dash_url,
                "rtmp": stream_info.rtmp_url,
                "websocket": stream_info.websocket_url,
                "player": stream_info.player_url,
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get stream status failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Get stream status failed: {str(e)}",
        )


@app.post(
    "/sessions/{session_id}/stream/stop",
    response_model=StreamStopResponse,
    tags=["Streaming"],
    summary="Stop streaming",
    description="Stop the active stream for this session.",
)
async def stop_stream(
    session_id: str,
):
    """Stop streaming for a session."""
    if not streaming_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Streaming not enabled",
        )
    
    try:
        streams = await streaming_manager.list_streams(session_id=session_id)
        if not streams:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active stream for session {session_id}",
            )
        
        stream_info = streams[0]
        final_info = await streaming_manager.stop_stream(stream_info.stream_id)
        
        if not final_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Stream not found",
            )
        
        return StreamStopResponse(
            success=True,
            stream_id=final_info.stream_id,
            duration_seconds=final_info.ended_at - final_info.started_at if final_info.ended_at else 0,
            total_viewers=len(final_info.viewer_ids),
            frames_sent=final_info.metrics.frames_sent,
            bytes_sent=final_info.metrics.bytes_sent,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stop stream failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stop stream failed: {str(e)}",
        )


# Recording management endpoints
@app.get(
    "/recordings",
    response_model=RecordingListResponse,
    tags=["Recording"],
    summary="List recordings",
    description="List all available recordings with optional filtering.",
)
async def list_recordings(
    session_id: Optional[str] = None,
    limit: int = 100,
):
    """List available recordings."""
    if not recording_storage:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recording storage not enabled",
        )
    
    try:
        recordings = await recording_storage.list(session_id=session_id, limit=limit)
        return RecordingListResponse(
            recordings=[rec.to_dict() for rec in recordings],
            total=len(recordings),
        )
    except Exception as e:
        logger.error(f"List recordings failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"List recordings failed: {str(e)}",
        )


@app.get(
    "/recordings/{recording_id}/download",
    response_model=RecordingDownloadResponse,
    tags=["Recording"],
    summary="Get recording download info",
    description="Get download information for a recording (presigned URL if S3).",
)
async def get_recording_download(
    recording_id: str,
):
    """Get download information for a recording."""
    if not recording_storage:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recording storage not enabled",
        )
    
    try:
        recording_info = await recording_storage.retrieve(recording_id)
        if not recording_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Recording not found: {recording_id}",
            )
        
        # Generate presigned URL if S3 storage
        download_url = None
        expires_at = None
        
        from flybrowser.service.cluster.storage import S3Storage
        if isinstance(recording_storage, S3Storage):
            download_url = await recording_storage.get_presigned_url(recording_id, expiration=86400)
            if download_url:
                expires_at = time.time() + 86400  # 24 hours
        else:
            # For local storage, provide direct path (in production, serve via static files)
            download_url = f"/recordings/{recording_id}/file"
        
        return RecordingDownloadResponse(
            recording_id=recording_info.recording_id,
            file_name=recording_info.file_name,
            file_size_bytes=recording_info.file_size_bytes,
            download_url=download_url,
            expires_at=expires_at,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get recording download failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Get recording download failed: {str(e)}",
        )


@app.delete(
    "/recordings/{recording_id}",
    tags=["Recording"],
    summary="Delete recording",
    description="Delete a recording from storage.",
)
async def delete_recording(
    recording_id: str,
):
    """Delete a recording."""
    if not recording_storage:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recording storage not enabled",
        )
    
    try:
        success = await recording_storage.delete(recording_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Recording not found: {recording_id}",
            )
        
        return {"success": True, "recording_id": recording_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete recording failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete recording failed: {str(e)}",
        )


# Stream serving endpoints (for HLS/DASH)
from fastapi.responses import PlainTextResponse, Response, HTMLResponse


@app.get(
    "/flybrowser/blank",
    response_class=HTMLResponse,
    tags=["Health"],
    summary="FlyBrowser blank page",
    description="Get the FlyBrowser custom blank page shown when the browser starts.",
)
async def get_blank_page():
    """Serve the FlyBrowser custom blank page.
    
    This page is displayed when the browser starts, showing a branded
    'waiting for agent' state instead of a plain white about:blank page.
    """
    html = render_blank_html(
        static_url="/static",
        inline_assets=False,
    )
    return HTMLResponse(content=html)


@app.get(
    "/streams/{stream_id}/player",
    response_class=HTMLResponse,
    tags=["Streaming"],
    summary="Embedded web player",
    description="Get an embedded web player page for the stream.",
)
async def get_stream_player(stream_id: str):
    """Serve embedded web player for the stream."""
    if not streaming_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Streaming not enabled",
        )
    
    try:
        stream_info = await streaming_manager.get_stream(stream_id)
        if not stream_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Stream not found",
            )
        
        # Determine protocol and URLs
        protocol = stream_info.protocol.value if hasattr(stream_info.protocol, 'value') else str(stream_info.protocol)
        hls_url = stream_info.hls_url or ''
        dash_url = stream_info.dash_url or ''
        
        # Get quality label from stream config
        config = stream_info.config
        quality_label = f"{config.width}x{config.height}@{config.frame_rate}fps"
        if hasattr(config, 'quality_profile'):
            profile_name = config.quality_profile.value if hasattr(config.quality_profile, 'value') else str(config.quality_profile)
            quality_label = f"{profile_name.upper()} ({config.width}x{config.height})"
        
        # Get quality profile for player optimization
        quality_profile = ""
        if hasattr(config, 'quality_profile'):
            quality_profile = config.quality_profile.value if hasattr(config.quality_profile, 'value') else str(config.quality_profile)
        
        # Render HTML with stream details using Jinja2 template
        html = render_player_html(
            stream_id=stream_id,
            protocol=protocol,
            hls_url=hls_url,
            dash_url=dash_url,
            quality=quality_label,
            quality_profile=quality_profile,
            static_url="/static",
            inline_assets=False,
        )
        
        return HTMLResponse(content=html)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get stream player failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get(
    "/streams/{stream_id}/playlist.m3u8",
    response_class=PlainTextResponse,
    tags=["Streaming"],
    summary="Get HLS playlist",
    description="Get the HLS playlist for a stream.",
)
async def get_hls_playlist(stream_id: str):
    """Serve HLS playlist."""
    if not streaming_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Streaming not enabled",
        )
    
    try:
        playlist = await streaming_manager.get_playlist(stream_id)
        if not playlist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Playlist not found",
            )
        
        return PlainTextResponse(content=playlist, media_type="application/vnd.apple.mpegurl")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get HLS playlist failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get(
    "/streams/{stream_id}/{segment_name}",
    response_class=Response,
    tags=["Streaming"],
    summary="Get HLS segment",
    description="Get an HLS segment file.",
)
async def get_hls_segment(stream_id: str, segment_name: str):
    """Serve HLS segment."""
    if not streaming_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Streaming not enabled",
        )
    
    # Security: Only allow .ts files
    if not segment_name.endswith(".ts"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid segment name",
        )
    
    try:
        segment_data = await streaming_manager.get_segment(stream_id, segment_name)
        if not segment_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Segment not found",
            )
        
        return Response(content=segment_data, media_type="video/mp2t")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get HLS segment failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
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
