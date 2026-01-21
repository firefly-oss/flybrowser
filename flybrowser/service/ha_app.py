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
High-Availability FastAPI Application for FlyBrowser.

This module provides an HA-aware FastAPI application that integrates with
the Raft consensus cluster. Features include:

- Automatic leader detection and redirect
- Session routing to correct nodes
- Cluster status endpoints
- Transparent failover handling

The app works in both standalone and cluster modes.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from flybrowser import __version__
from flybrowser.service.auth import verify_api_key
from flybrowser.service.cluster.ha_node import HAClusterNode, HANodeConfig
from flybrowser.service.cluster.raft import NodeRole
from flybrowser.service.cluster.exceptions import (
    NotLeaderError,
    SessionNotFoundError,
    NodeCapacityError,
    ConsistencyError,
)
from flybrowser.utils.logger import logger


# Global HA node instance
_ha_node: Optional[HAClusterNode] = None
_start_time: float = 0


def get_ha_node() -> HAClusterNode:
    """Get the HA cluster node instance."""
    if _ha_node is None:
        raise RuntimeError("HA node not initialized")
    return _ha_node


def create_ha_config_from_env() -> HANodeConfig:
    """Create HANodeConfig from environment variables."""
    peers_str = os.environ.get("FLYBROWSER_CLUSTER_PEERS", "")
    peers = [p.strip() for p in peers_str.split(",") if p.strip()]
    
    return HANodeConfig(
        node_id=os.environ.get("FLYBROWSER_NODE_ID", ""),
        api_host=os.environ.get("FLYBROWSER_API_HOST", "0.0.0.0"),
        api_port=int(os.environ.get("FLYBROWSER_API_PORT", "8000")),
        raft_host=os.environ.get("FLYBROWSER_RAFT_HOST", "0.0.0.0"),
        raft_port=int(os.environ.get("FLYBROWSER_RAFT_PORT", "4321")),
        peers=peers,
        data_dir=os.environ.get("FLYBROWSER_DATA_DIR", "./data"),
        max_sessions=int(os.environ.get("FLYBROWSER_MAX_SESSIONS", "10")),
    )


@asynccontextmanager
async def ha_lifespan(app: FastAPI):
    """Lifespan for HA-aware application."""
    global _ha_node, _start_time
    
    logger.info("Starting FlyBrowser HA service...")
    
    # Create and start HA node
    config = create_ha_config_from_env()
    _ha_node = HAClusterNode(config)
    await _ha_node.start()
    
    _start_time = time.time()
    logger.info(f"FlyBrowser HA service started (node: {_ha_node.node_id})")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FlyBrowser HA service...")
    await _ha_node.stop()
    logger.info("FlyBrowser HA service shut down")


def create_ha_app(config: Optional[HANodeConfig] = None) -> FastAPI:
    """Create an HA-aware FastAPI application.
    
    Args:
        config: Optional HANodeConfig. If not provided, reads from environment.
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="FlyBrowser HA API",
        description="High-Availability Browser Automation API",
        version=__version__,
        lifespan=ha_lifespan,
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store config for lifespan
    if config:
        app.state.ha_config = config
    
    # ==================== Middleware ====================
    
    @app.middleware("http")
    async def leader_redirect_middleware(request: Request, call_next):
        """Redirect write operations to leader if not leader."""
        # Skip for health/status endpoints
        if request.url.path in ["/health", "/cluster/status", "/raft/status"]:
            return await call_next(request)
        
        # Skip for GET requests (reads can be served by any node)
        if request.method == "GET":
            return await call_next(request)
        
        # Check if we're the leader for write operations
        node = get_ha_node()
        if not node.is_leader:
            leader_addr = node.get_leader_api_address()
            if leader_addr:
                # Redirect to leader
                redirect_url = f"http://{leader_addr}{request.url.path}"
                if request.url.query:
                    redirect_url += f"?{request.url.query}"
                return RedirectResponse(
                    url=redirect_url,
                    status_code=status.HTTP_307_TEMPORARY_REDIRECT,
                )
            else:
                # No leader known
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={"error": "No leader available", "retry_after": 1},
                )
        
        return await call_next(request)
    
    # ==================== Health Endpoints ====================
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        node = get_ha_node()
        return {
            "status": "healthy",
            "node_id": node.node_id,
            "role": node.role.value,
            "is_leader": node.is_leader,
        }

    @app.get("/cluster/status")
    async def cluster_status():
        """Get cluster status."""
        node = get_ha_node()
        return node.get_status()

    @app.get("/raft/status")
    async def raft_status():
        """Get Raft consensus status."""
        node = get_ha_node()
        return node._raft.get_status()

    @app.get("/cluster/nodes")
    async def list_nodes():
        """List all nodes in the cluster."""
        node = get_ha_node()
        nodes = node.state_machine.get_all_nodes()
        return {
            "nodes": [n.to_dict() for n in nodes],
            "leader_id": node.leader_id,
        }

    @app.get("/cluster/sessions")
    async def list_sessions(consistency: str = "eventual"):
        """List all sessions in the cluster.
        
        Args:
            consistency: Read consistency level (eventual or strong)
        """
        node = get_ha_node()
        
        # For strong consistency, route to leader
        if consistency == "strong" and not node.is_leader:
            leader_addr = node.get_leader_api_address()
            if leader_addr:
                return RedirectResponse(
                    url=f"http://{leader_addr}/cluster/sessions?consistency=strong",
                    status_code=status.HTTP_307_TEMPORARY_REDIRECT,
                )
        
        sessions = node.state_machine.get_all_sessions()
        return {
            "sessions": [s.to_dict() for s in sessions],
            "total": len(sessions),
            "consistency": consistency,
        }

    # ==================== Cluster Management Endpoints ====================

    @app.post("/cluster/nodes/{node_id}/drain")
    async def drain_node(node_id: str, request: Request):
        """Drain all sessions from a node.
        
        This migrates all sessions from the specified node to other healthy
        nodes in the cluster, preparing the node for maintenance or removal.
        """
        node = get_ha_node()
        
        if not node.is_leader:
            raise HTTPException(
                status_code=503,
                detail="This operation must be performed on the leader node"
            )
        
        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        force = body.get("force", False)
        
        # Get sessions on the target node
        all_sessions = node.state_machine.get_all_sessions()
        target_sessions = [s for s in all_sessions if s.node_id == node_id]
        
        if not target_sessions:
            return {
                "success": True,
                "migrated_sessions": 0,
                "failed_sessions": 0,
                "message": f"No sessions on node {node_id}",
            }
        
        # Try to migrate each session
        migrated = 0
        failed = 0
        errors = []
        
        for session in target_sessions:
            try:
                # Find a healthy target node
                nodes = node.state_machine.get_all_nodes()
                target = None
                for n in nodes:
                    if n.node_id != node_id and n.health == "healthy" and n.available_capacity > 0:
                        target = n
                        break
                
                if not target:
                    if force:
                        failed += 1
                        errors.append(f"Session {session.session_id}: No healthy target node")
                        continue
                    else:
                        return {
                            "success": False,
                            "error": "No healthy target nodes available for migration",
                            "migrated_sessions": migrated,
                            "failed_sessions": failed + len(target_sessions) - migrated,
                        }
                
                # Perform migration
                await node.migrate_session(session.session_id, target.node_id)
                migrated += 1
                
            except Exception as e:
                if force:
                    failed += 1
                    errors.append(f"Session {session.session_id}: {str(e)}")
                else:
                    return {
                        "success": False,
                        "error": str(e),
                        "migrated_sessions": migrated,
                        "failed_sessions": failed + 1,
                    }
        
        return {
            "success": failed == 0,
            "migrated_sessions": migrated,
            "failed_sessions": failed,
            "errors": errors if errors else None,
        }

    @app.post("/cluster/step-down")
    async def step_down(request: Request):
        """Request the leader to step down and transfer leadership.
        
        Optionally specify a target node to transfer leadership to.
        """
        node = get_ha_node()
        
        if not node.is_leader:
            return {
                "success": False,
                "error": "This node is not the leader",
                "current_leader": node.leader_id,
            }
        
        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        target_node = body.get("target_node")
        
        try:
            # Trigger leadership transfer via Raft node
            result = await node._raft._transfer_leadership(target_node)
            
            if result:
                return {
                    "success": True,
                    "message": "Leadership transfer initiated",
                    "new_leader": target_node or "pending",
                }
            else:
                return {
                    "success": False,
                    "error": "Leadership transfer failed",
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    @app.post("/cluster/rebalance")
    async def rebalance():
        """Trigger manual session rebalancing across the cluster.
        
        This moves sessions from overloaded nodes to underloaded ones.
        """
        node = get_ha_node()
        
        if not node.is_leader:
            raise HTTPException(
                status_code=503,
                detail="This operation must be performed on the leader node"
            )
        
        try:
            # Get all nodes and their load
            nodes = node.state_machine.get_all_nodes()
            healthy_nodes = [n for n in nodes if n.health == "healthy"]
            
            if len(healthy_nodes) < 2:
                return {
                    "success": True,
                    "sessions_moved": 0,
                    "message": "Not enough healthy nodes for rebalancing",
                }
            
            # Calculate average load
            total_sessions = sum(n.active_sessions for n in healthy_nodes)
            avg_sessions = total_sessions / len(healthy_nodes)
            
            # Find overloaded and underloaded nodes
            overloaded = [(n, n.active_sessions - avg_sessions) 
                         for n in healthy_nodes if n.active_sessions > avg_sessions * 1.2]
            underloaded = [(n, avg_sessions - n.active_sessions) 
                          for n in healthy_nodes if n.active_sessions < avg_sessions * 0.8 
                          and n.available_capacity > 0]
            
            if not overloaded or not underloaded:
                return {
                    "success": True,
                    "sessions_moved": 0,
                    "message": "Cluster is already balanced",
                }
            
            sessions_moved = 0
            
            # Move sessions from overloaded to underloaded
            for over_node, excess in sorted(overloaded, key=lambda x: -x[1]):
                node_sessions = [s for s in node.state_machine.get_all_sessions() 
                                if s.node_id == over_node.node_id]
                
                for session in node_sessions:
                    if excess <= 0:
                        break
                    
                    # Find best underloaded target
                    for under_node, deficit in sorted(underloaded, key=lambda x: -x[1]):
                        if deficit <= 0 or under_node.available_capacity <= 0:
                            continue
                        
                        try:
                            await node.migrate_session(session.session_id, under_node.node_id)
                            sessions_moved += 1
                            excess -= 1
                            # Update underloaded tracking
                            under_node.active_sessions += 1
                            under_node.available_capacity -= 1
                            break
                        except Exception:
                            continue
            
            return {
                "success": True,
                "sessions_moved": sessions_moved,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    # ==================== Session Endpoints ====================

    @app.post("/sessions")
    async def create_session(request: Request):
        """Create a new browser session.

        Accepts LLM configuration parameters in the request body:
        - llm_provider: LLM provider name (openai, anthropic, ollama)
        - llm_model: LLM model name (optional)
        - api_key: API key for the LLM provider
        - headless: Run browser in headless mode (default: true)
        - browser_type: Browser type (chromium, firefox, webkit)
        - client_id: Optional client identifier for session affinity
        """
        node = get_ha_node()

        body = await request.json() if request.headers.get("content-type") == "application/json" else {}

        # Extract LLM configuration from request body
        llm_config = {
            "llm_provider": body.get("llm_provider", "openai"),
            "llm_model": body.get("llm_model"),
            "api_key": body.get("api_key"),
            "headless": body.get("headless", True),
            "browser_type": body.get("browser_type", "chromium"),
        }
        client_id = body.get("client_id")

        try:
            session_id = await node.create_session(
                client_id=client_id,
                llm_config=llm_config,
            )

            # Get the node handling this session
            target_address = node.get_node_for_session(session_id)

            return {
                "session_id": session_id,
                "node_address": target_address,
                "status": "created",
            }
        except RuntimeError as e:
            if "Not leader" in str(e):
                leader_addr = node.get_leader_api_address()
                return RedirectResponse(
                    url=f"http://{leader_addr}/sessions",
                    status_code=status.HTTP_307_TEMPORARY_REDIRECT,
                )
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        """Delete a browser session."""
        node = get_ha_node()

        try:
            await node.delete_session(session_id)
            return {"status": "deleted", "session_id": session_id}
        except RuntimeError as e:
            if "Not leader" in str(e):
                leader_addr = node.get_leader_api_address()
                return RedirectResponse(
                    url=f"http://{leader_addr}/sessions/{session_id}",
                    status_code=status.HTTP_307_TEMPORARY_REDIRECT,
                )
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/sessions/{session_id}")
    async def get_session(
        session_id: str,
        consistency: str = "eventual",
    ):
        """Get session info and routing.
        
        Args:
            session_id: Session ID to look up
            consistency: Read consistency level:
                - "eventual": Read from any node (may be stale)
                - "strong": Read from leader (linearizable)
        """
        node = get_ha_node()
        
        # For strong consistency, route to leader
        if consistency == "strong" and not node.is_leader:
            leader_addr = node.get_leader_api_address()
            if leader_addr:
                return RedirectResponse(
                    url=f"http://{leader_addr}/sessions/{session_id}?consistency=strong",
                    status_code=status.HTTP_307_TEMPORARY_REDIRECT,
                )
            raise HTTPException(
                status_code=503,
                detail="Strong consistency requested but no leader available"
            )
        
        # For strong consistency on leader, verify we can serve reads
        if consistency == "strong" and node.is_leader:
            if not node._raft.can_serve_read():
                raise HTTPException(
                    status_code=503,
                    detail="Leader cannot serve reads (lease expired)"
                )

        session = node.state_machine.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        target_address = node.get_node_for_session(session_id)

        return {
            "session": session.to_dict(),
            "node_address": target_address,
            "consistency": consistency,
        }

    @app.get("/sessions/{session_id}/route")
    async def route_session(session_id: str):
        """Get the node address for a session (for client routing)."""
        node = get_ha_node()

        target_address = node.get_node_for_session(session_id)
        if not target_address:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "session_id": session_id,
            "node_address": target_address,
            "redirect_url": f"http://{target_address}",
        }

    # ==================== Session Operation Endpoints ====================
    # These endpoints handle browser operations and route to the correct node

    def _get_session_or_redirect(session_id: str):
        """Get session info and check if we should handle it locally or redirect."""
        node = get_ha_node()
        session = node.state_machine.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Check if session is on this node
        if session.node_id == node.node_id:
            return session, None  # Handle locally

        # Need to redirect to the correct node
        target_address = node.get_node_for_session(session_id)
        return session, target_address

    @app.post("/sessions/{session_id}/navigate")
    async def navigate(session_id: str, request: Request):
        """Navigate to a URL."""
        node = get_ha_node()
        session, redirect_addr = _get_session_or_redirect(session_id)

        if redirect_addr:
            return RedirectResponse(
                url=f"http://{redirect_addr}/sessions/{session_id}/navigate",
                status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            )

        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        url = body.get("url")
        wait_until = body.get("wait_until", "domcontentloaded")

        if not url:
            raise HTTPException(status_code=400, detail="URL is required")

        try:
            browser = node._session_manager.get_session(session_id)
            await browser.goto(url, wait_until=wait_until)

            title = await browser.page_controller.get_title()
            current_url = await browser.page_controller.get_url()

            return {
                "success": True,
                "url": current_url,
                "title": title,
            }
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found locally")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/sessions/{session_id}/extract")
    async def extract_data(session_id: str, request: Request):
        """Extract data from the current page."""
        node = get_ha_node()
        session, redirect_addr = _get_session_or_redirect(session_id)

        if redirect_addr:
            return RedirectResponse(
                url=f"http://{redirect_addr}/sessions/{session_id}/extract",
                status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            )

        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        instruction = body.get("instruction") or body.get("query")
        schema = body.get("schema")

        if not instruction:
            raise HTTPException(status_code=400, detail="Instruction/query is required")

        try:
            browser = node._session_manager.get_session(session_id)
            result = await browser.extract(instruction, schema=schema)

            return {
                "success": True,
                "data": result,
            }
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found locally")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/sessions/{session_id}/action")
    async def perform_action(session_id: str, request: Request):
        """Perform an action on the page."""
        node = get_ha_node()
        session, redirect_addr = _get_session_or_redirect(session_id)

        if redirect_addr:
            return RedirectResponse(
                url=f"http://{redirect_addr}/sessions/{session_id}/action",
                status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            )

        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        instruction = body.get("instruction")

        if not instruction:
            raise HTTPException(status_code=400, detail="Instruction is required")

        try:
            browser = node._session_manager.get_session(session_id)
            await browser.act(instruction)

            return {
                "success": True,
                "action_type": "act",
            }
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found locally")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/sessions/{session_id}/screenshot")
    async def take_screenshot(session_id: str, request: Request):
        """Take a screenshot of the current page."""
        node = get_ha_node()
        session, redirect_addr = _get_session_or_redirect(session_id)

        if redirect_addr:
            return RedirectResponse(
                url=f"http://{redirect_addr}/sessions/{session_id}/screenshot",
                status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            )

        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        full_page = body.get("full_page", False)

        try:
            browser = node._session_manager.get_session(session_id)
            result = await browser.screenshot(full_page=full_page)

            return {
                "success": True,
                **result,
            }
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found locally")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/sessions/{session_id}/recording/start")
    async def start_recording(session_id: str, request: Request):
        """Start recording the browser session."""
        node = get_ha_node()
        session, redirect_addr = _get_session_or_redirect(session_id)

        if redirect_addr:
            return RedirectResponse(
                url=f"http://{redirect_addr}/sessions/{session_id}/recording/start",
                status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            )

        try:
            browser = node._session_manager.get_session(session_id)
            result = await browser.start_recording()

            return {
                "success": True,
                **result,
            }
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found locally")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/sessions/{session_id}/recording/stop")
    async def stop_recording(session_id: str):
        """Stop recording and return recording data."""
        node = get_ha_node()
        session, redirect_addr = _get_session_or_redirect(session_id)

        if redirect_addr:
            return RedirectResponse(
                url=f"http://{redirect_addr}/sessions/{session_id}/recording/stop",
                status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            )

        try:
            browser = node._session_manager.get_session(session_id)
            result = await browser.stop_recording()

            return {
                "success": True,
                **result,
            }
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found locally")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/sessions/{session_id}/secure-fill")
    async def secure_fill(session_id: str, request: Request):
        """Securely fill a form field with a stored credential."""
        node = get_ha_node()
        session, redirect_addr = _get_session_or_redirect(session_id)

        if redirect_addr:
            return RedirectResponse(
                url=f"http://{redirect_addr}/sessions/{session_id}/secure-fill",
                status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            )

        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        selector = body.get("selector")
        credential_id = body.get("credential_id")
        clear_first = body.get("clear_first", True)

        if not selector or not credential_id:
            raise HTTPException(status_code=400, detail="selector and credential_id are required")

        try:
            browser = node._session_manager.get_session(session_id)
            success = await browser.secure_fill(selector, credential_id, clear_first)

            return {
                "success": success,
            }
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found locally")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/sessions/{session_id}/credentials")
    async def store_credential(session_id: str, request: Request):
        """Store a credential securely for later use."""
        node = get_ha_node()
        session, redirect_addr = _get_session_or_redirect(session_id)

        if redirect_addr:
            return RedirectResponse(
                url=f"http://{redirect_addr}/sessions/{session_id}/credentials",
                status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            )

        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        name = body.get("name")
        value = body.get("value")
        pii_type = body.get("pii_type", "password")

        if not name or not value:
            raise HTTPException(status_code=400, detail="name and value are required")

        try:
            browser = node._session_manager.get_session(session_id)
            from flybrowser.security.pii_handler import PIIType
            pii_type_enum = PIIType(pii_type)
            credential_id = browser.store_credential(name, value, pii_type_enum)

            return {
                "success": True,
                "credential_id": credential_id,
                "name": name,
                "pii_type": pii_type,
            }
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found locally")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/sessions/{session_id}/navigate-nl")
    async def navigate_natural_language(session_id: str, request: Request):
        """Navigate using natural language instructions."""
        node = get_ha_node()
        session, redirect_addr = _get_session_or_redirect(session_id)

        if redirect_addr:
            return RedirectResponse(
                url=f"http://{redirect_addr}/sessions/{session_id}/navigate-nl",
                status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            )

        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        instruction = body.get("instruction")
        use_vision = body.get("use_vision", True)

        if not instruction:
            raise HTTPException(status_code=400, detail="instruction is required")

        try:
            browser = node._session_manager.get_session(session_id)
            result = await browser.navigate(instruction, use_vision=use_vision)

            return {
                "success": result.get("success", False),
                "url": result.get("url"),
                "title": result.get("title"),
                "navigation_type": result.get("navigation_type"),
                "error": result.get("error"),
            }
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found locally")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/sessions/{session_id}/workflow")
    async def execute_workflow(session_id: str, request: Request):
        """Execute a multi-step workflow."""
        node = get_ha_node()
        session, redirect_addr = _get_session_or_redirect(session_id)

        if redirect_addr:
            return RedirectResponse(
                url=f"http://{redirect_addr}/sessions/{session_id}/workflow",
                status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            )

        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        workflow = body.get("workflow")
        variables = body.get("variables", {})

        if not workflow:
            raise HTTPException(status_code=400, detail="workflow is required")

        try:
            browser = node._session_manager.get_session(session_id)
            result = await browser.run_workflow(workflow, variables=variables)

            return {
                "success": result.get("success", False),
                "steps_completed": result.get("steps_completed", 0),
                "total_steps": result.get("total_steps", 0),
                "error": result.get("error"),
                "step_results": result.get("step_results", []),
                "variables": result.get("variables", {}),
            }
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found locally")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/sessions/{session_id}/monitor")
    async def monitor_condition(session_id: str, request: Request):
        """Monitor for a condition to be met."""
        node = get_ha_node()
        session, redirect_addr = _get_session_or_redirect(session_id)

        if redirect_addr:
            return RedirectResponse(
                url=f"http://{redirect_addr}/sessions/{session_id}/monitor",
                status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            )

        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        condition = body.get("condition")
        timeout = body.get("timeout", 30.0)
        poll_interval = body.get("poll_interval", 0.5)

        if not condition:
            raise HTTPException(status_code=400, detail="condition is required")

        try:
            browser = node._session_manager.get_session(session_id)
            result = await browser.monitor(condition, timeout=timeout, poll_interval=poll_interval)

            return {
                "success": result.get("success", False),
                "condition_met": result.get("condition_met", False),
                "elapsed_time": result.get("elapsed_time", 0.0),
                "error": result.get("error"),
                "details": result.get("details", {}),
            }
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found locally")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/pii/mask")
    async def mask_pii(request: Request):
        """Mask PII in text."""
        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        text = body.get("text", "")

        from flybrowser.security.pii_handler import PIIMasker
        masker = PIIMasker()
        masked_text = masker.mask_text(text)

        return {
            "original_length": len(text),
            "masked_text": masked_text,
        }

    return app


# Create default app instance
app = create_ha_app()

