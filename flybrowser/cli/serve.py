#!/usr/bin/env python3
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

"""FlyBrowser Server CLI.

Command-line interface for starting the FlyBrowser service.

Usage:
    flybrowser-serve [--host HOST] [--port PORT] [--reload]
    flybrowser-serve --cluster --node-id node1 --peers node2:4321,node3:4321
    
    Or with Python:
    python -m flybrowser.cli.serve

Environment Variables (for cluster mode):
    FLYBROWSER_CLUSTER_ENABLED=true
    FLYBROWSER_NODE_ID=node1
    FLYBROWSER_CLUSTER_PEERS=node2:4321,node3:4321
    FLYBROWSER_RAFT_PORT=4321
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def get_banner() -> str:
    """Get the FlyBrowser banner from banner.txt or fallback.

    Returns:
        The banner string
    """
    # Try to load from banner.txt
    banner_paths = [
        Path(__file__).parent.parent / "banner.txt",  # flybrowser/banner.txt
        Path(__file__).parent.parent.parent / "flybrowser" / "banner.txt",
    ]

    for banner_path in banner_paths:
        if banner_path.exists():
            try:
                return banner_path.read_text()
            except Exception:
                pass

    # Fallback banner (matches banner.txt)
    return r"""  _____.__         ___.
_/ ____\  | ___.__.\_ |_________  ______  _  ________ ___________
\   __\|  |<   |  | | __ \_  __ \/  _ \ \/ \/ /  ___// __ \_  __ \
 |  |  |  |_\___  | | \_\ \  | \(  <_> )     /\___ \\  ___/|  | \/
 |__|  |____/ ____| |___  /__|   \____/ \/\_//____  >\___  >__|
            \/          \/                        \/     \/"""


def main() -> None:
    """Main entry point for the serve command."""
    parser = argparse.ArgumentParser(
        description="Start the FlyBrowser service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  flybrowser-serve                    # Start standalone server
  flybrowser-serve --port 8080        # Custom port
  flybrowser-serve --reload           # Development mode with auto-reload
  flybrowser-serve --workers 4        # Production mode with 4 workers
  
  # Cluster mode (HA):
  flybrowser-serve --cluster --node-id node1 --peers node2:4321,node3:4321
  
  # Or via environment variables:
  FLYBROWSER_CLUSTER_ENABLED=true FLYBROWSER_NODE_ID=node1 flybrowser-serve
        """,
    )
    
    # Basic options
    parser.add_argument(
        "--host",
        default=os.environ.get("FLYBROWSER_HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("FLYBROWSER_PORT", "8000")),
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("FLYBROWSER_WORKERS", "1")),
        help="Number of worker processes (default: 1, ignored in cluster mode)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("FLYBROWSER_LOG_LEVEL", "info"),
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level (default: info)",
    )
    
    # Cluster mode options
    parser.add_argument(
        "--cluster",
        action="store_true",
        default=os.environ.get("FLYBROWSER_CLUSTER_ENABLED", "").lower() == "true",
        help="Enable cluster mode (HA)",
    )
    parser.add_argument(
        "--node-id",
        default=os.environ.get("FLYBROWSER_NODE_ID", ""),
        help="Unique node identifier (auto-generated if not provided)",
    )
    parser.add_argument(
        "--peers",
        default=os.environ.get("FLYBROWSER_CLUSTER_PEERS", ""),
        help="Comma-separated list of peer addresses (host:raft_port)",
    )
    parser.add_argument(
        "--raft-port",
        type=int,
        default=int(os.environ.get("FLYBROWSER_RAFT_PORT", "4321")),
        help="Raft consensus port (default: 4321)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("FLYBROWSER_DATA_DIR", "./data"),
        help="Data directory for persistent storage (default: ./data)",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=int(os.environ.get("FLYBROWSER_MAX_SESSIONS", "10")),
        help="Maximum sessions per node (default: 10)",
    )
    
    args = parser.parse_args()
    
    # Import uvicorn here to avoid import errors if not installed
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is not installed. Install it with:")
        print("  pip install uvicorn[standard]")
        sys.exit(1)
    
    # Determine which app to run
    if args.cluster:
        app_module = "flybrowser.service.ha_app:app"
        mode = "cluster"
        
        # Set environment variables for HA app
        if args.node_id:
            os.environ["FLYBROWSER_NODE_ID"] = args.node_id
        if args.peers:
            os.environ["FLYBROWSER_CLUSTER_PEERS"] = args.peers
        os.environ["FLYBROWSER_RAFT_PORT"] = str(args.raft_port)
        os.environ["FLYBROWSER_DATA_DIR"] = args.data_dir
        os.environ["FLYBROWSER_MAX_SESSIONS"] = str(args.max_sessions)
        os.environ["FLYBROWSER_API_HOST"] = args.host
        os.environ["FLYBROWSER_API_PORT"] = str(args.port)
    else:
        app_module = "flybrowser.service.app:app"
        mode = "standalone"
    
    print()
    print(get_banner())
    print()
    print("  Browser Automation Powered by LLM Agents")
    print()
    print(f"  Starting server...")
    print()
    print(f"  Mode:      {mode}")
    print(f"  Host:      {args.host}")
    print(f"  Port:      {args.port}")
    
    if args.cluster:
        print(f"  Node ID:   {args.node_id or '(auto-generated)'}")
        print(f"  Raft Port: {args.raft_port}")
        print(f"  Peers:     {args.peers or '(bootstrap node)'}")
        print(f"  Data Dir:  {args.data_dir}")
        print(f"  Max Sess:  {args.max_sessions}")
    else:
        print(f"  Workers:   {args.workers}")
    
    print(f"  Reload:    {args.reload}")
    print(f"  Log Level: {args.log_level}")
    print()
    print(f"  API Docs:  http://{args.host}:{args.port}/docs")
    print(f"  Health:    http://{args.host}:{args.port}/health")
    
    if args.cluster:
        print(f"  Cluster:   http://{args.host}:{args.port}/cluster/status")
    
    print()
    
    # Cluster mode always runs with 1 worker (state is managed internally)
    workers = 1 if args.cluster else (args.workers if not args.reload else 1)
    
    uvicorn.run(
        app_module,
        host=args.host,
        port=args.port,
        workers=workers,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()

