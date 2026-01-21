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
FlyBrowser Admin CLI.

This module provides administrative CLI commands for FlyBrowser:
- sessions: List and manage active sessions
- nodes: List and manage cluster nodes
- migrate: Migrate sessions between nodes
- backup: Backup cluster state
- restore: Restore cluster state

Usage:
    flybrowser-admin sessions list
    flybrowser-admin sessions kill <session_id>
    flybrowser-admin nodes list
    flybrowser-admin backup --output backup.json
    flybrowser-admin restore --input backup.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp


def print_json(data: dict) -> None:
    """Print data as formatted JSON."""
    print(json.dumps(data, indent=2, default=str))


def print_table(headers: List[str], rows: List[List[str]]) -> None:
    """Print data as a formatted table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


async def api_request(
    endpoint: str,
    method: str = "GET",
    path: str = "/",
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Make an API request."""
    url = f"{endpoint.rstrip('/')}{path}"
    async with aiohttp.ClientSession() as session:
        async with session.request(method, url, json=data) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                text = await resp.text()
                raise Exception(f"API error {resp.status}: {text}")


def cmd_sessions_list(args: argparse.Namespace) -> int:
    """List all sessions."""
    async def run():
        try:
            data = await api_request(args.endpoint, "GET", "/cluster/sessions")
            sessions = data.get("sessions", [])

            if args.json:
                print_json({"sessions": sessions, "total": len(sessions)})
            else:
                print(f"\n=== Active Sessions ({len(sessions)}) ===\n")
                if sessions:
                    headers = ["Session ID", "Node", "Status", "Client", "Age"]
                    rows = []
                    for s in sessions:
                        created = s.get("created_at", 0)
                        age = datetime.now().timestamp() - created if created else 0
                        age_str = f"{int(age // 60)}m {int(age % 60)}s"
                        rows.append([
                            s.get("session_id", "")[:16],
                            s.get("node_id", ""),
                            s.get("status", ""),
                            s.get("client_id", "N/A") or "N/A",
                            age_str,
                        ])
                    print_table(headers, rows)
                else:
                    print("No active sessions.")
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    return asyncio.run(run())


def cmd_sessions_kill(args: argparse.Namespace) -> int:
    """Kill a session."""
    async def run():
        try:
            await api_request(args.endpoint, "DELETE", f"/sessions/{args.session_id}")
            print(f"Session {args.session_id} terminated.")
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    return asyncio.run(run())


def cmd_nodes_list(args: argparse.Namespace) -> int:
    """List all nodes."""
    async def run():
        try:
            data = await api_request(args.endpoint, "GET", "/cluster/nodes")
            nodes = data.get("nodes", [])

            if args.json:
                print_json({"nodes": nodes, "total": len(nodes)})
            else:
                print(f"\n=== Cluster Nodes ({len(nodes)}) ===\n")
                if nodes:
                    headers = ["Node ID", "Address", "Health", "CPU", "Memory", "Sessions", "Capacity"]
                    rows = []
                    for n in nodes:
                        rows.append([
                            n.get("node_id", ""),
                            n.get("api_address", ""),
                            n.get("health", ""),
                            f"{n.get('cpu_percent', 0):.1f}%",
                            f"{n.get('memory_percent', 0):.1f}%",
                            str(n.get("active_sessions", 0)),
                            str(n.get("available_capacity", 0)),
                        ])
                    print_table(headers, rows)
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    return asyncio.run(run())


def cmd_nodes_drain(args: argparse.Namespace) -> int:
    """Drain all sessions from a node.
    
    This migrates all sessions from the specified node to other healthy nodes
    in the cluster, preparing the node for maintenance or removal.
    """
    async def run():
        try:
            # Get node info
            data = await api_request(args.endpoint, "GET", "/cluster/nodes")
            nodes = data.get("nodes", [])
            target_node = None
            for n in nodes:
                if n.get("node_id") == args.node_id or n.get("api_address") == args.node_id:
                    target_node = n
                    break
            
            if not target_node:
                print(f"Error: Node '{args.node_id}' not found", file=sys.stderr)
                return 1
            
            node_id = target_node.get("node_id")
            session_count = target_node.get("active_sessions", 0)
            
            if session_count == 0:
                print(f"Node {node_id} has no active sessions.")
                return 0
            
            print(f"Draining {session_count} sessions from node {node_id}...")
            
            # Trigger drain via API
            result = await api_request(
                args.endpoint, 
                "POST", 
                f"/cluster/nodes/{node_id}/drain",
                data={"force": args.force}
            )
            
            if result.get("success"):
                migrated = result.get("migrated_sessions", 0)
                failed = result.get("failed_sessions", 0)
                print(f"\nDrain complete:")
                print(f"  Migrated: {migrated}")
                if failed > 0:
                    print(f"  Failed:   {failed}")
                return 0 if failed == 0 else 1
            else:
                print(f"Drain failed: {result.get('error', 'Unknown error')}", file=sys.stderr)
                return 1
                
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    return asyncio.run(run())



def cmd_backup(args: argparse.Namespace) -> int:
    """Backup cluster state."""
    async def run():
        try:
            status = await api_request(args.endpoint, "GET", "/cluster/status")
            nodes = await api_request(args.endpoint, "GET", "/cluster/nodes")
            sessions = await api_request(args.endpoint, "GET", "/cluster/sessions")

            backup = {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "cluster_status": status,
                "nodes": nodes.get("nodes", []),
                "sessions": sessions.get("sessions", []),
            }

            output = args.output or f"flybrowser-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"

            with open(output, "w") as f:
                json.dump(backup, f, indent=2, default=str)

            print(f"Backup saved to: {output}")
            print(f"  Nodes: {len(backup['nodes'])}")
            print(f"  Sessions: {len(backup['sessions'])}")
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    return asyncio.run(run())


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="flybrowser-admin",
        description="FlyBrowser Administrative CLI",
    )
    parser.add_argument(
        "--endpoint", "-e",
        default="http://localhost:8000",
        help="Cluster endpoint URL",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output in JSON format",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # sessions command
    sessions_parser = subparsers.add_parser("sessions", help="Manage sessions")
    sessions_sub = sessions_parser.add_subparsers(dest="sessions_command")

    sessions_list = sessions_sub.add_parser("list", help="List sessions")
    sessions_list.set_defaults(func=cmd_sessions_list)

    sessions_kill = sessions_sub.add_parser("kill", help="Kill a session")
    sessions_kill.add_argument("session_id", help="Session ID to kill")
    sessions_kill.set_defaults(func=cmd_sessions_kill)

    # nodes command
    nodes_parser = subparsers.add_parser("nodes", help="Manage nodes")
    nodes_sub = nodes_parser.add_subparsers(dest="nodes_command")

    nodes_list = nodes_sub.add_parser("list", help="List nodes")
    nodes_list.set_defaults(func=cmd_nodes_list)

    nodes_drain = nodes_sub.add_parser("drain", help="Drain sessions from a node")
    nodes_drain.add_argument("node_id", help="Node ID or address to drain")
    nodes_drain.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force drain even if migrations fail",
    )
    nodes_drain.set_defaults(func=cmd_nodes_drain)

    # backup command
    backup_parser = subparsers.add_parser("backup", help="Backup cluster state")
    backup_parser.add_argument("--output", "-o", help="Output file path")
    backup_parser.set_defaults(func=cmd_backup)

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
