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
FlyBrowser Cluster Management CLI.

This module provides CLI commands for managing FlyBrowser clusters:
- join: Join an existing cluster
- leave: Gracefully leave a cluster
- status: Show cluster status and node health
- rebalance: Trigger manual rebalancing

Usage:
    flybrowser-cluster status
    flybrowser-cluster join --peers node1:4321,node2:4321
    flybrowser-cluster leave
    flybrowser-cluster rebalance
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import List, Optional

import aiohttp


def print_json(data: dict) -> None:
    """Print data as formatted JSON."""
    print(json.dumps(data, indent=2))


def print_table(headers: List[str], rows: List[List[str]]) -> None:
    """Print data as a formatted table."""
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Print header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))
    
    # Print rows
    for row in rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


async def get_cluster_status(endpoint: str) -> dict:
    """Get cluster status from endpoint."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{endpoint}/cluster/status") as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                raise Exception(f"Failed to get status: {resp.status}")


async def get_cluster_nodes(endpoint: str) -> List[dict]:
    """Get cluster nodes from endpoint."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{endpoint}/cluster/nodes") as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("nodes", [])
            else:
                raise Exception(f"Failed to get nodes: {resp.status}")


async def get_cluster_sessions(endpoint: str) -> List[dict]:
    """Get cluster sessions from endpoint."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{endpoint}/cluster/sessions") as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("sessions", [])
            else:
                raise Exception(f"Failed to get sessions: {resp.status}")


def cmd_status(args: argparse.Namespace) -> int:
    """Show cluster status."""
    endpoint = args.endpoint.rstrip("/")
    
    async def run():
        try:
            status = await get_cluster_status(endpoint)
            nodes = await get_cluster_nodes(endpoint)
            
            print("\n=== Cluster Status ===\n")
            print(f"Node ID:     {status.get('node_id', 'N/A')}")
            print(f"Role:        {status.get('role', 'N/A')}")
            print(f"Is Leader:   {status.get('is_leader', False)}")
            print(f"Leader ID:   {status.get('leader_id', 'N/A')}")
            
            if "cluster" in status:
                cluster = status["cluster"]
                print(f"\nCluster Stats:")
                print(f"  Total Nodes:      {cluster.get('node_count', 0)}")
                print(f"  Healthy Nodes:    {cluster.get('healthy_nodes', 0)}")
                print(f"  Total Capacity:   {cluster.get('total_capacity', 0)}")
                print(f"  Active Sessions:  {cluster.get('total_active_sessions', 0)}")
            
            if nodes:
                print("\n=== Cluster Nodes ===\n")
                headers = ["Node ID", "API Address", "Health", "Sessions", "Capacity", "Load"]
                rows = []
                for node in nodes:
                    rows.append([
                        node.get("node_id", ""),
                        node.get("api_address", ""),
                        node.get("health", "unknown"),
                        f"{node.get('active_sessions', 0)}/{node.get('max_sessions', 0)}",
                        str(node.get("available_capacity", 0)),
                        f"{node.get('load_score', 0):.2f}",
                    ])
                print_table(headers, rows)
            
            if args.json:
                print("\n=== Raw JSON ===\n")
                print_json(status)
            
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    return asyncio.run(run())


def cmd_nodes(args: argparse.Namespace) -> int:
    """List cluster nodes."""
    endpoint = args.endpoint.rstrip("/")
    
    async def run():
        try:
            nodes = await get_cluster_nodes(endpoint)
            
            if args.json:
                print_json({"nodes": nodes})
            else:
                print("\n=== Cluster Nodes ===\n")
                headers = ["Node ID", "API Address", "Raft Address", "Health", "CPU%", "Mem%", "Sessions"]
                rows = []
                for node in nodes:
                    rows.append([
                        node.get("node_id", ""),
                        node.get("api_address", ""),
                        node.get("raft_address", ""),
                        node.get("health", "unknown"),
                        f"{node.get('cpu_percent', 0):.1f}",
                        f"{node.get('memory_percent', 0):.1f}",
                        f"{node.get('active_sessions', 0)}/{node.get('max_sessions', 0)}",
                    ])
                print_table(headers, rows)
            
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    return asyncio.run(run())


def cmd_sessions(args: argparse.Namespace) -> int:
    """List cluster sessions."""
    endpoint = args.endpoint.rstrip("/")

    async def run():
        try:
            sessions = await get_cluster_sessions(endpoint)

            if args.json:
                print_json({"sessions": sessions})
            else:
                print(f"\n=== Cluster Sessions ({len(sessions)} total) ===\n")
                if sessions:
                    headers = ["Session ID", "Node ID", "Status", "Client ID", "Created"]
                    rows = []
                    for session in sessions:
                        rows.append([
                            session.get("session_id", "")[:12] + "...",
                            session.get("node_id", ""),
                            session.get("status", "unknown"),
                            session.get("client_id", "N/A") or "N/A",
                            str(session.get("created_at", ""))[:19],
                        ])
                    print_table(headers, rows)
                else:
                    print("No active sessions.")

            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    return asyncio.run(run())


def cmd_health(args: argparse.Namespace) -> int:
    """Check cluster health."""
    endpoint = args.endpoint.rstrip("/")

    async def run():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{endpoint}/health") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if args.json:
                            print_json(data)
                        else:
                            status = data.get("status", "unknown")
                            if status == "healthy":
                                print(f"✓ Cluster is healthy")
                                print(f"  Node: {data.get('node_id', 'N/A')}")
                                print(f"  Role: {data.get('role', 'N/A')}")
                                print(f"  Leader: {data.get('is_leader', False)}")
                            else:
                                print(f"✗ Cluster is {status}")
                        return 0 if status == "healthy" else 1
                    else:
                        print(f"✗ Health check failed: {resp.status}")
                        return 1
        except Exception as e:
            print(f"✗ Health check failed: {e}", file=sys.stderr)
            return 1

    return asyncio.run(run())


def cmd_step_down(args: argparse.Namespace) -> int:
    """Request the current leader to step down.
    
    This triggers a leadership transfer to another node in the cluster.
    Useful for planned maintenance or when you want to change the leader.
    """
    endpoint = args.endpoint.rstrip("/")

    async def run():
        try:
            # First check if this node is the leader
            status = await get_cluster_status(endpoint)
            if not status.get("is_leader"):
                print(f"This node is not the leader. Current leader: {status.get('leader_id', 'unknown')}")
                return 1
            
            print(f"Requesting leader {status.get('node_id')} to step down...")
            
            async with aiohttp.ClientSession() as session:
                data = {}
                if args.target:
                    data["target_node"] = args.target
                    
                async with session.post(
                    f"{endpoint}/cluster/step-down",
                    json=data,
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if result.get("success"):
                            new_leader = result.get("new_leader", "pending")
                            print(f"✓ Leadership transfer initiated")
                            print(f"  New leader: {new_leader}")
                            return 0
                        else:
                            print(f"✗ Step-down failed: {result.get('error', 'Unknown error')}")
                            return 1
                    else:
                        text = await resp.text()
                        print(f"✗ Step-down failed: {resp.status} - {text}")
                        return 1
                        
        except Exception as e:
            print(f"✗ Step-down failed: {e}", file=sys.stderr)
            return 1

    return asyncio.run(run())


def cmd_rebalance(args: argparse.Namespace) -> int:
    """Trigger manual session rebalancing across the cluster."""
    endpoint = args.endpoint.rstrip("/")

    async def run():
        try:
            print("Triggering cluster rebalance...")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{endpoint}/cluster/rebalance") as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if result.get("success"):
                            moved = result.get("sessions_moved", 0)
                            print(f"✓ Rebalance complete")
                            print(f"  Sessions moved: {moved}")
                            return 0
                        else:
                            print(f"✗ Rebalance failed: {result.get('error', 'Unknown error')}")
                            return 1
                    else:
                        text = await resp.text()
                        print(f"✗ Rebalance failed: {resp.status} - {text}")
                        return 1
                        
        except Exception as e:
            print(f"✗ Rebalance failed: {e}", file=sys.stderr)
            return 1

    return asyncio.run(run())


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="flybrowser-cluster",
        description="FlyBrowser Cluster Management CLI",
    )
    parser.add_argument(
        "--endpoint", "-e",
        default="http://localhost:8000",
        help="Cluster endpoint URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output in JSON format",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # status command
    status_parser = subparsers.add_parser("status", help="Show cluster status")
    status_parser.set_defaults(func=cmd_status)

    # nodes command
    nodes_parser = subparsers.add_parser("nodes", help="List cluster nodes")
    nodes_parser.set_defaults(func=cmd_nodes)

    # sessions command
    sessions_parser = subparsers.add_parser("sessions", help="List cluster sessions")
    sessions_parser.set_defaults(func=cmd_sessions)

    # health command
    health_parser = subparsers.add_parser("health", help="Check cluster health")
    health_parser.set_defaults(func=cmd_health)

    # step-down command
    step_down_parser = subparsers.add_parser("step-down", help="Request leader to step down")
    step_down_parser.add_argument(
        "--target", "-t",
        help="Target node ID to transfer leadership to (optional)",
    )
    step_down_parser.set_defaults(func=cmd_step_down)

    # rebalance command
    rebalance_parser = subparsers.add_parser("rebalance", help="Trigger session rebalancing")
    rebalance_parser.set_defaults(func=cmd_rebalance)

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
