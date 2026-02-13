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
FlyBrowser Session Management CLI Commands.

Provides CLI commands for creating, listing, inspecting, and managing
browser sessions in both embedded and server modes.

Usage:
    flybrowser session create [--provider openai] [--model gpt-4o] [--headless]
    flybrowser session list [--format table|json] [--status active|all]
    flybrowser session info <session-id>
    flybrowser session connect <session-id>
    flybrowser session exec <session-id> <command>
    flybrowser session close <session-id>
    flybrowser session close-all [--force]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

from flybrowser.cli.output import CLIOutput

# Module-level store for embedded sessions
_session_store: Dict[str, Any] = {}

# CLI output instance
_cli_output = CLIOutput()


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from sync context.

    Args:
        coro: The coroutine to run.

    Returns:
        The result of the coroutine.
    """
    return asyncio.run(coro)


async def create_session_embedded(
    provider: str = "openai",
    model: Optional[str] = None,
    headless: bool = True,
    name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a browser session in embedded mode (local browser).

    Uses the FlyBrowser SDK directly without a server.

    Args:
        provider: LLM provider name.
        model: LLM model name.
        headless: Whether to run in headless mode.
        name: Optional session name.
        api_key: LLM API key.

    Returns:
        Dictionary with session information.
    """
    # Lazy import to avoid import cycles
    from flybrowser.sdk import FlyBrowser

    session_id = f"embedded-{uuid.uuid4().hex[:12]}"

    browser = FlyBrowser(
        llm_provider=provider,
        llm_model=model,
        headless=headless,
        api_key=api_key or os.environ.get(f"{provider.upper()}_API_KEY"),
    )
    await browser.start()

    _session_store[session_id] = {
        "browser": browser,
        "name": name or session_id,
        "provider": provider,
        "model": model,
        "status": "active",
    }

    return {
        "session_id": session_id,
        "provider": provider,
        "model": model,
        "status": "active",
        "name": name or session_id,
        "mode": "embedded",
    }


async def create_session_server(
    endpoint: str,
    provider: str = "openai",
    model: Optional[str] = None,
    headless: bool = True,
    name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a browser session on a FlyBrowser server.

    Args:
        endpoint: Server endpoint URL.
        provider: LLM provider name.
        model: LLM model name.
        headless: Whether to run in headless mode.
        name: Optional session name.
        api_key: LLM API key.

    Returns:
        Dictionary with session information.
    """
    # Lazy import to avoid import cycles
    from flybrowser.client import FlyBrowserClient

    client = FlyBrowserClient(endpoint)
    await client.start()
    try:
        result = await client.create_session(
            llm_provider=provider,
            llm_model=model,
            api_key=api_key,
            headless=headless,
        )
        result["mode"] = "server"
        result["endpoint"] = endpoint
        if name:
            result["name"] = name
        return result
    finally:
        await client.stop()


def list_sessions(
    endpoint: str,
    status: str = "active",
) -> List[Dict[str, Any]]:
    """List sessions from a FlyBrowser server.

    Args:
        endpoint: Server endpoint URL.
        status: Filter by status ('active' or 'all').

    Returns:
        List of session dictionaries.
    """
    import aiohttp

    async def _fetch() -> List[Dict[str, Any]]:
        url = f"{endpoint.rstrip('/')}/sessions"
        params: Dict[str, str] = {}
        if status != "all":
            params["status"] = status

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("sessions", [])
                else:
                    text = await resp.text()
                    raise Exception(f"API error {resp.status}: {text}")

    return asyncio.run(_fetch())


async def exec_on_session(
    session_id: str,
    command: str,
    endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a command on a session.

    For server mode, uses the FlyBrowserClient. For embedded mode,
    uses the locally stored FlyBrowser instance.

    Args:
        session_id: The session ID.
        command: The command/instruction to run.
        endpoint: Server endpoint URL (None for embedded).

    Returns:
        Dictionary with result.
    """
    if endpoint:
        # Server mode - lazy import
        from flybrowser.client import FlyBrowserClient

        client = FlyBrowserClient(endpoint)
        await client.start()
        try:
            return await client.navigate_nl(session_id, command)
        finally:
            await client.stop()
    else:
        # Embedded mode
        session_data = _session_store.get(session_id)
        if not session_data:
            raise ValueError(f"Session not found: {session_id}")

        browser = session_data["browser"]
        return await browser.execute_task(command)


# ---------- Command Handlers ----------


def cmd_session_create(args: argparse.Namespace) -> int:
    """Handle 'flybrowser session create' command."""
    try:
        if args.endpoint:
            result = _run_async(
                create_session_server(
                    endpoint=args.endpoint,
                    provider=args.provider,
                    model=args.model,
                    headless=args.headless,
                    name=args.name,
                    api_key=args.api_key,
                )
            )
        else:
            result = _run_async(
                create_session_embedded(
                    provider=args.provider,
                    model=args.model,
                    headless=args.headless,
                    name=args.name,
                    api_key=args.api_key,
                )
            )

        _cli_output.print_summary("Session Created", {
            "Session ID": result.get("session_id", "unknown"),
            "Provider": result.get("provider", ""),
            "Model": result.get("model", "(default)"),
            "Mode": result.get("mode", "unknown"),
            "Status": result.get("status", "active"),
        })
        return 0

    except Exception as e:
        print(f"Error creating session: {e}", file=sys.stderr)
        return 1


def cmd_session_list(args: argparse.Namespace) -> int:
    """Handle 'flybrowser session list' command."""
    try:
        sessions = list_sessions(
            endpoint=args.endpoint,
            status=args.status,
        )

        if args.format == "json":
            print(json.dumps({"sessions": sessions, "total": len(sessions)}, indent=2))
        else:
            # Table format
            if not sessions:
                print("No active sessions.")
                return 0

            _cli_output.print_section(f"Sessions ({len(sessions)})")

            headers = ["Session ID", "Status", "Provider", "Model", "Created"]
            widths = [len(h) for h in headers]

            rows = []
            for s in sessions:
                row = [
                    s.get("session_id", "")[:20],
                    s.get("status", ""),
                    s.get("provider", ""),
                    s.get("model", ""),
                    s.get("created_at", ""),
                ]
                for i, cell in enumerate(row):
                    widths[i] = max(widths[i], len(str(cell)))
                rows.append(row)

            header_line = " | ".join(
                h.ljust(widths[i]) for i, h in enumerate(headers)
            )
            print(header_line)
            print("-" * len(header_line))
            for row in rows:
                print(
                    " | ".join(
                        str(cell).ljust(widths[i]) for i, cell in enumerate(row)
                    )
                )

        return 0

    except Exception as e:
        print(f"Error listing sessions: {e}", file=sys.stderr)
        return 1


def cmd_session_info(args: argparse.Namespace) -> int:
    """Handle 'flybrowser session info <session-id>' command."""
    try:
        import aiohttp

        async def _fetch_info() -> Dict[str, Any]:
            endpoint = args.endpoint
            url = f"{endpoint.rstrip('/')}/sessions/{args.session_id}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        text = await resp.text()
                        raise Exception(f"API error {resp.status}: {text}")

        info = _run_async(_fetch_info())

        _cli_output.print_summary(f"Session: {args.session_id}", {
            "Status": info.get("status", "unknown"),
            "Provider": info.get("provider", ""),
            "Model": info.get("model", ""),
            "Created": info.get("created_at", ""),
            "Node": info.get("node_id", ""),
        })
        return 0

    except Exception as e:
        print(f"Error getting session info: {e}", file=sys.stderr)
        return 1


def cmd_session_connect(args: argparse.Namespace) -> int:
    """Handle 'flybrowser session connect <session-id>' command."""
    try:
        print(f"Connecting to session {args.session_id}...")
        print(f"Endpoint: {args.endpoint}")
        print()
        print("Use 'flybrowser session exec' to run commands on this session.")
        print(
            f"  flybrowser session exec {args.session_id}"
            f" \"navigate to https://example.com\""
        )
        return 0

    except Exception as e:
        print(f"Error connecting to session: {e}", file=sys.stderr)
        return 1


def cmd_session_exec(args: argparse.Namespace) -> int:
    """Handle 'flybrowser session exec <session-id> <command>' command."""
    try:
        result = _run_async(
            exec_on_session(
                session_id=args.session_id,
                command=args.command,
                endpoint=args.endpoint,
            )
        )

        print(json.dumps(result, indent=2, default=str))
        return 0

    except Exception as e:
        print(f"Error executing command: {e}", file=sys.stderr)
        return 1


def cmd_session_close(args: argparse.Namespace) -> int:
    """Handle 'flybrowser session close <session-id>' command."""
    try:
        endpoint = args.endpoint

        if endpoint:
            async def _close_server() -> bool:
                from flybrowser.client import FlyBrowserClient

                client = FlyBrowserClient(endpoint)
                await client.start()
                try:
                    return await client.close_session(args.session_id)
                finally:
                    await client.stop()

            _run_async(_close_server())
        else:
            # Embedded mode
            session_data = _session_store.pop(args.session_id, None)
            if session_data:
                _run_async(session_data["browser"].stop())
            else:
                print(f"Session not found: {args.session_id}", file=sys.stderr)
                return 1

        print(f"Session {args.session_id} closed.")
        return 0

    except Exception as e:
        print(f"Error closing session: {e}", file=sys.stderr)
        return 1


def cmd_session_close_all(args: argparse.Namespace) -> int:
    """Handle 'flybrowser session close-all' command."""
    try:
        endpoint = args.endpoint

        if endpoint:
            import aiohttp

            async def _close_all() -> Dict[str, Any]:
                # First list sessions, then close each
                url = f"{endpoint.rstrip('/')}/sessions"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            sessions_list = data.get("sessions", [])
                        else:
                            sessions_list = []

                closed_count = 0
                from flybrowser.client import FlyBrowserClient

                client = FlyBrowserClient(endpoint)
                await client.start()
                try:
                    for s in sessions_list:
                        sid = s.get("session_id")
                        if sid:
                            try:
                                await client.close_session(sid)
                                closed_count += 1
                            except Exception:
                                pass
                finally:
                    await client.stop()

                return {"closed": closed_count}

            result = _run_async(_close_all())
        else:
            # Close embedded sessions
            closed_count = 0
            for sid, session_data in list(_session_store.items()):
                try:
                    _run_async(session_data["browser"].stop())
                    closed_count += 1
                except Exception:
                    pass
            _session_store.clear()
            result = {"closed": closed_count}

        closed = result.get("closed", 0)
        print(f"Closed {closed} session(s).")
        return 0

    except Exception as e:
        print(f"Error closing sessions: {e}", file=sys.stderr)
        return 1


# ---------- Argparse Subparser Registration ----------


def add_session_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add session management subcommands to the CLI parser.

    Args:
        subparsers: The subparsers action from the main parser.
    """
    session_parser = subparsers.add_parser(
        "session",
        help="Manage browser sessions",
    )
    session_subparsers = session_parser.add_subparsers(
        dest="session_command",
        help="Session commands",
    )

    # session create
    create_parser = session_subparsers.add_parser(
        "create",
        help="Create a new browser session",
    )
    create_parser.add_argument(
        "--provider", "-p",
        default=os.environ.get("FLYBROWSER_LLM_PROVIDER", "openai"),
        help="LLM provider (default: openai)",
    )
    create_parser.add_argument(
        "--model", "-m",
        default=os.environ.get("FLYBROWSER_LLM_MODEL"),
        help="LLM model (provider default if not specified)",
    )
    create_parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser in headless mode (default: True)",
    )
    create_parser.add_argument(
        "--no-headless",
        action="store_false",
        dest="headless",
        help="Run browser with visible UI",
    )
    create_parser.add_argument(
        "--name", "-n",
        default=None,
        help="Optional session name",
    )
    create_parser.add_argument(
        "--api-key",
        default=None,
        help="LLM API key (uses env var if not provided)",
    )
    create_parser.add_argument(
        "--endpoint", "-e",
        default=os.environ.get("FLYBROWSER_ENDPOINT"),
        help="FlyBrowser server endpoint (embedded mode if not set)",
    )
    create_parser.set_defaults(func=cmd_session_create)

    # session list
    list_parser = session_subparsers.add_parser(
        "list",
        help="List browser sessions",
    )
    list_parser.add_argument(
        "--format", "-f",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    list_parser.add_argument(
        "--status", "-s",
        choices=["active", "all"],
        default="active",
        help="Filter by status (default: active)",
    )
    list_parser.add_argument(
        "--endpoint", "-e",
        default=os.environ.get("FLYBROWSER_ENDPOINT", "http://localhost:8000"),
        help="FlyBrowser server endpoint",
    )
    list_parser.set_defaults(func=cmd_session_list)

    # session info
    info_parser = session_subparsers.add_parser(
        "info",
        help="Get session details",
    )
    info_parser.add_argument(
        "session_id",
        help="Session ID",
    )
    info_parser.add_argument(
        "--endpoint", "-e",
        default=os.environ.get("FLYBROWSER_ENDPOINT", "http://localhost:8000"),
        help="FlyBrowser server endpoint",
    )
    info_parser.set_defaults(func=cmd_session_info)

    # session connect
    connect_parser = session_subparsers.add_parser(
        "connect",
        help="Connect to an existing session",
    )
    connect_parser.add_argument(
        "session_id",
        help="Session ID",
    )
    connect_parser.add_argument(
        "--endpoint", "-e",
        default=os.environ.get("FLYBROWSER_ENDPOINT", "http://localhost:8000"),
        help="FlyBrowser server endpoint",
    )
    connect_parser.set_defaults(func=cmd_session_connect)

    # session exec
    exec_parser = session_subparsers.add_parser(
        "exec",
        help="Run a command on a session",
    )
    exec_parser.add_argument(
        "session_id",
        help="Session ID",
    )
    exec_parser.add_argument(
        "command",
        help="Command to run (natural language instruction)",
    )
    exec_parser.add_argument(
        "--endpoint", "-e",
        default=os.environ.get("FLYBROWSER_ENDPOINT", "http://localhost:8000"),
        help="FlyBrowser server endpoint",
    )
    exec_parser.set_defaults(func=cmd_session_exec)

    # session close
    close_parser = session_subparsers.add_parser(
        "close",
        help="Close a session",
    )
    close_parser.add_argument(
        "session_id",
        help="Session ID",
    )
    close_parser.add_argument(
        "--endpoint", "-e",
        default=os.environ.get("FLYBROWSER_ENDPOINT", "http://localhost:8000"),
        help="FlyBrowser server endpoint",
    )
    close_parser.set_defaults(func=cmd_session_close)

    # session close-all
    close_all_parser = session_subparsers.add_parser(
        "close-all",
        help="Close all sessions",
    )
    close_all_parser.add_argument(
        "--force",
        action="store_true",
        help="Force close without confirmation",
    )
    close_all_parser.add_argument(
        "--endpoint", "-e",
        default=os.environ.get("FLYBROWSER_ENDPOINT", "http://localhost:8000"),
        help="FlyBrowser server endpoint",
    )
    close_all_parser.set_defaults(func=cmd_session_close_all)
