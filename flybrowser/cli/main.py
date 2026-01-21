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

"""
FlyBrowser Unified CLI.

The main entry point for all FlyBrowser command-line functionality.

Usage:
    flybrowser                    # Launch interactive REPL
    flybrowser repl               # Launch interactive REPL (explicit)
    flybrowser setup [COMMAND]    # Installation/configuration wizard
    flybrowser serve [OPTIONS]    # Start the FlyBrowser service
    flybrowser cluster [COMMAND]  # Cluster management
    flybrowser admin [COMMAND]    # Administrative commands
    flybrowser doctor             # Diagnose installation issues
    flybrowser version            # Show version information
    flybrowser --help             # Show help

Examples:
    # Quick start - launch interactive REPL
    flybrowser

    # Configure FlyBrowser interactively
    flybrowser setup configure

    # Start the service
    flybrowser serve --port 8000

    # Check installation health
    flybrowser doctor
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path
from typing import List, Optional

from flybrowser.cli.output import CLIOutput
from flybrowser.llm.factory import LLMProviderFactory
from flybrowser.llm.provider_status import ProviderStatusLevel
from flybrowser.utils.logger import LogFormat, configure_logging


def get_version() -> str:
    """Get the FlyBrowser version."""
    try:
        import flybrowser
        return getattr(flybrowser, "__version__", "unknown")
    except ImportError:
        return "unknown"


# Global CLI output instance
cli_output = CLIOutput()


def get_banner() -> str:
    """Get the FlyBrowser banner from banner.txt or fallback.

    Returns:
        The banner string
    """
    return cli_output.get_banner()


def print_banner() -> None:
    """Print the FlyBrowser banner."""
    cli_output.print_banner(version=get_version())


def cmd_version(args: argparse.Namespace) -> int:
    """Show version information."""
    version = get_version()
    
    if args.json:
        import json
        import platform
        info = {
            "flybrowser": version,
            "python": platform.python_version(),
            "platform": platform.system(),
            "architecture": platform.machine(),
        }
        print(json.dumps(info, indent=2))
    else:
        print(f"FlyBrowser {version}")
    
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Run installation diagnostics."""
    # Print banner and summary
    cli_output.print_banner(version=get_version())
    
    # Show execution summary
    cli_output.print_summary("Diagnostics", {
        "Python": platform.python_version(),
        "Platform": f"{platform.system()} {platform.machine()}",
        "Working Dir": str(Path.cwd()),
    })
    
    checks = []
    
    cli_output.print_section("System Requirements")
    
    # Check 1: Python version
    py_version = platform.python_version()
    major, minor = map(int, py_version.split(".")[:2])
    if major >= 3 and minor >= 9:
        cli_output.print_status_line("OK", f"Python {py_version}", ok=True)
        checks.append(True)
    else:
        cli_output.print_status_line("FAIL", f"Python {py_version} (3.9+ required)", ok=False)
        checks.append(False)
    
    # Check 2: FlyBrowser import
    try:
        import flybrowser
        version = getattr(flybrowser, "__version__", "unknown")
        cli_output.print_status_line("OK", f"FlyBrowser {version} installed", ok=True)
        checks.append(True)
    except ImportError as e:
        cli_output.print_status_line("FAIL", f"FlyBrowser not installed: {e}", ok=False)
        checks.append(False)
    
    # Check 3: Playwright
    try:
        from playwright.sync_api import sync_playwright
        cli_output.print_status_line("OK", "Playwright installed", ok=True)
        checks.append(True)
    except ImportError:
        cli_output.print_status_line("FAIL", "Playwright not installed", ok=False)
        print("      Run: pip install playwright && playwright install")
        checks.append(False)
    
    # Check 4: Browser availability
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "--dry-run", "chromium"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            cli_output.print_status_line("OK", "Chromium browser available", ok=True)
            checks.append(True)
        else:
            cli_output.print_status_line("WARN", "Chromium may need installation", warn=True)
            print("       Run: playwright install chromium")
            checks.append(True)  # Not a hard failure
    except Exception:
        cli_output.print_status_line("WARN", "Could not verify browser installation", warn=True)
        checks.append(True)  # Not a hard failure
    
    # Check 5: LLM providers (DYNAMIC DISCOVERY)
    cli_output.print_section("LLM Provider Status")
    
    # Use dynamic provider discovery from factory
    provider_statuses = LLMProviderFactory.get_all_provider_statuses()
    
    for name, status in provider_statuses.items():
        if status.level == ProviderStatusLevel.OK:
            cli_output.print_status_line("OK", f"{status.name}: {status.message}", ok=True)
        elif status.level == ProviderStatusLevel.INFO:
            cli_output.print_status_line("INFO", f"{status.name}: {status.message}", info=True)
        elif status.level == ProviderStatusLevel.WARN:
            cli_output.print_status_line("WARN", f"{status.name}: {status.message}", warn=True)
        else:
            cli_output.print_status_line("FAIL", f"{status.name}: {status.message}", ok=False)
    
    # Show all registered providers and aliases
    providers = LLMProviderFactory.list_providers(include_aliases=False)
    aliases = LLMProviderFactory.get_aliases()
    
    print(f"\n  Providers: {', '.join(providers)}")
    if aliases:
        alias_strs = [f"{alias} → {target}" for alias, target in aliases.items()]
        print(f"  Aliases: {', '.join(alias_strs)}")
    
    # Check 6: Config files
    cli_output.print_section("Configuration")
    config_paths = [
        Path.cwd() / ".env",
        Path.home() / ".flybrowser" / "config",
        Path.home() / ".config" / "flybrowser" / "config",
    ]
    
    found_config = False
    for config_path in config_paths:
        if config_path.exists():
            cli_output.print_status_line("OK", f"Config found: {config_path}", ok=True)
            found_config = True
            break
    
    if not found_config:
        cli_output.print_status_line("INFO", "No config file found (using defaults)", info=True)
        print("       Run 'flybrowser setup configure' to create one")
    
    # Summary
    cli_output.print_section("Summary")
    cli_output.print_divider("=")
    if all(checks):
        cli_output.print_status_line("SUCCESS", "All checks passed!", ok=True)
        print("\nYou're ready to use FlyBrowser. Try:")
        print("  flybrowser repl         # Interactive mode")
        print("  flybrowser serve        # Start service")
        return 0
    else:
        cli_output.print_status_line("FAIL", "Some checks failed. Please fix the issues above.", ok=False)
        return 1


def cmd_repl(args: argparse.Namespace) -> int:
    """Launch the interactive REPL."""
    try:
        from flybrowser.cli.repl import FlyBrowserREPL
        
        # Apply logging config if specified
        if hasattr(args, 'log_level') and args.log_level:
            configure_logging(
                level=args.log_level,
                human_readable=getattr(args, 'human_readable', False),
            )
        
        cli_output.print_banner(version=get_version())
        
        # Show execution summary
        cli_output.print_summary("REPL Session", {
            "Provider": args.provider,
            "Model": args.model or "(default)",
            "Headless": args.headless,
        })
        
        cli_output.print_logs_header()
        
        repl = FlyBrowserREPL(
            llm_provider=args.provider,
            llm_model=args.model,
            headless=args.headless,
            api_key=args.api_key or os.environ.get(f"{args.provider.upper()}_API_KEY"),
        )
        repl.run()
        return 0
    except ImportError as e:
        print(f"Error: REPL dependencies not available: {e}")
        print("Install with: pip install prompt_toolkit")
        return 1
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0


def cmd_setup(args: argparse.Namespace, remaining: List[str]) -> int:
    """Delegate to setup CLI."""
    from flybrowser.cli import setup
    
    # Reconstruct argv for the setup module
    sys.argv = ["flybrowser-setup"] + remaining
    setup.main()
    return 0


def cmd_serve(args: argparse.Namespace, remaining: List[str]) -> int:
    """Delegate to serve CLI."""
    from flybrowser.cli import serve
    
    # Reconstruct argv for the serve module
    sys.argv = ["flybrowser-serve"] + remaining
    serve.main()
    return 0


def cmd_cluster(args: argparse.Namespace, remaining: List[str]) -> int:
    """Delegate to cluster CLI."""
    from flybrowser.cli import cluster
    
    # Reconstruct argv for the cluster module
    sys.argv = ["flybrowser-cluster"] + remaining
    sys.exit(cluster.main())


def cmd_admin(args: argparse.Namespace, remaining: List[str]) -> int:
    """Delegate to admin CLI."""
    from flybrowser.cli import admin
    
    # Reconstruct argv for the admin module
    sys.argv = ["flybrowser-admin"] + remaining
    sys.exit(admin.main())


def cmd_uninstall(args: argparse.Namespace, remaining: List[str]) -> int:
    """Delegate to uninstall CLI."""
    from flybrowser.cli import uninstall
    
    # Reconstruct argv for the uninstall module
    sys.argv = ["flybrowser-uninstall"] + remaining
    uninstall.main()
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="flybrowser",
        description="FlyBrowser - Browser Automation Powered by LLM Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  (none)      Launch interactive REPL (default)
  repl        Launch interactive REPL
  setup       Installation and configuration wizard
  serve       Start the FlyBrowser API service
  cluster     Cluster management commands
  admin       Administrative commands
  doctor      Diagnose installation issues
  version     Show version information

Examples:
  flybrowser                          # Start REPL
  flybrowser setup configure          # Interactive setup
  flybrowser serve --port 8000        # Start server
  flybrowser cluster status           # Check cluster
  flybrowser doctor                   # Run diagnostics

Documentation: https://flybrowser.dev/docs
""",
    )
    
    # Global logging options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.environ.get("FLYBROWSER_LOG_LEVEL", "INFO"),
        help="Set logging level (default: INFO)",
    )
    parser.add_argument(
        "--human-readable", "--human",
        action="store_true",
        default=os.environ.get("FLYBROWSER_LOG_FORMAT", "json").lower() == "human",
        help="Use human-readable log format instead of JSON",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # version command
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information",
    )
    version_parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output in JSON format",
    )
    version_parser.set_defaults(func=cmd_version)
    
    # doctor command
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Diagnose installation issues",
    )
    doctor_parser.set_defaults(func=cmd_doctor)
    
    # repl command
    repl_parser = subparsers.add_parser(
        "repl",
        help="Launch interactive REPL",
    )
    repl_parser.add_argument(
        "--provider", "-p",
        default=os.environ.get("FLYBROWSER_LLM_PROVIDER", "openai"),
        help="LLM provider (default: openai)",
    )
    repl_parser.add_argument(
        "--model", "-m",
        default=os.environ.get("FLYBROWSER_LLM_MODEL"),
        help="LLM model (provider default if not specified)",
    )
    repl_parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser in headless mode (default: True)",
    )
    repl_parser.add_argument(
        "--no-headless",
        action="store_false",
        dest="headless",
        help="Run browser with visible UI",
    )
    repl_parser.add_argument(
        "--api-key",
        help="LLM API key (uses env var if not provided)",
    )
    repl_parser.set_defaults(func=cmd_repl)
    
    # setup command (pass-through)
    setup_parser = subparsers.add_parser(
        "setup",
        help="Installation and configuration wizard",
        add_help=False,
    )
    
    # serve command (pass-through)
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the FlyBrowser API service",
        add_help=False,
    )
    
    # cluster command (pass-through)
    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Cluster management commands",
        add_help=False,
    )
    
    # admin command (pass-through)
    admin_parser = subparsers.add_parser(
        "admin",
        help="Administrative commands",
        add_help=False,
    )
    
    # uninstall command (pass-through)
    uninstall_parser = subparsers.add_parser(
        "uninstall",
        help="Uninstall FlyBrowser",
        add_help=False,
    )
    
    return parser


def main() -> None:
    """Main entry point for the unified CLI."""
    parser = create_parser()
    
    # Parse only the known args to allow pass-through for subcommands
    args, remaining = parser.parse_known_args()
    
    # Apply logging configuration
    configure_logging(
        level=getattr(args, 'log_level', 'INFO'),
        human_readable=getattr(args, 'human_readable', False),
    )
    
    # Handle commands that need pass-through
    if args.command == "setup":
        sys.exit(cmd_setup(args, remaining))
    elif args.command == "serve":
        sys.exit(cmd_serve(args, remaining))
    elif args.command == "cluster":
        sys.exit(cmd_cluster(args, remaining))
    elif args.command == "admin":
        sys.exit(cmd_admin(args, remaining))
    elif args.command == "uninstall":
        sys.exit(cmd_uninstall(args, remaining))
    
    # Handle other commands normally
    if args.command is None:
        # Default to REPL if no command given
        # First check if we're in interactive mode
        if sys.stdin.isatty():
            # Set default REPL args
            args.provider = os.environ.get("FLYBROWSER_LLM_PROVIDER", "openai")
            args.model = os.environ.get("FLYBROWSER_LLM_MODEL")
            args.headless = True
            args.api_key = None
            sys.exit(cmd_repl(args))
        else:
            # Not interactive, show help
            parser.print_help()
            sys.exit(0)
    elif hasattr(args, "func"):
        sys.exit(args.func(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
