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
FlyBrowser Setup CLI.

Provides installation and configuration commands for FlyBrowser.

Commands:
    install     Install FlyBrowser and dependencies
    configure   Interactive configuration wizard
    browsers    Install/manage Playwright browsers
    verify      Verify installation

Usage:
    python -m flybrowser.cli.setup install
    python -m flybrowser.cli.setup configure
    python -m flybrowser.cli.setup browsers install
    python -m flybrowser.cli.setup verify

    Or with the installed command:
    flybrowser-setup install
    flybrowser-setup configure
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Global flag to track if we've been interrupted
_interrupted = False


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle interrupt signals gracefully."""
    global _interrupted
    _interrupted = True
    signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    print(f"\n\n[INTERRUPTED] Received {signal_name}, cancelling...")
    print("[INFO] Setup cancelled. No changes were finalized.")
    sys.exit(128 + signum)  # Standard convention: 128 + signal number


def setup_signal_handlers() -> None:
    """Set up signal handlers for graceful cancellation."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def check_interrupted() -> None:
    """Check if we've been interrupted and exit if so."""
    global _interrupted
    if _interrupted:
        print("\n[INFO] Operation cancelled.")
        sys.exit(130)


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


def print_banner() -> None:
    """Print the FlyBrowser banner."""
    print()
    print(get_banner())
    print()


def prompt(message: str, default: Optional[str] = None, required: bool = False) -> str:
    """Prompt user for input.
    
    Handles KeyboardInterrupt (Ctrl+C) and EOFError (Ctrl+D) gracefully.
    """
    if default:
        message = f"{message} [{default}]: "
    else:
        message = f"{message}: "
    
    while True:
        try:
            check_interrupted()
            value = input(message).strip()
            if not value and default:
                return default
            if not value and required:
                print("  This field is required. Please enter a value.")
                continue
            return value
        except EOFError:
            # Handle Ctrl+D (EOF)
            print("\n")
            raise KeyboardInterrupt("EOF received")


def prompt_choice(message: str, choices: list, default: int = 0) -> str:
    """Prompt user to choose from options.
    
    Handles KeyboardInterrupt (Ctrl+C) and EOFError (Ctrl+D) gracefully.
    """
    print(f"\n{message}")
    for i, choice in enumerate(choices):
        marker = ">" if i == default else " "
        print(f"  {marker} {i + 1}. {choice}")
    
    while True:
        try:
            check_interrupted()
            value = input(f"Enter choice [1-{len(choices)}] (default: {default + 1}): ").strip()
            if not value:
                return choices[default]
            try:
                idx = int(value) - 1
                if 0 <= idx < len(choices):
                    return choices[idx]
            except ValueError:
                pass
            print(f"  Please enter a number between 1 and {len(choices)}")
        except EOFError:
            # Handle Ctrl+D (EOF)
            print("\n")
            raise KeyboardInterrupt("EOF received")


def prompt_bool(message: str, default: bool = True) -> bool:
    """Prompt user for yes/no.
    
    Handles KeyboardInterrupt (Ctrl+C) and EOFError (Ctrl+D) gracefully.
    """
    default_str = "Y/n" if default else "y/N"
    try:
        check_interrupted()
        value = input(f"{message} [{default_str}]: ").strip().lower()
        if not value:
            return default
        return value in ("y", "yes", "true", "1")
    except EOFError:
        # Handle Ctrl+D (EOF)
        print("\n")
        raise KeyboardInterrupt("EOF received")


def generate_env_file(config: Dict[str, Any], output_path: Path) -> None:
    """Generate .env file from configuration."""
    lines = [
        "# FlyBrowser Configuration",
        "# Generated by flybrowser-setup",
        "",
        "# Service Settings",
        f"FLYBROWSER_HOST={config.get('host', '0.0.0.0')}",
        f"FLYBROWSER_PORT={config.get('port', 8000)}",
        f"FLYBROWSER_ENV={config.get('env', 'production')}",
        f"FLYBROWSER_LOG_LEVEL={config.get('log_level', 'INFO')}",
        "",
        "# Session Settings",
        f"FLYBROWSER_MAX_SESSIONS={config.get('max_sessions', 100)}",
        f"FLYBROWSER_SESSION_TIMEOUT={config.get('session_timeout', 3600)}",
        "",
        "# Browser Pool Settings",
        f"FLYBROWSER_POOL__MIN_SIZE={config.get('pool_min_size', 1)}",
        f"FLYBROWSER_POOL__MAX_SIZE={config.get('pool_max_size', 10)}",
        f"FLYBROWSER_POOL__HEADLESS={str(config.get('headless', True)).lower()}",
        "",
        "# Deployment Mode",
        f"FLYBROWSER_DEPLOYMENT_MODE={config.get('deployment_mode', 'standalone')}",
        "",
    ]
    
    # Add cluster settings if in cluster mode
    if config.get("deployment_mode") == "cluster":
        lines.extend([
            "# Cluster Settings",
            f"FLYBROWSER_CLUSTER__ENABLED=true",
            f"FLYBROWSER_CLUSTER__NODE__ROLE={config.get('node_role', 'worker')}",
            f"FLYBROWSER_CLUSTER__COORDINATOR_HOST={config.get('coordinator_host', 'localhost')}",
            f"FLYBROWSER_CLUSTER__COORDINATOR_PORT={config.get('coordinator_port', 8001)}",
            "",
        ])
    
    # Add LLM provider settings
    lines.append("# LLM Provider Configuration")
    if config.get("default_llm_provider"):
        lines.append(f"FLYBROWSER_LLM_PROVIDER={config.get('default_llm_provider')}")
    if config.get("default_llm_model"):
        lines.append(f"FLYBROWSER_LLM_MODEL={config.get('default_llm_model')}")
    lines.append("")

    lines.append("# LLM Provider API Keys")
    if config.get("openai_api_key"):
        lines.append(f"OPENAI_API_KEY={config.get('openai_api_key', '')}")
    if config.get("anthropic_api_key"):
        lines.append(f"ANTHROPIC_API_KEY={config.get('anthropic_api_key', '')}")
    if config.get("google_api_key"):
        lines.append(f"GOOGLE_API_KEY={config.get('google_api_key', '')}")
    if config.get("ollama_base_url"):
        lines.append(f"OLLAMA_BASE_URL={config.get('ollama_base_url', 'http://localhost:11434')}")

    lines.append("")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n[OK] Configuration saved to {output_path}")


def generate_config(
    deployment_mode: str = "standalone",
    output_path: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Generate configuration programmatically.

    Args:
        deployment_mode: "standalone" or "cluster"
        output_path: Path to save .env file (optional)
        **kwargs: Additional configuration options

    Returns:
        Configuration dictionary
    """
    config = {
        "deployment_mode": deployment_mode,
        "host": kwargs.get("host", "0.0.0.0"),
        "port": kwargs.get("port", 8000),
        "env": kwargs.get("env", "production"),
        "log_level": kwargs.get("log_level", "INFO"),
        "max_sessions": kwargs.get("max_sessions", 100),
        "session_timeout": kwargs.get("session_timeout", 3600),
        "pool_min_size": kwargs.get("pool_min_size", 1),
        "pool_max_size": kwargs.get("pool_max_size", 10),
        "headless": kwargs.get("headless", True),
    }

    if deployment_mode == "cluster":
        config.update({
            "node_role": kwargs.get("node_role", "worker"),
            "coordinator_host": kwargs.get("coordinator_host", "localhost"),
            "coordinator_port": kwargs.get("coordinator_port", 8001),
        })

    if output_path:
        generate_env_file(config, Path(output_path))

    return config


def setup_wizard() -> Dict[str, Any]:
    """Run the interactive setup wizard.

    Returns:
        Configuration dictionary
    """
    print_banner()
    print("\nWelcome to the FlyBrowser Setup Wizard!")
    print("This wizard will help you configure FlyBrowser for your environment.\n")

    config: Dict[str, Any] = {}

    # Step 1: Deployment Mode
    print("=" * 60)
    print("STEP 1: Deployment Mode")
    print("=" * 60)

    mode = prompt_choice(
        "Select deployment mode:",
        ["Standalone (single node)", "Cluster (multi-node)"],
        default=0,
    )
    config["deployment_mode"] = "standalone" if "Standalone" in mode else "cluster"

    # Step 2: Service Configuration
    print("\n" + "=" * 60)
    print("STEP 2: Service Configuration")
    print("=" * 60)

    config["host"] = prompt("Host address", default="0.0.0.0")
    config["port"] = int(prompt("Port", default="8000"))
    config["env"] = prompt_choice(
        "Environment:",
        ["development", "staging", "production"],
        default=2,
    )
    config["log_level"] = prompt_choice(
        "Log level:",
        ["DEBUG", "INFO", "WARNING", "ERROR"],
        default=1,
    )

    # Step 3: Browser Pool Configuration
    print("\n" + "=" * 60)
    print("STEP 3: Browser Pool Configuration")
    print("=" * 60)

    config["pool_min_size"] = int(prompt("Minimum browser instances", default="1"))
    config["pool_max_size"] = int(prompt("Maximum browser instances", default="10"))
    config["max_sessions"] = int(prompt("Maximum concurrent sessions", default="100"))
    config["headless"] = prompt_bool("Run browsers in headless mode?", default=True)

    # Step 4: Cluster Configuration (if cluster mode)
    if config["deployment_mode"] == "cluster":
        print("\n" + "=" * 60)
        print("STEP 4: Cluster Configuration")
        print("=" * 60)

        role = prompt_choice(
            "Node role:",
            ["Coordinator (manages workers)", "Worker (handles browsers)"],
            default=1,
        )
        config["node_role"] = "coordinator" if "Coordinator" in role else "worker"

        if config["node_role"] == "worker":
            config["coordinator_host"] = prompt("Coordinator host", default="localhost")
            config["coordinator_port"] = int(prompt("Coordinator port", default="8001"))

    # Step 5: LLM Provider Configuration
    print("\n" + "=" * 60)
    print("STEP 5: LLM Provider Configuration")
    print("=" * 60)
    print("\nFlyBrowser supports multiple LLM providers for AI-powered automation.")
    print("You can configure one or more providers.\n")

    # Select default provider
    provider = prompt_choice(
        "Select default LLM provider:",
        ["OpenAI (GPT-4, GPT-3.5)", "Anthropic (Claude)", "Google (Gemini)", "Ollama (Local)", "Skip LLM setup"],
        default=0,
    )

    if "OpenAI" in provider:
        config["default_llm_provider"] = "openai"
        config["openai_api_key"] = prompt("OpenAI API Key", required=True)
        model = prompt_choice(
            "Select default OpenAI model:",
            ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            default=0,
        )
        config["default_llm_model"] = model

    elif "Anthropic" in provider:
        config["default_llm_provider"] = "anthropic"
        config["anthropic_api_key"] = prompt("Anthropic API Key", required=True)
        model = prompt_choice(
            "Select default Anthropic model:",
            ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"],
            default=0,
        )
        config["default_llm_model"] = model

    elif "Google" in provider:
        config["default_llm_provider"] = "gemini"
        config["google_api_key"] = prompt("Google AI API Key", required=True)
        model = prompt_choice(
            "Select default Gemini model:",
            ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
            default=0,
        )
        config["default_llm_model"] = model

    elif "Ollama" in provider:
        config["default_llm_provider"] = "ollama"
        config["ollama_base_url"] = prompt("Ollama base URL", default="http://localhost:11434")
        model = prompt("Ollama model name", default="llama3.2")
        config["default_llm_model"] = model

    # Ask about additional providers
    if "Skip" not in provider:
        if prompt_bool("\nConfigure additional LLM providers?", default=False):
            if config.get("default_llm_provider") != "openai":
                if prompt_bool("  Add OpenAI?", default=False):
                    config["openai_api_key"] = prompt("  OpenAI API Key", required=True)

            if config.get("default_llm_provider") != "anthropic":
                if prompt_bool("  Add Anthropic?", default=False):
                    config["anthropic_api_key"] = prompt("  Anthropic API Key", required=True)

            if config.get("default_llm_provider") != "gemini":
                if prompt_bool("  Add Google Gemini?", default=False):
                    config["google_api_key"] = prompt("  Google AI API Key", required=True)

            if config.get("default_llm_provider") != "ollama":
                if prompt_bool("  Add Ollama (local)?", default=False):
                    config["ollama_base_url"] = prompt("  Ollama base URL", default="http://localhost:11434")

    # Step 6: Save Configuration
    print("\n" + "=" * 60)
    print("STEP 6: Save Configuration")
    print("=" * 60)

    output_path = prompt("Output file path", default=".env")
    generate_env_file(config, Path(output_path))

    # Print summary
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"  Deployment Mode: {config['deployment_mode']}")
    print(f"  Service: {config['host']}:{config['port']}")
    print(f"  Environment: {config['env']}")
    print(f"  Browser Pool: {config['pool_min_size']}-{config['pool_max_size']} instances")
    print(f"  Max Sessions: {config['max_sessions']}")
    if config["deployment_mode"] == "cluster":
        print(f"  Node Role: {config['node_role']}")
        if config["node_role"] == "worker":
            print(f"  Coordinator: {config['coordinator_host']}:{config['coordinator_port']}")

    print("\n[OK] Setup complete! You can now start FlyBrowser with:")
    print(f"  uvicorn flybrowser.service.app:app --host {config['host']} --port {config['port']}")

    return config


def select_browsers_interactive() -> List[str]:
    """Interactive browser selection.
    
    Returns:
        List of selected browsers
    """
    print("\n" + "=" * 60)
    print("BROWSER SELECTION")
    print("=" * 60)
    print("\nSelect which Playwright browsers to install:")
    print("  Chromium: Google Chrome-based browser (recommended)")
    print("  Firefox: Mozilla Firefox browser")
    print("  WebKit: Safari-based browser\n")
    
    browsers = []
    
    if prompt_bool("Install Chromium?", default=True):
        browsers.append("chromium")
    
    if prompt_bool("Install Firefox?", default=False):
        browsers.append("firefox")
    
    if prompt_bool("Install WebKit?", default=False):
        browsers.append("webkit")
    
    if not browsers:
        print("\n[WARN] No browsers selected. Defaulting to Chromium.")
        browsers = ["chromium"]
    
    print(f"\n[INFO] Will install: {', '.join(browsers)}")
    return browsers


def install_optional_extras() -> List[str]:
    """Interactive selection of optional dependencies.
    
    Returns:
        List of extras to install
    """
    print("\n" + "=" * 60)
    print("OPTIONAL DEPENDENCIES")
    print("=" * 60)
    print("\nFlyBrowser has optional features you can install:")
    print("  jupyter: Support for Jupyter notebooks (nest_asyncio, ipython)")
    print("  repl: Enhanced interactive REPL experience")
    print("  dev: Development tools (pytest, black, ruff, mypy)\n")
    
    extras = []
    
    if prompt_bool("Install Jupyter notebook support?", default=False):
        extras.append("jupyter")
    
    if prompt_bool("Install enhanced REPL?", default=False):
        extras.append("repl")
    
    if prompt_bool("Install development tools?", default=False):
        extras.append("dev")
    
    if extras:
        print(f"\n[INFO] Will install extras: {', '.join(extras)}")
    else:
        print("\n[INFO] No optional extras selected.")
    
    return extras


def install_extras_pip(extras: List[str]) -> bool:
    """Install optional dependencies using pip.
    
    Args:
        extras: List of extras to install (e.g., ['jupyter', 'repl'])
    
    Returns:
        True if installation succeeded, False otherwise
    """
    if not extras:
        return True
    
    print(f"\n[INSTALL] Installing optional dependencies: {', '.join(extras)}")
    
    try:
        # Determine package manager
        try:
            import uv
            use_uv = True
        except ImportError:
            use_uv = False
        
        # Build install command
        extras_str = ",".join(extras)
        package_spec = f"flybrowser[{extras_str}]"
        
        if use_uv:
            cmd = [sys.executable, "-m", "uv", "pip", "install", "-e", f".[{extras_str}]"]
        else:
            cmd = [sys.executable, "-m", "pip", "install", "-e", f".[{extras_str}]"]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        try:
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print(f"[FAIL] Failed to install extras: {stderr}")
                return False
            print(f"[OK] Optional dependencies installed")
            return True
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Extra installation cancelled.")
        raise
    except Exception as e:
        print(f"[FAIL] Extra installation failed: {e}")
        return False


def install_browsers(browsers: Optional[List[str]] = None) -> bool:
    """Install Playwright browsers.

    Args:
        browsers: List of browsers to install. Defaults to ["chromium"].

    Returns:
        True if installation succeeded, False otherwise.
    """
    browsers = browsers or ["chromium"]
    print(f"\n[INSTALL] Installing Playwright browsers: {', '.join(browsers)}")

    try:
        for browser in browsers:
            check_interrupted()
            print(f"[INSTALL] Installing {browser}...")
            
            # Use Popen for better interrupt handling
            process = subprocess.Popen(
                [sys.executable, "-m", "playwright", "install", browser],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            try:
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    print(f"[FAIL] Failed to install {browser}: {stderr}")
                    return False
                print(f"[OK] Installed {browser}")
            except KeyboardInterrupt:
                # Kill the subprocess on interrupt
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise

        # Install system dependencies if needed
        if platform.system() == "Linux":
            check_interrupted()
            print("[INSTALL] Installing system dependencies...")
            
            process = subprocess.Popen(
                [sys.executable, "-m", "playwright", "install-deps"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            try:
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    print(f"[WARN] Warning: Could not install system deps: {stderr}")
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise

        return True
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Browser installation cancelled.")
        raise
    except Exception as e:
        print(f"[FAIL] Browser installation failed: {e}")
        return False


def verify_installation() -> bool:
    """Verify FlyBrowser installation.

    Returns:
        True if verification passed, False otherwise.
    """
    print("\n[CHECK] Verifying installation...")

    checks = []

    # Check FlyBrowser import
    try:
        import flybrowser
        version = getattr(flybrowser, "__version__", "unknown")
        print(f"[OK] FlyBrowser {version} installed")
        checks.append(True)
    except ImportError as e:
        print(f"[FAIL] FlyBrowser import failed: {e}")
        checks.append(False)

    # Check Playwright
    try:
        from playwright.sync_api import sync_playwright
        print("[OK] Playwright installed")
        checks.append(True)
    except ImportError:
        print("[FAIL] Playwright not installed")
        checks.append(False)

    # Check browser availability
    try:
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "--dry-run", "chromium"],
            capture_output=True,
            text=True,
        )
        if "chromium" in result.stdout.lower() or result.returncode == 0:
            print("[OK] Chromium browser available")
            checks.append(True)
        else:
            print("[WARN] Chromium browser may need installation")
            checks.append(True)  # Not a hard failure
    except Exception:
        print("[WARN] Could not verify browser installation")
        checks.append(True)  # Not a hard failure

    return all(checks)


def cmd_install(args: argparse.Namespace) -> int:
    """Handle the install command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        print_banner()
        print("\n[INSTALL] Installing FlyBrowser...\n")

        # Prompt for optional extras if interactive mode and not specified
        if args.interactive and not args.no_wizard:
            extras = install_optional_extras()
            if extras and not install_extras_pip(extras):
                print("\n[WARN] Optional dependencies had issues, but continuing...")

        # Select and install browsers unless skipped
        if not args.no_browsers:
            if args.interactive and not args.no_wizard:
                # Interactive browser selection
                browsers = select_browsers_interactive()
            else:
                # Use command line argument or default
                browsers = args.browsers.split(",") if args.browsers else ["chromium"]
            
            if not install_browsers(browsers):
                print("\n[WARN] Browser installation had issues, but continuing...")

        # Run configuration wizard if requested
        if args.configure:
            print("\n" + "=" * 60)
            setup_wizard()

        # Verify installation
        if verify_installation():
            print("\n" + "=" * 60)
            print("[SUCCESS] FlyBrowser installation complete!")
            print("=" * 60)
            print("\nNext steps:")
            print("  1. Configure: flybrowser-setup configure")
            print("  2. Start server: flybrowser-serve")
            print("  3. Check status: flybrowser-cluster status")
            return 0
        else:
            print("\n[FAIL] Installation verification failed")
            return 1
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Installation cancelled by user.")
        print("[INFO] Run 'flybrowser-setup install' to try again.")
        return 130  # Standard exit code for SIGINT


def cmd_configure(args: argparse.Namespace) -> int:
    """Handle the configure command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        setup_wizard()
        return 0
    except KeyboardInterrupt:
        print("\n\nConfiguration cancelled.")
        return 1


def cmd_browsers(args: argparse.Namespace) -> int:
    """Handle the browsers command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        if args.action == "install":
            browsers = args.browsers.split(",") if args.browsers else ["chromium"]
            return 0 if install_browsers(browsers) else 1
        elif args.action == "list":
            print("Available browsers: chromium, firefox, webkit")
            return 0
        else:
            print(f"Unknown action: {args.action}")
            return 1
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Browser operation cancelled by user.")
        return 130


def cmd_verify(args: argparse.Namespace) -> int:
    """Handle the verify command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    return 0 if verify_installation() else 1


def setup_jupyter_kernel() -> bool:
    """Install Jupyter kernel for FlyBrowser.
    
    Returns:
        True if successful, False otherwise
    """
    print("[INFO] Setting up Jupyter kernel...")
    
    # Check if we're in a venv
    venv_python = Path(sys.executable)
    if not (venv_python.parent.parent / "pyvenv.cfg").exists():
        # Check alternative venv location
        if not (venv_python.parent.parent / "bin" / "activate").exists():
            print("[WARN] Not running in a virtual environment")
            print("[INFO] Jupyter kernel setup works best with venv installation")
    
    # Check if jupyter and ipykernel are installed
    try:
        import jupyter  # noqa: F401
        import ipykernel  # noqa: F401
    except ImportError:
        print("[INFO] Jupyter and ipykernel not found, installing...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet",
                 "jupyter", "ipykernel", "nest_asyncio"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("[OK] Installed jupyter, ipykernel, and nest_asyncio")
        except subprocess.CalledProcessError:
            print("[FAIL] Failed to install Jupyter packages")
            return False
    
    # Register the kernel
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ipykernel", "install",
             "--user", "--name=flybrowser", "--display-name=FlyBrowser"],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            print("[OK] Jupyter kernel installed successfully")
            print("")
            print("To use in Jupyter:")
            print("  1. Start Jupyter: jupyter notebook")
            print("  2. Select kernel: Kernel → Change Kernel → FlyBrowser")
            print("  3. Import: from flybrowser import FlyBrowser")
            print("")
            return True
        else:
            print(f"[FAIL] Failed to register kernel: {result.stderr}")
            return False
    except Exception as e:
        print(f"[FAIL] Error registering kernel: {e}")
        return False


def uninstall_jupyter_kernel() -> bool:
    """Remove Jupyter kernel for FlyBrowser.
    
    Returns:
        True if successful, False otherwise
    """
    print("[INFO] Removing Jupyter kernel...")
    
    # Check if kernel exists
    try:
        result = subprocess.run(
            ["jupyter", "kernelspec", "list"],
            capture_output=True,
            text=True,
        )
        
        if "flybrowser" not in result.stdout:
            print("[INFO] FlyBrowser kernel not found")
            return True
    except FileNotFoundError:
        print("[WARN] jupyter command not found")
        return False
    
    # Remove the kernel
    try:
        subprocess.check_call(
            ["jupyter", "kernelspec", "uninstall", "flybrowser", "-y"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[OK] Jupyter kernel removed")
        return True
    except subprocess.CalledProcessError:
        print("[FAIL] Failed to remove kernel")
        return False


def check_jupyter_status() -> bool:
    """Check Jupyter kernel installation status.
    
    Returns:
        True if kernel is installed, False otherwise
    """
    print("[INFO] Checking Jupyter kernel status...")
    print("")
    
    # Check if jupyter is available
    jupyter_available = shutil.which("jupyter") is not None
    print(f"  Jupyter command: {'✓ Available' if jupyter_available else '✗ Not found'}")
    
    if not jupyter_available:
        print("")
        print("[INFO] Jupyter not installed")
        print("[TIP] Install with: pip install jupyter ipykernel")
        return False
    
    # Check if kernel is registered
    try:
        result = subprocess.run(
            ["jupyter", "kernelspec", "list"],
            capture_output=True,
            text=True,
        )
        
        kernel_installed = "flybrowser" in result.stdout
        print(f"  FlyBrowser kernel: {'✓ Installed' if kernel_installed else '✗ Not installed'}")
        
        if kernel_installed:
            # Extract kernel path from output
            for line in result.stdout.split("\n"):
                if "flybrowser" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        print(f"  Kernel location: {parts[1]}")
            print("")
            print("[OK] Jupyter kernel is properly configured")
            print("")
            print("Usage:")
            print("  1. Start Jupyter: jupyter notebook")
            print("  2. Select FlyBrowser kernel in notebook")
            return True
        else:
            print("")
            print("[INFO] Kernel not installed")
            print("[TIP] Install with: flybrowser setup jupyter install")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def fix_jupyter_kernel() -> bool:
    """Fix/reinstall Jupyter kernel.
    
    Returns:
        True if successful, False otherwise
    """
    print("[INFO] Fixing Jupyter kernel installation...")
    
    # Uninstall existing kernel (ignore errors)
    print("[INFO] Removing existing kernel...")
    try:
        subprocess.run(
            ["jupyter", "kernelspec", "uninstall", "flybrowser", "-y"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except:
        pass
    
    # Reinstall
    print("[INFO] Reinstalling kernel...")
    return setup_jupyter_kernel()


def cmd_jupyter(args: argparse.Namespace) -> int:
    """Handle the jupyter command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        if args.jupyter_action == "install":
            return 0 if setup_jupyter_kernel() else 1
        elif args.jupyter_action == "uninstall":
            return 0 if uninstall_jupyter_kernel() else 1
        elif args.jupyter_action == "status":
            return 0 if check_jupyter_status() else 1
        elif args.jupyter_action == "fix":
            return 0 if fix_jupyter_kernel() else 1
        else:
            print(f"Unknown action: {args.jupyter_action}")
            return 1
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Jupyter operation cancelled by user.")
        return 130


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="flybrowser-setup",
        description="FlyBrowser Setup and Configuration Tool",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Install command
    install_parser = subparsers.add_parser(
        "install",
        help="Install FlyBrowser and dependencies",
    )
    install_parser.add_argument(
        "--no-browsers",
        action="store_true",
        help="Skip Playwright browser installation",
    )
    install_parser.add_argument(
        "--browsers",
        type=str,
        default="chromium",
        help="Comma-separated list of browsers to install (default: chromium)",
    )
    install_parser.add_argument(
        "--configure",
        action="store_true",
        help="Run configuration wizard after installation",
    )
    install_parser.add_argument(
        "--interactive",
        action="store_true",
        default=True,
        help="Run interactive prompts for options (default: True)",
    )
    install_parser.add_argument(
        "--no-wizard",
        action="store_true",
        help="Skip interactive prompts and use defaults",
    )
    install_parser.set_defaults(func=cmd_install)

    # Configure command
    configure_parser = subparsers.add_parser(
        "configure",
        help="Run interactive configuration wizard",
    )
    configure_parser.set_defaults(func=cmd_configure)

    # Browsers command
    browsers_parser = subparsers.add_parser(
        "browsers",
        help="Manage Playwright browsers",
    )
    browsers_parser.add_argument(
        "action",
        choices=["install", "list"],
        help="Action to perform",
    )
    browsers_parser.add_argument(
        "--browsers",
        type=str,
        default="chromium",
        help="Comma-separated list of browsers (default: chromium)",
    )
    browsers_parser.set_defaults(func=cmd_browsers)

    # Verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify FlyBrowser installation",
    )
    verify_parser.set_defaults(func=cmd_verify)

    # Jupyter command
    jupyter_parser = subparsers.add_parser(
        "jupyter",
        help="Manage Jupyter kernel integration",
    )
    jupyter_parser.add_argument(
        "jupyter_action",
        choices=["install", "uninstall", "status", "fix"],
        help="Jupyter action to perform",
    )
    jupyter_parser.set_defaults(func=cmd_jupyter)

    return parser


def main() -> None:
    """Main entry point for the setup CLI."""
    # Set up signal handlers for graceful cancellation
    setup_signal_handlers()
    
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        # Default to interactive wizard for backward compatibility
        try:
            setup_wizard()
        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Setup cancelled by user.")
            print("[INFO] Run 'flybrowser-setup' to try again.")
            sys.exit(130)  # Standard exit code for SIGINT
    else:
        try:
            sys.exit(args.func(args))
        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Operation cancelled by user.")
            sys.exit(130)


if __name__ == "__main__":
    main()

