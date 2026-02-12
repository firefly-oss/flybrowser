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
FlyBrowser Uninstall CLI.

Provides uninstallation functionality for FlyBrowser.

Usage:
    python -m flybrowser.cli.uninstall
    python -m flybrowser.cli.uninstall --all
    python -m flybrowser.cli.uninstall --keep-data

    Or with the installed command:
    flybrowser uninstall
    flybrowser-uninstall --all
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


# ANSI color codes
class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    NC = "\033[0m"  # No Color


# Default paths
DEFAULT_INSTALL_DIR = "/usr/local/bin"
DEFAULT_VENV_DIR = Path.home() / ".flybrowser" / "venv"
DEFAULT_DATA_DIR = Path.home() / ".flybrowser"
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "flybrowser"


def print_info(msg: str) -> None:
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ {msg}{Colors.NC}")


def print_success(msg: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}[ok] {msg}{Colors.NC}")


def print_warning(msg: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}[warning] {msg}{Colors.NC}")


def print_error(msg: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}[fail] {msg}{Colors.NC}")


def print_step(msg: str) -> None:
    """Print a step message."""
    print(f"{Colors.CYAN}🗑 {msg}{Colors.NC}")


def get_banner() -> str:
    """Get the FlyBrowser banner."""
    return r"""  _____.__         ___.
_/ ____\  | ___.__.\\_ |_________  ______  _  ________ ___________
\   __\|  |<   |  | | __ \_  __ \/  _ \ \/ \/ /  ___// __ \_  __ \
 |  |  |  |_\___  | | \_\ \  | \(  <_> )     /\___ \\  ___/|  | \/
 |__|  |____/ ____| |___  /__|   \____/ \/\_//____  >\___  >__|
            \/          \/                        \/     \/"""


def print_banner() -> None:
    """Print the FlyBrowser banner."""
    print()
    print(f"{Colors.BLUE}{get_banner()}{Colors.NC}")
    print()
    print(f"{Colors.RED} Uninstaller{Colors.NC}")
    print()


def confirm(message: str, default: bool = False) -> bool:
    """Ask for user confirmation."""
    prompt = "[Y/n]" if default else "[y/N]"
    try:
        response = input(f"{message} {prompt} ").strip().lower()
        if not response:
            return default
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def detect_os() -> str:
    """Detect the operating system."""
    system = platform.system().lower()
    if system == "darwin":
        return "darwin"
    elif system == "linux":
        return "linux"
    elif system == "windows":
        return "windows"
    return "unknown"


def remove_services() -> None:
    """Remove system services."""
    print_step("Checking for services...")
    
    os_type = detect_os()
    removed = False
    
    # macOS launchd
    if os_type == "darwin":
        plist_path = Path.home() / "Library" / "LaunchAgents" / "dev.flybrowser.plist"
        if plist_path.exists():
            print_info("Found launchd service")
            try:
                subprocess.run(
                    ["launchctl", "unload", str(plist_path)],
                    capture_output=True,
                    check=False,
                )
                plist_path.unlink()
                print_success("Removed launchd service")
                removed = True
            except Exception as e:
                print_warning(f"Could not remove launchd service: {e}")
    
    # Linux systemd
    if os_type == "linux":
        service_path = Path("/etc/systemd/system/flybrowser.service")
        if service_path.exists():
            print_info("Found systemd service")
            try:
                subprocess.run(
                    ["sudo", "systemctl", "stop", "flybrowser"],
                    capture_output=True,
                    check=False,
                )
                subprocess.run(
                    ["sudo", "systemctl", "disable", "flybrowser"],
                    capture_output=True,
                    check=False,
                )
                subprocess.run(
                    ["sudo", "rm", "-f", str(service_path)],
                    check=True,
                )
                subprocess.run(
                    ["sudo", "systemctl", "daemon-reload"],
                    check=True,
                )
                print_success("Removed systemd service")
                removed = True
            except Exception as e:
                print_warning(f"Could not remove systemd service: {e}")
    
    if not removed:
        print_info("No services found")


def remove_cli_wrappers(install_dir: str = DEFAULT_INSTALL_DIR) -> None:
    """Remove CLI wrapper scripts."""
    print_step("Removing CLI commands...")
    
    wrappers = [
        "flybrowser",
        "flybrowser-serve",
        "flybrowser-setup",
        "flybrowser-cluster",
        "flybrowser-admin",
        "flybrowser-uninstall",
    ]
    
    install_path = Path(install_dir)
    removed = 0
    
    for wrapper in wrappers:
        wrapper_path = install_path / wrapper
        if wrapper_path.exists():
            try:
                if os.access(install_dir, os.W_OK):
                    wrapper_path.unlink()
                else:
                    subprocess.run(
                        ["sudo", "rm", "-f", str(wrapper_path)],
                        check=True,
                    )
                print_success(f"Removed {wrapper_path}")
                removed += 1
            except Exception as e:
                print_warning(f"Could not remove {wrapper_path}: {e}")
    
    if removed == 0:
        print_info(f"No CLI commands found in {install_dir}")
    else:
        print_success(f"Removed {removed} CLI command(s)")


def remove_venv(venv_dir: Path = DEFAULT_VENV_DIR) -> None:
    """Remove the virtual environment."""
    print_step("Removing virtual environment...")
    
    if venv_dir.exists():
        try:
            shutil.rmtree(venv_dir)
            print_success(f"Removed {venv_dir}")
        except Exception as e:
            print_error(f"Could not remove virtual environment: {e}")
    else:
        print_info(f"Virtual environment not found at {venv_dir}")


def remove_data(
    data_dir: Path = DEFAULT_DATA_DIR,
    config_dir: Path = DEFAULT_CONFIG_DIR,
) -> None:
    """Remove data and configuration directories."""
    print_step("Removing data and configuration...")
    
    removed = False
    
    # Remove data directory
    if data_dir.exists():
        try:
            shutil.rmtree(data_dir)
            print_success(f"Removed {data_dir}")
            removed = True
        except Exception as e:
            print_error(f"Could not remove data directory: {e}")
    
    # Remove config directory
    if config_dir.exists():
        try:
            shutil.rmtree(config_dir)
            print_success(f"Removed {config_dir}")
            removed = True
        except Exception as e:
            print_error(f"Could not remove config directory: {e}")
    
    # Check for .env in current directory
    env_file = Path.cwd() / ".env"
    if env_file.exists():
        try:
            content = env_file.read_text()
            if "FLYBROWSER" in content:
                if confirm("Remove .env file in current directory?"):
                    env_file.unlink()
                    print_success("Removed .env")
                    removed = True
        except Exception:
            pass
    
    if not removed:
        print_info("No data directories found")


def print_summary(keep_data: bool, remove_all: bool) -> None:
    """Print uninstall summary."""
    print()
    print(f"{Colors.GREEN}{'═' * 60}{Colors.NC}")
    print(f"{Colors.GREEN}[ok] FlyBrowser has been uninstalled{Colors.NC}")
    print(f"{Colors.GREEN}{'═' * 60}{Colors.NC}")
    print()
    
    if keep_data or not remove_all:
        print_info("Configuration and data were preserved")
    else:
        print_info("All FlyBrowser files have been removed")
    
    print()
    print_info("To reinstall FlyBrowser:")
    print("  curl -fsSL https://get.flybrowser.dev | bash")
    print("  # or")
    print("  git clone https://github.com/fireflyresearch/flybrowser && cd flybrowser && ./install.sh")
    print()


def check_installation(
    venv_dir: Path = DEFAULT_VENV_DIR,
    install_dir: str = DEFAULT_INSTALL_DIR,
) -> bool:
    """Check if FlyBrowser is installed."""
    install_path = Path(install_dir)
    return venv_dir.exists() or (install_path / "flybrowser").exists()


def run_uninstall(
    remove_all: bool = False,
    keep_data: bool = False,
    force: bool = False,
    install_dir: str = DEFAULT_INSTALL_DIR,
    venv_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    config_dir: Optional[Path] = None,
) -> int:
    """Run the uninstall process.
    
    Args:
        remove_all: Remove everything including data
        keep_data: Keep configuration and data
        force: Don't ask for confirmation
        install_dir: Directory containing CLI wrappers
        venv_dir: Virtual environment directory
        data_dir: Data directory
        config_dir: Configuration directory
    
    Returns:
        Exit code (0 for success)
    """
    venv_dir = venv_dir or DEFAULT_VENV_DIR
    data_dir = data_dir or DEFAULT_DATA_DIR
    config_dir = config_dir or DEFAULT_CONFIG_DIR
    
    print_banner()
    
    # Check if installed
    if not check_installation(venv_dir, install_dir):
        print_warning("FlyBrowser does not appear to be installed")
        print_info(f"Checked: {venv_dir}")
        print_info(f"Checked: {install_dir}/flybrowser")
        return 0
    
    # Show what will be removed
    print()
    print(f"{Colors.BOLD}The following will be removed:{Colors.NC}")
    print()
    
    install_path = Path(install_dir)
    if (install_path / "flybrowser").exists():
        print(f"  • CLI commands in {install_dir}")
    
    if venv_dir.exists():
        print(f"  • Virtual environment: {venv_dir}")
    
    os_type = detect_os()
    if os_type == "darwin":
        plist = Path.home() / "Library" / "LaunchAgents" / "dev.flybrowser.plist"
        if plist.exists():
            print("  • macOS launchd service")
    if os_type == "linux":
        if Path("/etc/systemd/system/flybrowser.service").exists():
            print("  • Linux systemd service")
    
    if not keep_data:
        if remove_all:
            if data_dir.exists():
                print(f"  • Data directory: {data_dir}")
            if config_dir.exists():
                print(f"  • Config directory: {config_dir}")
        else:
            print()
            print(f"{Colors.DIM}  (Data and configuration will be preserved unless --all is used){Colors.NC}")
    
    print()
    
    # Confirm
    if not force:
        if not confirm("Proceed with uninstall?"):
            print_info("Uninstall cancelled")
            return 0
    
    print()
    
    # Perform uninstall
    remove_services()
    remove_cli_wrappers(install_dir)
    remove_venv(venv_dir)
    
    # Handle data removal
    if not keep_data:
        if remove_all:
            remove_data(data_dir, config_dir)
        elif not force:
            print()
            if confirm("Also remove data and configuration (~/.flybrowser)?"):
                remove_data(data_dir, config_dir)
            else:
                print_info("Keeping data and configuration")
    
    print_summary(keep_data, remove_all)
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="flybrowser-uninstall",
        description="Uninstall FlyBrowser and its components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  flybrowser uninstall                  # Interactive uninstall
  flybrowser uninstall --all            # Remove everything
  flybrowser uninstall --keep-data      # Keep configuration
  flybrowser uninstall --force          # No confirmation
""",
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        dest="remove_all",
        help="Remove everything including data and configuration",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep configuration and data (remove only binaries)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Don't ask for confirmation",
    )
    parser.add_argument(
        "--install-dir",
        default=DEFAULT_INSTALL_DIR,
        help=f"CLI installation directory (default: {DEFAULT_INSTALL_DIR})",
    )
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=DEFAULT_VENV_DIR,
        help=f"Virtual environment directory (default: {DEFAULT_VENV_DIR})",
    )
    
    return parser


def main() -> None:
    """Main entry point for the uninstall CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        exit_code = run_uninstall(
            remove_all=args.remove_all,
            keep_data=args.keep_data,
            force=args.force,
            install_dir=args.install_dir,
            venv_dir=args.venv_dir,
        )
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n")
        print_info("Uninstall cancelled by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
