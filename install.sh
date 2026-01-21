#!/usr/bin/env bash
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

# ============================================================================
# FlyBrowser Installation Script
# ============================================================================
#
# Best-in-class installation supporting multiple scenarios:
#   - One-liner: curl -fsSL get.flybrowser.dev | bash
#   - Source installation: ./install.sh (when cloned from GitHub)
#
# Note: FlyBrowser is installed from GitHub source, not PyPI.
#
# Features:
#   - Auto-detects source vs remote installation
#   - Prefers uv for faster installs (falls back to pip)
#   - Comprehensive dependency checking
#   - Interactive configuration wizard (optional)
#   - System service installation (optional)
#   - Uninstall capability
#
# Usage:
#   # One-liner installation (recommended)
#   curl -fsSL https://get.flybrowser.dev | bash
#
#   # Source installation (after git clone)
#   ./install.sh
#
#   # With options
#   ./install.sh --install-dir /opt/flybrowser --with-service
#   ./install.sh --dev              # Development mode from source
#   ./install.sh --version 1.26.0   # Specific version
#   ./install.sh --uninstall        # Remove FlyBrowser
#
# Options:
#   --install-dir DIR     CLI installation directory (default: /usr/local/bin)
#   --venv-dir DIR        Virtual environment directory (default: ~/.flybrowser/venv)
#   --version VER         Install specific version (default: latest)
#   --dev                 Development mode (install from current directory)
#   --with-service        Install as system service (systemd/launchd)
#   --cluster             Configure for cluster mode
#   --no-playwright       Skip Playwright browser installation
#   --no-wizard           Skip interactive configuration wizard
#   --use-pip             Force pip instead of uv
#   --uninstall           Uninstall FlyBrowser
#   --help                Show this help message
#
# Environment Variables:
#   FLYBROWSER_NO_MODIFY_PATH    Don't modify PATH
#   FLYBROWSER_ACCEPT_DEFAULTS   Accept all defaults (non-interactive)
#
# ============================================================================

set -euo pipefail

# Script version
INSTALLER_VERSION="2.0.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

# Emoji support detection
if [[ "${TERM:-}" == *"256color"* ]] || [[ "${COLORTERM:-}" == "truecolor" ]]; then
    EMOJI_CHECK="✓"
    EMOJI_CROSS="✗"
    EMOJI_ARROW="→"
    EMOJI_WARN="⚠"
    EMOJI_INFO="ℹ"
    EMOJI_ROCKET="🚀"
    EMOJI_PACKAGE="📦"
    EMOJI_GEAR="⚙"
else
    EMOJI_CHECK="[OK]"
    EMOJI_CROSS="[FAIL]"
    EMOJI_ARROW="->"
    EMOJI_WARN="[WARN]"
    EMOJI_INFO="[INFO]"
    EMOJI_ROCKET="[*]"
    EMOJI_PACKAGE="[PKG]"
    EMOJI_GEAR="[CFG]"
fi

# Default values
INSTALL_DIR="/usr/local/bin"
VENV_DIR="$HOME/.flybrowser/venv"
DATA_DIR="$HOME/.flybrowser"
WITH_SERVICE=false
CLUSTER_MODE=false
DEV_MODE=false
INSTALL_PLAYWRIGHT=true
RUN_WIZARD=true
FORCE_PIP=false
UNINSTALL=false
FLYBROWSER_VERSION="latest"
GITHUB_REPO="firefly-oss/flybrowsers"
GITHUB_RAW_URL="https://raw.githubusercontent.com/$GITHUB_REPO/main"

# Installation context
INSTALLATION_CONTEXT="unknown"  # source, remote, or unknown
SOURCE_DIR=""
TEMP_DIR=""
PACKAGE_MANAGER="pip"  # pip or uv

# Track if we're in the middle of an operation
INSTALLATION_IN_PROGRESS=false
INTERRUPTED=false

# Cleanup on exit
cleanup() {
    local exit_code=$?
    
    # Don't show messages if we're being interrupted (handled by interrupt_handler)
    if [ "$INTERRUPTED" = true ]; then
        return
    fi
    
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
    
    # If installation was in progress and we're exiting with an error, show cleanup message
    if [ "$INSTALLATION_IN_PROGRESS" = true ] && [ $exit_code -ne 0 ]; then
        echo ""
        print_warning "Installation did not complete successfully"
    fi
}

# Handle interrupt signals (Ctrl+C, SIGTERM, etc.)
interrupt_handler() {
    local signal=$1
    INTERRUPTED=true
    
    echo ""  # New line after ^C
    print_warning "Installation interrupted by user (signal: $signal)"
    
    # Cleanup temp directory
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        print_info "Cleaning up temporary files..."
        rm -rf "$TEMP_DIR"
    fi
    
    # If venv was partially created, offer to clean it up
    if [ "$INSTALLATION_IN_PROGRESS" = true ] && [ -d "$VENV_DIR" ]; then
        echo ""
        print_info "Partial installation detected at $VENV_DIR"
        print_info "Run './install.sh' again to restart, or './install.sh --uninstall' to clean up"
    fi
    
    echo ""
    print_info "Installation cancelled. No changes were finalized."
    
    # Exit with appropriate code (128 + signal number is convention)
    # SIGINT=2, SIGTERM=15
    case $signal in
        INT)  exit 130 ;;  # 128 + 2
        TERM) exit 143 ;;  # 128 + 15
        *)    exit 1 ;;
    esac
}

# Set up signal traps
trap cleanup EXIT
trap 'interrupt_handler INT' INT
trap 'interrupt_handler TERM' TERM

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Linux*)     OS="linux";;
        Darwin*)    OS="darwin";;
        CYGWIN*|MINGW*|MSYS*) OS="windows";;
        *)          OS="unknown";;
    esac
    echo "$OS"
}

# Print colored message
print_msg() {
    local color=$1
    local msg=$2
    echo -e "${color}${msg}${NC}"
}

print_info() { print_msg "$BLUE" "$EMOJI_INFO $1"; }
print_success() { print_msg "$GREEN" "$EMOJI_CHECK $1"; }
print_warning() { print_msg "$YELLOW" "$EMOJI_WARN $1"; }
print_error() { print_msg "$RED" "$EMOJI_CROSS $1"; }
print_step() { print_msg "$CYAN" "$EMOJI_ARROW $1"; }

# Print a progress spinner
spinner() {
    local pid="$1"
    local delay=0.1
    local spinstr='|/-\'
    while kill -0 "$pid" 2>/dev/null; do
        # Check if we've been interrupted
        if [ "$INTERRUPTED" = true ]; then
            printf "    \b\b\b\b"
            return
        fi
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        spinstr=$temp${spinstr%"$temp"}
        sleep "$delay" || return  # sleep can be interrupted
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Detect installation context (source vs remote)
detect_installation_context() {
    # Check if we're in a git repository with flybrowser source
    if [ -f "pyproject.toml" ] && [ -d "flybrowser" ] && [ -f "flybrowser/__init__.py" ]; then
        # Verify it's actually the flybrowser project
        if grep -q 'name = "flybrowser"' pyproject.toml 2>/dev/null; then
            INSTALLATION_CONTEXT="source"
            SOURCE_DIR="$(pwd)"
            print_info "Detected source installation from: $SOURCE_DIR"
            return
        fi
    fi
    
    # Check if script is being piped (curl | bash)
    if [ ! -t 0 ]; then
        INSTALLATION_CONTEXT="remote"
        print_info "Detected remote installation (piped from curl)"
        return
    fi
    
    # Default: treat as remote installation
    INSTALLATION_CONTEXT="remote"
    print_info "Installation context: remote"
}

# Clone repository to temp directory (for remote/curl installation)
clone_to_temp() {
    # Skip if we're already in source directory
    if [ "$INSTALLATION_CONTEXT" = "source" ]; then
        SOURCE_DIR="$(pwd)"
        return
    fi
    
    # Check if git is available
    if ! command -v git &> /dev/null; then
        print_error "Git is required for installation"
        print_info "Install git and try again:"
        print_info "  macOS: xcode-select --install"
        print_info "  Linux: sudo apt install git"
        exit 1
    fi
    
    print_step "Cloning FlyBrowser repository..."
    TEMP_DIR=$(mktemp -d)
    
    local branch="main"
    if [ "$FLYBROWSER_VERSION" != "latest" ]; then
        branch="v$FLYBROWSER_VERSION"
    fi
    
    if git clone --depth 1 --branch "$branch" "https://github.com/$GITHUB_REPO.git" "$TEMP_DIR" 2>/dev/null; then
        SOURCE_DIR="$TEMP_DIR"
        print_success "Repository cloned"
    else
        print_error "Failed to clone repository from GitHub"
        print_info "Check your internet connection and try again"
        print_info "Or clone manually: git clone https://github.com/$GITHUB_REPO.git"
        exit 1
    fi
}

# Check if uv is available and working
check_uv() {
    if [ "$FORCE_PIP" = true ]; then
        PACKAGE_MANAGER="pip"
        return
    fi
    
    if command -v uv &> /dev/null; then
        # Verify uv works
        if uv --version &>/dev/null; then
            PACKAGE_MANAGER="uv"
            print_info "Using uv for faster installation"
            return
        fi
    fi
    
    PACKAGE_MANAGER="pip"
}

# Install uv if user wants it
install_uv_if_requested() {
    if [ "$FORCE_PIP" = true ]; then
        return
    fi
    
    if command -v uv &> /dev/null; then
        return
    fi
    
    # Only offer to install uv in interactive mode
    if [ -t 0 ] && [ "${FLYBROWSER_ACCEPT_DEFAULTS:-}" != "true" ]; then
        echo ""
        print_info "uv is a fast Python package installer (10-100x faster than pip)"
        read -p "Would you like to install uv? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Installing uv..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.cargo/bin:$PATH"
            PACKAGE_MANAGER="uv"
            print_success "uv installed"
        fi
    fi
}

# Print banner - uses canonical banner from flybrowser/banner.txt
print_banner() {
    echo ""
    # Check if we're in the repo and banner.txt exists
    if [ -f "flybrowser/banner.txt" ]; then
        print_msg "$BLUE" "$(cat flybrowser/banner.txt)"
    else
        # Fallback banner (matches flybrowser/banner.txt)
        print_msg "$BLUE" '  _____.__         ___.'
        print_msg "$BLUE" '_/ ____\  | ___.__.\_ |_________  ______  _  ________ ___________'
        print_msg "$BLUE" '\   __\|  |<   |  | | __ \_  __ \/  _ \ \/ \/ /  ___// __ \_  __ \'
        print_msg "$BLUE" ' |  |  |  |_\___  | | \_\ \  | \(  <_> )     /\___ \\  ___/|  | \/'
        print_msg "$BLUE" ' |__|  |____/ ____| |___  /__|   \____/ \/\_//____  >\___  >__|'
        print_msg "$BLUE" '            \/          \/                        \/     \/'
    fi
    echo ""
    print_msg "$GREEN" " Browser Automation Powered by LLM Agents"
    echo ""
}

# Check for required commands
check_requirements() {
    print_step "Checking system requirements..."
    echo ""

    local all_ok=true

    # Python check
    if command -v python3 &> /dev/null; then
        local python_version
        python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
        local major
        local minor
        major=$(echo "$python_version" | cut -d. -f1)
        minor=$(echo "$python_version" | cut -d. -f2)
        
        # Check for Python 3.9+
        if [ "$major" -eq 3 ] && [ "$minor" -ge 9 ]; then
            print_success "Python $python_version"
        elif [ "$major" -gt 3 ]; then
            print_success "Python $python_version"
        else
            print_error "Python $python_version (3.9+ required)"
            all_ok=false
        fi
    else
        print_error "Python 3 not found"
        all_ok=false
    fi

    # pip check
    if command -v pip3 &> /dev/null || command -v pip &> /dev/null; then
        local pip_version
        pip_version=$(pip3 --version 2>/dev/null | cut -d' ' -f2) || pip_version=$(pip --version 2>/dev/null | cut -d' ' -f2) || pip_version="unknown"
        print_success "pip $pip_version"
    else
        print_error "pip not found"
        all_ok=false
    fi

    # git check (optional but recommended)
    if command -v git &> /dev/null; then
        local git_version
        git_version=$(git --version | cut -d' ' -f3)
        print_success "git $git_version"
    else
        print_warning "git not found (optional, needed for --dev mode)"
    fi

    # curl check (optional)
    if command -v curl &> /dev/null; then
        print_success "curl available"
    else
        print_warning "curl not found (optional)"
    fi

    # uv check
    check_uv
    if [ "$PACKAGE_MANAGER" = "uv" ]; then
        local uv_version
        uv_version=$(uv --version 2>/dev/null | awk '{print $2}') || uv_version="unknown"
        print_success "uv $uv_version (fast mode)"
    else
        print_info "uv not found (using pip)"
    fi

    echo ""
    
    if [ "$all_ok" = false ]; then
        print_error "Missing required dependencies. Please install them and try again."
        echo ""
        print_info "Install Python: https://www.python.org/downloads/"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_step "Creating virtual environment..."
    
    mkdir -p "$DATA_DIR"
    
    if [ -d "$VENV_DIR" ]; then
        print_warning "Virtual environment exists, recreating..."
        rm -rf "$VENV_DIR"
    fi
    
    if [ "$PACKAGE_MANAGER" = "uv" ]; then
        uv venv "$VENV_DIR" --python python3 2>/dev/null || python3 -m venv "$VENV_DIR"
    else
        python3 -m venv "$VENV_DIR"
    fi
    
    print_success "Virtual environment created at $VENV_DIR"
}

# Install FlyBrowser from source
install_flybrowser() {
    print_step "Installing FlyBrowser from source..."
    
    # Activate venv
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    
    # Determine install command based on package manager
    local install_cmd="pip install"
    if [ "$PACKAGE_MANAGER" = "uv" ]; then
        install_cmd="uv pip install"
    fi
    
    # Upgrade pip first
    $install_cmd --upgrade pip > /dev/null 2>&1 || true
    
    # Determine installation path
    local install_path="$SOURCE_DIR"
    if [ -z "$install_path" ]; then
        install_path="."
    fi
    
    if [ ! -f "$install_path/pyproject.toml" ]; then
        print_error "pyproject.toml not found at $install_path"
        print_info "Make sure you're in the FlyBrowser repository directory"
        exit 1
    fi
    
    print_info "Installing from: $install_path"
    
    # Install with appropriate extras
    local extras="dev"
    if [ "$DEV_MODE" = true ]; then
        extras="dev,repl"
    fi
    
    if [ "$PACKAGE_MANAGER" = "uv" ]; then
        (cd "$install_path" && uv pip install -e ".[$extras]")
    else
        pip install -e "$install_path[$extras]"
    fi
    
    # Get installed version
    local installed_version
    installed_version=$(python -c "import flybrowser; print(flybrowser.__version__)" 2>/dev/null) || installed_version="unknown"
    print_success "FlyBrowser $installed_version installed"
}

# Install Playwright browsers (deprecated - now handled by flybrowser-setup CLI)
# Kept for backward compatibility if called directly
install_playwright() {
    if [ "$INSTALL_PLAYWRIGHT" = false ]; then
        print_warning "Skipping Playwright browser installation"
        return
    fi

    # Delegate to CLI if available, otherwise use direct command
    source "$VENV_DIR/bin/activate"
    if python -m flybrowser.cli.setup browsers install 2>/dev/null; then
        print_success "Playwright browsers installed via CLI"
    else
        print_info "Installing Playwright browsers directly..."
        playwright install chromium
        print_success "Playwright browsers installed"
    fi
}

# Create CLI wrapper scripts
create_wrappers() {
    print_step "Creating CLI commands..."

    # Check if we have write permission
    if [ ! -w "$INSTALL_DIR" ]; then
        print_warning "Need sudo to write to $INSTALL_DIR"
        SUDO="sudo"
    else
        SUDO=""
    fi

    # Create main unified flybrowser command
    $SUDO tee "$INSTALL_DIR/flybrowser" > /dev/null << EOF
#!/usr/bin/env bash
# FlyBrowser Unified CLI
# Generated by install.sh v$INSTALLER_VERSION
source "$VENV_DIR/bin/activate"
exec python -m flybrowser.cli.main "\$@"
EOF
    $SUDO chmod +x "$INSTALL_DIR/flybrowser"

    # Create wrapper for flybrowser-serve (legacy compatibility)
    $SUDO tee "$INSTALL_DIR/flybrowser-serve" > /dev/null << EOF
#!/usr/bin/env bash
source "$VENV_DIR/bin/activate"
exec python -m flybrowser.cli.serve "\$@"
EOF
    $SUDO chmod +x "$INSTALL_DIR/flybrowser-serve"

    # Create wrapper for flybrowser-setup (legacy compatibility)
    $SUDO tee "$INSTALL_DIR/flybrowser-setup" > /dev/null << EOF
#!/usr/bin/env bash
source "$VENV_DIR/bin/activate"
exec python -m flybrowser.cli.setup "\$@"
EOF
    $SUDO chmod +x "$INSTALL_DIR/flybrowser-setup"

    # Create wrapper for flybrowser-cluster (legacy compatibility)
    $SUDO tee "$INSTALL_DIR/flybrowser-cluster" > /dev/null << EOF
#!/usr/bin/env bash
source "$VENV_DIR/bin/activate"
exec python -m flybrowser.cli.cluster "\$@"
EOF
    $SUDO chmod +x "$INSTALL_DIR/flybrowser-cluster"

    # Create wrapper for flybrowser-admin (legacy compatibility)
    $SUDO tee "$INSTALL_DIR/flybrowser-admin" > /dev/null << EOF
#!/usr/bin/env bash
source "$VENV_DIR/bin/activate"
exec python -m flybrowser.cli.admin "\$@"
EOF
    $SUDO chmod +x "$INSTALL_DIR/flybrowser-admin"

    print_success "CLI commands created in $INSTALL_DIR"
    print_info "Main command: flybrowser"
}

# Install systemd service (Linux)
install_systemd_service() {
    print_info "Installing systemd service..."

    sudo tee /etc/systemd/system/flybrowser.service > /dev/null << EOF
[Unit]
Description=FlyBrowser Browser Automation Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME
Environment="PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$VENV_DIR/bin/python -m flybrowser.cli.serve
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable flybrowser

    print_success "Systemd service installed"
    print_info "Start with: sudo systemctl start flybrowser"
}

# Install launchd service (macOS)
install_launchd_service() {
    print_info "Installing launchd service..."

    local plist_path="$HOME/Library/LaunchAgents/dev.flybrowser.plist"
    mkdir -p "$HOME/Library/LaunchAgents"

    cat > "$plist_path" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>dev.flybrowser</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV_DIR/bin/python</string>
        <string>-m</string>
        <string>flybrowser.cli.serve</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$HOME/.flybrowser/logs/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>$HOME/.flybrowser/logs/stderr.log</string>
</dict>
</plist>
EOF

    mkdir -p "$HOME/.flybrowser/logs"

    print_success "Launchd service installed"
    print_info "Start with: launchctl load $plist_path"
}

# Install system service
install_service() {
    if [ "$WITH_SERVICE" = false ]; then
        return
    fi

    local os=$(detect_os)

    case "$os" in
        linux)
            install_systemd_service
            ;;
        darwin)
            install_launchd_service
            ;;
        *)
            print_warning "Service installation not supported on $os"
            ;;
    esac
}

# Uninstall FlyBrowser
uninstall_flybrowser() {
    print_banner
    print_warning "Uninstalling FlyBrowser..."
    echo ""
    
    local os=$(detect_os)
    
    # Stop and remove services
    if [ "$os" = "darwin" ]; then
        local plist_path="$HOME/Library/LaunchAgents/dev.flybrowser.plist"
        if [ -f "$plist_path" ]; then
            launchctl unload "$plist_path" 2>/dev/null || true
            rm -f "$plist_path"
            print_success "Removed launchd service"
        fi
    elif [ "$os" = "linux" ]; then
        if [ -f "/etc/systemd/system/flybrowser.service" ]; then
            sudo systemctl stop flybrowser 2>/dev/null || true
            sudo systemctl disable flybrowser 2>/dev/null || true
            sudo rm -f /etc/systemd/system/flybrowser.service
            sudo systemctl daemon-reload
            print_success "Removed systemd service"
        fi
    fi
    
    # Remove CLI wrappers
    local wrappers=("flybrowser" "flybrowser-serve" "flybrowser-setup" "flybrowser-cluster" "flybrowser-admin")
    for wrapper in "${wrappers[@]}"; do
        if [ -f "$INSTALL_DIR/$wrapper" ]; then
            sudo rm -f "$INSTALL_DIR/$wrapper" 2>/dev/null || rm -f "$INSTALL_DIR/$wrapper"
            print_success "Removed $INSTALL_DIR/$wrapper"
        fi
    done
    
    # Remove virtual environment and data
    if [ -d "$DATA_DIR" ]; then
        local reply
        read -r -p "Remove data directory ($DATA_DIR)? This includes config and logs. [y/N] " -n 1 reply
        echo
        if [[ "$reply" =~ ^[Yy]$ ]]; then
            rm -rf "$DATA_DIR"
            print_success "Removed $DATA_DIR"
        else
            print_info "Kept $DATA_DIR"
        fi
    fi
    
    echo ""
    print_success "FlyBrowser uninstalled"
    print_info "To reinstall: curl -fsSL https://get.flybrowser.dev | bash"
}

# Show usage
show_usage() {
    cat << 'EOF'
FlyBrowser Installation Script

Usage: install.sh [OPTIONS]

Installation Options:
  --install-dir DIR     CLI installation directory (default: /usr/local/bin)
  --venv-dir DIR        Virtual environment directory (default: ~/.flybrowser/venv)
  --version VER         Install specific version (default: latest)
  --dev                 Development mode (install from source)
  --use-pip             Force pip instead of uv
  --no-playwright       Skip Playwright browser installation
  --no-wizard           Skip interactive configuration wizard

Service Options:
  --with-service        Install as system service (systemd/launchd)
  --cluster             Configure for cluster mode

Other:
  --uninstall           Uninstall FlyBrowser
  --help                Show this help message

Examples:
  # One-liner installation (recommended)
  curl -fsSL https://get.flybrowser.dev | bash

  # Standard installation from source
  ./install.sh

  # Development installation
  ./install.sh --dev

  # Install specific version
  ./install.sh --version 1.26.0

  # Install with system service
  ./install.sh --with-service

  # Uninstall
  ./install.sh --uninstall

Environment Variables:
  FLYBROWSER_NO_MODIFY_PATH    Don't modify PATH
  FLYBROWSER_ACCEPT_DEFAULTS   Accept all defaults (non-interactive)

Documentation: https://flybrowser.dev/docs
GitHub: https://github.com/firefly-oss/flybrowsers
EOF
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --install-dir)
                INSTALL_DIR="$2"
                shift 2
                ;;
            --venv-dir)
                VENV_DIR="$2"
                shift 2
                ;;
            --version)
                FLYBROWSER_VERSION="$2"
                shift 2
                ;;
            --with-service)
                WITH_SERVICE=true
                shift
                ;;
            --cluster)
                CLUSTER_MODE=true
                shift
                ;;
            --dev)
                DEV_MODE=true
                shift
                ;;
            --no-playwright)
                INSTALL_PLAYWRIGHT=false
                shift
                ;;
            --no-wizard)
                RUN_WIZARD=false
                shift
                ;;
            --use-pip)
                FORCE_PIP=true
                shift
                ;;
            --uninstall)
                UNINSTALL=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo ""
                show_usage
                exit 1
                ;;
        esac
    done
}

# Run flybrowser-setup CLI for browser installation and verification
run_setup_cli() {
    print_step "Running setup CLI..."
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"

    # Build CLI arguments as array for proper handling
    local -a cli_args=("install")

    if [ "$INSTALL_PLAYWRIGHT" = false ]; then
        cli_args+=("--no-browsers")
    fi

    # Run the CLI with proper quoting
    python -m flybrowser.cli.setup "${cli_args[@]}" || {
        print_warning "Setup CLI had issues, continuing..."
    }
}

# Verify installation using CLI
verify_installation() {
    print_step "Verifying installation..."
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"

    if python -m flybrowser.cli.setup verify 2>/dev/null; then
        print_success "Installation verified"
    else
        # Fallback to basic check
        if python -c "import flybrowser; print('FlyBrowser ' + flybrowser.__version__)" 2>/dev/null; then
            print_success "Installation verified (basic)"
        else
            print_error "Installation verification failed"
            exit 1
        fi
    fi
}

# Print completion message
print_completion() {
    echo ""
    echo "============================================================"
    print_msg "$GREEN" "$EMOJI_ROCKET FlyBrowser installation complete!"
    echo "============================================================"
    echo ""
    
    # Get version
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    local version
    version=$(python -c "import flybrowser; print(flybrowser.__version__)" 2>/dev/null) || version="unknown"
    print_info "Version: $version"
    echo ""
    
    print_msg "$BOLD" "Quick Start:"
    echo "  flybrowser                    # Launch interactive REPL"
    echo "  flybrowser setup configure    # Run configuration wizard"
    echo "  flybrowser serve              # Start the API server"
    echo "  flybrowser doctor             # Check installation health"
    echo ""
    
    print_msg "$BOLD" "Legacy Commands (still supported):"
    echo "  flybrowser-setup configure    # Configuration wizard"
    echo "  flybrowser-serve              # Start server"
    echo "  flybrowser-cluster status     # Check cluster"
    echo ""
    
    print_msg "$BOLD" "Resources:"
    echo "  Documentation: https://flybrowser.dev/docs"
    echo "  GitHub:        https://github.com/$GITHUB_REPO"
    echo "  Discord:       https://discord.gg/flybrowser"
    echo ""
}

# Main installation flow
main() {
    parse_args "$@"
    
    # Handle uninstall
    if [ "$UNINSTALL" = true ]; then
        uninstall_flybrowser
        exit 0
    fi

    print_banner
    
    # Detect installation context
    detect_installation_context
    
    # Clone repository if not in source directory
    clone_to_temp

    echo ""
    print_msg "$BOLD" "Installation Settings:"
    echo "  Install directory:    $INSTALL_DIR"
    echo "  Virtual environment:  $VENV_DIR"
    echo "  Version:              $FLYBROWSER_VERSION"
    echo "  Development mode:     $DEV_MODE"
    echo "  Install service:      $WITH_SERVICE"
    echo "  Package manager:      $PACKAGE_MANAGER"
    echo ""

    # Phase 1: Check requirements
    check_requirements
    
    # Offer to install uv if not available
    install_uv_if_requested
    
    # Mark installation as in progress (for graceful interrupt handling)
    INSTALLATION_IN_PROGRESS=true
    
    # Phase 2: Create venv and install
    create_venv
    install_flybrowser
    create_wrappers

    # Phase 3: Setup browsers and config
    run_setup_cli

    # Phase 4: Service installation
    install_service

    # Phase 5: Verify and complete
    verify_installation
    
    # Mark installation as complete
    INSTALLATION_IN_PROGRESS=false
    
    print_completion
}

# Run main
main "$@"

