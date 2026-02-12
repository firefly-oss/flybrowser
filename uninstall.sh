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
# FlyBrowser Uninstall Script
# ============================================================================
#
# Removes FlyBrowser and all its components from the system.
#
# Usage:
#   ./uninstall.sh                    # Interactive uninstall
#   ./uninstall.sh --all              # Remove everything (including data)
#   ./uninstall.sh --keep-data        # Keep configuration and data
#   ./uninstall.sh --help             # Show help
#
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

# Emoji support detection
if [[ "${TERM:-}" == *"256color"* ]] || [[ "${COLORTERM:-}" == "truecolor" ]]; then
    EMOJI_CHECK="✓"
    EMOJI_CROSS="✗"
    EMOJI_WARN="⚠"
    EMOJI_INFO="ℹ"
    EMOJI_TRASH="🗑"
else
    EMOJI_CHECK="[OK]"
    EMOJI_CROSS="[FAIL]"
    EMOJI_WARN="[WARN]"
    EMOJI_INFO="[INFO]"
    EMOJI_TRASH="[DEL]"
fi

# Default paths
INSTALL_DIR="/usr/local/bin"
VENV_DIR="$HOME/.flybrowser/venv"
DATA_DIR="$HOME/.flybrowser"
CONFIG_DIR="$HOME/.config/flybrowser"
USER_SITE_DIR="$HOME/.local/lib"
JUPYTER_KERNEL_DIR="$HOME/.local/share/jupyter/kernels/flybrowser"

# Installation mode detection
INSTALL_MODE="unknown"  # venv, system, user, or unknown

# Options
REMOVE_DATA=false
KEEP_DATA=false
FORCE=false
QUIET=false

# Print functions
print_msg() {
    local color=$1
    local msg=$2
    if [ "$QUIET" = false ]; then
        echo -e "${color}${msg}${NC}"
    fi
}

print_info() { print_msg "$BLUE" "$EMOJI_INFO $1"; }
print_success() { print_msg "$GREEN" "$EMOJI_CHECK $1"; }
print_warning() { print_msg "$YELLOW" "$EMOJI_WARN $1"; }
print_error() { print_msg "$RED" "$EMOJI_CROSS $1"; }
print_step() { print_msg "$CYAN" "$EMOJI_TRASH $1"; }

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Linux*)     echo "linux";;
        Darwin*)    echo "darwin";;
        CYGWIN*|MINGW*|MSYS*) echo "windows";;
        *)          echo "unknown";;
    esac
}

# Detect installation mode
detect_install_mode() {
    local found_any=false
    local detected_modes=()
    
    # Check for venv installation
    if [ -d "$VENV_DIR" ]; then
        detected_modes+=("venv")
        found_any=true
    fi
    
    # Check user site-packages (most common for pip install --user)
    if python3 -m pip list --user 2>/dev/null | grep -q "^flybrowser"; then
        detected_modes+=("user")
        found_any=true
    elif [ -n "$(find "$HOME/.local/lib" -type d -name "flybrowser" 2>/dev/null | head -n 1)" ]; then
        detected_modes+=("user")
        found_any=true
    fi
    
    # Check system site-packages
    if python3 -m pip list 2>/dev/null | grep -q "^flybrowser" && \
       python3 -c "import flybrowser, os; install_loc = os.path.dirname(flybrowser.__file__); exit(0 if install_loc.startswith(('/usr/', '/opt/', '/Library/')) else 1)" 2>/dev/null; then
        detected_modes+=("system")
        found_any=true
    fi
    
    # Check for CLI binaries in /usr/local/bin or ~/.local/bin
    if [ -f "$INSTALL_DIR/flybrowser" ] || [ -f "$HOME/.local/bin/flybrowser" ]; then
        found_any=true
    fi
    
    # Set installation mode based on findings
    if [ "$found_any" = false ]; then
        INSTALL_MODE="none"
    elif [ ${#detected_modes[@]} -eq 1 ]; then
        INSTALL_MODE="${detected_modes[0]}"
    elif [ ${#detected_modes[@]} -gt 1 ]; then
        # Multiple installations detected - use "multiple" mode
        INSTALL_MODE="multiple"
    else
        INSTALL_MODE="unknown"
    fi
}

# Print banner
print_banner() {
    if [ "$QUIET" = false ]; then
        echo ""
        print_msg "$BLUE" '  _____.__         ___.'
        print_msg "$BLUE" '_/ ____\  | ___.__.\\_ |_________  ______  _  ________ ___________'
        print_msg "$BLUE" '\\   __\\|  |<   |  | | __ \\_  __ \\/  _ \\ \\/ \\/ /  ___// __ \\_  __ \\'
        print_msg "$BLUE" ' |  |  |  |_\\___  | | \\_\\ \\  | \\(  <_> )     /\\___ \\\\  ___/|  | \\/'
        print_msg "$BLUE" ' |__|  |____/ ____| |___  /__|   \\____/ \\/\\_//____  >\\___  >__|'
        print_msg "$BLUE" '            \\/          \\/                        \\/     \\/'
        echo ""
        print_msg "$RED" " Uninstaller"
        echo ""
    fi
}

# Show usage
show_usage() {
    cat << 'EOF'
FlyBrowser Uninstall Script

Usage: uninstall.sh [OPTIONS]

Options:
  --all             Remove everything including data and configuration
  --keep-data       Keep configuration and data (remove only binaries)
  --force, -f       Don't ask for confirmation
  --quiet, -q       Suppress output
  --help, -h        Show this help message

Examples:
  # Interactive uninstall (asks what to remove)
  ./uninstall.sh

  # Remove everything without asking
  ./uninstall.sh --all --force

  # Remove only binaries, keep config
  ./uninstall.sh --keep-data

Paths that may be removed (depending on installation mode):
  Virtual Env Mode:
    - CLI Commands:     /usr/local/bin/flybrowser*
    - Virtual Env:      ~/.flybrowser/venv
  
  System-wide Mode:
    - System Python packages
  
  User Mode:
    - User packages:    ~/.local/lib/python*/site-packages/flybrowser
    - CLI Commands:     ~/.local/bin/flybrowser*
  
  Common:
    - Data Directory:   ~/.flybrowser
    - Config Directory: ~/.config/flybrowser
    - Jupyter Kernel:   ~/.local/share/jupyter/kernels/flybrowser
    - macOS Service:    ~/Library/LaunchAgents/dev.flybrowser.plist
    - Linux Service:    /etc/systemd/system/flybrowser.service
EOF
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                REMOVE_DATA=true
                shift
                ;;
            --keep-data)
                KEEP_DATA=true
                shift
                ;;
            --force|-f)
                FORCE=true
                shift
                ;;
            --quiet|-q)
                QUIET=true
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

# Ask for confirmation
confirm() {
    local message=$1
    local default=${2:-n}
    
    if [ "$FORCE" = true ]; then
        return 0
    fi
    
    local prompt
    if [ "$default" = "y" ]; then
        prompt="[Y/n]"
    else
        prompt="[y/N]"
    fi
    
    read -p "$message $prompt " -n 1 -r reply
    echo
    
    if [ -z "$reply" ]; then
        reply=$default
    fi
    
    [[ "$reply" =~ ^[Yy]$ ]]
}

# Stop and remove services
remove_services() {
    local os=$(detect_os)
    local removed=false
    
    print_step "Checking for services..."
    
    # macOS launchd
    if [ "$os" = "darwin" ]; then
        local plist_path="$HOME/Library/LaunchAgents/dev.flybrowser.plist"
        if [ -f "$plist_path" ]; then
            print_info "Found launchd service"
            launchctl unload "$plist_path" 2>/dev/null || true
            rm -f "$plist_path"
            print_success "Removed launchd service"
            removed=true
        fi
    fi
    
    # Linux systemd
    if [ "$os" = "linux" ]; then
        if [ -f "/etc/systemd/system/flybrowser.service" ]; then
            print_info "Found systemd service"
            sudo systemctl stop flybrowser 2>/dev/null || true
            sudo systemctl disable flybrowser 2>/dev/null || true
            sudo rm -f /etc/systemd/system/flybrowser.service
            sudo systemctl daemon-reload
            print_success "Removed systemd service"
            removed=true
        fi
    fi
    
    if [ "$removed" = false ]; then
        print_info "No services found"
    fi
}

# Remove CLI wrappers
remove_cli_wrappers() {
    print_step "Removing CLI commands..."
    
    local wrappers=(
        "flybrowser"
        "flybrowser-serve"
        "flybrowser-setup"
        "flybrowser-cluster"
        "flybrowser-admin"
        "flybrowser-uninstall"
    )
    
    local removed=0
    
    for wrapper in "${wrappers[@]}"; do
        local path="$INSTALL_DIR/$wrapper"
        if [ -f "$path" ]; then
            if [ -w "$INSTALL_DIR" ]; then
                rm -f "$path"
            else
                sudo rm -f "$path"
            fi
            print_success "Removed $path"
            ((removed++))
        fi
    done
    
    if [ $removed -eq 0 ]; then
        print_info "No CLI commands found in $INSTALL_DIR"
    else
        print_success "Removed $removed CLI command(s)"
    fi
}

# Remove virtual environment
remove_venv() {
    if [ "$INSTALL_MODE" != "venv" ]; then
        return
    fi
    
    print_step "Removing virtual environment..."
    
    if [ -d "$VENV_DIR" ]; then
        rm -rf "$VENV_DIR"
        print_success "Removed $VENV_DIR"
    else
        print_info "Virtual environment not found at $VENV_DIR"
    fi
}

# Remove system-wide installation
remove_system_install() {
    if [ "$INSTALL_MODE" != "system" ]; then
        return
    fi
    
    print_step "Removing system-wide installation..."
    
    if python3 -c "import flybrowser" 2>/dev/null; then
        print_info "Uninstalling flybrowser package (system-wide)..."
        python3 -m pip uninstall -y flybrowser 2>/dev/null || {
            print_warning "Could not uninstall with pip, trying with PEP 668 flag"
            python3 -m pip uninstall -y --break-system-packages flybrowser 2>/dev/null || true
        }
        print_success "Removed system-wide installation"
    else
        print_info "No system-wide installation found"
    fi
}

# Remove user installation
remove_user_install() {
    if [ "$INSTALL_MODE" != "user" ]; then
        return
    fi
    
    print_step "Removing user installation..."
    
    if python3 -c "import flybrowser" 2>/dev/null; then
        print_info "Uninstalling flybrowser package (user)..."
        python3 -m pip uninstall -y flybrowser 2>/dev/null || true
        print_success "Removed user installation"
    else
        print_info "No user installation found"
    fi
    
    # Remove .local/bin scripts if they exist
    local user_bin="$HOME/.local/bin"
    if [ -d "$user_bin" ]; then
        local removed=0
        for cmd in flybrowser flybrowser-setup flybrowser-serve flybrowser-cluster flybrowser-admin; do
            if [ -f "$user_bin/$cmd" ]; then
                rm -f "$user_bin/$cmd"
                ((removed++))
            fi
        done
        if [ $removed -gt 0 ]; then
            print_success "Removed $removed command(s) from $user_bin"
        fi
    fi
}

# Remove Jupyter kernel
remove_jupyter_kernel() {
    print_step "Checking for Jupyter kernel..."
    
    local removed=false
    
    # Try using jupyter kernelspec if available
    if command -v jupyter &> /dev/null; then
        if jupyter kernelspec list 2>/dev/null | grep -q "flybrowser"; then
            print_info "Found Jupyter kernel via kernelspec"
            jupyter kernelspec uninstall -f flybrowser 2>/dev/null || {
                print_warning "Could not uninstall via jupyter kernelspec, removing manually"
                rm -rf "$JUPYTER_KERNEL_DIR" 2>/dev/null || true
            }
            print_success "Removed Jupyter kernel"
            removed=true
        fi
    fi
    
    # Also check and remove the directory directly
    if [ -d "$JUPYTER_KERNEL_DIR" ]; then
        rm -rf "$JUPYTER_KERNEL_DIR"
        if [ "$removed" = false ]; then
            print_success "Removed Jupyter kernel directory"
            removed=true
        fi
    fi
    
    # Check for any other jupyter kernel directories in alternative locations
    local alt_kernel_dirs=(
        "$HOME/.jupyter/kernels/flybrowser"
        "$HOME/Library/Jupyter/kernels/flybrowser"
    )
    
    for kernel_dir in "${alt_kernel_dirs[@]}"; do
        if [ -d "$kernel_dir" ]; then
            rm -rf "$kernel_dir"
            print_success "Removed kernel from $kernel_dir"
            removed=true
        fi
    done
    
    if [ "$removed" = false ]; then
        print_info "No Jupyter kernel found"
    fi
}

# Remove data and configuration
remove_data() {
    print_step "Removing data and configuration..."
    
    local removed=false
    
    # Remove data directory
    if [ -d "$DATA_DIR" ]; then
        rm -rf "$DATA_DIR"
        print_success "Removed $DATA_DIR"
        removed=true
    fi
    
    # Remove config directory
    if [ -d "$CONFIG_DIR" ]; then
        rm -rf "$CONFIG_DIR"
        print_success "Removed $CONFIG_DIR"
        removed=true
    fi
    
    # Remove .env in current directory if it's a flybrowser config
    if [ -f ".env" ] && grep -q "FLYBROWSER" ".env" 2>/dev/null; then
        if confirm "Remove .env file in current directory?"; then
            rm -f ".env"
            print_success "Removed .env"
            removed=true
        fi
    fi
    
    if [ "$removed" = false ]; then
        print_info "No data directories found"
    fi
}

# Print summary
print_summary() {
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    print_msg "$GREEN" "$EMOJI_CHECK FlyBrowser has been uninstalled"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    if [ "$KEEP_DATA" = true ] || [ "$REMOVE_DATA" = false ]; then
        print_info "Configuration and data were preserved"
        print_info "To reinstall: curl -fsSL https://get.flybrowser.dev | bash"
    else
        print_info "All FlyBrowser files have been removed"
    fi
    echo ""
    
    print_info "To reinstall FlyBrowser:"
    echo "  curl -fsSL https://get.flybrowser.dev | bash"
    echo "  # or"
    echo "  git clone https://github.com/fireflyresearch/flybrowser && cd flybrowser && ./install.sh"
    echo ""
}

# Main uninstall flow
main() {
    parse_args "$@"
    
    print_banner
    
    # Detect installation mode
    detect_install_mode
    
    # Check if FlyBrowser is installed
    if [ "$INSTALL_MODE" = "none" ]; then
        print_warning "FlyBrowser does not appear to be installed"
        print_info "Checked: Virtual environment ($VENV_DIR)"
        print_info "Checked: System Python (python3 -m pip list)"
        print_info "Checked: User Python (~/.local)"
        exit 0
    fi
    
    print_info "Detected installation mode: $INSTALL_MODE"
    
    # Show what will be removed
    echo ""
    print_msg "$BOLD" "The following will be removed:"
    echo ""
    
    case "$INSTALL_MODE" in
        venv)
            if [ -f "$INSTALL_DIR/flybrowser" ]; then
                echo "  • CLI commands in $INSTALL_DIR"
            fi
            if [ -d "$VENV_DIR" ]; then
                echo "  • Virtual environment: $VENV_DIR"
            fi
            ;;
        system)
            echo "  • System-wide FlyBrowser package"
            ;;
        user)
            echo "  • User-installed FlyBrowser package (~/.local)"
            echo "  • CLI commands in ~/.local/bin"
            ;;
        multiple)
            echo "  • Multiple installations detected:"
            if [ -d "$VENV_DIR" ]; then
                echo "    - Virtual environment: $VENV_DIR"
            fi
            if python3 -m pip list --user 2>/dev/null | grep -q "^flybrowser"; then
                echo "    - User package (~/.local)"
            fi
            if python3 -m pip list 2>/dev/null | grep -q "^flybrowser"; then
                echo "    - System package"
            fi
            if [ -f "$INSTALL_DIR/flybrowser" ]; then
                echo "    - CLI commands in $INSTALL_DIR"
            fi
            if [ -f "$HOME/.local/bin/flybrowser" ]; then
                echo "    - CLI commands in ~/.local/bin"
            fi
            ;;
        *)
            echo "  • All detected FlyBrowser installations"
            ;;
    esac
    
    if [ -d "$JUPYTER_KERNEL_DIR" ]; then
        echo "  • Jupyter kernel"
    fi
    
    local os=$(detect_os)
    if [ "$os" = "darwin" ] && [ -f "$HOME/Library/LaunchAgents/dev.flybrowser.plist" ]; then
        echo "  • macOS launchd service"
    fi
    if [ "$os" = "linux" ] && [ -f "/etc/systemd/system/flybrowser.service" ]; then
        echo "  • Linux systemd service"
    fi
    
    if [ "$KEEP_DATA" = false ]; then
        if [ "$REMOVE_DATA" = true ]; then
            if [ -d "$DATA_DIR" ]; then
                echo "  • Data directory: $DATA_DIR"
            fi
            if [ -d "$CONFIG_DIR" ]; then
                echo "  • Config directory: $CONFIG_DIR"
            fi
        else
            echo ""
            print_msg "$DIM" "  (Data and configuration will be preserved unless --all is used)"
        fi
    fi
    
    echo ""
    
    # Confirm uninstall
    if ! confirm "Proceed with uninstall?"; then
        print_info "Uninstall cancelled"
        exit 0
    fi
    
    echo ""
    
    # Perform uninstall based on mode
    remove_services
    
    case "$INSTALL_MODE" in
        venv)
            remove_cli_wrappers
            remove_venv
            ;;
        system)
            remove_system_install
            ;;
        user)
            remove_user_install
            ;;
        multiple)
            print_info "Removing all detected installations..."
            remove_cli_wrappers
            remove_venv
            remove_system_install
            remove_user_install
            ;;
        *)
            print_warning "Unknown installation mode, attempting all removal methods"
            remove_cli_wrappers
            remove_venv
            remove_system_install
            remove_user_install
            ;;
    esac
    
    # Remove Jupyter kernel if installed
    remove_jupyter_kernel
    
    # Handle data removal
    if [ "$KEEP_DATA" = false ]; then
        if [ "$REMOVE_DATA" = true ]; then
            remove_data
        elif [ "$FORCE" = false ]; then
            echo ""
            if confirm "Also remove data and configuration (~/.flybrowser)?"; then
                remove_data
            else
                print_info "Keeping data and configuration"
            fi
        fi
    fi
    
    print_summary
}

# Run main
main "$@"
