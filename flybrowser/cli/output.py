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

"""CLI output formatting utilities for consistent command output."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class CLIOutput:
    """
    Unified CLI output formatting for FlyBrowser commands.
    
    Provides consistent output formatting across all CLI commands including:
    - Banner display from banner.txt
    - Execution summaries
    - Log section separators
    - Section headers
    
    All commands should follow the pattern:
    1. Print banner
    2. Print execution summary
    3. Print logs header
    4. Actual logs/output
    """
    
    # ANSI color codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    
    def __init__(self, use_colors: bool = True):
        """
        Initialize CLI output formatter.
        
        Args:
            use_colors: Whether to use ANSI colors (auto-detected if stdout is a TTY)
        """
        self.use_colors = use_colors and sys.stdout.isatty()
        self._banner_cache: Optional[str] = None
    
    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if self.use_colors:
            return f"{color}{text}{self.RESET}"
        return text
    
    def _bold(self, text: str) -> str:
        """Make text bold if colors are enabled."""
        if self.use_colors:
            return f"{self.BOLD}{text}{self.RESET}"
        return text
    
    def _dim(self, text: str) -> str:
        """Make text dim if colors are enabled."""
        if self.use_colors:
            return f"{self.DIM}{text}{self.RESET}"
        return text
    
    def get_banner(self) -> str:
        """
        Load the banner from banner.txt.
        
        Returns:
            Banner ASCII art string
        """
        if self._banner_cache is not None:
            return self._banner_cache
        
        # Try to find banner.txt relative to this file
        banner_paths = [
            Path(__file__).parent.parent / "banner.txt",
            Path(__file__).parent.parent.parent / "flybrowser" / "banner.txt",
        ]
        
        for banner_path in banner_paths:
            if banner_path.exists():
                self._banner_cache = banner_path.read_text().rstrip()
                return self._banner_cache
        
        # Fallback banner
        self._banner_cache = r"""  _____.__         ___.
_/ ____\  | ___.__.\\_ |_________  ______  _  ________ ___________
\   __\|  |<   |  | | __ \_  __ \/  _ \ \/ \/ /  ___// __ \_  __ \
 |  |  |  |_\___  | | \_\ \  | \(  <_> )     /\___ \\  ___/|  | \/
 |__|  |____/ ____| |___  /__|   \____/ \/\_//____  >\___  >__|
            \/          \/                        \/     \/"""
        return self._banner_cache
    
    def print_banner(self, version: Optional[str] = None, tagline: Optional[str] = None) -> None:
        """
        Print the FlyBrowser banner.
        
        Args:
            version: Optional version string to display
            tagline: Optional tagline to display (default: "Browser Automation Powered by LLM Agents")
        """
        print()
        print(self._color(self.get_banner(), self.CYAN))
        print()
        if tagline is None:
            tagline = "Browser Automation Powered by LLM Agents"
        print(f"  {self._dim(tagline)}")
        if version:
            print(f"  {self._dim(f'Version: {version}')}")
        print()
    
    def print_summary(
        self,
        title: str,
        items: Dict[str, Any],
        show_divider: bool = True,
    ) -> None:
        """
        Print an execution summary with key-value pairs.
        
        Args:
            title: Summary title (e.g., "Execution Summary", "Configuration")
            items: Dictionary of key-value pairs to display
            show_divider: Whether to show a divider line after the summary
        """
        print(self._bold(f"▶ {title}"))
        print()
        
        # Calculate padding for alignment
        max_key_len = max(len(str(k)) for k in items.keys()) if items else 0
        
        for key, value in items.items():
            key_str = str(key).ljust(max_key_len)
            value_str = str(value) if value is not None else "-"
            print(f"  {self._dim(key_str)}  {value_str}")
        
        print()
        if show_divider:
            self.print_divider()
    
    def print_section(self, title: str) -> None:
        """
        Print a section header.
        
        Args:
            title: Section title
        """
        print()
        print(self._bold(f"--- {title} ---"))
        print()
    
    def print_logs_header(self) -> None:
        """Print the logs section header with arrow separator."""
        width = 50
        arrow = "↓"
        label = " LOGS START HERE "
        padding = (width - len(label)) // 2
        
        line = f"{arrow * padding}{label}{arrow * padding}"
        
        print()
        print(self._dim(line))
        print()
    
    def print_divider(self, char: str = "─", width: int = 50) -> None:
        """
        Print a horizontal divider.
        
        Args:
            char: Character to use for the divider
            width: Width of the divider
        """
        print(self._dim(char * width))
    
    def print_status_line(
        self,
        status: str,
        message: str,
        ok: bool = True,
        info: bool = False,
        warn: bool = False,
    ) -> None:
        """
        Print a status line with icon.
        
        Args:
            status: Status icon text (e.g., "OK", "FAIL", "INFO")
            message: Status message
            ok: Whether this is a success status (green)
            info: Whether this is an info status (blue)
            warn: Whether this is a warning status (yellow)
        """
        if ok:
            icon = self._color(f"[{status}]", self.GREEN)
        elif info:
            icon = self._color(f"[{status}]", self.BLUE)
        elif warn:
            icon = self._color(f"[{status}]", self.YELLOW)
        else:
            icon = f"[{status}]"
        
        print(f"{icon} {message}")
    
    def print_list(self, title: str, items: List[str], bullet: str = "•") -> None:
        """
        Print a bulleted list.
        
        Args:
            title: List title
            items: List items
            bullet: Bullet character
        """
        if title:
            print(self._bold(title))
        for item in items:
            print(f"  {bullet} {item}")


# Global instance for convenience
output = CLIOutput()
