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
Browser Automation Scope Validator.

Validates that tasks and operations are within the valid scope of browser automation,
rejecting requests that attempt to access filesystem, execute system commands, or
perform other operations outside the browser context.
"""

from __future__ import annotations

import re
from typing import Optional, List, Tuple
from urllib.parse import urlparse


class BrowserScopeValidator:
    """
    Validates that tasks are appropriate for browser automation.
    
    This validator ensures that:
    - Tasks are related to web browser operations
    - No filesystem or system operations are requested
    - No network operations outside browser context
    - URLs are safe and valid
    
    Example:
        >>> validator = BrowserScopeValidator()
        >>> is_valid, error = validator.validate_task("Navigate to example.com and extract data")
        >>> assert is_valid
        >>> is_valid, error = validator.validate_task("Delete all files in /tmp")
        >>> assert not is_valid
    """
    
    # Operations that are explicitly prohibited
    PROHIBITED_KEYWORDS = [
        # Filesystem operations
        "delete file", "remove file", "save to disk", "write to file",
        "read file", "open file", "file system", "create file",
        "modify file", "rename file", "move file", "copy file",
        "/etc/", "/tmp/", "/var/", "~/.ssh", "c:\\", "d:\\",
        
        # System operations
        "execute command", "run command", "system command", "shell command",
        "run script", "execute script", "bash", "powershell", "cmd.exe",
        "kill process", "start process", "sudo", "chmod", "chown",
        
        # Network operations (outside browser)
        "send email", "smtp", "send message", "make api call",
        "http request", "curl", "wget", "download and save",
        "ftp upload", "ssh connection", "telnet",
        
        # Database operations
        "query database", "sql query", "insert into", "delete from",
        "update database", "create table", "drop table", "mysql",
        "postgres", "mongodb", "redis",
        
        # Security-sensitive operations
        "access password", "steal data", "inject code", "xss attack",
        "sql injection", "exploit", "hack", "breach",
    ]
    
    # Keywords that indicate valid browser automation
    BROWSER_KEYWORDS = [
        # Navigation
        "navigate", "visit", "go to", "open", "load page", "browse",
        
        # Interaction
        "click", "type", "fill", "submit", "select", "check", "uncheck",
        "hover", "scroll", "drag", "drop", "press", "focus",
        
        # Extraction
        "extract", "scrape", "get data", "find", "search", "locate",
        "read text", "get content", "parse", "collect",
        
        # Page elements
        "button", "link", "form", "input", "field", "element",
        "menu", "dropdown", "checkbox", "radio", "textarea",
        
        # Web concepts
        "website", "webpage", "url", "browser", "page", "site",
        "html", "dom", "css selector", "xpath",
        
        # Common actions
        "login", "signup", "register", "search for", "add to cart",
        "checkout", "download", "upload", "view", "explore",
    ]
    
    # URL schemes that are allowed
    ALLOWED_URL_SCHEMES = ["http", "https"]
    
    # URL schemes that are prohibited
    PROHIBITED_URL_SCHEMES = [
        "file", "ftp", "ftps", "javascript", "data", "vbscript",
        "about", "blob", "filesystem",
    ]
    
    def validate_task(
        self,
        task: str,
        skip_browser_keyword_check: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that a task is appropriate for browser automation.
        
        Args:
            task: The task description to validate
            skip_browser_keyword_check: When True, skip the browser keyword validation.
                This should be True when called from SDK methods like extract(), act(),
                agent() etc. since these methods already define the browser operation type.
                Prohibited keyword checks are ALWAYS performed for security.
            
        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
            
        Example:
            >>> validator = BrowserScopeValidator()
            >>> is_valid, error = validator.validate_task("Click the login button")
            >>> print(is_valid)
            True
            >>> # SDK method call - skip browser keyword check
            >>> is_valid, error = validator.validate_task(
            ...     "Get the titles and scores",
            ...     skip_browser_keyword_check=True
            ... )
            >>> print(is_valid)
            True
        """
        if not task or not task.strip():
            return False, "Task cannot be empty"
        
        task_lower = task.lower()
        
        # Check for prohibited keywords
        for keyword in self.PROHIBITED_KEYWORDS:
            if keyword in task_lower:
                return False, (
                    f"Task contains prohibited operation: '{keyword}'. "
                    f"FlyBrowser only supports browser automation. "
                    f"Cannot perform filesystem, system, or network operations outside the browser."
                )
        
        # Check for URLs and validate them
        urls = re.findall(r'https?://[^\s\)]+', task, re.IGNORECASE)
        for url in urls:
            is_valid, error = self.validate_url(url)
            if not is_valid:
                return False, error
        
        # If called from SDK methods, skip the browser keyword check
        # The SDK method (extract, act, agent, etc.) already defines the operation type
        # We only need to check for prohibited/dangerous operations
        if skip_browser_keyword_check:
            return True, None
        
        # Check if task seems browser-related
        has_browser_keyword = any(keyword in task_lower for keyword in self.BROWSER_KEYWORDS)
        has_url = bool(urls)
        
        # If neither browser keywords nor URLs, likely not a browser task
        if not has_browser_keyword and not has_url:
            # Special case: very short tasks might be unclear
            if len(task.split()) <= 3:
                return False, (
                    "Task is too vague. Please specify a browser action "
                    "(e.g., 'click', 'navigate', 'extract') or include a website URL."
                )
            
            return False, (
                "Task doesn't appear to be browser automation. "
                "Please specify:\n"
                "- A website URL to visit (e.g., https://example.com)\n"
                "- A browser action (e.g., click, type, extract, navigate)\n"
                "- A web page element (e.g., button, link, form)"
            )
        
        return True, None
    
    def validate_url(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that a URL is safe for browser automation.
        
        Args:
            url: The URL to validate
            
        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
            
        Example:
            >>> validator = BrowserScopeValidator()
            >>> is_valid, error = validator.validate_url("https://example.com")
            >>> print(is_valid)
            True
            >>> is_valid, error = validator.validate_url("file:///etc/passwd")
            >>> print(is_valid)
            False
        """
        if not url or not url.strip():
            return False, "URL cannot be empty"
        
        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            return False, f"Invalid URL format: {e}"
        
        scheme = parsed.scheme.lower()
        
        # Check for prohibited schemes
        if scheme in self.PROHIBITED_URL_SCHEMES:
            return False, (
                f"URL scheme '{scheme}://' is not allowed. "
                f"Only HTTP and HTTPS URLs are supported for browser automation."
            )
        
        # Check for allowed schemes
        if scheme and scheme not in self.ALLOWED_URL_SCHEMES:
            return False, (
                f"URL scheme '{scheme}://' is not supported. "
                f"Only http:// and https:// URLs are allowed."
            )
        
        # Check for localhost/private IPs if you want to restrict
        # (Optional - comment out if you want to allow localhost)
        hostname = parsed.hostname
        if hostname:
            hostname_lower = hostname.lower()
            # Check for obvious malicious patterns
            if any(dangerous in hostname_lower for dangerous in [
                "javascript:", "data:", "vbscript:",
            ]):
                return False, f"URL contains dangerous pattern: {hostname}"
        
        return True, None
    
    def validate_tool_operation(
        self,
        tool_name: str,
        parameters: dict,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that a tool operation is safe and within browser scope.
        
        Args:
            tool_name: Name of the tool being executed
            parameters: Parameters for the tool
            
        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
        """
        # Prohibited tool names (if any tools exist that shouldn't)
        prohibited_tools = [
            "execute_js_arbitrary",
            "run_system_command",
            "save_file",
            "delete_file",
            "send_email",
            "make_api_call",
        ]
        
        if tool_name in prohibited_tools:
            return False, (
                f"Tool '{tool_name}' is prohibited for security reasons. "
                f"Only browser automation tools are available."
            )
        
        # Validate URL parameter if present
        if "url" in parameters:
            url = parameters["url"]
            if isinstance(url, str):
                is_valid, error = self.validate_url(url)
                if not is_valid:
                    return False, f"Invalid URL in tool parameters: {error}"
        
        # Validate selector parameters don't contain dangerous patterns
        if "selector" in parameters:
            selector = str(parameters["selector"])
            # Check for script injection attempts
            if any(dangerous in selector.lower() for dangerous in [
                "javascript:", "<script", "onerror=", "onclick=",
            ]):
                return False, (
                    f"Selector contains potentially dangerous pattern. "
                    f"Please use standard CSS selectors or XPath."
                )
        
        return True, None
    
    def get_validation_hints(self, task: str) -> List[str]:
        """
        Get helpful hints for fixing an invalid task.
        
        Args:
            task: The task that failed validation
            
        Returns:
            List of suggestions to fix the task
        """
        hints = []
        task_lower = task.lower()
        
        # Suggest URL if missing
        if not re.search(r'https?://', task):
            hints.append(
                "Consider adding a website URL (e.g., https://example.com) "
                "to specify where the automation should happen."
            )
        
        # Suggest browser actions if missing
        browser_action_words = ["click", "type", "navigate", "extract", "scrape"]
        if not any(action in task_lower for action in browser_action_words):
            hints.append(
                "Add a browser action verb like 'navigate to', 'click on', "
                "'type in', or 'extract from' to clarify what to do."
            )
        
        # Check for common mistakes
        if "download" in task_lower and "save" in task_lower:
            hints.append(
                "Note: FlyBrowser can navigate to download links, but cannot "
                "save files to your local disk. Consider 'navigate to download link' instead."
            )
        
        if any(word in task_lower for word in ["file", "folder", "directory"]):
            hints.append(
                "FlyBrowser operates within web browsers only. "
                "It cannot access your local filesystem."
            )
        
        if "email" in task_lower:
            hints.append(
                "FlyBrowser can interact with webmail interfaces (Gmail, Outlook, etc.) "
                "but cannot send emails directly via SMTP."
            )
        
        return hints


# Singleton instance for easy access
_validator_instance: Optional[BrowserScopeValidator] = None


def get_scope_validator() -> BrowserScopeValidator:
    """Get the global scope validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = BrowserScopeValidator()
    return _validator_instance
