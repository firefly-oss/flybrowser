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

"""Logging configuration for FlyBrowser."""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class LogFormat(str, Enum):
    """Supported log output formats."""
    
    JSON = "json"              # Structured JSON format (default)
    HUMAN = "human"            # Human-readable colored format
    TEXT = "text"              # Plain text format (legacy)


class JsonFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.
    
    Outputs logs as single-line JSON objects with consistent fields.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add source location
        if record.pathname and record.lineno:
            log_data["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "message", "taskName"
            }
        }
        if extra_fields:
            log_data["extra"] = extra_fields
        
        return json.dumps(log_data, default=str)


class HumanFormatter(logging.Formatter):
    """
    Human-readable log formatter with colors and clean layout.
    
    Uses ANSI colors for different log levels and provides a clean,
    easy-to-read format for terminal output.
    """
    
    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    def __init__(self, use_colors: bool = True):
        """Initialize formatter with optional color support."""
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human readability."""
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Level with color
        level = record.levelname
        if self.use_colors:
            color = self.COLORS.get(level, "")
            level_str = f"{color}{self.BOLD}[{level:>8}]{self.RESET}"
            time_str = f"{self.DIM}{timestamp}{self.RESET}"
        else:
            level_str = f"[{level:>8}]"
            time_str = timestamp
        
        # Message
        message = record.getMessage()
        
        # Format output
        output = f"{time_str} {level_str} {message}"
        
        # Add exception if present
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            if self.use_colors:
                exc_text = f"{self.COLORS['ERROR']}{exc_text}{self.RESET}"
            output += f"\n{exc_text}"
        
        return output


class TextFormatter(logging.Formatter):
    """Plain text log formatter (legacy format)."""
    
    def __init__(self):
        """Initialize with standard format."""
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )


def get_log_level(level_str: str) -> int:
    """
    Convert log level string to logging constant.
    
    Args:
        level_str: Log level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Logging level constant
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(level_str.upper(), logging.INFO)


def get_formatter(log_format: LogFormat, use_colors: bool = True) -> logging.Formatter:
    """
    Get the appropriate formatter for the specified format.
    
    Args:
        log_format: Log format type
        use_colors: Whether to use colors (for human format)
    
    Returns:
        Configured formatter instance
    """
    if log_format == LogFormat.JSON:
        return JsonFormatter()
    elif log_format == LogFormat.HUMAN:
        return HumanFormatter(use_colors=use_colors)
    else:
        return TextFormatter()


def configure_logging(
    level: str = "INFO",
    log_format: LogFormat = LogFormat.JSON,
    human_readable: bool = False,
) -> None:
    """
    Configure global logging settings for FlyBrowser.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format type (JSON, HUMAN, TEXT)
        human_readable: If True, forces human-readable format regardless of log_format
    
    Note:
        This function reconfigures the global logger. Call it early in your
        application startup to set logging preferences.
    """
    global logger
    
    # Determine format
    if human_readable:
        log_format = LogFormat.HUMAN
    
    # Get log level
    log_level = get_log_level(level)
    
    # Reconfigure logger
    logger.setLevel(log_level)
    logger.handlers.clear()
    
    # Create handler with appropriate formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(get_formatter(log_format))
    
    logger.addHandler(handler)


def setup_logger(
    name: str = "flybrowser",
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup and configure a logger for FlyBrowser.

    Args:
        name: Logger name
        level: Logging level
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance
    """
    log = logging.getLogger(name)
    log.setLevel(level)

    # Remove existing handlers
    log.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Determine formatter based on environment or default to JSON
    env_format = os.environ.get("FLYBROWSER_LOG_FORMAT", "json").lower()
    env_level = os.environ.get("FLYBROWSER_LOG_LEVEL", "")
    
    if env_level:
        level = get_log_level(env_level)
        log.setLevel(level)
        handler.setLevel(level)
    
    if format_string is not None:
        # Use custom format string if provided
        formatter = logging.Formatter(format_string)
    elif env_format == "human":
        formatter = HumanFormatter()
    elif env_format == "text":
        formatter = TextFormatter()
    else:
        # Default to JSON
        formatter = JsonFormatter()
    
    handler.setFormatter(formatter)
    log.addHandler(handler)

    return log


# Default logger instance
logger = setup_logger()

