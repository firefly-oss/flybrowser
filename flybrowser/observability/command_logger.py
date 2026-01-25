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
Structured Command Logging for FlyBrowser.

This module provides comprehensive logging of all commands and operations
for debugging and session replay. It captures:

- SDK method calls with parameters and results
- LLM interactions (prompts, responses, token counts)
- ReAct tool executions with timing
- Error traces with full context
- Page state at each step

Example:
    >>> from flybrowser.observability import CommandLogger
    >>> 
    >>> logger = CommandLogger(session_id="my-session")
    >>> 
    >>> # Log a command
    >>> with logger.log_command("goto", {"url": "https://example.com"}) as entry:
    ...     # Execute command
    ...     pass
    >>> 
    >>> # Get session history
    >>> history = logger.get_history()
    >>> history.export_json("session.json")
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
import uuid
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import sqlite3

from flybrowser.utils.logger import logger


class CommandType(str, Enum):
    """Types of commands that can be logged."""
    # SDK methods
    GOTO = "goto"
    ACT = "act"
    EXTRACT = "extract"
    NAVIGATE = "navigate"
    OBSERVE = "observe"
    AGENT = "agent"
    SCREENSHOT = "screenshot"
    
    # Internal operations
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    PAGE_NAVIGATION = "page_navigation"
    ELEMENT_INTERACTION = "element_interaction"
    
    # Stealth operations
    CAPTCHA_DETECTED = "captcha_detected"
    CAPTCHA_SOLVED = "captcha_solved"
    PROXY_ROTATED = "proxy_rotated"
    
    # Errors
    ERROR = "error"
    WARNING = "warning"


class LogLevel(str, Enum):
    """Log levels for filtering."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class PageState:
    """Captured page state at a point in time."""
    url: str = ""
    title: str = ""
    viewport: Dict[str, int] = field(default_factory=dict)
    scroll_position: Dict[str, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "viewport": self.viewport,
            "scroll_position": self.scroll_position,
            "timestamp": self.timestamp,
        }


@dataclass
class LLMUsage:
    """LLM usage statistics for a request."""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "cached": self.cached,
        }


@dataclass
class CommandEntry:
    """
    A single command log entry.
    
    Contains all information about a command execution including
    parameters, results, timing, and any associated page state.
    """
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    parent_id: Optional[str] = None  # For nested commands
    
    # Command info
    command_type: CommandType = CommandType.ACT
    name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Execution info
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    duration_ms: float = 0.0
    
    # Result
    success: bool = True
    result: Any = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # Context
    page_state_before: Optional[PageState] = None
    page_state_after: Optional[PageState] = None
    
    # LLM usage (if applicable)
    llm_usage: Optional[LLMUsage] = None
    llm_prompt: Optional[str] = None
    llm_response: Optional[str] = None
    
    # Tool calls (for agent operations)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    level: LogLevel = LogLevel.INFO
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(
        self, 
        success: bool = True, 
        result: Any = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Mark command as complete."""
        self.ended_at = time.time()
        self.duration_ms = (self.ended_at - self.started_at) * 1000
        self.success = success
        self.result = result
        
        if error:
            self.success = False
            self.error_message = str(error)
            self.error_traceback = traceback.format_exc()
            self.level = LogLevel.ERROR
    
    def add_tool_call(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any],
        result: Any = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Add a tool call to this command."""
        self.tool_calls.append({
            "tool_name": tool_name,
            "parameters": parameters,
            "result": result,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "parent_id": self.parent_id,
            "command_type": self.command_type.value,
            "name": self.name,
            "parameters": self.parameters,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "result": self._serialize_result(self.result),
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
            "page_state_before": self.page_state_before.to_dict() if self.page_state_before else None,
            "page_state_after": self.page_state_after.to_dict() if self.page_state_after else None,
            "llm_usage": self.llm_usage.to_dict() if self.llm_usage else None,
            "llm_prompt": self.llm_prompt,
            "llm_response": self.llm_response,
            "tool_calls": self.tool_calls,
            "level": self.level.value,
            "tags": self.tags,
            "metadata": self.metadata,
        }
    
    def _serialize_result(self, result: Any) -> Any:
        """Serialize result for JSON."""
        if result is None:
            return None
        if isinstance(result, (str, int, float, bool)):
            return result
        if isinstance(result, (list, tuple)):
            return [self._serialize_result(item) for item in result]
        if isinstance(result, dict):
            return {k: self._serialize_result(v) for k, v in result.items()}
        # For complex objects, convert to string
        try:
            return str(result)
        except Exception:
            return "<non-serializable>"


class SessionHistory:
    """
    Queryable session command history.
    
    Provides methods to filter, search, and export command history.
    """
    
    def __init__(self, entries: Optional[List[CommandEntry]] = None):
        self._entries = entries or []
    
    def add(self, entry: CommandEntry) -> None:
        """Add an entry to history."""
        self._entries.append(entry)
    
    def __len__(self) -> int:
        return len(self._entries)
    
    def __iter__(self):
        return iter(self._entries)
    
    @property
    def entries(self) -> List[CommandEntry]:
        """Get all entries."""
        return list(self._entries)
    
    def filter(
        self,
        command_type: Optional[CommandType] = None,
        level: Optional[LogLevel] = None,
        success: Optional[bool] = None,
        tags: Optional[List[str]] = None,
        time_range: Optional[tuple] = None,
    ) -> "SessionHistory":
        """
        Filter entries by criteria.
        
        Args:
            command_type: Filter by command type
            level: Filter by log level
            success: Filter by success status
            tags: Filter by tags (entry must have all)
            time_range: Filter by time range (start, end) as timestamps
            
        Returns:
            New SessionHistory with filtered entries
        """
        filtered = self._entries
        
        if command_type:
            filtered = [e for e in filtered if e.command_type == command_type]
        
        if level:
            filtered = [e for e in filtered if e.level == level]
        
        if success is not None:
            filtered = [e for e in filtered if e.success == success]
        
        if tags:
            filtered = [e for e in filtered if all(t in e.tags for t in tags)]
        
        if time_range:
            start, end = time_range
            filtered = [e for e in filtered if start <= e.started_at <= end]
        
        return SessionHistory(filtered)
    
    def get_errors(self) -> "SessionHistory":
        """Get all error entries."""
        return self.filter(success=False)
    
    def get_llm_calls(self) -> "SessionHistory":
        """Get all LLM-related entries."""
        return SessionHistory([
            e for e in self._entries 
            if e.command_type in (CommandType.LLM_REQUEST, CommandType.LLM_RESPONSE)
            or e.llm_usage is not None
        ])
    
    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """Get all tool calls across entries."""
        calls = []
        for entry in self._entries:
            calls.extend(entry.tool_calls)
        return calls
    
    def get_total_duration_ms(self) -> float:
        """Get total execution duration."""
        return sum(e.duration_ms for e in self._entries)
    
    def get_total_llm_usage(self) -> Dict[str, Any]:
        """Get aggregated LLM usage statistics."""
        total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "calls_count": 0,
            "cached_calls": 0,
        }
        
        for entry in self._entries:
            if entry.llm_usage:
                total["prompt_tokens"] += entry.llm_usage.prompt_tokens
                total["completion_tokens"] += entry.llm_usage.completion_tokens
                total["total_tokens"] += entry.llm_usage.total_tokens
                total["cost_usd"] += entry.llm_usage.cost_usd
                total["calls_count"] += 1
                if entry.llm_usage.cached:
                    total["cached_calls"] += 1
        
        return total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entries": [e.to_dict() for e in self._entries],
            "summary": {
                "total_entries": len(self._entries),
                "total_duration_ms": self.get_total_duration_ms(),
                "errors_count": len(self.get_errors()),
                "llm_usage": self.get_total_llm_usage(),
            },
        }
    
    def export_json(self, path: Union[str, Path]) -> None:
        """Export history to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"[COMMAND_LOG] Exported {len(self)} entries to {path}")
    
    def export_sqlite(self, path: Union[str, Path]) -> None:
        """Export history to SQLite database."""
        path = Path(path)
        
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS commands (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                parent_id TEXT,
                command_type TEXT,
                name TEXT,
                parameters TEXT,
                started_at REAL,
                ended_at REAL,
                duration_ms REAL,
                success INTEGER,
                result TEXT,
                error_message TEXT,
                level TEXT,
                tags TEXT,
                metadata TEXT
            )
        """)
        
        # Insert entries
        for entry in self._entries:
            cursor.execute("""
                INSERT OR REPLACE INTO commands VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id,
                entry.session_id,
                entry.parent_id,
                entry.command_type.value,
                entry.name,
                json.dumps(entry.parameters),
                entry.started_at,
                entry.ended_at,
                entry.duration_ms,
                1 if entry.success else 0,
                json.dumps(entry._serialize_result(entry.result)),
                entry.error_message,
                entry.level.value,
                json.dumps(entry.tags),
                json.dumps(entry.metadata),
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"[COMMAND_LOG] Exported {len(self)} entries to {path}")


class CommandLogger:
    """
    Main command logging service.
    
    Captures all commands and operations for a session with
    structured logging and queryable history.
    
    Example:
        >>> logger = CommandLogger(session_id="my-session")
        >>> 
        >>> # Context manager for commands
        >>> with logger.log_command("goto", {"url": "..."}) as entry:
        ...     # Execute
        ...     entry.result = "success"
        >>> 
        >>> # Async context manager
        >>> async with logger.log_async("extract", {"query": "..."}) as entry:
        ...     result = await extract()
        ...     entry.result = result
        >>> 
        >>> # Get history
        >>> history = logger.get_history()
    """
    
    # Global instance for singleton pattern
    _instance: Optional["CommandLogger"] = None
    
    def __init__(
        self, 
        session_id: Optional[str] = None,
        enabled: bool = True,
        max_entries: int = 10000,
        log_llm_prompts: bool = False,
        log_llm_responses: bool = False,
    ):
        """
        Initialize command logger.
        
        Args:
            session_id: Session identifier (auto-generated if None)
            enabled: Whether logging is enabled
            max_entries: Maximum entries to keep in memory
            log_llm_prompts: Whether to log full LLM prompts
            log_llm_responses: Whether to log full LLM responses
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.enabled = enabled
        self.max_entries = max_entries
        self.log_llm_prompts = log_llm_prompts
        self.log_llm_responses = log_llm_responses
        
        self._history = SessionHistory()
        self._current_entry: Optional[CommandEntry] = None
        self._entry_stack: List[CommandEntry] = []
        self._lock = asyncio.Lock()
        
        logger.info(f"[COMMAND_LOG] Initialized for session {self.session_id}")
    
    @classmethod
    def get_instance(cls) -> "CommandLogger":
        """Get the global CommandLogger instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def set_instance(cls, instance: "CommandLogger") -> None:
        """Set the global CommandLogger instance."""
        cls._instance = instance
    
    @contextmanager
    def log_command(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        command_type: CommandType = CommandType.ACT,
        tags: Optional[List[str]] = None,
    ):
        """
        Context manager for logging a command.
        
        Args:
            name: Command name
            parameters: Command parameters
            command_type: Type of command
            tags: Optional tags
            
        Yields:
            CommandEntry to populate with results
        """
        if not self.enabled:
            yield CommandEntry()
            return
        
        entry = CommandEntry(
            session_id=self.session_id,
            parent_id=self._entry_stack[-1].id if self._entry_stack else None,
            command_type=command_type,
            name=name,
            parameters=parameters or {},
            tags=tags or [],
        )
        
        self._entry_stack.append(entry)
        self._current_entry = entry
        
        try:
            yield entry
            if not entry.ended_at:
                entry.complete(success=True)
        except Exception as e:
            entry.complete(success=False, error=e)
            raise
        finally:
            self._entry_stack.pop()
            self._current_entry = self._entry_stack[-1] if self._entry_stack else None
            self._add_entry(entry)
    
    @asynccontextmanager
    async def log_async(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        command_type: CommandType = CommandType.ACT,
        tags: Optional[List[str]] = None,
    ):
        """Async context manager for logging a command."""
        if not self.enabled:
            yield CommandEntry()
            return
        
        async with self._lock:
            entry = CommandEntry(
                session_id=self.session_id,
                parent_id=self._entry_stack[-1].id if self._entry_stack else None,
                command_type=command_type,
                name=name,
                parameters=parameters or {},
                tags=tags or [],
            )
            
            self._entry_stack.append(entry)
            self._current_entry = entry
        
        try:
            yield entry
            if not entry.ended_at:
                entry.complete(success=True)
        except Exception as e:
            entry.complete(success=False, error=e)
            raise
        finally:
            async with self._lock:
                self._entry_stack.pop()
                self._current_entry = self._entry_stack[-1] if self._entry_stack else None
                self._add_entry(entry)
    
    def log_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        usage: Dict[str, Any],
    ) -> CommandEntry:
        """
        Log an LLM API call.
        
        Args:
            model: Model name
            prompt: Prompt sent
            response: Response received
            usage: Token usage stats
            
        Returns:
            Created CommandEntry
        """
        if not self.enabled:
            return CommandEntry()
        
        entry = CommandEntry(
            session_id=self.session_id,
            parent_id=self._current_entry.id if self._current_entry else None,
            command_type=CommandType.LLM_REQUEST,
            name=f"llm_call_{model}",
            llm_usage=LLMUsage(
                model=model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                cost_usd=usage.get("cost_usd", 0.0),
                cached=usage.get("cached", False),
            ),
            llm_prompt=prompt if self.log_llm_prompts else None,
            llm_response=response if self.log_llm_responses else None,
        )
        entry.complete(success=True)
        
        # Also add to parent entry if exists
        if self._current_entry:
            self._current_entry.llm_usage = entry.llm_usage
            if self.log_llm_prompts:
                self._current_entry.llm_prompt = prompt
            if self.log_llm_responses:
                self._current_entry.llm_response = response
        
        self._add_entry(entry)
        return entry
    
    def log_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Any = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Log a tool call."""
        if not self.enabled:
            return
        
        if self._current_entry:
            self._current_entry.add_tool_call(
                tool_name=tool_name,
                parameters=parameters,
                result=result,
                duration_ms=duration_ms,
            )
    
    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> CommandEntry:
        """Log an error."""
        entry = CommandEntry(
            session_id=self.session_id,
            parent_id=self._current_entry.id if self._current_entry else None,
            command_type=CommandType.ERROR,
            name="error",
            level=LogLevel.ERROR,
            metadata=context or {},
        )
        entry.complete(success=False, error=error)
        self._add_entry(entry)
        return entry
    
    def log_warning(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> CommandEntry:
        """Log a warning."""
        entry = CommandEntry(
            session_id=self.session_id,
            parent_id=self._current_entry.id if self._current_entry else None,
            command_type=CommandType.WARNING,
            name="warning",
            level=LogLevel.WARNING,
            result=message,
            metadata=context or {},
        )
        entry.complete(success=True)
        self._add_entry(entry)
        return entry
    
    def log_page_state(self, state: PageState, position: str = "after") -> None:
        """Log page state to current entry."""
        if self._current_entry:
            if position == "before":
                self._current_entry.page_state_before = state
            else:
                self._current_entry.page_state_after = state
    
    def _add_entry(self, entry: CommandEntry) -> None:
        """Add entry to history with size limit."""
        self._history.add(entry)
        
        # Enforce max entries
        if len(self._history) > self.max_entries:
            # Remove oldest entries
            self._history._entries = self._history._entries[-self.max_entries:]
    
    def get_history(self) -> SessionHistory:
        """Get session command history."""
        return self._history
    
    def get_current_entry(self) -> Optional[CommandEntry]:
        """Get the current active command entry."""
        return self._current_entry
    
    def clear(self) -> None:
        """Clear all history."""
        self._history = SessionHistory()
        self._entry_stack.clear()
        self._current_entry = None
    
    def export(self, path: Union[str, Path], format: str = "json") -> None:
        """
        Export history to file.
        
        Args:
            path: Output file path
            format: Export format (json, sqlite)
        """
        if format == "json":
            self._history.export_json(path)
        elif format == "sqlite":
            self._history.export_sqlite(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
