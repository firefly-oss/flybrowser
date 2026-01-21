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
Raft Log Implementation for FlyBrowser HA Cluster.

This module provides persistent log storage for the Raft consensus algorithm.
The log stores commands that are replicated across the cluster and applied
to the state machine once committed.

Features:
- Append-only log with persistence
- Log compaction via snapshots
- Efficient range queries
- Crash recovery
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flybrowser.service.cluster.raft.messages import LogEntry


@dataclass
class Snapshot:
    """A snapshot of the state machine at a point in time.
    
    Attributes:
        last_included_index: The last log index included in snapshot
        last_included_term: The term of last_included_index
        data: Serialized state machine data
        created_at: Timestamp when snapshot was created
    """
    last_included_index: int
    last_included_term: int
    data: bytes
    created_at: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (data as base64)."""
        import base64
        return {
            "last_included_index": self.last_included_index,
            "last_included_term": self.last_included_term,
            "data": base64.b64encode(self.data).decode("utf-8"),
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Snapshot":
        """Deserialize from dictionary."""
        import base64
        return cls(
            last_included_index=d["last_included_index"],
            last_included_term=d["last_included_term"],
            data=base64.b64decode(d["data"]),
            created_at=d.get("created_at", 0.0),
        )


class RaftLog:
    """Persistent log storage for Raft consensus.
    
    The log is 1-indexed (first entry is at index 1).
    Index 0 is reserved for the "null" entry before the log starts.
    
    Thread-safe for concurrent access.
    
    Example:
        >>> log = RaftLog("./raft_data")
        >>> log.append(LogEntry(term=1, index=1, command={"op": "set", "key": "x", "value": 1}))
        >>> entry = log.get(1)
        >>> log.commit(1)
    """
    
    def __init__(self, data_dir: str) -> None:
        """Initialize the Raft log.
        
        Args:
            data_dir: Directory for persistent storage
        """
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        
        self._log_file = self._data_dir / "raft.log"
        self._meta_file = self._data_dir / "raft.meta"
        self._snapshot_file = self._data_dir / "raft.snapshot"
        
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        # Executor for running blocking I/O in async contexts
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="raft_log")
        
        # In-memory log (entries after snapshot)
        self._entries: List[LogEntry] = []
        
        # Snapshot state
        self._snapshot: Optional[Snapshot] = None
        self._snapshot_last_index: int = 0
        self._snapshot_last_term: int = 0
        
        # Persistent state
        self._current_term: int = 0
        self._voted_for: Optional[str] = None
        self._commit_index: int = 0
        
        # Load from disk
        self._load()
    
    def _load(self) -> None:
        """Load log and metadata from disk."""
        # Load metadata
        if self._meta_file.exists():
            try:
                with open(self._meta_file, "r") as f:
                    meta = json.load(f)
                    self._current_term = meta.get("current_term", 0)
                    self._voted_for = meta.get("voted_for")
                    self._commit_index = meta.get("commit_index", 0)
            except (json.JSONDecodeError, IOError):
                pass
        
        # Load snapshot
        if self._snapshot_file.exists():
            try:
                with open(self._snapshot_file, "r") as f:
                    snap_data = json.load(f)
                    self._snapshot = Snapshot.from_dict(snap_data)
                    self._snapshot_last_index = self._snapshot.last_included_index
                    self._snapshot_last_term = self._snapshot.last_included_term
            except (json.JSONDecodeError, IOError):
                pass
        
        # Load log entries
        if self._log_file.exists():
            try:
                with open(self._log_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entry = LogEntry.from_json(line)
                            self._entries.append(entry)
            except (json.JSONDecodeError, IOError):
                pass
    
    def _persist_meta(self) -> None:
        """Persist metadata to disk."""
        meta = {
            "current_term": self._current_term,
            "voted_for": self._voted_for,
            "commit_index": self._commit_index,
        }
        with open(self._meta_file, "w") as f:
            json.dump(meta, f)

    def _persist_log(self) -> None:
        """Persist log entries to disk."""
        with open(self._log_file, "w") as f:
            for entry in self._entries:
                f.write(entry.to_json() + "\n")

    def _append_to_log_file(self, entry: LogEntry) -> None:
        """Append a single entry to the log file."""
        with open(self._log_file, "a") as f:
            f.write(entry.to_json() + "\n")

    # ==================== Public API ====================

    @property
    def current_term(self) -> int:
        """Get current term."""
        with self._lock:
            return self._current_term

    @current_term.setter
    def current_term(self, value: int) -> None:
        """Set current term and persist."""
        with self._lock:
            self._current_term = value
            self._persist_meta()

    @property
    def voted_for(self) -> Optional[str]:
        """Get voted_for in current term."""
        with self._lock:
            return self._voted_for

    @voted_for.setter
    def voted_for(self, value: Optional[str]) -> None:
        """Set voted_for and persist."""
        with self._lock:
            self._voted_for = value
            self._persist_meta()

    @property
    def commit_index(self) -> int:
        """Get commit index."""
        with self._lock:
            return self._commit_index

    @commit_index.setter
    def commit_index(self, value: int) -> None:
        """Set commit index and persist."""
        with self._lock:
            self._commit_index = value
            self._persist_meta()

    @property
    def last_index(self) -> int:
        """Get the index of the last log entry."""
        with self._lock:
            if self._entries:
                return self._entries[-1].index
            return self._snapshot_last_index

    @property
    def last_term(self) -> int:
        """Get the term of the last log entry."""
        with self._lock:
            if self._entries:
                return self._entries[-1].term
            return self._snapshot_last_term

    @property
    def first_index(self) -> int:
        """Get the index of the first log entry (after snapshot)."""
        with self._lock:
            if self._entries:
                return self._entries[0].index
            return self._snapshot_last_index + 1

    def __len__(self) -> int:
        """Get number of entries in log (after snapshot)."""
        with self._lock:
            return len(self._entries)

    def get(self, index: int) -> Optional[LogEntry]:
        """Get a log entry by index."""
        with self._lock:
            if index <= self._snapshot_last_index:
                return None  # Entry is in snapshot

            # Convert to array index
            array_idx = index - self._snapshot_last_index - 1
            if 0 <= array_idx < len(self._entries):
                return self._entries[array_idx]
            return None

    def get_term(self, index: int) -> int:
        """Get the term of a log entry by index."""
        with self._lock:
            if index == 0:
                return 0
            if index == self._snapshot_last_index:
                return self._snapshot_last_term

            entry = self.get(index)
            return entry.term if entry else 0

    def get_range(self, start_index: int, end_index: Optional[int] = None) -> List[LogEntry]:
        """Get a range of log entries [start_index, end_index]."""
        with self._lock:
            if end_index is None:
                end_index = self.last_index

            result = []
            for idx in range(start_index, end_index + 1):
                entry = self.get(idx)
                if entry:
                    result.append(entry)
            return result

    def append(self, entry: LogEntry) -> None:
        """Append a new entry to the log."""
        with self._lock:
            self._entries.append(entry)
            self._append_to_log_file(entry)

    def append_entries(self, entries: List[LogEntry]) -> None:
        """Append multiple entries to the log."""
        with self._lock:
            for entry in entries:
                self._entries.append(entry)
            self._persist_log()

    def truncate_after(self, index: int) -> None:
        """Remove all entries after the given index."""
        with self._lock:
            if index < self._snapshot_last_index:
                return  # Can't truncate into snapshot

            # Find entries to keep
            new_entries = []
            for entry in self._entries:
                if entry.index <= index:
                    new_entries.append(entry)

            self._entries = new_entries
            self._persist_log()

    def matches(self, index: int, term: int) -> bool:
        """Check if log contains entry at index with matching term."""
        with self._lock:
            if index == 0:
                return True  # Empty log always matches
            if index == self._snapshot_last_index:
                return self._snapshot_last_term == term

            entry = self.get(index)
            return entry is not None and entry.term == term

    def find_conflict(self, index: int, term: int) -> Tuple[Optional[int], Optional[int]]:
        """Find conflict info for AppendEntries optimization.

        Returns:
            (conflict_index, conflict_term) or (None, None) if no conflict
        """
        with self._lock:
            entry = self.get(index)
            if entry is None:
                # We don't have this entry, conflict at our last index + 1
                return (self.last_index + 1, None)

            if entry.term != term:
                # Term mismatch, find first entry of conflicting term
                conflict_term = entry.term
                conflict_index = index

                for e in self._entries:
                    if e.term == conflict_term:
                        conflict_index = e.index
                        break

                return (conflict_index, conflict_term)

            return (None, None)

    # ==================== Snapshot Methods ====================

    def create_snapshot(self, state_data: bytes, last_index: int, last_term: int) -> Snapshot:
        """Create a snapshot and compact the log.

        Args:
            state_data: Serialized state machine data
            last_index: Last log index included in snapshot
            last_term: Term of last_index

        Returns:
            The created Snapshot
        """
        import time

        with self._lock:
            snapshot = Snapshot(
                last_included_index=last_index,
                last_included_term=last_term,
                data=state_data,
                created_at=time.time(),
            )

            # Save snapshot to disk
            with open(self._snapshot_file, "w") as f:
                json.dump(snapshot.to_dict(), f)

            # Remove entries included in snapshot
            new_entries = [e for e in self._entries if e.index > last_index]
            self._entries = new_entries

            # Update snapshot state
            self._snapshot = snapshot
            self._snapshot_last_index = last_index
            self._snapshot_last_term = last_term

            # Persist compacted log
            self._persist_log()

            return snapshot

    def install_snapshot(self, snapshot: Snapshot) -> None:
        """Install a snapshot received from leader.

        Args:
            snapshot: The snapshot to install
        """
        with self._lock:
            # Save snapshot to disk
            with open(self._snapshot_file, "w") as f:
                json.dump(snapshot.to_dict(), f)

            # Clear log entries before snapshot
            new_entries = [e for e in self._entries if e.index > snapshot.last_included_index]
            self._entries = new_entries

            # Update snapshot state
            self._snapshot = snapshot
            self._snapshot_last_index = snapshot.last_included_index
            self._snapshot_last_term = snapshot.last_included_term

            # Update commit index if needed
            if self._commit_index < snapshot.last_included_index:
                self._commit_index = snapshot.last_included_index

            # Persist
            self._persist_log()
            self._persist_meta()

    def get_snapshot(self) -> Optional[Snapshot]:
        """Get the current snapshot."""
        with self._lock:
            return self._snapshot

    def needs_snapshot(self, threshold: int) -> bool:
        """Check if log needs compaction via snapshot."""
        with self._lock:
            return len(self._entries) >= threshold

    def set_term_and_vote(self, term: int, voted_for: Optional[str]) -> None:
        """Atomically set term and voted_for."""
        with self._lock:
            self._current_term = term
            self._voted_for = voted_for
            self._persist_meta()

    def get_entries_for_follower(
        self, next_index: int, max_entries: int = 100
    ) -> Tuple[int, int, List[LogEntry]]:
        """Get entries to send to a follower.

        Args:
            next_index: The next index the follower needs
            max_entries: Maximum entries to return

        Returns:
            (prev_log_index, prev_log_term, entries)
        """
        with self._lock:
            prev_log_index = next_index - 1
            prev_log_term = self.get_term(prev_log_index)

            entries = []
            for i in range(next_index, min(next_index + max_entries, self.last_index + 1)):
                entry = self.get(i)
                if entry:
                    entries.append(entry)

            return (prev_log_index, prev_log_term, entries)

    # ==================== Async Wrappers ====================
    # These methods run blocking I/O operations in a thread pool
    # to avoid blocking the async event loop.

    async def append_async(self, entry: LogEntry) -> None:
        """Async version of append - runs I/O in executor."""
        async with self._async_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self.append, entry)

    async def append_entries_async(self, entries: List[LogEntry]) -> None:
        """Async version of append_entries - runs I/O in executor."""
        async with self._async_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self.append_entries, entries)

    async def truncate_after_async(self, index: int) -> None:
        """Async version of truncate_after - runs I/O in executor."""
        async with self._async_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self.truncate_after, index)

    async def create_snapshot_async(
        self, state_data: bytes, last_index: int, last_term: int
    ) -> Snapshot:
        """Async version of create_snapshot - runs I/O in executor."""
        async with self._async_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                partial(self.create_snapshot, state_data, last_index, last_term),
            )

    async def install_snapshot_async(self, snapshot: Snapshot) -> None:
        """Async version of install_snapshot - runs I/O in executor."""
        async with self._async_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self.install_snapshot, snapshot)

    async def set_term_and_vote_async(self, term: int, voted_for: Optional[str]) -> None:
        """Async version of set_term_and_vote - runs I/O in executor."""
        async with self._async_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor, self.set_term_and_vote, term, voted_for
            )

    def close(self) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=True)
