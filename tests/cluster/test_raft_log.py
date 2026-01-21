# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Raft log."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from flybrowser.service.cluster.raft.log import RaftLog, Snapshot
from flybrowser.service.cluster.raft.messages import LogEntry


class TestSnapshot:
    """Tests for Snapshot."""

    def test_snapshot_creation(self):
        """Test creating a snapshot."""
        snap = Snapshot(
            last_included_index=10,
            last_included_term=2,
            data=b"test state data",
            created_at=1000.0
        )
        
        assert snap.last_included_index == 10
        assert snap.last_included_term == 2
        assert snap.data == b"test state data"
        assert snap.created_at == 1000.0

    def test_snapshot_to_dict(self):
        """Test snapshot serialization."""
        snap = Snapshot(
            last_included_index=10,
            last_included_term=2,
            data=b"test",
        )
        
        d = snap.to_dict()
        
        assert d["last_included_index"] == 10
        assert d["last_included_term"] == 2
        assert "data" in d  # Base64 encoded

    def test_snapshot_from_dict(self):
        """Test snapshot deserialization."""
        import base64
        
        d = {
            "last_included_index": 10,
            "last_included_term": 2,
            "data": base64.b64encode(b"test").decode("utf-8"),
            "created_at": 1000.0,
        }
        
        snap = Snapshot.from_dict(d)
        
        assert snap.last_included_index == 10
        assert snap.last_included_term == 2
        assert snap.data == b"test"


class TestRaftLog:
    """Tests for RaftLog."""

    def test_init(self):
        """Test RaftLog initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = RaftLog(tmpdir)
            
            assert log.current_term == 0
            assert log._commit_index == 0

    def test_current_term_property(self):
        """Test current_term property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = RaftLog(tmpdir)
            
            log.current_term = 5
            
            assert log.current_term == 5

    def test_append_entry(self):
        """Test appending log entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = RaftLog(tmpdir)
            
            entry = LogEntry(
                term=1,
                index=1,
                command={"op": "set", "key": "x", "value": 1}
            )
            
            log.append(entry)
            
            assert len(log._entries) == 1
            assert log._entries[0].term == 1

    def test_get_entry(self):
        """Test getting log entry by index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = RaftLog(tmpdir)
            
            entry = LogEntry(term=1, index=1, command={"op": "test"})
            log.append(entry)
            
            retrieved = log.get(1)
            
            assert retrieved is not None
            assert retrieved.term == 1
            assert retrieved.command["op"] == "test"

    def test_get_entry_not_found(self):
        """Test getting non-existent entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = RaftLog(tmpdir)
            
            result = log.get(100)
            
            assert result is None

    def test_persistence(self):
        """Test log persistence across restarts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate log
            log1 = RaftLog(tmpdir)
            log1.current_term = 3
            log1.append(LogEntry(term=1, index=1, command={"op": "test1"}))
            log1.append(LogEntry(term=2, index=2, command={"op": "test2"}))
            
            # Create new log instance from same directory
            log2 = RaftLog(tmpdir)
            
            assert log2.current_term == 3
            assert len(log2._entries) == 2

    def test_voted_for_persistence(self):
        """Test voted_for persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log1 = RaftLog(tmpdir)
            log1._voted_for = "node-123"
            log1._persist_meta()
            
            log2 = RaftLog(tmpdir)
            
            assert log2._voted_for == "node-123"


class TestRaftLogRange:
    """Tests for RaftLog range operations."""

    def test_get_range(self):
        """Test getting range of entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = RaftLog(tmpdir)
            
            for i in range(1, 6):
                log.append(LogEntry(term=1, index=i, command={"i": i}))
            
            entries = log.get_range(2, 4)
            
            assert len(entries) == 3
            assert entries[0].index == 2
            assert entries[-1].index == 4

    def test_last_index(self):
        """Test last_index property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = RaftLog(tmpdir)
            
            assert log.last_index == 0
            
            log.append(LogEntry(term=1, index=1, command={}))
            log.append(LogEntry(term=1, index=2, command={}))
            
            assert log.last_index == 2

    def test_last_term(self):
        """Test last_term property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = RaftLog(tmpdir)
            
            assert log.last_term == 0
            
            log.append(LogEntry(term=3, index=1, command={}))
            
            assert log.last_term == 3


class TestRaftLogSnapshot:
    """Tests for RaftLog snapshot operations."""

    def test_install_snapshot(self):
        """Test installing a snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = RaftLog(tmpdir)
            
            # Add some entries
            for i in range(1, 11):
                log.append(LogEntry(term=1, index=i, command={"i": i}))
            
            # Install snapshot
            snap = Snapshot(
                last_included_index=5,
                last_included_term=1,
                data=b"state data"
            )
            
            log.install_snapshot(snap)
            
            assert log._snapshot is not None
            assert log._snapshot_last_index == 5

    def test_create_snapshot(self):
        """Test creating a snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = RaftLog(tmpdir)
            
            for i in range(1, 11):
                log.append(LogEntry(term=1, index=i, command={"i": i}))
            
            log.create_snapshot(b"state data", last_index=5, last_term=1)
            
            assert log._snapshot is not None
            assert log._snapshot_last_index == 5
            assert log._snapshot_last_term == 1
