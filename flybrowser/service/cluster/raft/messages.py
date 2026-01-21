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
Raft RPC Messages for FlyBrowser HA Cluster.

This module defines the message types used in Raft consensus protocol:
- RequestVote: Used during leader election
- AppendEntries: Used for log replication and heartbeats
- InstallSnapshot: Used for log compaction

All messages are serializable to JSON for transport over HTTP.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class MessageType(str, Enum):
    """Types of Raft RPC messages."""
    REQUEST_VOTE = "request_vote"
    REQUEST_VOTE_RESPONSE = "request_vote_response"
    PRE_VOTE = "pre_vote"
    PRE_VOTE_RESPONSE = "pre_vote_response"
    APPEND_ENTRIES = "append_entries"
    APPEND_ENTRIES_RESPONSE = "append_entries_response"
    INSTALL_SNAPSHOT = "install_snapshot"
    INSTALL_SNAPSHOT_RESPONSE = "install_snapshot_response"
    TIMEOUT_NOW = "timeout_now"
    TIMEOUT_NOW_RESPONSE = "timeout_now_response"
    CLIENT_REQUEST = "client_request"
    CLIENT_RESPONSE = "client_response"


@dataclass
class LogEntry:
    """A single entry in the Raft log.
    
    Attributes:
        term: The term when entry was received by leader
        index: Position in the log (1-indexed)
        command: The command to apply to state machine
        client_id: Optional client identifier for deduplication
        request_id: Optional request identifier for deduplication
    """
    term: int
    index: int
    command: Dict[str, Any]
    client_id: Optional[str] = None
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "term": self.term,
            "index": self.index,
            "command": self.command,
            "client_id": self.client_id,
            "request_id": self.request_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        """Deserialize from dictionary."""
        return cls(
            term=data["term"],
            index=data["index"],
            command=data["command"],
            client_id=data.get("client_id"),
            request_id=data.get("request_id"),
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, data: str) -> "LogEntry":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(data))


@dataclass
class RaftMessage:
    """Base class for all Raft messages."""
    message_type: MessageType
    term: int
    sender_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "message_type": self.message_type.value,
            "term": self.term,
            "sender_id": self.sender_id,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class RequestVoteRequest(RaftMessage):
    """RequestVote RPC request (Section 5.2 of Raft paper).
    
    Sent by candidates to gather votes during election.
    
    Attributes:
        term: Candidate's term
        sender_id: Candidate requesting vote
        last_log_index: Index of candidate's last log entry
        last_log_term: Term of candidate's last log entry
        is_pre_vote: True if this is a pre-vote request (doesn't increment term)
    """
    last_log_index: int = 0
    last_log_term: int = 0
    is_pre_vote: bool = False
    
    def __post_init__(self) -> None:
        if self.is_pre_vote:
            self.message_type = MessageType.PRE_VOTE
        else:
            self.message_type = MessageType.REQUEST_VOTE
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "last_log_index": self.last_log_index,
            "last_log_term": self.last_log_term,
            "is_pre_vote": self.is_pre_vote,
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequestVoteRequest":
        is_pre_vote = data.get("is_pre_vote", False) or data.get("message_type") == MessageType.PRE_VOTE.value
        return cls(
            message_type=MessageType.PRE_VOTE if is_pre_vote else MessageType.REQUEST_VOTE,
            term=data["term"],
            sender_id=data["sender_id"],
            last_log_index=data.get("last_log_index", 0),
            last_log_term=data.get("last_log_term", 0),
            is_pre_vote=is_pre_vote,
        )


@dataclass
class RequestVoteResponse(RaftMessage):
    """RequestVote RPC response.

    Attributes:
        term: Current term, for candidate to update itself
        sender_id: Node sending the response
        vote_granted: True if candidate received vote
        is_pre_vote: True if this is a pre-vote response
    """
    vote_granted: bool = False
    is_pre_vote: bool = False

    def __post_init__(self) -> None:
        if self.is_pre_vote:
            self.message_type = MessageType.PRE_VOTE_RESPONSE
        else:
            self.message_type = MessageType.REQUEST_VOTE_RESPONSE

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["vote_granted"] = self.vote_granted
        data["is_pre_vote"] = self.is_pre_vote
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequestVoteResponse":
        is_pre_vote = data.get("is_pre_vote", False) or data.get("message_type") == MessageType.PRE_VOTE_RESPONSE.value
        return cls(
            message_type=MessageType.PRE_VOTE_RESPONSE if is_pre_vote else MessageType.REQUEST_VOTE_RESPONSE,
            term=data["term"],
            sender_id=data["sender_id"],
            vote_granted=data.get("vote_granted", False),
            is_pre_vote=is_pre_vote,
        )


@dataclass
class AppendEntriesRequest(RaftMessage):
    """AppendEntries RPC request (Section 5.3 of Raft paper).

    Used for log replication and as heartbeat (when entries is empty).

    Attributes:
        term: Leader's term
        sender_id: Leader's ID (so follower can redirect clients)
        prev_log_index: Index of log entry immediately preceding new ones
        prev_log_term: Term of prev_log_index entry
        entries: Log entries to store (empty for heartbeat)
        leader_commit: Leader's commit index
    """
    prev_log_index: int = 0
    prev_log_term: int = 0
    entries: List[Dict[str, Any]] = field(default_factory=list)
    leader_commit: int = 0

    def __post_init__(self) -> None:
        self.message_type = MessageType.APPEND_ENTRIES

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "prev_log_index": self.prev_log_index,
            "prev_log_term": self.prev_log_term,
            "entries": self.entries,
            "leader_commit": self.leader_commit,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppendEntriesRequest":
        return cls(
            message_type=MessageType.APPEND_ENTRIES,
            term=data["term"],
            sender_id=data["sender_id"],
            prev_log_index=data.get("prev_log_index", 0),
            prev_log_term=data.get("prev_log_term", 0),
            entries=data.get("entries", []),
            leader_commit=data.get("leader_commit", 0),
        )


@dataclass
class AppendEntriesResponse(RaftMessage):
    """AppendEntries RPC response.

    Attributes:
        term: Current term, for leader to update itself
        sender_id: Node sending the response
        success: True if follower contained entry matching prev_log_index/term
        match_index: The index of the highest log entry known to be replicated
        conflict_index: On failure, the first index of the conflicting term
        conflict_term: On failure, the term of the conflicting entry
    """
    success: bool = False
    match_index: int = 0
    conflict_index: Optional[int] = None
    conflict_term: Optional[int] = None

    def __post_init__(self) -> None:
        self.message_type = MessageType.APPEND_ENTRIES_RESPONSE

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "success": self.success,
            "match_index": self.match_index,
            "conflict_index": self.conflict_index,
            "conflict_term": self.conflict_term,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppendEntriesResponse":
        return cls(
            message_type=MessageType.APPEND_ENTRIES_RESPONSE,
            term=data["term"],
            sender_id=data["sender_id"],
            success=data.get("success", False),
            match_index=data.get("match_index", 0),
            conflict_index=data.get("conflict_index"),
            conflict_term=data.get("conflict_term"),
        )


@dataclass
class InstallSnapshotRequest(RaftMessage):
    """InstallSnapshot RPC request (Section 7 of Raft paper).

    Used to send snapshot chunks to slow followers.

    Attributes:
        term: Leader's term
        sender_id: Leader's ID
        last_included_index: The snapshot replaces all entries up through this index
        last_included_term: Term of last_included_index
        offset: Byte offset where chunk is positioned in snapshot file
        data: Raw bytes of the snapshot chunk (base64 encoded)
        done: True if this is the last chunk
    """
    last_included_index: int = 0
    last_included_term: int = 0
    offset: int = 0
    data: str = ""  # Base64 encoded
    done: bool = False

    def __post_init__(self) -> None:
        self.message_type = MessageType.INSTALL_SNAPSHOT

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "last_included_index": self.last_included_index,
            "last_included_term": self.last_included_term,
            "offset": self.offset,
            "data": self.data,
            "done": self.done,
        })
        return data


@dataclass
class InstallSnapshotResponse(RaftMessage):
    """InstallSnapshot RPC response.

    Attributes:
        term: Current term, for leader to update itself
        sender_id: Node sending the response
    """

    def __post_init__(self) -> None:
        self.message_type = MessageType.INSTALL_SNAPSHOT_RESPONSE

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstallSnapshotResponse":
        return cls(
            message_type=MessageType.INSTALL_SNAPSHOT_RESPONSE,
            term=data["term"],
            sender_id=data["sender_id"],
        )


@dataclass
class TimeoutNowRequest(RaftMessage):
    """TimeoutNow RPC request for leadership transfer.

    Sent by the current leader to a designated successor to
    trigger immediate election timeout.

    Attributes:
        term: Leader's term
        sender_id: Leader's ID
    """

    def __post_init__(self) -> None:
        self.message_type = MessageType.TIMEOUT_NOW

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeoutNowRequest":
        return cls(
            message_type=MessageType.TIMEOUT_NOW,
            term=data["term"],
            sender_id=data["sender_id"],
        )


@dataclass
class TimeoutNowResponse(RaftMessage):
    """TimeoutNow RPC response.

    Attributes:
        term: Current term
        sender_id: Node sending the response
        success: True if the node will start an election
    """
    success: bool = False

    def __post_init__(self) -> None:
        self.message_type = MessageType.TIMEOUT_NOW_RESPONSE

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["success"] = self.success
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeoutNowResponse":
        return cls(
            message_type=MessageType.TIMEOUT_NOW_RESPONSE,
            term=data["term"],
            sender_id=data["sender_id"],
            success=data.get("success", False),
        )


def parse_message(data: Dict[str, Any]) -> RaftMessage:
    """Parse a dictionary into the appropriate message type."""
    msg_type = MessageType(data.get("message_type", ""))

    parsers = {
        MessageType.REQUEST_VOTE: RequestVoteRequest.from_dict,
        MessageType.REQUEST_VOTE_RESPONSE: RequestVoteResponse.from_dict,
        MessageType.PRE_VOTE: RequestVoteRequest.from_dict,
        MessageType.PRE_VOTE_RESPONSE: RequestVoteResponse.from_dict,
        MessageType.APPEND_ENTRIES: AppendEntriesRequest.from_dict,
        MessageType.APPEND_ENTRIES_RESPONSE: AppendEntriesResponse.from_dict,
        MessageType.INSTALL_SNAPSHOT_RESPONSE: InstallSnapshotResponse.from_dict,
        MessageType.TIMEOUT_NOW: TimeoutNowRequest.from_dict,
        MessageType.TIMEOUT_NOW_RESPONSE: TimeoutNowResponse.from_dict,
    }

    parser = parsers.get(msg_type)
    if parser:
        return parser(data)

    raise ValueError(f"Unknown message type: {msg_type}")

