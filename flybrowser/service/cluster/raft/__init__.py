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
Custom Raft Consensus Implementation for FlyBrowser HA Cluster.

This module provides a production-grade Raft consensus implementation built
from scratch for leader election, log replication, and distributed state management.

The implementation follows the Raft paper (Ongaro & Ousterhout, 2014) with
optimizations for browser automation workloads.

Key Components:
- RaftNode: Core Raft node with leader election and log replication
- RaftLog: Persistent log storage with snapshotting
- StateMachine: Replicated state machine for cluster-wide data
- RaftConfig: Configuration for Raft consensus parameters

Features:
- Leader election with randomized timeouts
- Log replication with consistency guarantees
- Automatic leader failover
- Log compaction via snapshots
- Dynamic cluster membership changes
- Persistent state for crash recovery
"""

from flybrowser.service.cluster.raft.config import RaftConfig
from flybrowser.service.cluster.raft.log import LogEntry, RaftLog
from flybrowser.service.cluster.raft.messages import (
    AppendEntriesRequest,
    AppendEntriesResponse,
    RequestVoteRequest,
    RequestVoteResponse,
    RaftMessage,
)
from flybrowser.service.cluster.raft.node import RaftNode, NodeRole
from flybrowser.service.cluster.raft.state_machine import StateMachine
from flybrowser.service.cluster.raft.transport import RaftTransport

__all__ = [
    "AppendEntriesRequest",
    "AppendEntriesResponse",
    "LogEntry",
    "NodeRole",
    "RaftConfig",
    "RaftLog",
    "RaftMessage",
    "RaftNode",
    "RaftTransport",
    "RequestVoteRequest",
    "RequestVoteResponse",
    "StateMachine",
]

