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
Raft Consensus Node for FlyBrowser HA Cluster.

This module implements the core Raft consensus algorithm including:
- Leader election with randomized timeouts
- Log replication with consistency guarantees
- Commit index advancement
- Snapshot-based log compaction

The implementation follows the Raft paper (Ongaro & Ousterhout, 2014).
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from flybrowser.service.cluster.raft.config import RaftConfig
from flybrowser.service.cluster.raft.log import RaftLog
from flybrowser.service.cluster.raft.messages import (
    AppendEntriesRequest,
    AppendEntriesResponse,
    LogEntry,
    MessageType,
    RequestVoteRequest,
    RequestVoteResponse,
    TimeoutNowRequest,
    TimeoutNowResponse,
)
from flybrowser.service.cluster.raft.state_machine import StateMachine, CommandType
from flybrowser.service.cluster.raft.transport import RaftTransport
from flybrowser.service.cluster.exceptions import (
    NotLeaderError,
    CommandTimeoutError,
    ClusterConfigurationError,
)
from flybrowser.utils.logger import logger


class ConfigurationState(str, Enum):
    """State of cluster configuration change."""
    STABLE = "stable"  # Normal operation
    JOINT = "joint"    # Joint consensus (C_old,new)
    TRANSITIONING = "transitioning"  # Moving to new config


class NodeRole(str, Enum):
    """Role of a node in the Raft cluster."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class RaftNode:
    """A node in the Raft consensus cluster.
    
    Implements the Raft consensus algorithm for leader election and log replication.
    
    Example:
        >>> config = RaftConfig(
        ...     node_id="node1",
        ...     bind_host="0.0.0.0",
        ...     bind_port=4321,
        ...     cluster_nodes=["node2:4322", "node3:4323"],
        ... )
        >>> node = RaftNode(config)
        >>> await node.start()
        >>> 
        >>> # Submit a command (only works on leader)
        >>> result = await node.submit_command({"type": "set", "key": "x", "value": 1})
        >>> 
        >>> await node.stop()
    """
    
    def __init__(self, config: RaftConfig) -> None:
        """Initialize the Raft node.
        
        Args:
            config: Raft configuration
        """
        self.config = config
        self.node_id = config.node_id
        
        # Persistent state (on stable storage)
        self._log = RaftLog(config.data_dir)
        
        # Volatile state
        self._role = NodeRole.FOLLOWER
        self._leader_id: Optional[str] = None
        self._commit_index = 0
        self._last_applied = 0
        
        # Leader state (reinitialized after election)
        self._next_index: Dict[str, int] = {}  # peer -> next log index to send
        self._match_index: Dict[str, int] = {}  # peer -> highest replicated index
        
        # Timing
        self._last_heartbeat = time.time()
        self._election_timeout = self._random_election_timeout()
        
        # State machine
        self._state_machine = StateMachine()
        
        # Transport
        self._transport = RaftTransport(
            host=config.bind_host,
            port=config.bind_port,
            timeout_ms=config.rpc_timeout_ms,
        )
        
        # Peer addresses (host:port for Raft RPC)
        self._peers: List[str] = []
        for node_spec in config.cluster_nodes:
            parts = node_spec.split(":")
            if len(parts) >= 2:
                self._peers.append(f"{parts[0]}:{parts[1]}")
            else:
                self._peers.append(f"{node_spec}:4321")
        
        # Background tasks
        self._running = False
        self._election_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._apply_task: Optional[asyncio.Task] = None
        
        # Pending client requests (for leader)
        self._pending_requests: Dict[int, Tuple[asyncio.Future, float]] = {}  # index -> (future, timestamp)
        
        # Lock for state modifications
        self._lock = asyncio.Lock()
        
        # Callbacks
        self._on_leader_change: Optional[Callable[[Optional[str]], None]] = None
        self._on_role_change: Optional[Callable[[NodeRole], None]] = None
        
        # Configuration change state (for joint consensus)
        self._config_state = ConfigurationState.STABLE
        self._old_peers: List[str] = []  # Peers before config change
        self._new_peers: List[str] = []  # Peers after config change
        
        # Pre-vote state
        self._pre_vote_enabled = True
        
        # Leadership transfer state
        self._transfer_target: Optional[str] = None
        self._transferring_leadership = False
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Lease-based read safety
        self._last_successful_heartbeat = time.time()
        self._lease_duration = config.election_timeout_min_ms / 1000.0
    
    # ==================== Properties ====================
    
    @property
    def role(self) -> NodeRole:
        """Get current role."""
        return self._role
    
    @property
    def is_leader(self) -> bool:
        """Check if this node is the leader."""
        return self._role == NodeRole.LEADER
    
    @property
    def leader_id(self) -> Optional[str]:
        """Get the current leader's ID."""
        return self._leader_id
    
    @property
    def current_term(self) -> int:
        """Get current term."""
        return self._log.current_term
    
    @property
    def state_machine(self) -> StateMachine:
        """Get the state machine."""
        return self._state_machine
    
    @property
    def cluster_size(self) -> int:
        """Get total cluster size."""
        return len(self._peers) + 1

    @property
    def quorum_size(self) -> int:
        """Get quorum size (majority)."""
        return (self.cluster_size // 2) + 1

    def _get_quorum_for_config(self, peers: List[str]) -> int:
        """Get quorum size for a specific configuration."""
        size = len(peers) + 1  # +1 for self
        return (size // 2) + 1

    def _check_joint_quorum(self, responding_peers: set) -> bool:
        """Check if we have quorum in joint consensus mode.
        
        In joint consensus, we need majority in BOTH old and new configurations.
        """
        if self._config_state != ConfigurationState.JOINT:
            # Normal quorum check
            votes = len(responding_peers) + 1  # +1 for self
            return votes >= self.quorum_size
        
        # Joint consensus: need majority in both old AND new configs
        old_votes = sum(1 for p in responding_peers if p in self._old_peers) + 1
        new_votes = sum(1 for p in responding_peers if p in self._new_peers) + 1
        
        old_quorum = self._get_quorum_for_config(self._old_peers)
        new_quorum = self._get_quorum_for_config(self._new_peers)
        
        return old_votes >= old_quorum and new_votes >= new_quorum

    def can_serve_read(self) -> bool:
        """Check if this leader can safely serve reads.
        
        Uses lease-based approach to ensure linearizable reads.
        """
        if self._role != NodeRole.LEADER:
            return False
        # Check if we've received heartbeat acks within lease period
        return time.time() - self._last_successful_heartbeat < self._lease_duration

    def set_callbacks(
        self,
        on_leader_change: Optional[Callable[[Optional[str]], None]] = None,
        on_role_change: Optional[Callable[[NodeRole], None]] = None,
    ) -> None:
        """Set callbacks for state changes."""
        self._on_leader_change = on_leader_change
        self._on_role_change = on_role_change

    def update_peers(self, peers: List[str]) -> None:
        """Update the list of cluster peers dynamically.

        This is used for dynamic cluster membership changes via discovery.
        For leader, this triggers a proper configuration change via joint consensus.
        For followers, this updates the peer list directly (will be overwritten by leader).

        Args:
            peers: List of peer addresses in host:raft_port:api_port or host:raft_port format
        """
        new_peers = []
        for node_spec in peers:
            parts = node_spec.split(":")
            if len(parts) >= 2:
                new_peers.append(f"{parts[0]}:{parts[1]}")
            else:
                new_peers.append(f"{node_spec}:4321")

        # Only update if changed
        if set(new_peers) != set(self._peers):
            old_count = len(self._peers)
            
            if self._role == NodeRole.LEADER and self._config_state == ConfigurationState.STABLE:
                # Leader should use joint consensus for config changes
                # Start async task for config change
                asyncio.create_task(self._change_configuration(new_peers))
            else:
                # Followers update directly (leader will sync via log)
                self._peers = new_peers
                logger.info(f"Updated Raft peers: {old_count} -> {len(self._peers)} nodes")

                # Reinitialize leader state for new peers
                if self._role == NodeRole.LEADER:
                    for peer in self._peers:
                        if peer not in self._next_index:
                            self._next_index[peer] = self._log.last_index + 1
                            self._match_index[peer] = 0

    async def _change_configuration(self, new_peers: List[str]) -> None:
        """Change cluster configuration using joint consensus (Raft Section 6).
        
        This implements the two-phase configuration change:
        1. Enter joint consensus (C_old,new) - need majority in both
        2. Commit and transition to C_new
        """
        if self._config_state != ConfigurationState.STABLE:
            logger.warning("Configuration change already in progress")
            return
        
        if self._role != NodeRole.LEADER:
            logger.warning("Only leader can initiate configuration change")
            return
        
        logger.info(f"Starting configuration change: {len(self._peers)} -> {len(new_peers)} peers")
        
        # Phase 1: Enter joint consensus
        self._old_peers = list(self._peers)
        self._new_peers = new_peers
        self._config_state = ConfigurationState.JOINT
        
        # Replicate joint config to both old and new members
        combined_peers = list(set(self._old_peers + self._new_peers))
        self._peers = combined_peers
        
        # Initialize state for any new peers
        for peer in self._peers:
            if peer not in self._next_index:
                self._next_index[peer] = self._log.last_index + 1
                self._match_index[peer] = 0
        
        try:
            # Append joint config entry
            await self.submit_command({
                "type": CommandType.SET_CONFIG.value,
                "key": "_cluster_config",
                "value": {
                    "state": "joint",
                    "old_peers": self._old_peers,
                    "new_peers": self._new_peers,
                },
            }, timeout=10.0)
            
            logger.info("Joint configuration committed, transitioning to new config")
            
            # Phase 2: Transition to new config
            self._config_state = ConfigurationState.TRANSITIONING
            self._peers = new_peers
            
            # Append final config entry
            await self.submit_command({
                "type": CommandType.SET_CONFIG.value,
                "key": "_cluster_config",
                "value": {
                    "state": "stable",
                    "peers": new_peers,
                },
            }, timeout=10.0)
            
            # Configuration change complete
            self._config_state = ConfigurationState.STABLE
            self._old_peers = []
            self._new_peers = []
            
            logger.info(f"Configuration change complete: {len(new_peers)} peers")
            
        except Exception as e:
            # Rollback on failure
            logger.error(f"Configuration change failed: {e}")
            self._peers = self._old_peers
            self._config_state = ConfigurationState.STABLE
            self._old_peers = []
            self._new_peers = []

    def _random_election_timeout(self) -> float:
        """Generate a random election timeout."""
        min_ms = self.config.election_timeout_min_ms
        max_ms = self.config.election_timeout_max_ms
        return random.randint(min_ms, max_ms) / 1000.0

    # ==================== Lifecycle ====================

    async def start(self) -> None:
        """Start the Raft node."""
        if self._running:
            return

        logger.info(f"Starting Raft node {self.node_id}")

        # Set up transport handler
        self._transport.set_handler(self._handle_rpc_sync)
        await self._transport.start()

        self._running = True

        # Start background tasks
        self._election_task = asyncio.create_task(self._election_loop())
        self._apply_task = asyncio.create_task(self._apply_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # If single node, become leader immediately
        if self.cluster_size == 1:
            await self._become_leader()

        logger.info(f"Raft node {self.node_id} started as {self._role.value}")

    async def stop(self) -> None:
        """Stop the Raft node."""
        if not self._running:
            return

        logger.info(f"Stopping Raft node {self.node_id}")
        
        # Transfer leadership if we're the leader
        if self._role == NodeRole.LEADER and self._peers:
            try:
                await self._transfer_leadership()
            except Exception as e:
                logger.warning(f"Leadership transfer failed during shutdown: {e}")
        
        self._running = False

        # Cancel background tasks
        for task in [self._election_task, self._heartbeat_task, self._apply_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        await self._transport.stop()
        logger.info(f"Raft node {self.node_id} stopped")

    async def _cleanup_loop(self) -> None:
        """Background loop to clean up stale pending requests."""
        while self._running:
            try:
                await asyncio.sleep(10.0)  # Check every 10 seconds
                await self._cleanup_pending_requests()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _cleanup_pending_requests(self) -> None:
        """Clean up stale pending requests to prevent memory leaks."""
        now = time.time()
        stale_timeout = 60.0  # 1 minute
        
        stale_indices = []
        for index, (future, timestamp) in list(self._pending_requests.items()):
            # Clean up if:
            # 1. Request is old AND index has been applied
            # 2. Request is very old (> 2x timeout)
            if index <= self._last_applied:
                stale_indices.append(index)
            elif now - timestamp > stale_timeout * 2:
                stale_indices.append(index)
                if not future.done():
                    future.set_exception(CommandTimeoutError(
                        "Request expired",
                        index=index,
                        timeout=stale_timeout * 2,
                    ))
        
        for index in stale_indices:
            self._pending_requests.pop(index, None)
        
        if stale_indices:
            logger.debug(f"Cleaned up {len(stale_indices)} stale pending requests")

    # ==================== RPC Handlers ====================

    def _handle_rpc_sync(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous RPC handler (called from transport)."""
        msg_type = MessageType(message.get("message_type", ""))

        if msg_type == MessageType.REQUEST_VOTE:
            request = RequestVoteRequest.from_dict(message)
            response = self._handle_request_vote(request)
            return response.to_dict()

        elif msg_type == MessageType.PRE_VOTE:
            request = RequestVoteRequest.from_dict(message)
            response = self._handle_pre_vote(request)
            return response.to_dict()

        elif msg_type == MessageType.APPEND_ENTRIES:
            request = AppendEntriesRequest.from_dict(message)
            response = self._handle_append_entries(request)
            return response.to_dict()

        elif msg_type == MessageType.TIMEOUT_NOW:
            request = TimeoutNowRequest.from_dict(message)
            response = self._handle_timeout_now(request)
            return response.to_dict()

        else:
            return {"error": f"Unknown message type: {msg_type}"}

    def _handle_pre_vote(self, request: RequestVoteRequest) -> RequestVoteResponse:
        """Handle Pre-Vote RPC (Raft thesis Section 9.6).
        
        Pre-vote doesn't change state - it's just a check if election would succeed.
        This prevents disruptive elections from partitioned nodes.
        """
        current_term = self._log.current_term
        
        # For pre-vote, we check if we WOULD vote, without actually voting
        # Key difference: we don't update term or voted_for
        
        # Reply false if term < currentTerm (pre-vote uses +1 term)
        if request.term < current_term:
            return RequestVoteResponse(
                message_type=MessageType.PRE_VOTE_RESPONSE,
                term=current_term,
                sender_id=self.node_id,
                vote_granted=False,
                is_pre_vote=True,
            )
        
        # Check if we've heard from a leader recently
        # If so, don't grant pre-vote (prevents disruption)
        if time.time() - self._last_heartbeat < self._election_timeout:
            return RequestVoteResponse(
                message_type=MessageType.PRE_VOTE_RESPONSE,
                term=current_term,
                sender_id=self.node_id,
                vote_granted=False,
                is_pre_vote=True,
            )
        
        # Check if candidate's log is at least as up-to-date
        last_log_term = self._log.last_term
        last_log_index = self._log.last_index
        
        log_ok = (
            request.last_log_term > last_log_term or
            (request.last_log_term == last_log_term and request.last_log_index >= last_log_index)
        )
        
        return RequestVoteResponse(
            message_type=MessageType.PRE_VOTE_RESPONSE,
            term=current_term,
            sender_id=self.node_id,
            vote_granted=log_ok,
            is_pre_vote=True,
        )

    def _handle_timeout_now(self, request: TimeoutNowRequest) -> TimeoutNowResponse:
        """Handle TimeoutNow RPC for leadership transfer.
        
        When we receive this, we should immediately start an election.
        """
        current_term = self._log.current_term
        
        if request.term < current_term:
            return TimeoutNowResponse(
                message_type=MessageType.TIMEOUT_NOW_RESPONSE,
                term=current_term,
                sender_id=self.node_id,
                success=False,
            )
        
        # Start election immediately (skip pre-vote for transfers)
        logger.info(f"Received TimeoutNow from {request.sender_id}, starting immediate election")
        
        # Schedule election on the event loop
        asyncio.create_task(self._immediate_election())
        
        return TimeoutNowResponse(
            message_type=MessageType.TIMEOUT_NOW_RESPONSE,
            term=current_term,
            sender_id=self.node_id,
            success=True,
        )

    async def _immediate_election(self) -> None:
        """Start an immediate election (for leadership transfer)."""
        # Skip pre-vote for immediate elections
        self._pre_vote_enabled = False
        await self._become_candidate()
        self._pre_vote_enabled = True

    def _handle_request_vote(self, request: RequestVoteRequest) -> RequestVoteResponse:
        """Handle RequestVote RPC (Section 5.2).

        1. Reply false if term < currentTerm
        2. If votedFor is null or candidateId, and candidate's log is at
           least as up-to-date as receiver's log, grant vote
        """
        current_term = self._log.current_term

        # Rule 1: Reply false if term < currentTerm
        if request.term < current_term:
            return RequestVoteResponse(
                message_type=MessageType.REQUEST_VOTE_RESPONSE,
                term=current_term,
                sender_id=self.node_id,
                vote_granted=False,
            )

        # If RPC request contains term > currentTerm, update and convert to follower
        if request.term > current_term:
            self._log.set_term_and_vote(request.term, None)
            self._become_follower(request.term)

        # Rule 2: Check if we can vote for this candidate
        voted_for = self._log.voted_for
        can_vote = voted_for is None or voted_for == request.sender_id

        # Check if candidate's log is at least as up-to-date
        last_log_term = self._log.last_term
        last_log_index = self._log.last_index

        log_ok = (
            request.last_log_term > last_log_term or
            (request.last_log_term == last_log_term and request.last_log_index >= last_log_index)
        )

        if can_vote and log_ok:
            # Grant vote
            self._log.voted_for = request.sender_id
            self._last_heartbeat = time.time()  # Reset election timeout

            logger.debug(f"Granted vote to {request.sender_id} for term {request.term}")

            return RequestVoteResponse(
                message_type=MessageType.REQUEST_VOTE_RESPONSE,
                term=self._log.current_term,
                sender_id=self.node_id,
                vote_granted=True,
            )

        return RequestVoteResponse(
            message_type=MessageType.REQUEST_VOTE_RESPONSE,
            term=self._log.current_term,
            sender_id=self.node_id,
            vote_granted=False,
        )

    def _handle_append_entries(self, request: AppendEntriesRequest) -> AppendEntriesResponse:
        """Handle AppendEntries RPC (Section 5.3).

        1. Reply false if term < currentTerm
        2. Reply false if log doesn't contain entry at prevLogIndex with prevLogTerm
        3. If existing entry conflicts with new one, delete it and all following
        4. Append any new entries not already in the log
        5. If leaderCommit > commitIndex, set commitIndex = min(leaderCommit, last new entry)
        """
        current_term = self._log.current_term

        # Rule 1: Reply false if term < currentTerm
        if request.term < current_term:
            return AppendEntriesResponse(
                message_type=MessageType.APPEND_ENTRIES_RESPONSE,
                term=current_term,
                sender_id=self.node_id,
                success=False,
            )

        # Valid leader heartbeat - reset election timeout
        self._last_heartbeat = time.time()

        # Update term if needed
        if request.term > current_term:
            self._log.set_term_and_vote(request.term, None)

        # Convert to follower if not already
        if self._role != NodeRole.FOLLOWER:
            self._become_follower(request.term)

        # Update leader ID
        if self._leader_id != request.sender_id:
            self._leader_id = request.sender_id
            if self._on_leader_change:
                self._on_leader_change(self._leader_id)

        # Rule 2: Check log consistency
        if not self._log.matches(request.prev_log_index, request.prev_log_term):
            # Find conflict info for optimization
            conflict_index, conflict_term = self._log.find_conflict(
                request.prev_log_index, request.prev_log_term
            )
            return AppendEntriesResponse(
                message_type=MessageType.APPEND_ENTRIES_RESPONSE,
                term=self._log.current_term,
                sender_id=self.node_id,
                success=False,
                conflict_index=conflict_index,
                conflict_term=conflict_term,
            )

        # Rules 3 & 4: Process entries
        if request.entries:
            for entry_data in request.entries:
                entry = LogEntry.from_dict(entry_data)
                existing = self._log.get(entry.index)

                if existing is None:
                    # Append new entry
                    self._log.append(entry)
                elif existing.term != entry.term:
                    # Conflict - truncate and append
                    self._log.truncate_after(entry.index - 1)
                    self._log.append(entry)
                # else: entry already exists with same term, skip

        # Rule 5: Update commit index
        if request.leader_commit > self._commit_index:
            self._commit_index = min(request.leader_commit, self._log.last_index)
            self._log.commit_index = self._commit_index

        return AppendEntriesResponse(
            message_type=MessageType.APPEND_ENTRIES_RESPONSE,
            term=self._log.current_term,
            sender_id=self.node_id,
            success=True,
            match_index=self._log.last_index,
        )

    # ==================== Role Transitions ====================

    def _become_follower(self, term: int) -> None:
        """Transition to follower state."""
        old_role = self._role
        self._role = NodeRole.FOLLOWER

        # Cancel heartbeat task if we were leader
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

        if old_role != NodeRole.FOLLOWER:
            logger.info(f"Node {self.node_id} became follower in term {term}")
            if self._on_role_change:
                self._on_role_change(NodeRole.FOLLOWER)

    async def _become_candidate(self) -> None:
        """Transition to candidate state and start election."""
        # First, run pre-vote if enabled (prevents disruptive elections)
        if self._pre_vote_enabled and self._peers:
            if not await self._run_pre_vote():
                # Pre-vote failed, don't start real election
                self._last_heartbeat = time.time()
                self._election_timeout = self._random_election_timeout()
                return
        
        self._role = NodeRole.CANDIDATE
        self._leader_id = None

        # Increment term and vote for self
        new_term = self._log.current_term + 1
        self._log.set_term_and_vote(new_term, self.node_id)

        logger.info(f"Node {self.node_id} became candidate for term {new_term}")

        if self._on_role_change:
            self._on_role_change(NodeRole.CANDIDATE)

        # Start election
        await self._run_election()

    async def _run_pre_vote(self) -> bool:
        """Run pre-vote phase before real election (Raft thesis Section 9.6).
        
        Returns True if pre-vote succeeds (we should start real election).
        Returns False if pre-vote fails (we should not disrupt cluster).
        """
        if not self._peers:
            return True  # Single node, skip pre-vote
        
        # Pre-vote request uses term+1 but doesn't actually increment term
        request = RequestVoteRequest(
            message_type=MessageType.PRE_VOTE,
            term=self._log.current_term + 1,
            sender_id=self.node_id,
            last_log_index=self._log.last_index,
            last_log_term=self._log.last_term,
            is_pre_vote=True,
        )
        
        # Send to all peers
        requests = {peer: request for peer in self._peers}
        responses = await self._transport.send_request_vote_batch(requests)
        
        # Count pre-votes (we vote for ourselves)
        votes = 1
        responding_peers = set()
        
        for peer, response in responses.items():
            if response is None:
                continue
            
            responding_peers.add(peer)
            if response.vote_granted:
                votes += 1
        
        # Check if we have quorum (considering joint consensus)
        if self._config_state == ConfigurationState.JOINT:
            success = self._check_joint_quorum(responding_peers)
        else:
            success = votes >= self.quorum_size
        
        if success:
            logger.debug(f"Pre-vote succeeded with {votes} votes")
        else:
            logger.debug(f"Pre-vote failed with {votes} votes (need {self.quorum_size})")
        
        return success

    async def _become_leader(self) -> None:
        """Transition to leader state."""
        self._role = NodeRole.LEADER
        self._leader_id = self.node_id

        logger.info(f"Node {self.node_id} became leader for term {self._log.current_term}")

        # Initialize leader state
        next_index = self._log.last_index + 1
        for peer in self._peers:
            self._next_index[peer] = next_index
            self._match_index[peer] = 0

        # Append no-op entry to commit entries from previous terms
        noop_entry = LogEntry(
            term=self._log.current_term,
            index=self._log.last_index + 1,
            command={"type": "noop"},
        )
        self._log.append(noop_entry)

        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        if self._on_role_change:
            self._on_role_change(NodeRole.LEADER)

        if self._on_leader_change:
            self._on_leader_change(self.node_id)
        
        # Reset lease timer
        self._last_successful_heartbeat = time.time()

    # ==================== Leadership Transfer ====================

    async def _transfer_leadership(self, target_id: Optional[str] = None) -> bool:
        """Transfer leadership to another node.
        
        Args:
            target_id: Specific node to transfer to, or None for best candidate
            
        Returns:
            True if transfer was initiated successfully
        """
        if self._role != NodeRole.LEADER:
            logger.warning("Cannot transfer leadership: not leader")
            return False
        
        if not self._peers:
            logger.warning("Cannot transfer leadership: no peers")
            return False
        
        if self._transferring_leadership:
            logger.warning("Leadership transfer already in progress")
            return False
        
        self._transferring_leadership = True
        
        try:
            # Select target: either specified or most up-to-date follower
            if target_id:
                target_peer = None
                for peer in self._peers:
                    if target_id in peer:
                        target_peer = peer
                        break
                if not target_peer:
                    logger.warning(f"Transfer target {target_id} not found")
                    return False
            else:
                # Find most up-to-date peer
                best_peer = None
                best_match = -1
                for peer, match_idx in self._match_index.items():
                    if match_idx > best_match:
                        best_match = match_idx
                        best_peer = peer
                target_peer = best_peer
            
            if not target_peer:
                logger.warning("No suitable transfer target")
                return False
            
            logger.info(f"Transferring leadership to {target_peer}")
            
            # First, make sure target is up to date
            # Send AppendEntries to catch them up
            for _ in range(3):  # Try a few times
                await self._send_append_entries_to_all()
                await asyncio.sleep(0.1)
                if self._match_index.get(target_peer, 0) >= self._log.last_index:
                    break
            
            # Send TimeoutNow to trigger immediate election
            request = TimeoutNowRequest(
                message_type=MessageType.TIMEOUT_NOW,
                term=self._log.current_term,
                sender_id=self.node_id,
            )
            
            response = await self._transport._send_rpc(target_peer, request.to_dict())
            
            if response and response.get("success"):
                logger.info(f"Leadership transfer to {target_peer} initiated")
                # We'll step down when we receive AppendEntries from new leader
                return True
            else:
                logger.warning(f"Leadership transfer to {target_peer} failed")
                return False
                
        finally:
            self._transferring_leadership = False

    async def transfer_leadership(self, target_id: Optional[str] = None) -> bool:
        """Public API for leadership transfer."""
        return await self._transfer_leadership(target_id)

    # ==================== Election ====================

    async def _election_loop(self) -> None:
        """Background loop that triggers elections on timeout."""
        while self._running:
            try:
                await asyncio.sleep(0.05)  # Check every 50ms

                if self._role == NodeRole.LEADER:
                    continue  # Leaders don't need election timeout

                elapsed = time.time() - self._last_heartbeat
                if elapsed >= self._election_timeout:
                    # Election timeout - start election
                    self._election_timeout = self._random_election_timeout()
                    await self._become_candidate()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Election loop error: {e}")
                await asyncio.sleep(0.1)

    async def _run_election(self) -> None:
        """Run a leader election."""
        if not self._peers:
            # Single node cluster - become leader immediately
            await self._become_leader()
            return

        # Prepare RequestVote request
        request = RequestVoteRequest(
            message_type=MessageType.REQUEST_VOTE,
            term=self._log.current_term,
            sender_id=self.node_id,
            last_log_index=self._log.last_index,
            last_log_term=self._log.last_term,
        )

        # Send to all peers
        requests = {peer: request for peer in self._peers}
        responses = await self._transport.send_request_vote_batch(requests)

        # Count votes (we vote for ourselves)
        votes = 1

        for peer, response in responses.items():
            if response is None:
                continue

            # Check if we should step down
            if response.term > self._log.current_term:
                self._log.set_term_and_vote(response.term, None)
                self._become_follower(response.term)
                return

            if response.vote_granted:
                votes += 1

        # Check if we won
        if self._role == NodeRole.CANDIDATE and votes >= self.quorum_size:
            await self._become_leader()
        else:
            # Election failed, reset timeout and try again
            self._election_timeout = self._random_election_timeout()
            self._last_heartbeat = time.time()

    # ==================== Heartbeat / Log Replication ====================

    async def _heartbeat_loop(self) -> None:
        """Background loop that sends heartbeats/AppendEntries as leader."""
        while self._running and self._role == NodeRole.LEADER:
            try:
                await self._send_append_entries_to_all()
                await asyncio.sleep(self.config.heartbeat_interval_ms / 1000.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(0.1)

    async def _send_append_entries_to_all(self) -> None:
        """Send AppendEntries to all peers."""
        if not self._peers:
            return

        requests = {}
        for peer in self._peers:
            next_idx = self._next_index.get(peer, self._log.last_index + 1)
            prev_log_index, prev_log_term, entries = self._log.get_entries_for_follower(
                next_idx, self.config.max_entries_per_request
            )

            request = AppendEntriesRequest(
                message_type=MessageType.APPEND_ENTRIES,
                term=self._log.current_term,
                sender_id=self.node_id,
                prev_log_index=prev_log_index,
                prev_log_term=prev_log_term,
                entries=[e.to_dict() for e in entries],
                leader_commit=self._commit_index,
            )
            requests[peer] = request

        responses = await self._transport.send_append_entries_batch(requests)

        successful_responses = 0
        for peer, response in responses.items():
            if response is None:
                continue

            # Check if we should step down
            if response.term > self._log.current_term:
                self._log.set_term_and_vote(response.term, None)
                self._become_follower(response.term)
                return

            if response.success:
                # Update next_index and match_index
                self._match_index[peer] = response.match_index
                self._next_index[peer] = response.match_index + 1
                successful_responses += 1
            else:
                # Decrement next_index and retry
                if response.conflict_index is not None:
                    self._next_index[peer] = response.conflict_index
                else:
                    self._next_index[peer] = max(1, self._next_index.get(peer, 1) - 1)

        # Update lease timer if we got quorum responses
        if successful_responses + 1 >= self.quorum_size:  # +1 for self
            self._last_successful_heartbeat = time.time()

        # Update commit index
        self._update_commit_index()

    def _update_commit_index(self) -> None:
        """Update commit index based on match_index values."""
        if self._role != NodeRole.LEADER:
            return

        # Find the highest index replicated on a majority
        match_indices = list(self._match_index.values()) + [self._log.last_index]
        match_indices.sort(reverse=True)

        for n in range(self._log.last_index, self._commit_index, -1):
            # Count how many nodes have this entry
            count = sum(1 for idx in match_indices if idx >= n)
            count += 1  # Include self

            # Check if majority and entry is from current term
            if count >= self.quorum_size:
                entry = self._log.get(n)
                if entry and entry.term == self._log.current_term:
                    self._commit_index = n
                    self._log.commit_index = n
                    break

    # ==================== Apply Loop ====================

    async def _apply_loop(self) -> None:
        """Background loop that applies committed entries to state machine."""
        while self._running:
            try:
                await asyncio.sleep(0.01)  # Check every 10ms

                while self._last_applied < self._commit_index:
                    self._last_applied += 1
                    entry = self._log.get(self._last_applied)

                    if entry:
                        # Apply with timeout to prevent blocking
                        try:
                            result = await asyncio.wait_for(
                                asyncio.to_thread(
                                    self._state_machine.apply, entry.command, entry.index
                                ),
                                timeout=5.0
                            )
                        except asyncio.TimeoutError:
                            logger.error(f"State machine apply timeout for index {entry.index}")
                            result = None

                        # Resolve pending request if we're leader
                        if self._last_applied in self._pending_requests:
                            future, _ = self._pending_requests.pop(self._last_applied)
                            if not future.done():
                                future.set_result(result)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Apply loop error: {e}")
                await asyncio.sleep(0.1)

    # ==================== Client API ====================

    async def submit_command(
        self,
        command: Dict[str, Any],
        timeout: float = 5.0,
    ) -> Any:
        """Submit a command to be replicated and applied.

        Args:
            command: The command to apply to the state machine
            timeout: Timeout in seconds

        Returns:
            Result from the state machine

        Raises:
            NotLeaderError: If not leader
            CommandTimeoutError: If command times out
            ClusterConfigurationError: If cluster is in configuration transition
        """
        if self._role != NodeRole.LEADER:
            raise NotLeaderError(
                leader_id=self._leader_id,
                leader_address=self.get_leader_address(),
            )
        
        # Check if we're in a config transition that blocks writes
        if self._config_state == ConfigurationState.TRANSITIONING:
            raise ClusterConfigurationError(
                "Cluster configuration transition in progress",
                in_transition=True,
            )

        async with self._lock:
            # Create log entry
            entry = LogEntry(
                term=self._log.current_term,
                index=self._log.last_index + 1,
                command=command,
            )

            # Append to log
            self._log.append(entry)

            # Create future for result with timestamp
            future: asyncio.Future = asyncio.get_event_loop().create_future()
            self._pending_requests[entry.index] = (future, time.time())

        try:
            # Wait for commit and apply
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            # Remove pending request
            self._pending_requests.pop(entry.index, None)
            raise CommandTimeoutError(
                "Command timed out waiting for commit",
                index=entry.index,
                timeout=timeout,
            )

    def get_leader_address(self) -> Optional[str]:
        """Get the current leader's API address.

        Returns:
            Leader's API address or None if unknown
        """
        if self._leader_id == self.node_id:
            return f"{self.config.api_host}:{self.config.api_port}"

        # Find leader in peers
        for node_spec in self.config.cluster_nodes:
            parts = node_spec.split(":")
            # Assuming format: host:raft_port:api_port or host:raft_port
            if len(parts) >= 3:
                return f"{parts[0]}:{parts[2]}"
            elif len(parts) == 2:
                # Assume API port is raft_port - 321
                return f"{parts[0]}:{int(parts[1]) - 321}"

        return None

    def get_status(self) -> Dict[str, Any]:
        """Get node status."""
        return {
            "node_id": self.node_id,
            "role": self._role.value,
            "term": self._log.current_term,
            "leader_id": self._leader_id,
            "commit_index": self._commit_index,
            "last_applied": self._last_applied,
            "log_length": len(self._log),
            "cluster_size": self.cluster_size,
            "peers": self._peers,
            "config_state": self._config_state.value,
            "can_serve_read": self.can_serve_read(),
            "pre_vote_enabled": self._pre_vote_enabled,
        }
