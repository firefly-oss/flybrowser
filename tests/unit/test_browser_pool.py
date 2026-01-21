# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BrowserPool and related classes."""

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flybrowser.core.browser_pool import (
    BrowserPool,
    BrowserSession,
    BrowserSessionState,
    Job,
    JobQueue,
    JobState,
    PoolConfig,
)
from flybrowser.exceptions import BrowserError


class TestPoolConfig:
    """Tests for PoolConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PoolConfig()
        
        assert config.min_size == 1
        assert config.max_size == 10
        assert config.idle_timeout_seconds == 300.0
        assert config.max_session_age_seconds == 3600.0
        assert config.startup_timeout_seconds == 30.0
        assert config.shutdown_timeout_seconds == 10.0
        assert config.health_check_interval_seconds == 60.0
        assert config.headless is True
        assert config.browser_type == "chromium"
        assert config.browser_options == {}

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = PoolConfig(
            min_size=2,
            max_size=20,
            headless=False,
            browser_type="firefox"
        )
        
        assert config.min_size == 2
        assert config.max_size == 20
        assert config.headless is False
        assert config.browser_type == "firefox"

    def test_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(os.environ, {
            "FLYBROWSER_POOL_MIN_SIZE": "3",
            "FLYBROWSER_POOL_MAX_SIZE": "15",
            "FLYBROWSER_POOL_HEADLESS": "false",
            "FLYBROWSER_POOL_BROWSER_TYPE": "webkit"
        }):
            config = PoolConfig.from_env()
            
            assert config.min_size == 3
            assert config.max_size == 15
            assert config.headless is False
            assert config.browser_type == "webkit"


class TestBrowserSession:
    """Tests for BrowserSession."""

    def test_default_state(self):
        """Test default session state."""
        manager = MagicMock()
        session = BrowserSession(id="test-123", manager=manager)
        
        assert session.id == "test-123"
        assert session.state == BrowserSessionState.IDLE
        assert session.job_count == 0
        assert session.error_count == 0
        assert session.current_job_id is None

    def test_is_expired(self):
        """Test session expiration check."""
        manager = MagicMock()
        session = BrowserSession(id="test-123", manager=manager)
        session.created_at = time.time() - 4000  # 4000 seconds ago
        
        assert session.is_expired(3600) is True  # Expired (>1 hour)
        assert session.is_expired(5000) is False  # Not expired

    def test_is_idle_timeout(self):
        """Test session idle timeout check."""
        manager = MagicMock()
        session = BrowserSession(id="test-123", manager=manager)
        session.state = BrowserSessionState.IDLE
        session.last_used_at = time.time() - 400  # 400 seconds ago
        
        assert session.is_idle_timeout(300) is True  # Timed out (>5 min)
        assert session.is_idle_timeout(500) is False  # Not timed out

    def test_is_idle_timeout_not_idle(self):
        """Test idle timeout returns False when not idle."""
        manager = MagicMock()
        session = BrowserSession(id="test-123", manager=manager)
        session.state = BrowserSessionState.BUSY
        session.last_used_at = time.time() - 400
        
        assert session.is_idle_timeout(300) is False  # Not idle


class TestJob:
    """Tests for Job dataclass."""

    def test_default_values(self):
        """Test default job values."""
        job = Job()
        
        assert job.id is not None
        assert job.func is None
        assert job.args == ()
        assert job.kwargs == {}
        assert job.state == JobState.PENDING
        assert job.priority == 0
        assert job.timeout_seconds == 300.0

    def test_custom_values(self):
        """Test job with custom values."""
        func = AsyncMock()
        job = Job(
            func=func,
            args=(1, 2),
            kwargs={"key": "value"},
            priority=5,
            timeout_seconds=60.0
        )
        
        assert job.func is func
        assert job.args == (1, 2)
        assert job.kwargs == {"key": "value"}
        assert job.priority == 5
        assert job.timeout_seconds == 60.0


class TestJobQueue:
    """Tests for JobQueue."""

    @pytest.mark.asyncio
    async def test_put_and_get(self):
        """Test adding and retrieving jobs."""
        queue = JobQueue()
        job = Job(func=AsyncMock())
        
        job_id = await queue.put(job)
        
        assert job_id == job.id
        assert queue.size() == 1
        
        retrieved = await queue.get()
        assert retrieved is job

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test jobs are retrieved by priority."""
        queue = JobQueue()
        
        low_priority = Job(func=AsyncMock(), priority=1)
        high_priority = Job(func=AsyncMock(), priority=10)
        
        await queue.put(low_priority)
        await queue.put(high_priority)
        
        # High priority should come first
        first = await queue.get()
        assert first is high_priority
        
        second = await queue.get()
        assert second is low_priority

    @pytest.mark.asyncio
    async def test_get_job_by_id(self):
        """Test getting job by ID."""
        queue = JobQueue()
        job = Job(func=AsyncMock())
        await queue.put(job)
        
        retrieved = queue.get_job(job.id)
        assert retrieved is job
        
        # Non-existent job
        assert queue.get_job("non-existent") is None

    @pytest.mark.asyncio
    async def test_cancel_job(self):
        """Test cancelling a job."""
        queue = JobQueue()
        job = Job(func=AsyncMock())
        await queue.put(job)
        
        result = await queue.cancel(job.id)
        
        assert result is True
        assert job.state == JobState.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_non_pending_job(self):
        """Test cancelling a non-pending job returns False."""
        queue = JobQueue()
        job = Job(func=AsyncMock())
        job.state = JobState.RUNNING
        queue._jobs[job.id] = job
        
        result = await queue.cancel(job.id)
        assert result is False

    def test_get_stats(self):
        """Test getting queue statistics."""
        queue = JobQueue()
        
        # Add jobs in various states
        pending = Job(func=AsyncMock())
        running = Job(func=AsyncMock())
        running.state = JobState.RUNNING
        completed = Job(func=AsyncMock())
        completed.state = JobState.COMPLETED
        
        queue._jobs[pending.id] = pending
        queue._jobs[running.id] = running
        queue._jobs[completed.id] = completed
        
        stats = queue.get_stats()
        
        assert stats["pending"] == 1
        assert stats["running"] == 1
        assert stats["completed"] == 1
        assert stats["total"] == 3


class TestBrowserPool:
    """Tests for BrowserPool."""

    def test_init_default_config(self):
        """Test pool initialization with default config."""
        pool = BrowserPool()
        
        assert pool.config.min_size == 1
        assert pool.config.max_size == 10
        assert pool._running is False

    def test_init_custom_config(self):
        """Test pool initialization with custom config."""
        config = PoolConfig(min_size=3, max_size=15)
        pool = BrowserPool(config)
        
        assert pool.config.min_size == 3
        assert pool.config.max_size == 15

    @pytest.mark.asyncio
    async def test_start_creates_min_sessions(self):
        """Test start creates minimum number of sessions."""
        config = PoolConfig(min_size=2, max_size=5)
        pool = BrowserPool(config)
        
        with patch.object(pool, "_create_session", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MagicMock()
            
            await pool.start()
            
            # Should create min_size sessions
            assert mock_create.await_count == 2
            
            await pool.stop()

    @pytest.mark.asyncio
    async def test_stop_destroys_all_sessions(self):
        """Test stop destroys all sessions."""
        pool = BrowserPool()
        pool._running = True
        
        # Add mock sessions
        session1 = MagicMock()
        session1.id = "session-1"
        session2 = MagicMock()
        session2.id = "session-2"
        pool._sessions = {"session-1": session1, "session-2": session2}
        
        with patch.object(pool, "_destroy_session", new_callable=AsyncMock) as mock_destroy:
            await pool.stop()
            
            assert pool._running is False
            assert mock_destroy.await_count == 2

    @pytest.mark.asyncio
    async def test_submit_job(self):
        """Test submitting a job to the pool."""
        pool = BrowserPool()
        pool._running = True
        
        async def test_func(browser):
            return "result"
        
        with patch.object(pool, "_maybe_scale_up", new_callable=AsyncMock):
            job_id = await pool.submit(test_func, priority=5)
            
            assert job_id is not None
            job = pool._job_queue.get_job(job_id)
            assert job.priority == 5

    @pytest.mark.asyncio
    async def test_cancel_job(self):
        """Test cancelling a job."""
        pool = BrowserPool()
        pool._running = True
        
        async def test_func(browser):
            return "result"
        
        with patch.object(pool, "_maybe_scale_up", new_callable=AsyncMock):
            job_id = await pool.submit(test_func)
            
            result = await pool.cancel_job(job_id)
            assert result is True

    def test_get_stats(self):
        """Test getting pool statistics."""
        config = PoolConfig(min_size=1, max_size=5)
        pool = BrowserPool(config)
        
        # Add mock sessions
        idle_session = MagicMock()
        idle_session.state = BrowserSessionState.IDLE
        busy_session = MagicMock()
        busy_session.state = BrowserSessionState.BUSY
        
        pool._sessions = {"idle": idle_session, "busy": busy_session}
        
        stats = pool.get_stats()
        
        assert stats["total_sessions"] == 2
        assert stats["idle_sessions"] == 1
        assert stats["busy_sessions"] == 1
        assert stats["min_size"] == 1
        assert stats["max_size"] == 5

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        pool = BrowserPool()
        
        with patch.object(pool, "start", new_callable=AsyncMock) as mock_start:
            with patch.object(pool, "stop", new_callable=AsyncMock) as mock_stop:
                async with pool:
                    mock_start.assert_awaited_once()
                
                mock_stop.assert_awaited_once()


class TestBrowserSessionState:
    """Tests for BrowserSessionState enum."""

    def test_states(self):
        """Test all session states exist."""
        assert BrowserSessionState.IDLE == "idle"
        assert BrowserSessionState.BUSY == "busy"
        assert BrowserSessionState.STARTING == "starting"
        assert BrowserSessionState.STOPPING == "stopping"
        assert BrowserSessionState.ERROR == "error"


class TestJobState:
    """Tests for JobState enum."""

    def test_states(self):
        """Test all job states exist."""
        assert JobState.PENDING == "pending"
        assert JobState.RUNNING == "running"
        assert JobState.COMPLETED == "completed"
        assert JobState.FAILED == "failed"
        assert JobState.CANCELLED == "cancelled"
