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

"""Session manager for handling multiple browser sessions."""

import asyncio
import time
import uuid
from typing import Any, Dict, Optional

from flybrowser.sdk import FlyBrowser
from flybrowser.utils.logger import logger


class SessionManager:
    """Manages multiple browser sessions."""

    def __init__(self, max_sessions: int = 100, session_timeout: int = 3600):
        """
        Initialize the session manager.

        Args:
            max_sessions: Maximum number of concurrent sessions
            session_timeout: Session timeout in seconds
        """
        self.sessions: Dict[str, FlyBrowser] = {}
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self._cleanup_task: Optional[asyncio.Task] = None
        self._total_requests = 0

        # Start cleanup task
        self._start_cleanup_task()

    def _start_cleanup_task(self) -> None:
        """Start the background cleanup task."""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_expired_sessions()

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        now = time.time()
        expired = []

        for session_id, metadata in self.session_metadata.items():
            if now - metadata["last_activity"] > self.session_timeout:
                expired.append(session_id)

        for session_id in expired:
            logger.info(f"Cleaning up expired session: {session_id}")
            await self.delete_session(session_id)

    async def create_session(
        self,
        llm_provider: str,
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        headless: bool = True,
        browser_type: str = "chromium",
        session_id: Optional[str] = None,
    ) -> str:
        """
        Create a new browser session.

        Args:
            llm_provider: LLM provider name
            llm_model: LLM model name
            api_key: API key for LLM provider
            headless: Run browser in headless mode
            browser_type: Browser type
            session_id: Optional session ID (generated if not provided)

        Returns:
            Session ID

        Raises:
            RuntimeError: If max sessions reached
        """
        if len(self.sessions) >= self.max_sessions:
            raise RuntimeError(f"Maximum sessions ({self.max_sessions}) reached")

        # Use provided session_id or generate a new one
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Create FlyBrowser instance
        browser = FlyBrowser(
            llm_provider=llm_provider,
            llm_model=llm_model,
            api_key=api_key,
            headless=headless,
            browser_type=browser_type,
        )

        # Start the browser
        await browser.start()

        # Store session
        self.sessions[session_id] = browser
        self.session_metadata[session_id] = {
            "created_at": time.time(),
            "last_activity": time.time(),
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "browser_type": browser_type,
        }

        logger.info(f"Created session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> FlyBrowser:
        """
        Get a browser session.

        Args:
            session_id: Session ID

        Returns:
            FlyBrowser instance

        Raises:
            KeyError: If session not found
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session not found: {session_id}")

        # Update last activity
        self.session_metadata[session_id]["last_activity"] = time.time()
        self._total_requests += 1

        return self.sessions[session_id]

    async def delete_session(self, session_id: str) -> None:
        """
        Delete a browser session.

        Args:
            session_id: Session ID

        Raises:
            KeyError: If session not found
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session not found: {session_id}")

        browser = self.sessions[session_id]
        
        try:
            await browser.stop()
        except Exception as e:
            logger.error(f"Error stopping browser for session {session_id}: {e}")

        del self.sessions[session_id]
        del self.session_metadata[session_id]

        logger.info(f"Deleted session: {session_id}")

    async def cleanup_all(self) -> None:
        """Clean up all sessions."""
        logger.info("Cleaning up all sessions...")
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Stop all browsers
        for session_id in list(self.sessions.keys()):
            try:
                await self.delete_session(session_id)
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")

        logger.info("All sessions cleaned up")

    def get_active_session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self.sessions)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get session manager statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "active_sessions": len(self.sessions),
            "max_sessions": self.max_sessions,
            "total_requests": self._total_requests,
            "session_timeout": self.session_timeout,
        }

