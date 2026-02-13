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

"""Authentication and authorization for the API service.

Provides both API key and JWT-based RBAC authentication. The RBACAuthManager
wraps fireflyframework-genai's RBACManager to provide JWT tokens with
admin/operator/viewer roles, while maintaining backward compatibility with
the existing API key validation flow.
"""

import logging
import secrets
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from fireflyframework_genai.security.rbac import RBACManager
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Bearer token authentication
bearer_scheme = HTTPBearer(auto_error=False)


class APIKey(BaseModel):
    """API Key model."""

    key: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    rate_limit: Optional[int] = None
    enabled: bool = True


class APIKeyManager:
    """Manages API keys for authentication."""

    def __init__(self):
        """Initialize the API key manager."""
        self.keys: dict[str, APIKey] = {}
        
        # Create a default API key for development
        default_key = "flybrowser_dev_" + secrets.token_urlsafe(32)
        self.keys[default_key] = APIKey(
            key=default_key,
            name="Development Key",
            created_at=datetime.now(),
            enabled=True,
        )

    def create_key(
        self,
        name: str,
        expires_in_days: Optional[int] = None,
        rate_limit: Optional[int] = None,
    ) -> APIKey:
        """
        Create a new API key.

        Args:
            name: Key name/description
            expires_in_days: Days until expiration (None for no expiration)
            rate_limit: Rate limit for this key

        Returns:
            APIKey instance
        """
        key = "flybrowser_" + secrets.token_urlsafe(32)
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        api_key = APIKey(
            key=key,
            name=name,
            created_at=datetime.now(),
            expires_at=expires_at,
            rate_limit=rate_limit,
            enabled=True,
        )

        self.keys[key] = api_key
        return api_key

    def validate_key(self, key: str) -> Optional[APIKey]:
        """
        Validate an API key.

        Args:
            key: API key to validate

        Returns:
            APIKey if valid, None otherwise
        """
        if key not in self.keys:
            return None

        api_key = self.keys[key]

        # Check if enabled
        if not api_key.enabled:
            return None

        # Check expiration
        if api_key.expires_at and datetime.now() > api_key.expires_at:
            return None

        return api_key

    def revoke_key(self, key: str) -> bool:
        """
        Revoke an API key.

        Args:
            key: API key to revoke

        Returns:
            True if revoked, False if not found
        """
        if key in self.keys:
            self.keys[key].enabled = False
            return True
        return False

    def list_keys(self) -> list[APIKey]:
        """
        List all API keys.

        Returns:
            List of APIKey instances
        """
        return list(self.keys.values())


# ---------------------------------------------------------------------------
# RBAC role definitions
# ---------------------------------------------------------------------------

_ROLES = {
    "admin": ["*"],
    "operator": [
        "sessions.create",
        "sessions.delete",
        "sessions.list",
        "sessions.get",
        "sessions.navigate",
        "sessions.extract",
        "sessions.act",
        "sessions.agent",
        "sessions.screenshot",
        "sessions.observe",
        "sessions.stream",
        "recordings.list",
        "recordings.download",
    ],
    "viewer": [
        "sessions.list",
        "sessions.get",
        "recordings.list",
        "recordings.download",
    ],
}


class RBACAuthManager:
    """JWT-based RBAC authentication manager.

    Wraps :class:`fireflyframework_genai.security.rbac.RBACManager` to provide
    JWT token creation/validation with admin, operator, and viewer roles.
    Also maintains backward compatibility with API key validation through
    an internal :class:`APIKeyManager`.

    Example::

        mgr = RBACAuthManager()
        token = mgr.create_token(user_id="alice", roles=["operator"])
        claims = mgr.validate_token(token)
        if mgr.has_permission("operator", "sessions.create"):
            ...
    """

    def __init__(
        self,
        jwt_secret: str | None = None,
        *,
        token_expiry_hours: int = 24,
    ) -> None:
        # Use a deterministic-but-safe default secret for local development.
        # Production deployments should inject a real secret via config/env.
        self._jwt_secret = jwt_secret or secrets.token_urlsafe(64)

        self._rbac = RBACManager(
            jwt_secret=self._jwt_secret,
            token_expiry_hours=token_expiry_hours,
            roles=_ROLES,
        )

        # Internal API key manager for backward compatibility
        self._api_key_manager = APIKeyManager()

    # -- Token methods -------------------------------------------------------

    def create_token(
        self,
        user_id: str,
        roles: list[str],
        *,
        custom_claims: dict[str, Any] | None = None,
    ) -> str:
        """Create a signed JWT token.

        Args:
            user_id: Unique identifier for the user.
            roles: List of role names (e.g. ``["admin"]``).
            custom_claims: Optional extra claims embedded in the JWT.

        Returns:
            Signed JWT token string.
        """
        return self._rbac.create_token(
            user_id=user_id,
            roles=roles,
            custom_claims=custom_claims,
        )

    def validate_token(self, token: str) -> dict[str, Any] | None:
        """Validate a JWT token and return its claims.

        Unlike the underlying :pymethod:`RBACManager.validate_token`, this
        method returns ``None`` on any validation failure instead of raising.

        Args:
            token: JWT token string.

        Returns:
            Claims dictionary if valid, ``None`` otherwise.
        """
        if not token:
            return None
        try:
            return self._rbac.validate_token(token)
        except (ValueError, Exception):  # noqa: BLE001
            logger.debug("Token validation failed for token of length %d", len(token))
            return None

    # -- Permission helpers --------------------------------------------------

    def has_permission(self, role: str, permission: str) -> bool:
        """Check whether *role* grants *permission*.

        This is a convenience wrapper that constructs a minimal claims dict
        so callers can check a single role without needing a full JWT.

        Args:
            role: Role name (e.g. ``"operator"``).
            permission: Permission string (e.g. ``"sessions.create"``).

        Returns:
            ``True`` if the role includes the permission.
        """
        claims = {"roles": [role]}
        return self._rbac.has_permission(claims, permission)

    # -- API key backward compatibility --------------------------------------

    def validate_api_key(self, key: str) -> APIKey | None:
        """Validate an API key via the internal key manager.

        Args:
            key: Raw API key string.

        Returns:
            :class:`APIKey` if valid, ``None`` otherwise.
        """
        return self._api_key_manager.validate_key(key)

    @property
    def dev_api_key(self) -> str:
        """Return the development API key for backward compatibility."""
        keys = self._api_key_manager.list_keys()
        if keys:
            return keys[0].key
        # Shouldn't happen, but create one if list is empty
        new_key = self._api_key_manager.create_key("Development Key (fallback)")
        return new_key.key


# ---------------------------------------------------------------------------
# Global singleton instances
# ---------------------------------------------------------------------------

# Global API key manager instance (preserved for backward compatibility)
api_key_manager = APIKeyManager()

# Global RBAC auth manager instance
rbac_manager = RBACAuthManager()


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> APIKey:
    """
    Verify API key from header.

    Args:
        api_key: API key from header

    Returns:
        APIKey instance

    Raises:
        HTTPException: If authentication fails
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    validated_key = api_key_manager.validate_key(api_key)
    if not validated_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return validated_key


async def verify_bearer_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> str:
    """
    Verify bearer token.

    Args:
        credentials: Bearer token credentials

    Returns:
        Token string

    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # For now, treat bearer token same as API key
    validated_key = api_key_manager.validate_key(credentials.credentials)
    if not validated_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials

