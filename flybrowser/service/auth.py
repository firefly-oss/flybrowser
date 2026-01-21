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

"""Authentication and authorization for the API service."""

import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

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


# Global API key manager instance
api_key_manager = APIKeyManager()


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

