# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RBAC authentication integration."""

import pytest

from flybrowser.service.auth import (
    APIKeyManager,
    RBACAuthManager,
    api_key_manager,
    rbac_manager,
    verify_api_key,
    verify_bearer_token,
)


class TestRBACAuthManagerTokenCreation:
    """Tests for RBACAuthManager token creation."""

    def test_create_admin_token(self):
        """Test creating a token with admin role."""
        manager = RBACAuthManager()
        token = manager.create_token(user_id="admin-user", roles=["admin"])

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_operator_token(self):
        """Test creating a token with operator role."""
        manager = RBACAuthManager()
        token = manager.create_token(user_id="operator-user", roles=["operator"])

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_viewer_token(self):
        """Test creating a token with viewer role."""
        manager = RBACAuthManager()
        token = manager.create_token(user_id="viewer-user", roles=["viewer"])

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0


class TestRBACAuthManagerTokenValidation:
    """Tests for RBACAuthManager token validation."""

    def test_validate_admin_token(self):
        """Test validating an admin token returns correct claims with roles."""
        manager = RBACAuthManager()
        token = manager.create_token(user_id="admin-user", roles=["admin"])

        claims = manager.validate_token(token)

        assert claims is not None
        assert claims["sub"] == "admin-user"
        assert "admin" in claims["roles"]

    def test_validate_operator_token(self):
        """Test validating an operator token returns correct claims."""
        manager = RBACAuthManager()
        token = manager.create_token(user_id="op-user", roles=["operator"])

        claims = manager.validate_token(token)

        assert claims is not None
        assert claims["sub"] == "op-user"
        assert "operator" in claims["roles"]

    def test_validate_viewer_token(self):
        """Test validating a viewer token returns correct claims."""
        manager = RBACAuthManager()
        token = manager.create_token(user_id="view-user", roles=["viewer"])

        claims = manager.validate_token(token)

        assert claims is not None
        assert claims["sub"] == "view-user"
        assert "viewer" in claims["roles"]

    def test_invalid_token_returns_none(self):
        """Test that an invalid token returns None (not raises)."""
        manager = RBACAuthManager()

        result = manager.validate_token("not-a-valid-jwt-token")

        assert result is None

    def test_empty_token_returns_none(self):
        """Test that an empty token returns None."""
        manager = RBACAuthManager()

        result = manager.validate_token("")

        assert result is None


class TestRBACAuthManagerPermissions:
    """Tests for RBACAuthManager role permission checks."""

    def test_admin_can_sessions_create(self):
        """Test admin role has sessions.create permission."""
        manager = RBACAuthManager()

        assert manager.has_permission("admin", "sessions.create") is True

    def test_admin_can_any_permission(self):
        """Test admin role has wildcard permission."""
        manager = RBACAuthManager()

        assert manager.has_permission("admin", "sessions.create") is True
        assert manager.has_permission("admin", "sessions.delete") is True
        assert manager.has_permission("admin", "recordings.list") is True
        assert manager.has_permission("admin", "some.random.permission") is True

    def test_operator_can_sessions_create(self):
        """Test operator role has sessions.create permission."""
        manager = RBACAuthManager()

        assert manager.has_permission("operator", "sessions.create") is True

    def test_operator_can_sessions_navigate(self):
        """Test operator role has sessions.navigate permission."""
        manager = RBACAuthManager()

        assert manager.has_permission("operator", "sessions.navigate") is True

    def test_operator_cannot_admin_only(self):
        """Test operator role cannot access admin-only permissions."""
        manager = RBACAuthManager()

        assert manager.has_permission("operator", "admin.manage_users") is False
        assert manager.has_permission("operator", "some.random.permission") is False

    def test_viewer_can_sessions_list(self):
        """Test viewer role has sessions.list permission."""
        manager = RBACAuthManager()

        assert manager.has_permission("viewer", "sessions.list") is True

    def test_viewer_can_recordings_list(self):
        """Test viewer role has recordings.list permission."""
        manager = RBACAuthManager()

        assert manager.has_permission("viewer", "recordings.list") is True

    def test_viewer_cannot_sessions_create(self):
        """Test viewer role cannot create sessions."""
        manager = RBACAuthManager()

        assert manager.has_permission("viewer", "sessions.create") is False

    def test_viewer_cannot_sessions_delete(self):
        """Test viewer role cannot delete sessions."""
        manager = RBACAuthManager()

        assert manager.has_permission("viewer", "sessions.delete") is False

    def test_unknown_role_has_no_permissions(self):
        """Test that an unknown role has no permissions."""
        manager = RBACAuthManager()

        assert manager.has_permission("nonexistent", "sessions.list") is False


class TestBackwardCompatibility:
    """Tests for backward compatibility with API key authentication."""

    def test_backward_compat_api_key_still_works(self):
        """Test that the existing API key manager still works."""
        # The module-level api_key_manager singleton must still exist
        assert api_key_manager is not None
        assert isinstance(api_key_manager, APIKeyManager)

        # Existing create/validate flow should still work
        key = api_key_manager.create_key("test-key")
        validated = api_key_manager.validate_key(key.key)
        assert validated is not None
        assert validated.name == "test-key"

    def test_rbac_manager_singleton_exists(self):
        """Test that the module-level rbac_manager singleton exists."""
        assert rbac_manager is not None
        assert isinstance(rbac_manager, RBACAuthManager)

    def test_rbac_validate_api_key(self):
        """Test that RBACAuthManager can also validate API keys for backward compat."""
        manager = RBACAuthManager()

        # Get the dev API key
        dev_key = manager.dev_api_key
        assert dev_key is not None
        assert isinstance(dev_key, str)
        assert dev_key.startswith("flybrowser_dev_")

        # Validate it via the RBAC manager
        result = manager.validate_api_key(dev_key)
        assert result is not None

    def test_rbac_validate_api_key_invalid(self):
        """Test that invalid API keys return None via RBACAuthManager."""
        manager = RBACAuthManager()

        result = manager.validate_api_key("invalid-key-that-does-not-exist")
        assert result is None

    def test_dev_api_key_property(self):
        """Test the dev_api_key property returns the first key."""
        manager = RBACAuthManager()

        dev_key = manager.dev_api_key
        assert dev_key is not None
        assert isinstance(dev_key, str)

    def test_verify_api_key_function_exists(self):
        """Test that the verify_api_key FastAPI dependency still exists."""
        assert callable(verify_api_key)

    def test_verify_bearer_token_function_exists(self):
        """Test that the verify_bearer_token FastAPI dependency still exists."""
        assert callable(verify_bearer_token)


class TestRBACAuthManagerIntegration:
    """Integration tests for the full token lifecycle."""

    def test_full_token_lifecycle(self):
        """Test create -> validate -> check permissions flow."""
        manager = RBACAuthManager()

        # Create token
        token = manager.create_token(user_id="test-user", roles=["operator"])

        # Validate token
        claims = manager.validate_token(token)
        assert claims is not None
        assert claims["sub"] == "test-user"

        # Check permissions using the role from claims
        for role in claims["roles"]:
            assert manager.has_permission(role, "sessions.create") is True
            assert manager.has_permission(role, "sessions.list") is True
            assert manager.has_permission(role, "admin.manage_users") is False

    def test_multi_role_token(self):
        """Test token with multiple roles."""
        manager = RBACAuthManager()

        token = manager.create_token(
            user_id="multi-role-user", roles=["viewer", "operator"]
        )

        claims = manager.validate_token(token)
        assert claims is not None
        assert "viewer" in claims["roles"]
        assert "operator" in claims["roles"]
