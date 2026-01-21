# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PII handling."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flybrowser.security.pii_handler import (
    PIIConfig,
    PIIField,
    PIIHandler,
    PIIMasker,
    PIIType,
    SecureCredential,
    SecureCredentialStore,
    SensitiveDataMarker,
    secure_zero_memory,
)


class TestPIIType:
    """Tests for PIIType enum."""

    def test_all_types_exist(self):
        """Test all PII types are defined."""
        assert PIIType.PASSWORD == "password"
        assert PIIType.USERNAME == "username"
        assert PIIType.EMAIL == "email"
        assert PIIType.PHONE == "phone"
        assert PIIType.SSN == "ssn"
        assert PIIType.CREDIT_CARD == "credit_card"
        assert PIIType.CVV == "cvv"
        assert PIIType.ADDRESS == "address"
        assert PIIType.NAME == "name"
        assert PIIType.DATE_OF_BIRTH == "date_of_birth"
        assert PIIType.API_KEY == "api_key"
        assert PIIType.TOKEN == "token"
        assert PIIType.CUSTOM == "custom"


class TestPIIConfig:
    """Tests for PIIConfig."""

    def test_default_values(self):
        """Test default config values."""
        config = PIIConfig()
        
        assert config.enabled is True
        assert config.mask_in_logs is True
        assert config.mask_in_llm_prompts is True
        assert config.encryption_enabled is True
        assert config.mask_character == "*"
        assert config.mask_length == 8
        assert config.preserve_format is True
        assert config.auto_detect_pii is True
        assert config.use_placeholders_for_llm is True
        assert config.placeholder_prefix == "{{CREDENTIAL:"
        assert config.placeholder_suffix == "}}"

    def test_custom_values(self):
        """Test config with custom values."""
        config = PIIConfig(
            enabled=False,
            mask_character="#",
            mask_length=6,
            credential_timeout=300,
        )
        
        assert config.enabled is False
        assert config.mask_character == "#"
        assert config.mask_length == 6
        assert config.credential_timeout == 300


class TestPIIField:
    """Tests for PIIField."""

    def test_field_creation(self):
        """Test PIIField creation."""
        field = PIIField(
            name="password",
            pii_type=PIIType.PASSWORD,
            selector="#password-input",
        )
        
        assert field.name == "password"
        assert field.pii_type == PIIType.PASSWORD
        assert field.selector == "#password-input"
        assert field.is_sensitive is True


class TestSecureCredential:
    """Tests for SecureCredential."""

    def test_credential_creation(self):
        """Test SecureCredential creation."""
        cred = SecureCredential(
            name="test_password",
            pii_type=PIIType.PASSWORD,
            encrypted_value=b"encrypted",
        )
        
        assert cred.name == "test_password"
        assert cred.pii_type == PIIType.PASSWORD
        assert cred.encrypted_value == b"encrypted"
        assert cred.id is not None
        assert cred.created_at is not None

    def test_repr_safe(self):
        """Test repr doesn't expose sensitive data."""
        cred = SecureCredential(
            name="password",
            pii_type=PIIType.PASSWORD,
            _original_value="secret123",
        )
        
        repr_str = repr(cred)
        assert "secret123" not in repr_str
        assert "password" in repr_str


class TestPIIMasker:
    """Tests for PIIMasker."""

    def test_mask_email(self):
        """Test email masking."""
        masker = PIIMasker()
        
        text = "Contact me at john.doe@example.com"
        result = masker.mask_text(text)
        
        assert "john.doe@example.com" not in result
        assert "@" in result  # Format preserved

    def test_mask_phone(self):
        """Test phone number masking."""
        masker = PIIMasker()
        
        text = "Call me at 555-123-4567"
        result = masker.mask_text(text)
        
        assert "555-123-4567" not in result
        assert "-" in result  # Format preserved

    def test_mask_credit_card(self):
        """Test credit card masking."""
        masker = PIIMasker()
        
        text = "Card: 4111-1111-1111-1111"
        result = masker.mask_text(text)
        
        assert "4111-1111-1111" not in result
        # Last 4 digits may be preserved
        assert "1111" in result

    def test_mask_ssn(self):
        """Test SSN masking."""
        masker = PIIMasker()
        
        text = "SSN: 123-45-6789"
        result = masker.mask_text(text)
        
        assert "123-45-6789" not in result

    def test_mask_api_key(self):
        """Test API key masking."""
        masker = PIIMasker()
        
        text = "Use key: sk-1234567890abcdef1234567890"
        result = masker.mask_text(text)
        
        assert "sk-1234567890abcdef1234567890" not in result

    def test_mask_disabled(self):
        """Test masking when disabled."""
        config = PIIConfig(enabled=False)
        masker = PIIMasker(config)
        
        text = "Email: test@example.com"
        result = masker.mask_text(text)
        
        assert result == text

    def test_mask_dict(self):
        """Test dictionary masking."""
        masker = PIIMasker()
        
        data = {
            "username": "john",
            "password": "secret123",
            "email": "john@example.com",
        }
        
        result = masker.mask_dict(data)
        
        assert result["password"] == "********"
        assert "secret123" not in str(result)

    def test_is_sensitive_field(self):
        """Test sensitive field detection."""
        masker = PIIMasker()
        
        assert masker.is_sensitive_field("password") is True
        assert masker.is_sensitive_field("user_password") is True
        assert masker.is_sensitive_field("api_key") is True
        assert masker.is_sensitive_field("credit_card") is True
        assert masker.is_sensitive_field("username") is False


class TestSecureCredentialStore:
    """Tests for SecureCredentialStore."""

    def test_store_and_retrieve(self):
        """Test storing and retrieving credentials."""
        store = SecureCredentialStore()
        
        cred_id = store.store("password", "secret123", PIIType.PASSWORD)
        
        assert cred_id is not None
        
        value = store.retrieve(cred_id)
        assert value == "secret123"

    def test_retrieve_by_name(self):
        """Test retrieving by name."""
        store = SecureCredentialStore()
        
        store.store("email", "user@example.com", PIIType.EMAIL)
        
        value = store.retrieve_by_name("email")
        assert value == "user@example.com"

    def test_get_placeholder(self):
        """Test placeholder generation."""
        store = SecureCredentialStore()
        
        cred_id = store.store("email", "user@example.com", PIIType.EMAIL)
        
        placeholder = store.get_placeholder(cred_id)
        assert placeholder == "{{CREDENTIAL:email}}"

    def test_delete(self):
        """Test credential deletion."""
        store = SecureCredentialStore()
        
        cred_id = store.store("password", "secret", PIIType.PASSWORD)
        assert store.retrieve(cred_id) == "secret"
        
        result = store.delete(cred_id)
        assert result is True
        
        assert store.retrieve(cred_id) is None

    def test_list_credentials(self):
        """Test listing credentials."""
        store = SecureCredentialStore()
        
        store.store("email", "user@example.com", PIIType.EMAIL)
        store.store("password", "secret", PIIType.PASSWORD)
        
        creds = store.list_credentials()
        
        assert len(creds) == 2
        # Should not contain actual values
        assert all("user@example.com" not in str(c) for c in creds)
        assert all("secret" not in str(c) for c in creds)

    def test_credential_timeout(self):
        """Test credential expiration."""
        config = PIIConfig(credential_timeout=1)  # 1 second
        store = SecureCredentialStore(config)
        
        cred_id = store.store("password", "secret", PIIType.PASSWORD)
        
        # Should work immediately
        assert store.retrieve(cred_id) == "secret"
        
        # Force expiration
        store._credentials[cred_id].created_at = time.time() - 10
        
        # Should return None after timeout
        assert store.retrieve(cred_id) is None

    def test_cleanup_expired(self):
        """Test cleanup of expired credentials."""
        config = PIIConfig(credential_timeout=1)
        store = SecureCredentialStore(config)
        
        # Create expired credential
        cred_id = store.store("password", "secret", PIIType.PASSWORD)
        store._credentials[cred_id].created_at = time.time() - 10
        
        count = store.cleanup_expired()
        
        assert count == 1
        assert cred_id not in store._credentials


class TestSensitiveDataMarker:
    """Tests for SensitiveDataMarker."""

    def test_mark_field(self):
        """Test marking a field as sensitive."""
        marker = SensitiveDataMarker()
        
        field = marker.mark_field("password", PIIType.PASSWORD, "#pass-input")
        
        assert field.name == "password"
        assert field.pii_type == PIIType.PASSWORD
        assert field.selector == "#pass-input"

    def test_is_marked(self):
        """Test checking if field is marked."""
        marker = SensitiveDataMarker()
        
        assert marker.is_marked("password") is False
        
        marker.mark_field("password", PIIType.PASSWORD)
        
        assert marker.is_marked("password") is True

    def test_is_selector_marked(self):
        """Test checking if selector is marked."""
        marker = SensitiveDataMarker()
        
        marker.mark_field("password", PIIType.PASSWORD, "#pass-input")
        
        assert marker.is_selector_marked("#pass-input") is True
        assert marker.is_selector_marked("#other") is False

    def test_unmark_field(self):
        """Test unmarking a field."""
        marker = SensitiveDataMarker()
        
        marker.mark_field("password", PIIType.PASSWORD, "#pass")
        assert marker.is_marked("password") is True
        
        result = marker.unmark_field("password")
        
        assert result is True
        assert marker.is_marked("password") is False
        assert marker.is_selector_marked("#pass") is False


class TestPIIHandler:
    """Tests for PIIHandler."""

    def test_store_and_retrieve_credential(self):
        """Test storing and retrieving credentials."""
        handler = PIIHandler()
        
        cred_id = handler.store_credential("email", "user@example.com", PIIType.EMAIL)
        
        value = handler.retrieve_credential(cred_id)
        assert value == "user@example.com"

    def test_delete_credential(self):
        """Test deleting credentials."""
        handler = PIIHandler()
        
        cred_id = handler.store_credential("password", "secret", PIIType.PASSWORD)
        
        result = handler.delete_credential(cred_id)
        assert result is True
        
        assert handler.retrieve_credential(cred_id) is None

    def test_mark_sensitive_field(self):
        """Test marking sensitive fields."""
        handler = PIIHandler()
        
        field = handler.mark_sensitive_field("password", PIIType.PASSWORD, "#pass")
        
        assert field.name == "password"
        assert handler.is_sensitive_field("password") is True

    def test_mask_for_llm(self):
        """Test masking for LLM."""
        handler = PIIHandler()
        
        text = "My email is test@example.com"
        result = handler.mask_for_llm(text)
        
        assert "test@example.com" not in result

    def test_mask_for_log(self):
        """Test masking for logs."""
        handler = PIIHandler()
        
        # PII masker masks credit cards and other patterns, but not arbitrary strings
        text = "Card is 4111-1111-1111-1111, email is user@example.com"
        result = handler.mask_for_log(text)
        
        assert "4111-1111-1111" not in result
        assert "user@example.com" not in result

    @pytest.mark.asyncio
    async def test_secure_fill(self):
        """Test secure form filling."""
        handler = PIIHandler()
        cred_id = handler.store_credential("password", "secret123")
        
        mock_page = MagicMock()
        mock_page.fill = AsyncMock()
        
        result = await handler.secure_fill(mock_page, "#password", cred_id)
        
        assert result is True
        mock_page.fill.assert_awaited()
        # Verify the actual value was used
        call_args = mock_page.fill.call_args_list
        assert any("secret123" in str(args) for args in call_args)

    @pytest.mark.asyncio
    async def test_secure_fill_not_found(self):
        """Test secure fill with unknown credential."""
        handler = PIIHandler()
        mock_page = MagicMock()
        
        result = await handler.secure_fill(mock_page, "#password", "unknown-id")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_secure_type(self):
        """Test secure typing."""
        handler = PIIHandler()
        cred_id = handler.store_credential("password", "secret123")
        
        mock_page = MagicMock()
        mock_page.type = AsyncMock()
        
        result = await handler.secure_type(mock_page, "#password", cred_id, delay=10)
        
        assert result is True
        mock_page.type.assert_awaited_once()

    def test_replace_values_with_placeholders(self):
        """Test replacing values with placeholders."""
        handler = PIIHandler()
        
        handler.store_credential("email", "user@example.com", PIIType.EMAIL)
        handler.store_credential("password", "secret123", PIIType.PASSWORD)
        
        text = "Login with user@example.com and password secret123"
        result = handler.replace_values_with_placeholders(text)
        
        assert "user@example.com" not in result
        assert "secret123" not in result
        assert "{{CREDENTIAL:email}}" in result
        assert "{{CREDENTIAL:password}}" in result

    def test_resolve_placeholders(self):
        """Test resolving placeholders back to values."""
        handler = PIIHandler()
        
        handler.store_credential("email", "user@example.com", PIIType.EMAIL)
        handler.store_credential("password", "secret123", PIIType.PASSWORD)
        
        text = "Type {{CREDENTIAL:password}} into #pass and {{CREDENTIAL:email}} into #email"
        result = handler.resolve_placeholders(text)
        
        assert "{{CREDENTIAL:password}}" not in result
        assert "{{CREDENTIAL:email}}" not in result
        assert "secret123" in result
        assert "user@example.com" in result

    def test_has_placeholders(self):
        """Test checking for placeholders."""
        handler = PIIHandler()
        
        assert handler.has_placeholders("Normal text") is False
        assert handler.has_placeholders("Use {{CREDENTIAL:email}}") is True

    def test_extract_placeholders(self):
        """Test extracting placeholder names."""
        handler = PIIHandler()
        
        text = "Login with {{CREDENTIAL:email}} and {{CREDENTIAL:password}}"
        names = handler.extract_placeholders(text)
        
        assert "email" in names
        assert "password" in names

    def test_create_secure_instruction(self):
        """Test creating secure instruction."""
        handler = PIIHandler()
        
        instruction = "Login with user@example.com and secret123"
        credentials = {
            "email": "user@example.com",
            "password": "secret123",
        }
        
        safe_instruction, cred_ids = handler.create_secure_instruction(
            instruction, credentials
        )
        
        assert "user@example.com" not in safe_instruction
        assert "secret123" not in safe_instruction
        assert "email" in cred_ids
        assert "password" in cred_ids

    def test_clear_all_credentials(self):
        """Test clearing all credentials."""
        handler = PIIHandler()
        
        handler.store_credential("email", "user@example.com")
        handler.store_credential("password", "secret")
        
        assert len(handler.list_credentials()) == 2
        
        handler.clear_all_credentials()
        
        assert len(handler.list_credentials()) == 0

    def test_infer_pii_type(self):
        """Test PII type inference from name."""
        handler = PIIHandler()
        
        assert handler._infer_pii_type("password") == PIIType.PASSWORD
        assert handler._infer_pii_type("user_password") == PIIType.PASSWORD
        assert handler._infer_pii_type("email_address") == PIIType.EMAIL
        assert handler._infer_pii_type("phone_number") == PIIType.PHONE
        assert handler._infer_pii_type("ssn") == PIIType.SSN
        assert handler._infer_pii_type("credit_card") == PIIType.CREDIT_CARD
        assert handler._infer_pii_type("api_key") == PIIType.API_KEY
        assert handler._infer_pii_type("auth_token") == PIIType.TOKEN
        assert handler._infer_pii_type("username") == PIIType.USERNAME
        assert handler._infer_pii_type("random_field") == PIIType.CUSTOM
