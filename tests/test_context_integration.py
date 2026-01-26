# Copyright 2026 Firefly Software Solutions Inc.
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
Integration tests for context system flow.

Tests the complete flow: SDK -> API Models -> Agent Memory -> Tool Access
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, Optional

from flybrowser.agents.context import (
    ActionContext,
    ContextBuilder,
    ContextType,
    FileUploadSpec,
)
from flybrowser.agents.memory import AgentMemory
from flybrowser.agents.tools.base import BaseTool, ToolMetadata, ToolResult


class MockTool(BaseTool):
    """Mock tool for testing context access."""
    
    def __init__(self):
        super().__init__()
        self._received_context = None
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="mock_tool",
            description="Mock tool for testing",
            parameters=[],
            expected_context_types=["form_data", "filters"],
        )
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Store received context for testing."""
        self._received_context = self.get_user_context()
        return ToolResult.success_result(data={"context_received": bool(self._received_context)})


class TestContextMemoryStorage:
    """Test context storage in agent memory."""
    
    def test_store_context_dict_in_memory(self):
        """Test storing context dict in agent memory."""
        memory = AgentMemory()
        
        context_dict = {
            "form_data": {"email": "test@example.com"},
            "filters": {"price_max": 100},
        }
        
        memory.working.set_scratch("user_context", context_dict)
        
        retrieved = memory.working.get_scratch("user_context")
        assert retrieved == context_dict
        assert retrieved["form_data"]["email"] == "test@example.com"
    
    def test_store_action_context_as_dict(self):
        """Test storing ActionContext converted to dict."""
        memory = AgentMemory()
        
        context = ContextBuilder()\
            .with_form_data({"email": "test@example.com"})\
            .with_filters({"price_max": 100})\
            .build()
        
        context_dict = context.to_dict()
        memory.working.set_scratch("user_context", context_dict)
        
        retrieved = memory.working.get_scratch("user_context")
        assert "form_data" in retrieved
        assert retrieved["form_data"]["email"] == "test@example.com"
    
    def test_context_persists_through_memory_operations(self):
        """Test context persists through various memory operations."""
        memory = AgentMemory()
        
        context_dict = {"form_data": {"test": "value"}}
        memory.working.set_scratch("user_context", context_dict)
        
        # Perform other memory operations
        memory.working.set_scratch("other_key", "other_value")
        memory.working.set_scratch("another_key", 123)
        
        # Context should still be retrievable
        retrieved = memory.working.get_scratch("user_context")
        assert retrieved["form_data"]["test"] == "value"


class TestToolContextAccess:
    """Test tools accessing context from memory."""
    
    def test_tool_gets_user_context_from_memory(self):
        """Test tool can access user context from memory."""
        memory = AgentMemory()
        tool = MockTool()
        
        # Set up memory with context
        context_dict = {"form_data": {"email": "test@example.com"}}
        memory.working.set_scratch("user_context", context_dict)
        
        # Inject memory into tool
        tool.set_memory(memory)
        
        # Tool should be able to get context
        user_context = tool.get_user_context()
        assert user_context is not None
        assert user_context["form_data"]["email"] == "test@example.com"
    
    def test_tool_returns_empty_without_memory(self):
        """Test tool returns empty dict when no memory set."""
        tool = MockTool()
        
        # Without memory set, should return empty dict
        user_context = tool.get_user_context()
        assert user_context == {}
    
    def test_tool_returns_empty_without_context(self):
        """Test tool returns empty dict when no context in memory."""
        memory = AgentMemory()
        tool = MockTool()
        tool.set_memory(memory)
        
        # Memory exists but no user_context
        user_context = tool.get_user_context()
        assert user_context == {}
    
    @pytest.mark.asyncio
    async def test_tool_execute_with_context(self):
        """Test tool execute can access context."""
        memory = AgentMemory()
        tool = MockTool()
        
        # Set up context
        context_dict = {"filters": {"price_max": 100}}
        memory.working.set_scratch("user_context", context_dict)
        tool.set_memory(memory)
        
        # Execute tool
        result = await tool.execute()
        
        assert result.success
        assert result.data["context_received"] is True
        assert tool._received_context == context_dict


class TestContextAPIModels:
    """Test context in API request models."""
    
    def test_action_request_with_context(self):
        """Test ActionRequest model accepts context."""
        from flybrowser.service.models import ActionRequest
        
        request = ActionRequest(
            instruction="Fill the form",
            context={
                "form_data": {"email": "test@example.com"},
                "files": [{"field": "resume", "path": "/tmp/test.pdf"}],
            }
        )
        
        assert request.context is not None
        assert request.context["form_data"]["email"] == "test@example.com"
    
    def test_extract_request_with_context(self):
        """Test ExtractRequest model accepts context."""
        from flybrowser.service.models import ExtractRequest
        
        request = ExtractRequest(
            query="Get products",
            context={
                "filters": {"price_max": 100},
                "preferences": {"sort_by": "price"},
            }
        )
        
        assert request.context is not None
        assert request.context["filters"]["price_max"] == 100
    
    def test_observe_request_with_context(self):
        """Test ObserveRequest model accepts context."""
        from flybrowser.service.models import ObserveRequest
        
        request = ObserveRequest(
            query="Find buttons",
            context={
                "filters": {"type": "submit"},
            }
        )
        
        assert request.context is not None
        assert request.context["filters"]["type"] == "submit"
    
    def test_agent_request_with_full_context(self):
        """Test AgentRequest model accepts full context."""
        from flybrowser.service.models import AgentRequest
        
        request = AgentRequest(
            task="Complete registration",
            context={
                "form_data": {"email": "test@example.com"},
                "files": [{"field": "cv", "path": "/tmp/cv.pdf"}],
                "filters": {"category": "tech"},
                "preferences": {"timeout_seconds": 60},
                "conditions": {"requires_login": False},
                "constraints": {"max_retries": 3},
                "metadata": {"request_id": "abc123"},
            }
        )
        
        assert request.context is not None
        assert "form_data" in request.context
        assert "files" in request.context
        assert "filters" in request.context
        assert "preferences" in request.context
        assert "conditions" in request.context
        assert "constraints" in request.context
        assert "metadata" in request.context


class TestContextConversionFlow:
    """Test context conversion throughout the system."""
    
    def test_builder_to_dict_to_memory_flow(self):
        """Test complete flow from builder to memory storage."""
        # User creates context with builder
        context = ContextBuilder()\
            .with_form_data({"email": "user@example.com"})\
            .with_filters({"price_max": 100})\
            .with_preferences({"sort_by": "price"})\
            .build()
        
        # Convert to dict (as SDK would do)
        context_dict = context.to_dict()
        
        # Store in memory (as SDK methods do)
        memory = AgentMemory()
        memory.working.set_scratch("user_context", context_dict)
        
        # Tool retrieves context
        tool = MockTool()
        tool.set_memory(memory)
        user_context = tool.get_user_context()
        
        # Verify complete flow
        assert user_context["form_data"]["email"] == "user@example.com"
        assert user_context["filters"]["price_max"] == 100
        assert user_context["preferences"]["sort_by"] == "price"
    
    def test_dict_context_passes_through_unchanged(self):
        """Test dict context passes through unchanged."""
        # Direct dict context (legacy style)
        context_dict = {
            "form_data": {"field1": "value1"},
            "custom_key": "custom_value",
        }
        
        memory = AgentMemory()
        memory.working.set_scratch("user_context", context_dict)
        
        tool = MockTool()
        tool.set_memory(memory)
        user_context = tool.get_user_context()
        
        # Should be identical
        assert user_context == context_dict
        assert user_context["custom_key"] == "custom_value"
    
    def test_action_context_validation_in_flow(self):
        """Test ActionContext validation during conversion."""
        # Create context
        context = ContextBuilder()\
            .with_form_data({"email": "test@example.com"})\
            .build()
        
        # Validate (as SDK methods do)
        from flybrowser.agents.context import ContextValidator
        is_valid, errors = ContextValidator.validate(context)
        
        assert is_valid
        assert len(errors) == 0
        
        # Convert and store
        context_dict = context.to_dict()
        assert "form_data" in context_dict


class TestToolSpecificContextUsage:
    """Test how specific tools would use context."""
    
    def test_type_tool_form_data_extraction(self):
        """Test extracting form_data as TypeTool would."""
        memory = AgentMemory()
        memory.working.set_scratch("user_context", {
            "form_data": {
                "input[name=email]": "user@example.com",
                "input[name=password]": "secret123",
            }
        })
        
        tool = MockTool()
        tool.set_memory(memory)
        user_context = tool.get_user_context()
        
        # Tool would extract form_data
        form_data = user_context.get("form_data", {})
        assert "input[name=email]" in form_data
        assert form_data["input[name=email]"] == "user@example.com"
    
    def test_upload_tool_files_extraction(self):
        """Test extracting files as UploadFileTool would."""
        memory = AgentMemory()
        memory.working.set_scratch("user_context", {
            "files": [
                {"field": "resume", "path": "/tmp/resume.pdf", "mime_type": "application/pdf"},
                {"field": "photo", "path": "/tmp/photo.jpg", "mime_type": "image/jpeg"},
            ]
        })
        
        tool = MockTool()
        tool.set_memory(memory)
        user_context = tool.get_user_context()
        
        # Tool would extract files
        files = user_context.get("files", [])
        assert len(files) == 2
        assert files[0]["field"] == "resume"
        assert files[1]["mime_type"] == "image/jpeg"
    
    def test_extract_tool_filters_extraction(self):
        """Test extracting filters as ExtractTextTool would."""
        memory = AgentMemory()
        memory.working.set_scratch("user_context", {
            "filters": {"price_max": 100, "category": "electronics"},
            "preferences": {"max_items": 10, "sort_by": "price"},
        })
        
        tool = MockTool()
        tool.set_memory(memory)
        user_context = tool.get_user_context()
        
        # Tool would extract filters and preferences
        filters = user_context.get("filters", {})
        preferences = user_context.get("preferences", {})
        
        assert filters["price_max"] == 100
        assert preferences["sort_by"] == "price"
    
    def test_navigate_tool_conditions_extraction(self):
        """Test extracting conditions as NavigateTool would."""
        memory = AgentMemory()
        memory.working.set_scratch("user_context", {
            "conditions": {"requires_login": False, "max_redirects": 3},
            "constraints": {"timeout_seconds": 30},
        })
        
        tool = MockTool()
        tool.set_memory(memory)
        user_context = tool.get_user_context()
        
        # Tool would extract conditions and constraints
        conditions = user_context.get("conditions", {})
        constraints = user_context.get("constraints", {})
        
        assert conditions["requires_login"] is False
        assert constraints["timeout_seconds"] == 30


class TestContextToolMetadata:
    """Test tool metadata for context types."""
    
    def test_tool_expected_context_types(self):
        """Test tools declaring expected context types."""
        tool = MockTool()
        metadata = tool.metadata
        
        assert hasattr(metadata, 'expected_context_types')
        assert "form_data" in metadata.expected_context_types
        assert "filters" in metadata.expected_context_types
    
    def test_type_tool_metadata(self):
        """Test TypeTool declares form_data context type."""
        from flybrowser.agents.tools.interaction import TypeTool
        
        # Mock page controller
        mock_page = MagicMock()
        tool = TypeTool(page_controller=mock_page)
        
        metadata = tool.metadata
        assert "form_data" in (metadata.expected_context_types or [])
    
    def test_navigate_tool_metadata(self):
        """Test NavigateTool declares conditions context type."""
        from flybrowser.agents.tools.navigation import NavigateTool
        
        # Mock page controller
        mock_page = MagicMock()
        tool = NavigateTool(page_controller=mock_page)
        
        metadata = tool.metadata
        assert "conditions" in (metadata.expected_context_types or [])
        assert "constraints" in (metadata.expected_context_types or [])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
