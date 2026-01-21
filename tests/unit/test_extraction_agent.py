# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ExtractionAgent."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from flybrowser.agents.extraction_agent import ExtractionAgent
from flybrowser.exceptions import ExtractionError


class TestExtractionAgentInit:
    """Tests for ExtractionAgent initialization."""

    def test_init_stores_components(self):
        """Test initialization stores all components."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = ExtractionAgent(mock_page, mock_detector, mock_llm)
        
        assert agent.page is mock_page
        assert agent.detector is mock_detector
        assert agent.llm is mock_llm

    def test_init_with_pii_handler(self):
        """Test initialization with PIIHandler."""
        mock_page = MagicMock()
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        mock_pii = MagicMock()
        
        agent = ExtractionAgent(
            mock_page, mock_detector, mock_llm,
            pii_handler=mock_pii
        )
        
        assert agent.pii_handler is mock_pii


class TestExtractionAgentExecute:
    """Tests for ExtractionAgent.execute()."""

    @pytest.mark.asyncio
    async def test_execute_text_extraction(self):
        """Test text-based extraction."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(return_value="https://example.com")
        mock_page.get_title = AsyncMock(return_value="Example")
        mock_page.get_html = AsyncMock(return_value="<html><body>Test</body></html>")
        
        mock_detector = MagicMock()
        
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value=MagicMock(content=json.dumps({"title": "Example"}))
        )
        
        agent = ExtractionAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.execute("Extract the title", use_vision=False)
        
        assert "title" in result
        assert result["title"] == "Example"

    @pytest.mark.asyncio
    async def test_execute_vision_extraction(self):
        """Test vision-based extraction."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(return_value="https://example.com")
        mock_page.get_title = AsyncMock(return_value="Example")
        mock_page.get_html = AsyncMock(return_value="<html></html>")
        mock_page.screenshot = AsyncMock(return_value=b"screenshot_data")
        
        mock_detector = MagicMock()
        
        mock_llm = MagicMock()
        mock_llm.generate_with_vision = AsyncMock(
            return_value=MagicMock(content=json.dumps({"colors": ["blue", "white"]}))
        )
        
        agent = ExtractionAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.execute("What colors are on the page?", use_vision=True)
        
        assert "colors" in result
        mock_page.screenshot.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_structured_extraction(self):
        """Test structured extraction with schema."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(return_value="https://example.com")
        mock_page.get_title = AsyncMock(return_value="Example")
        mock_page.get_html = AsyncMock(return_value="<html></html>")
        
        mock_detector = MagicMock()
        
        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock(
            return_value={"products": [{"name": "Item1", "price": 10.99}]}
        )
        
        agent = ExtractionAgent(mock_page, mock_detector, mock_llm)
        
        schema = {
            "type": "object",
            "properties": {
                "products": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "number"}
                        }
                    }
                }
            }
        }
        
        result = await agent.execute("Extract all products", schema=schema)
        
        assert "products" in result
        assert len(result["products"]) == 1
        assert result["products"][0]["name"] == "Item1"

    @pytest.mark.asyncio
    async def test_execute_returns_text_on_invalid_json(self):
        """Test execute returns text when LLM returns invalid JSON."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(return_value="https://example.com")
        mock_page.get_title = AsyncMock(return_value="Example")
        mock_page.get_html = AsyncMock(return_value="<html></html>")
        
        mock_detector = MagicMock()
        
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value=MagicMock(content="The title is 'Example Domain'")
        )
        
        agent = ExtractionAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.execute("What is the title?", use_vision=False)
        
        assert "extracted_text" in result
        assert "Example Domain" in result["extracted_text"]

    @pytest.mark.asyncio
    async def test_execute_raises_extraction_error(self):
        """Test execute raises ExtractionError on failure."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(side_effect=Exception("Page error"))
        
        mock_detector = MagicMock()
        mock_llm = MagicMock()
        
        agent = ExtractionAgent(mock_page, mock_detector, mock_llm)
        
        with pytest.raises(ExtractionError, match="Failed to extract data"):
            await agent.execute("Extract data")

    @pytest.mark.asyncio
    async def test_execute_truncates_long_html(self):
        """Test execute truncates very long HTML content."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(return_value="https://example.com")
        mock_page.get_title = AsyncMock(return_value="Example")
        # Create HTML longer than 8000 characters
        long_html = "<html>" + "x" * 10000 + "</html>"
        mock_page.get_html = AsyncMock(return_value=long_html)
        
        mock_detector = MagicMock()
        
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value=MagicMock(content=json.dumps({"result": "ok"}))
        )
        
        agent = ExtractionAgent(mock_page, mock_detector, mock_llm)
        
        result = await agent.execute("Extract something", use_vision=False)
        
        # Should still work despite long HTML
        assert result == {"result": "ok"}
        mock_llm.generate.assert_awaited_once()


class TestExtractionAgentWithPII:
    """Tests for ExtractionAgent with PII handling."""

    @pytest.mark.asyncio
    async def test_masks_query_for_llm(self):
        """Test query is masked when sent to LLM."""
        mock_page = MagicMock()
        mock_page.get_url = AsyncMock(return_value="https://example.com")
        mock_page.get_title = AsyncMock(return_value="Example")
        mock_page.get_html = AsyncMock(return_value="<html></html>")
        
        mock_detector = MagicMock()
        
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value=MagicMock(content=json.dumps({"result": "ok"}))
        )
        
        mock_pii = MagicMock()
        mock_pii.replace_values_with_placeholders = MagicMock(
            return_value="Extract my {{CREDENTIAL:email}} info"
        )
        mock_pii.mask_for_log = MagicMock(return_value="Extract my ***@***.com info")
        
        agent = ExtractionAgent(
            mock_page, mock_detector, mock_llm,
            pii_handler=mock_pii
        )
        
        await agent.execute("Extract my user@example.com info", use_vision=False)
        
        # Verify masking was called
        mock_pii.replace_values_with_placeholders.assert_called()
