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

"""Unit tests for text normalization utilities."""

import pytest
from flybrowser.utils.text_normalizer import normalize_text, normalize_data, is_normalized


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_empty_string(self):
        """Test empty string remains empty."""
        assert normalize_text("") == ""

    def test_ascii_unchanged(self):
        """Test that pure ASCII text is unchanged."""
        text = "Hello, World! This is a test."
        assert normalize_text(text) == text

    def test_en_dash_to_hyphen(self):
        """Test en-dash converts to hyphen."""
        assert normalize_text("EU–INC") == "EU-INC"

    def test_em_dash_to_hyphen(self):
        """Test em-dash converts to hyphen."""
        assert normalize_text("A — B") == "A - B"

    def test_smart_double_quotes(self):
        """Test smart double quotes convert to straight quotes."""
        assert normalize_text('"quoted"') == '"quoted"'
        assert normalize_text('«quoted»') == '"quoted"'

    def test_smart_single_quotes(self):
        """Test smart single quotes convert to apostrophe."""
        assert normalize_text("it's") == "it's"
        assert normalize_text("'quoted'") == "'quoted'"

    def test_ellipsis(self):
        """Test ellipsis character converts to three dots."""
        assert normalize_text("Wait…") == "Wait..."

    def test_non_breaking_space(self):
        """Test non-breaking space converts to regular space."""
        assert normalize_text("hello\xa0world") == "hello world"

    def test_zero_width_chars_removed(self):
        """Test zero-width characters are removed."""
        # Zero-width space
        assert normalize_text("hello\u200bworld") == "helloworld"
        # Zero-width joiner
        assert normalize_text("test\u200dtext") == "testtext"

    def test_multiple_spaces_collapsed(self):
        """Test multiple spaces collapse to single space."""
        assert normalize_text("hello    world") == "hello world"
        assert normalize_text("a  b   c    d") == "a b c d"

    def test_leading_trailing_whitespace_trimmed(self):
        """Test leading and trailing whitespace is trimmed."""
        assert normalize_text("  hello  ") == "hello"
        assert normalize_text("\n\thello\t\n") == "hello"

    def test_preserves_accented_characters(self):
        """Test that accented letters are preserved."""
        assert normalize_text("café") == "café"
        assert normalize_text("naïve") == "naïve"
        assert normalize_text("jalapeño") == "jalapeño"
        assert normalize_text("Zürich") == "Zürich"

    def test_preserves_currency_symbols(self):
        """Test that currency symbols are preserved."""
        assert normalize_text("$99.99") == "$99.99"
        assert normalize_text("€50") == "€50"
        assert normalize_text("£100") == "£100"
        assert normalize_text("¥500") == "¥500"

    def test_preserves_math_symbols(self):
        """Test that math symbols are preserved."""
        assert normalize_text("±5%") == "±5%"
        assert normalize_text("2×3") == "2×3"
        assert normalize_text("10÷2") == "10÷2"
        assert normalize_text("90°") == "90°"

    def test_hacker_news_title(self):
        """Test real-world Hacker News title normalization."""
        input_text = "Show HN: ChartGPU – WebGPU-powered charting"
        expected = "Show HN: ChartGPU - WebGPU-powered charting"
        assert normalize_text(input_text) == expected

    def test_complex_mixed_text(self):
        """Test complex text with multiple normalization issues."""
        input_text = "EU\u2013INC \u2014 A \u201cnew\u201d entity\u2026  \u200b"
        expected = 'EU-INC - A "new" entity...'
        assert normalize_text(input_text) == expected

    def test_non_string_passthrough(self):
        """Test that non-string types are returned unchanged."""
        assert normalize_text(123) == 123
        assert normalize_text(None) is None
        assert normalize_text([1, 2, 3]) == [1, 2, 3]


class TestNormalizeData:
    """Tests for normalize_data function."""

    def test_string_normalized(self):
        """Test that strings are normalized."""
        assert normalize_data("EU–INC") == "EU-INC"

    def test_dict_values_normalized(self):
        """Test that dictionary values are normalized."""
        data = {"title": "EU–INC — A new entity"}
        expected = {"title": "EU-INC - A new entity"}
        assert normalize_data(data) == expected

    def test_dict_keys_unchanged(self):
        """Test that dictionary keys are NOT normalized."""
        # Keys should remain unchanged - only values are normalized
        data = {"title–name": "Value"}
        result = normalize_data(data)
        assert "title–name" in result  # Key preserved

    def test_list_normalized(self):
        """Test that list items are normalized."""
        data = ["EU–INC", "Item — Two"]
        expected = ["EU-INC", "Item - Two"]
        assert normalize_data(data) == expected

    def test_nested_structure(self):
        """Test deeply nested structure is fully normalized."""
        data = {
            "items": [
                {"title": "First\u2013Item", "score": "100 points"},
                {"title": "Second \u2014 Item", "score": "50 points"},
            ],
            "metadata": {
                "source": "Test\u2026",
                "nested": {"deep": "Value \u201cquoted\u201d"},
            },
        }
        expected = {
            "items": [
                {"title": "First-Item", "score": "100 points"},
                {"title": "Second - Item", "score": "50 points"},
            ],
            "metadata": {
                "source": "Test...",
                "nested": {"deep": 'Value "quoted"'},
            },
        }
        assert normalize_data(data) == expected

    def test_extracted_data_structure(self):
        """Test normalization of typical extraction result."""
        data = {
            "extracted_data": [
                {"title": "Show HN: ChartGPU – WebGPU-powered", "score": "147 points"},
                {"title": "EU–INC — A new entity", "score": "526 points"},
            ]
        }
        expected = {
            "extracted_data": [
                {"title": "Show HN: ChartGPU - WebGPU-powered", "score": "147 points"},
                {"title": "EU-INC - A new entity", "score": "526 points"},
            ]
        }
        assert normalize_data(data) == expected

    def test_tuple_normalized(self):
        """Test that tuples are normalized and remain tuples."""
        data = ("EU–INC", "Item — Two")
        result = normalize_data(data)
        assert result == ("EU-INC", "Item - Two")
        assert isinstance(result, tuple)

    def test_numbers_unchanged(self):
        """Test that numbers are unchanged."""
        data = {"price": 99.99, "count": 10}
        assert normalize_data(data) == data

    def test_none_unchanged(self):
        """Test that None values are unchanged."""
        data = {"title": None, "value": "test"}
        expected = {"title": None, "value": "test"}
        assert normalize_data(data) == expected


class TestIsNormalized:
    """Tests for is_normalized function."""

    def test_normalized_text_returns_true(self):
        """Test that already normalized text returns True."""
        assert is_normalized("Hello, World!") is True
        assert is_normalized("Normal text here") is True
        assert is_normalized("$99.99") is True

    def test_en_dash_returns_false(self):
        """Test that text with en-dash returns False."""
        assert is_normalized("EU–INC") is False

    def test_em_dash_returns_false(self):
        """Test that text with em-dash returns False."""
        assert is_normalized("A — B") is False

    def test_smart_quotes_returns_false(self):
        """Test that text with smart quotes returns False."""
        assert is_normalized('\u201cquoted\u201d') is False  # curly double quotes

    def test_multiple_spaces_returns_false(self):
        """Test that text with multiple spaces returns False."""
        assert is_normalized("hello  world") is False

    def test_leading_trailing_whitespace_returns_false(self):
        """Test that text with leading/trailing whitespace returns False."""
        assert is_normalized(" hello") is False
        assert is_normalized("hello ") is False

    def test_invisible_chars_returns_false(self):
        """Test that text with invisible characters returns False."""
        assert is_normalized("hello\u200bworld") is False

    def test_non_string_returns_true(self):
        """Test that non-string types return True."""
        assert is_normalized(123) is True
        assert is_normalized(None) is True
