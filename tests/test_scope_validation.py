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
Comprehensive tests for React scope validation and guardrails.

Tests the following enhancements:
1. BrowserScopeValidator - validates tasks and URLs
2. SDK integration - act(), extract(), auto(), execute_task() validation
3. Tool-level URL validation in ReactAgent
4. Goal interpreter fast-path patterns
"""

import pytest
from flybrowser.agents.scope_validator import get_scope_validator, BrowserScopeValidator
from flybrowser.agents.goal_interpreter import GoalInterpreter


class TestBrowserScopeValidator:
    """Test suite for BrowserScopeValidator."""
    
    def setup_method(self):
        """Get validator instance before each test."""
        self.validator = get_scope_validator()
    
    def test_singleton_pattern(self):
        """Validator should be a singleton."""
        validator1 = get_scope_validator()
        validator2 = get_scope_validator()
        assert validator1 is validator2
    
    def test_valid_browser_tasks(self):
        """Valid browser automation tasks should pass validation."""
        valid_tasks = [
            "Navigate to https://example.com",
            "Click the login button",
            "Extract text from the page",
            "Fill out the registration form",
            "Scroll down to see more content",
            "Search for python tutorials on Google",
            "Take a screenshot of the page",
            "Wait for page to load",
            "Type username into the input field",
        ]
        
        for task in valid_tasks:
            is_valid, error = self.validator.validate_task(task)
            assert is_valid, f"Task should be valid: '{task}' - Error: {error}"
    
    def test_prohibited_filesystem_operations(self):
        """Filesystem operations should be rejected."""
        invalid_tasks = [
            "Delete file /etc/passwd",
            "Read file from /var/log/system.log",
            "Write data to file.txt",
            "Create directory /tmp/test",
            "Remove folder /home/user",
        ]
        
        for task in invalid_tasks:
            is_valid, error = self.validator.validate_task(task)
            assert not is_valid, f"Task should be invalid: '{task}'"
            assert error is not None
            assert "prohibited" in error.lower() or "browser automation" in error.lower()
    
    def test_prohibited_system_commands(self):
        """System commands should be rejected."""
        invalid_tasks = [
            "Execute system command ls -la",
            "Run python script exploit.py",
            "Execute bash script",
            "Run sudo command",
            "Kill process 1234",
        ]
        
        for task in invalid_tasks:
            is_valid, error = self.validator.validate_task(task)
            assert not is_valid, f"Task should be invalid: '{task}'"
            assert error is not None
    
    def test_prohibited_network_operations(self):
        """Direct network operations should be rejected."""
        invalid_tasks = [
            "Send email to admin@example.com",
            "SSH into server 192.168.1.1",
            "FTP upload file to server",
            "Connect to database",
            "Make API request to endpoint",
        ]
        
        for task in invalid_tasks:
            is_valid, error = self.validator.validate_task(task)
            assert not is_valid, f"Task should be invalid: '{task}'"
            assert error is not None
    
    def test_prohibited_security_exploits(self):
        """Security exploits should be rejected."""
        invalid_tasks = [
            "Inject SQL into database",
            "Run exploit script",
            "Execute XSS attack",
            "Bypass authentication",
            "Crack password hash",
        ]
        
        for task in invalid_tasks:
            is_valid, error = self.validator.validate_task(task)
            assert not is_valid, f"Task should be invalid: '{task}'"
            assert error is not None
    
    def test_valid_http_urls(self):
        """HTTP and HTTPS URLs should pass validation."""
        valid_urls = [
            "https://example.com",
            "http://example.com",
            "https://subdomain.example.com",
            "https://example.com:8080/path",
            "http://localhost:3000",
            "https://example.com/path?query=value#fragment",
        ]
        
        for url in valid_urls:
            is_valid, error = self.validator.validate_url(url)
            assert is_valid, f"URL should be valid: '{url}' - Error: {error}"
    
    def test_prohibited_url_schemes(self):
        """Non-HTTP URL schemes should be rejected."""
        invalid_urls = [
            "file:///etc/passwd",
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "ftp://example.com/file.txt",
            "ssh://server.com",
            "tel:+1234567890",
            "mailto:test@example.com",
        ]
        
        for url in invalid_urls:
            is_valid, error = self.validator.validate_url(url)
            assert not is_valid, f"URL should be invalid: '{url}'"
            assert error is not None
            assert "scheme" in error.lower() or "not allowed" in error.lower()
    
    def test_empty_and_invalid_inputs(self):
        """Empty and malformed inputs should be rejected."""
        # Empty task
        is_valid, error = self.validator.validate_task("")
        assert not is_valid
        assert "empty" in error.lower()
        
        # Empty URL
        is_valid, error = self.validator.validate_url("")
        assert not is_valid
        assert "empty" in error.lower()
        
        # Malformed URL
        is_valid, error = self.validator.validate_url("not a url")
        assert not is_valid
    
    def test_validation_hints(self):
        """Validator should provide helpful hints."""
        hints = self.validator.get_validation_hints("Do something with data")
        assert hints is not None
        assert len(hints) > 0
        # Hints should suggest adding URL or browser action
        hints_text = " ".join(hints).lower()
        assert "url" in hints_text or "browser" in hints_text or "action" in hints_text
    
    def test_skip_browser_keyword_check(self):
        """SDK methods should be able to skip browser keyword validation."""
        # Without skip - should fail (no browser keywords)
        is_valid, error = self.validator.validate_task("Get the top 5 items")
        assert not is_valid, "Should fail without browser keywords"
        
        # With skip - should pass (SDK method already defines operation)
        is_valid, error = self.validator.validate_task(
            "Get the top 5 items",
            skip_browser_keyword_check=True
        )
        assert is_valid, f"Should pass with skip_browser_keyword_check=True: {error}"
        
        # Prohibited keywords should still be blocked even with skip
        is_valid, error = self.validator.validate_task(
            "delete file /etc/passwd",
            skip_browser_keyword_check=True
        )
        assert not is_valid, "Prohibited keywords should still be blocked"


class TestGoalInterpreterFastPath:
    """Test suite for GoalInterpreter fast-path patterns."""
    
    def setup_method(self):
        """Create interpreter instance before each test."""
        self.interpreter = GoalInterpreter()
    
    def test_navigation_patterns(self):
        """Navigation goals should map to navigate tool."""
        test_cases = [
            ("Navigate to https://example.com", {"url": "https://example.com"}),
            ("Go to https://google.com", {"url": "https://google.com"}),
            ("Visit https://github.com", {"url": "https://github.com"}),
            ("Open https://reddit.com", {"url": "https://reddit.com"}),
        ]
        
        for goal, expected_params in test_cases:
            action = self.interpreter.parse_goal(goal)
            assert action is not None, f"Should match: '{goal}'"
            assert action.tool_name == "navigate"
            assert action.parameters == expected_params
    
    def test_page_state_patterns(self):
        """Page state goals should map to get_page_state tool."""
        test_cases = [
            "Get page state",
            "Inspect page",
            "Identify navigation links",
            "List all links",
            "Find navigation",
        ]
        
        for goal in test_cases:
            action = self.interpreter.parse_goal(goal)
            assert action is not None, f"Should match: '{goal}'"
            assert action.tool_name == "get_page_state"
    
    def test_scroll_patterns(self):
        """Scroll goals should map to scroll tool with correct parameters."""
        test_cases = [
            ("Scroll down", {"direction": "down"}),
            ("Scroll up", {"direction": "up"}),
            ("Page down", {"direction": "down"}),
            ("Page up", {"direction": "up"}),
            ("Scroll to bottom", {"direction": "down", "amount": "full"}),
            ("Scroll to top", {"direction": "up", "amount": "full"}),
        ]
        
        for goal, expected_params in test_cases:
            action = self.interpreter.parse_goal(goal)
            assert action is not None, f"Should match: '{goal}'"
            assert action.tool_name == "scroll"
            assert action.parameters == expected_params
    
    def test_wait_patterns(self):
        """Wait goals should map to wait tool with correct parameters."""
        test_cases = [
            ("Wait 5 seconds", {"seconds": 5}),
            ("Wait 10 seconds", {"seconds": 10}),
            ("Wait for page load", {"condition": "page_load"}),
            ("Wait for page", {"condition": "page_load"}),
        ]
        
        for goal, expected_params in test_cases:
            action = self.interpreter.parse_goal(goal)
            assert action is not None, f"Should match: '{goal}'"
            assert action.tool_name == "wait"
            assert action.parameters == expected_params
    
    def test_click_patterns_with_selectors(self):
        """Click goals with explicit selectors should map to click tool."""
        test_cases = [
            ("Click #login-btn", {"selector": "#login-btn"}),
            ("Click .submit", {"selector": ".submit"}),
            ("Click button#submit", {"selector": "#submit"}),
        ]
        
        for goal, expected_params in test_cases:
            action = self.interpreter.parse_goal(goal)
            assert action is not None, f"Should match: '{goal}'"
            assert action.tool_name == "click"
            assert action.parameters == expected_params
    
    def test_type_patterns_with_selectors(self):
        """Type goals with explicit selectors and text should map to type tool."""
        test_cases = [
            ('Type "hello" in #search', {"selector": "#search", "text": "hello"}),
            ("Type 'world' in .input", {"selector": ".input", "text": "world"}),
            ('Fill "username" into #login', {"selector": "#login", "text": "username"}),
            ("Enter 'test' in #email", {"selector": "#email", "text": "test"}),
        ]
        
        for goal, expected_params in test_cases:
            action = self.interpreter.parse_goal(goal)
            assert action is not None, f"Should match: '{goal}'"
            assert action.tool_name == "type"
            assert action.parameters == expected_params
    
    def test_extract_patterns(self):
        """Extract goals should map to extract_text tool."""
        test_cases = [
            "Extract text from page",
            "Get text from the page",
            "Read text content",
        ]
        
        for goal in test_cases:
            action = self.interpreter.parse_goal(goal)
            assert action is not None, f"Should match: '{goal}'"
            assert action.tool_name == "extract_text"
    
    def test_screenshot_patterns(self):
        """Screenshot goals should map to screenshot tool."""
        test_cases = [
            "Take screenshot",
            "Capture screen",
            "Screenshot",
        ]
        
        for goal in test_cases:
            action = self.interpreter.parse_goal(goal)
            assert action is not None, f"Should match: '{goal}'"
            assert action.tool_name == "screenshot"
    
    def test_refresh_patterns(self):
        """Refresh goals should map to refresh tool."""
        test_cases = [
            "Refresh",
            "Reload page",
            "Reload the page",
        ]
        
        for goal in test_cases:
            action = self.interpreter.parse_goal(goal)
            assert action is not None, f"Should match: '{goal}'"
            assert action.tool_name == "refresh"
    
    def test_ambiguous_goals_need_llm(self):
        """Ambiguous goals should return None (need LLM reasoning)."""
        ambiguous_goals = [
            "Click the login button",  # No explicit selector
            "Fill in the username field",  # No explicit selector or text
            "Find the submit button and click it",  # Complex, multi-step
            "Extract product prices",  # Needs context
        ]
        
        for goal in ambiguous_goals:
            action = self.interpreter.parse_goal(goal)
            assert action is None, f"Should not match (needs LLM): '{goal}'"
    
    def test_vision_skip_logic(self):
        """Should correctly determine when to skip vision."""
        # Should skip vision for navigation
        assert self.interpreter.should_skip_vision("Navigate to https://example.com")
        assert self.interpreter.should_skip_vision("Go to https://google.com")
        
        # Should skip vision for page state inspection
        assert self.interpreter.should_skip_vision("Get page state")
        assert self.interpreter.should_skip_vision("Inspect page")
        
        # Should NOT skip vision for other actions
        assert not self.interpreter.should_skip_vision("Click the button")
        assert not self.interpreter.should_skip_vision("Extract data")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
