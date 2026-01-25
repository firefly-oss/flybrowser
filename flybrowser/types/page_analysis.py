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
Type definitions for intelligent page analysis system.

This module provides types for the hybrid heuristic/LLM/VLM page analysis
system that intelligently detects interactive elements on web pages.

The analysis system works in three tiers:
    1. JavaScript Heuristics (fast, free, 80% of cases)
    2. LLM HTML Analysis (smart, ~$0.001 per page, 20% of cases)
    3. VLM Visual Analysis (most accurate, ~$0.01 per page, opt-in)

Classes:
    AnalysisMethod: Enum of detection methods
    ElementType: Types of interactive elements
    ElementPurpose: Semantic purpose classification
    InteractiveElement: Represents a detected element
    AnalysisResult: Complete analysis results with metadata
    PageAnalysisConfig: Configuration for analysis behavior

Example:
    >>> config = PageAnalysisConfig(enable_llm_html_analysis=True)
    >>> analyzer = PageAnalyzer(llm_provider, config)
    >>> result = await analyzer.analyze_page(page)
    >>> print(f"Found {len(result.all_elements)} elements via {result.method}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class AnalysisMethod(str, Enum):
    """Method used to detect interactive elements."""
    
    HEURISTIC = "heuristic"      # JavaScript-based detection
    LLM_HTML = "llm_html"        # LLM analysis of HTML structure
    VLM_VISUAL = "vlm_visual"    # Vision model analysis of screenshot
    HYBRID = "hybrid"            # Combination of methods


class ElementType(str, Enum):
    """Type of interactive element."""
    
    BUTTON = "button"
    LINK = "link"
    MENU = "menu"
    MENU_TOGGLE = "menu_toggle"
    FORM_INPUT = "form_input"
    FORM_SUBMIT = "form_submit"
    DROPDOWN = "dropdown"
    TAB = "tab"
    MODAL_TRIGGER = "modal_trigger"
    UNKNOWN = "unknown"


class ElementPurpose(str, Enum):
    """Semantic purpose of an element."""
    
    NAVIGATION = "navigation"
    LANGUAGE_SWITCH = "language_switch"
    SEARCH = "search"
    LOGIN = "login"
    SIGNUP = "signup"
    MENU_CONTROL = "menu_control"
    FORM_SUBMISSION = "form_submission"
    CONTENT_ACTION = "content_action"
    UNKNOWN = "unknown"


@dataclass
class InteractiveElement:
    """
    Represents an interactive element detected on a page.
    
    Can be detected via JavaScript heuristics, LLM HTML analysis,
    or VLM visual analysis.
    """
    
    # Identity
    element_id: str                          # Unique identifier
    element_type: ElementType                # Type of element
    purpose: ElementPurpose = ElementPurpose.UNKNOWN
    
    # Content
    text: str = ""                           # Visible text content
    aria_label: Optional[str] = None         # Accessibility label
    title: Optional[str] = None              # Title attribute
    
    # Selection
    selector: Optional[str] = None           # CSS selector (if available)
    xpath: Optional[str] = None              # XPath (if available)
    bounding_box: Optional[Dict[str, float]] = None  # {x, y, width, height}
    
    # State
    is_visible: bool = True                  # Element is visible
    is_enabled: bool = True                  # Element is enabled
    is_in_viewport: bool = False             # Element is in current viewport
    
    # Analysis metadata
    detected_by: AnalysisMethod = AnalysisMethod.HEURISTIC
    confidence: float = 1.0                  # Confidence score (0.0-1.0)
    reasoning: str = ""                      # Why this element was identified
    
    # Navigation context
    href: Optional[str] = None               # Link URL (for links)
    target: Optional[str] = None             # Link target (_blank, etc.)
    
    # Additional attributes
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """
    Result of page analysis containing detected elements.
    
    Includes metadata about which analysis method was used
    and overall confidence in the results.
    """
    
    # Elements detected
    buttons: List[InteractiveElement] = field(default_factory=list)
    links: List[InteractiveElement] = field(default_factory=list)
    menu_controls: List[InteractiveElement] = field(default_factory=list)
    forms: List[InteractiveElement] = field(default_factory=list)
    all_elements: List[InteractiveElement] = field(default_factory=list)
    
    # Analysis metadata
    method: AnalysisMethod = AnalysisMethod.HEURISTIC
    methods_used: List[AnalysisMethod] = field(default_factory=list)
    overall_confidence: float = 1.0
    
    # Performance metrics
    analysis_time_ms: float = 0.0
    token_count: int = 0                     # For LLM/VLM analysis
    cost_usd: float = 0.0                    # Estimated cost
    
    # Issues detected
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def get_menu_buttons(self) -> List[InteractiveElement]:
        """Get all menu toggle/control buttons."""
        return [e for e in self.all_elements 
                if e.element_type in [ElementType.MENU, ElementType.MENU_TOGGLE]]
    
    def get_navigation_links(self) -> List[InteractiveElement]:
        """Get all navigation links (excluding language switchers)."""
        return [e for e in self.links 
                if e.purpose != ElementPurpose.LANGUAGE_SWITCH]
    
    def get_language_switches(self) -> List[InteractiveElement]:
        """Get language switcher elements."""
        return [e for e in self.all_elements 
                if e.purpose == ElementPurpose.LANGUAGE_SWITCH]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "method": self.method.value,
            "methods_used": [m.value for m in self.methods_used],
            "confidence": self.overall_confidence,
            "buttons_count": len(self.buttons),
            "links_count": len(self.links),
            "menu_controls_count": len(self.menu_controls),
            "navigation_links": [
                {"text": e.text, "href": e.href, "confidence": e.confidence}
                for e in self.get_navigation_links()[:10]
            ],
            "menu_buttons": [
                {"text": e.text or e.aria_label, "type": e.element_type.value}
                for e in self.get_menu_buttons()
            ],
            "language_switches": [
                {"text": e.text, "href": e.href}
                for e in self.get_language_switches()
            ],
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "performance": {
                "time_ms": self.analysis_time_ms,
                "tokens": self.token_count,
                "cost_usd": self.cost_usd,
            }
        }


# PageAnalysisConfig has been moved to flybrowser.agents.config
# Import it from there:
#   from flybrowser.agents.config import PageAnalysisConfig
