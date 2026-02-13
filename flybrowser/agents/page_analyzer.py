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
Page Analyzer - Multi-Screenshot Vision Analysis

Analyzes multiple screenshots of a webpage to build comprehensive understanding
including sections, navigation, content structure, and summaries.

Uses StructuredLLMWrapper for reliable JSON output with automatic repair.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from flybrowser.agents.page_map import PageMap, PageSection, SectionType
from flybrowser.agents.config import PageExplorationConfig
from flybrowser.agents.structured_llm import StructuredLLMWrapper
from flybrowser.agents.schemas import PAGE_ANALYSIS_SCHEMA

if TYPE_CHECKING:
    from flybrowser.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class PageAnalyzer:
    """
    Analyzes PageMaps with multiple screenshots using vision models.
    
    Sends all screenshots to LLM with spatial context to extract:
    - Page sections (header, nav, content, footer)
    - Navigation elements and links
    - Content hierarchy
    - Overall summary
    """
    
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        prompt_manager: Any,
        config: Optional[PageExplorationConfig] = None,
    ) -> None:
        """
        Initialize page analyzer.
        
        Args:
            llm_provider: LLM provider with vision capabilities
            prompt_manager: Prompt manager for templates
            config: Page exploration configuration
        """
        self.llm = llm_provider
        self.prompt_manager = prompt_manager
        self.config = config or PageExplorationConfig()
    
    async def analyze_page_map(self, page_map: PageMap) -> PageMap:
        """
        Analyze PageMap screenshots and populate structural understanding.
        
        Args:
            page_map: PageMap with captured screenshots
            
        Returns:
            Updated PageMap with sections, navigation, and summary populated
        """
        if not self.config.enable_multi_screenshot_analysis:
            logger.info("[PageAnalyzer] Multi-screenshot analysis disabled")
            return page_map
        
        if len(page_map.screenshots) < self.config.min_screenshots_for_analysis:
            logger.warning(
                f"[PageAnalyzer] Insufficient screenshots "
                f"({len(page_map.screenshots)} < {self.config.min_screenshots_for_analysis}), "
                f"skipping analysis"
            )
            return page_map
        
        logger.info(
            f"🔬 [PageAnalyzer] Analyzing {len(page_map.screenshots)} screenshots "
            f"for {page_map.url}"
        )
        
        try:
            # Build analysis prompt with spatial context
            prompt = self._build_analysis_prompt(page_map)
            
            # Prepare screenshot images
            images = [
                screenshot.image_data 
                for screenshot in page_map.screenshots
            ]
            
            # Send to vision model with structured output
            logger.info(
                f"[PageAnalyzer] Sending {len(images)} images to LLM "
                f"(total: {sum(len(img) for img in images) // 1024}KB)"
            )
            
            # Use StructuredLLMWrapper for reliable JSON output with repair
            wrapper = StructuredLLMWrapper(
                llm_provider=self.llm,
                max_repair_attempts=2,
                repair_temperature=0.1,
            )
            
            analysis_data = await wrapper.generate_structured_with_vision(
                prompt=prompt["user"],
                image_data=images,
                schema=PAGE_ANALYSIS_SCHEMA,
                system_prompt=prompt["system"],
                temperature=self.config.analysis_temperature,
                max_tokens=self.config.analysis_max_tokens,
            )
            
            # Parse structured response into PageSection objects
            analysis_result = self._parse_structured_analysis(analysis_data, page_map)
            
            # Populate PageMap
            page_map.sections = analysis_result.get("sections", [])
            page_map.navigation_structure = analysis_result.get("navigation_structure", {})
            page_map.summary = analysis_result.get("summary", "")
            page_map.main_content_area = analysis_result.get("main_content_area")
            page_map.analysis_complete = True
            
            logger.info(
                f" [PageAnalyzer] Analysis complete: "
                f"{len(page_map.sections)} sections identified, "
                f"{len(page_map.get_all_navigation_links())} links found"
            )
            
            return page_map
            
        except ValueError as e:
            # Structured output validation failed after repair attempts
            logger.error(f"[PageAnalyzer] Structured output validation failed: {e}")
            return page_map
        except Exception as e:
            logger.exception(f"[PageAnalyzer] Analysis failed: {e}")
            # Return original PageMap on failure
            return page_map
    
    def _build_analysis_prompt(self, page_map: PageMap) -> Dict[str, str]:
        """
        Build prompt for multi-screenshot analysis.
        
        Args:
            page_map: PageMap to analyze
            
        Returns:
            Dictionary with system and user prompts
        """
        # Build screenshot position context
        screenshot_positions = []
        for screenshot in page_map.screenshots:
            area = screenshot.visible_area
            screenshot_positions.append({
                "index": screenshot.index,
                "scroll_y": screenshot.scroll_position.y,
                "showing_range": f"{area['top']}-{area['bottom']}px",
                "percentage": f"{(area['top'] / page_map.total_height * 100):.1f}%"
            })
        
        # Try to get template from PromptManager
        try:
            prompts = self.prompt_manager.get_prompt(
                "page_multi_screenshot_analysis",
                page_url=page_map.url,
                page_title=page_map.title,
                total_height=page_map.total_height,
                total_width=page_map.total_width,
                num_screenshots=len(page_map.screenshots),
                screenshot_positions=json.dumps(screenshot_positions, indent=2),
                coverage=f"{page_map.get_coverage_percentage():.1f}%"
            )
            return prompts
        except Exception as e:
            logger.warning(f"[PageAnalyzer] Failed to load template, using fallback: {e}")
            # Fallback prompt
            return {
                "system": self._get_fallback_system_prompt(),
                "user": self._get_fallback_user_prompt(page_map, screenshot_positions)
            }
    
    def _get_fallback_system_prompt(self) -> str:
        """Fallback system prompt if template not found."""
        return """You are an expert web page analyzer. Your task is to analyze multiple screenshots of a webpage taken at different scroll positions and provide comprehensive structural understanding.

You will receive multiple screenshots showing different parts of the same webpage from top to bottom. Each screenshot is labeled with its scroll position.

Your analysis should identify:
1. **Sections**: Distinct visual/functional sections (header, navigation, hero, content areas, footer)
2. **Navigation**: All navigation elements and their links
3. **Content Structure**: Hierarchy and organization of content
4. **Summary**: Brief overview of page purpose and key information

Respond in JSON format with this structure:
{
  "sections": [
    {
      "type": "header|navigation|hero|content|sidebar|footer",
      "name": "Section name",
      "description": "What this section contains",
      "scroll_range": {"start_y": 0, "end_y": 800},
      "screenshot_indices": [0, 1],
      "elements": ["Element 1", "Element 2"],
      "navigation_links": [{"text": "Link", "description": "Where it goes"}]
    }
  ],
  "navigation_structure": {
    "main_menu": ["Link 1", "Link 2"],
    "footer_links": ["About", "Contact"]
  },
  "main_content_area": {"start_y": 800, "end_y": 3500},
  "summary": "Overall page summary"
}"""
    
    def _get_fallback_user_prompt(
        self,
        page_map: PageMap,
        screenshot_positions: List[Dict[str, Any]]
    ) -> str:
        """Fallback user prompt if template not found."""
        return f"""Analyze these screenshots of the webpage: **{page_map.title}**

URL: {page_map.url}
Page dimensions: {page_map.total_width}x{page_map.total_height}px
Coverage: {page_map.get_coverage_percentage():.1f}%

Screenshot positions:
{json.dumps(screenshot_positions, indent=2)}

The screenshots show the page from top to bottom at these scroll positions. Analyze the ENTIRE page structure across all screenshots and provide comprehensive JSON analysis following the schema."""
    
    def _parse_structured_analysis(
        self,
        analysis_data: Dict[str, Any],
        page_map: PageMap
    ) -> Dict[str, Any]:
        """
        Parse structured LLM response into analysis result with PageSection objects.
        
        The StructuredLLMWrapper has already validated the JSON schema,
        so we just need to convert to domain objects.
        
        Args:
            analysis_data: Validated structured response from LLM
            page_map: Original PageMap for context
            
        Returns:
            Dictionary with sections, navigation, summary
        """
        # Convert sections to PageSection objects
        sections = []
        for section_data in analysis_data.get("sections", []):
            try:
                section_type = SectionType(section_data.get("type", "unknown"))
            except ValueError:
                section_type = SectionType.UNKNOWN
            
            section = PageSection(
                type=section_type,
                name=section_data.get("name", "Unnamed Section"),
                description=section_data.get("description", ""),
                scroll_range=section_data.get("scroll_range", {"start_y": 0, "end_y": 0}),
                screenshot_indices=section_data.get("screenshot_indices", []),
                elements=section_data.get("elements", []),
                navigation_links=section_data.get("navigation_links", []),
            )
            sections.append(section)
        
        return {
            "sections": sections,
            "navigation_structure": analysis_data.get("navigation_structure", {}),
            "summary": analysis_data.get("summary", ""),
            "main_content_area": analysis_data.get("main_content_area"),
        }
