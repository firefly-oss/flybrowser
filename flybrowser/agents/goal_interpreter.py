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
Goal Interpreter for FlyBrowser ReAct Agent.

Uses LLM to intelligently interpret goals and determine the best action,
similar to how Stagehand handles intent recognition.

Only uses fast-path for trivial cases like explicit URL navigation.
All other goals go through the main ReAct loop with full LLM reasoning.
"""

import re
import logging
from typing import Optional

from flybrowser.agents.types import Action

logger = logging.getLogger(__name__)


class GoalInterpreter:
    """
    Goal interpreter that only fast-paths trivial navigation goals.
    
    All other goals are delegated to the main ReAct loop where the LLM
    has full context (including vision if available) to make intelligent
    decisions about which tool to use.
    """
    
    def parse_goal(self, goal: str, current_goal_desc: Optional[str] = None) -> Optional[Action]:
        """
        Parse a goal and return a direct Action ONLY for trivial URL navigation.
        
        All other goals are delegated to the main ReAct loop where the LLM
        has full context (page state, vision, history) to make intelligent
        decisions about which tool to use.
        
        Args:
            goal: Full task description
            current_goal_desc: More specific goal description from planner
            
        Returns:
            Action for trivial navigation goals, None for everything else
        """
        # Use current_goal_desc if available (more specific than full task)
        goal_text = current_goal_desc or goal
        goal_lower = goal_text.lower().strip()
        
        logger.debug(f"[GoalInterpreter] Parsing goal: {goal_text}")
        
        # ONLY FAST-PATH: Explicit URL navigation
        # This is deterministic and doesn't need LLM context
        action = self._extract_navigation_url(goal_text, goal_lower)
        if action:
            logger.info(f" [GoalInterpreter] Fast-path navigation: {action.parameters.get('url')}")
            return action
        
        # ALL other goals go through the main ReAct loop with full LLM reasoning
        # The LLM will see:
        # - Current page state (via memory context)
        # - Screenshot (if vision enabled)
        # - Available tools with descriptions
        # - Execution plan context
        # This allows intelligent tool selection based on actual page content
        logger.debug(f"[GoalInterpreter] Delegating to ReAct loop for LLM reasoning")
        return None
    
    def _extract_navigation_url(self, goal_text: str, goal_lower: str) -> Optional[Action]:
        """
        Extract URL from navigation goals - the ONLY fast-path.
        
        This handles trivial cases like:
        - "Navigate to https://example.com"
        - "Go to https://example.com"
        - "Visit https://example.com"
        
        Returns:
            Action with navigate tool if URL found in navigation context, None otherwise
        """
        # Look for explicit URLs in the goal
        url_pattern = r'https?://[^\s)>\]"\']+(?:\.[^\s)>\]"\']+)*'
        matches = re.findall(url_pattern, goal_text, re.IGNORECASE)
        
        if matches:
            # Check if this is a navigation context (not just mentioning a URL)
            nav_keywords = [
                'navigate', 'go to', 'visit', 'open', 'access', 
                'load', 'browse', 'url', 'website', 'site'
            ]
            if any(keyword in goal_lower for keyword in nav_keywords):
                url = matches[0].rstrip('.,;:')  # Clean trailing punctuation
                return Action(
                    tool_name="navigate",
                    parameters={"url": url}
                )
        
        return None
    
    def should_skip_vision(self, goal: str) -> bool:
        """
        Determine if vision should be skipped for this goal.
        
        For explicit navigation goals with URLs, we don't need to see
        the current page - we're about to leave it anyway!
        
        Returns:
            True if vision should be skipped for this goal
        """
        goal_lower = goal.lower()
        
        # Skip vision only for explicit navigation goals with URLs
        if self._extract_navigation_url(goal, goal_lower):
            return True
        
        # All other goals benefit from vision context
        return False
