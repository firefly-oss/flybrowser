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
Prompt manager with A/B testing and performance tracking.

This module provides the PromptManager class which orchestrates prompt template
usage with advanced features like A/B testing and performance tracking.

Features:
- Centralized prompt retrieval and rendering
- A/B testing with weighted variant selection
- Performance tracking (usage count, success rate)
- Automatic variant selection based on performance
- Integration with PromptRegistry for template storage

The PromptManager enables data-driven prompt optimization by tracking which
prompt variants perform best and automatically adjusting selection weights.

Example:
    >>> from flybrowser.prompts.registry import PromptRegistry
    >>> from flybrowser.prompts.manager import PromptManager
    >>>
    >>> registry = PromptRegistry()
    >>> manager = PromptManager(registry)
    >>>
    >>> # Get and render a prompt
    >>> prompt = manager.get_prompt(
    ...     "data_extraction",
    ...     data_type="emails",
    ...     content="Contact us at..."
    ... )
    >>>
    >>> # Record success for performance tracking
    >>> manager.record_success("data_extraction")
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional

from flybrowser.prompts.registry import PromptRegistry
from flybrowser.prompts.template import PromptTemplate
from flybrowser.utils.logger import logger

# Optional integration with ExperimentManager
try:
    from flybrowser.prompts.experiments import get_experiment_manager
    EXPERIMENTS_AVAILABLE = True
except ImportError:
    EXPERIMENTS_AVAILABLE = False


class PromptManager:
    """
    Manages prompts with A/B testing and performance tracking.

    This class provides a high-level interface for working with prompt templates.
    It handles template retrieval, rendering, A/B testing, and performance tracking.

    Attributes:
        registry: PromptRegistry instance for template storage
        ab_tests: Mapping of template names to their variant IDs
        ab_weights: Mapping of template names to variant selection weights

    Example:
        >>> manager = PromptManager()
        >>>
        >>> # Simple usage
        >>> prompt = manager.get_prompt(
        ...     "data_extraction",
        ...     data_type="product names",
        ...     content="<html>...</html>"
        ... )
        >>>
        >>> # Setup A/B test
        >>> manager.setup_ab_test(
        ...     "data_extraction",
        ...     variants=["v1", "v2"],
        ...     weights={"v1": 0.5, "v2": 0.5}
        ... )
        >>>
        >>> # Get prompt (will select variant based on weights)
        >>> prompt = manager.get_prompt("data_extraction", enable_ab_testing=True)
        >>>
        >>> # Record success
        >>> manager.record_success("data_extraction", variant="v1")
    """

    def __init__(self, registry: Optional[PromptRegistry] = None, enable_experiments: bool = True) -> None:
        """
        Initialize the prompt manager.

        Args:
            registry: PromptRegistry instance for template storage.
                If not provided, creates a new default registry.
            enable_experiments: Whether to enable ExperimentManager integration

        Example:
            >>> from flybrowser.prompts.registry import PromptRegistry
            >>> registry = PromptRegistry()
            >>> manager = PromptManager(registry)

            Or with default registry:
            >>> manager = PromptManager()
        """
        self.registry = registry or PromptRegistry()
        self.ab_tests: Dict[str, List[str]] = {}  # template_name -> [variant_ids]
        self.ab_weights: Dict[str, Dict[str, float]] = {}  # template_name -> {variant: weight}
        self._experiment_manager = None
        self._enable_experiments = enable_experiments and EXPERIMENTS_AVAILABLE
        self._active_renders: Dict[str, Dict[str, Any]] = {}  # Track active renders for metrics

    def get_prompt(
        self,
        name: str,
        version: Optional[str] = None,
        enable_ab_testing: bool = True,
        experiment_id: Optional[str] = None,
        **variables: Any,
    ) -> Dict[str, str]:
        """
        Get and render a prompt template.

        Args:
            name: Template name
            version: Template version
            enable_ab_testing: Whether to use A/B testing
            experiment_id: Optional experiment ID for tracking
            **variables: Variables to render the template

        Returns:
            Dictionary with rendered prompts
        """
        variant_used = None
        start_time = time.time()
        
        # Check for active experiments if enabled
        if self._enable_experiments:
            em = self._get_experiment_manager()
            if em and experiment_id:
                # Get active experiment
                try:
                    experiments = em.list_experiments()
                    for exp in experiments:
                        if exp['id'] == experiment_id and exp['status'] == 'active':
                            # Select variant from experiment
                            variant_used = em.select_variant(experiment_id)
                            logger.debug(f"Selected variant '{variant_used}' from experiment '{experiment_id}'")
                            break
                except Exception as e:
                    logger.debug(f"No active experiment tracking: {e}")
        
        # Get template (with A/B testing if enabled)
        if enable_ab_testing and name in self.ab_tests:
            template = self._select_variant(name)
        else:
            template = self.registry.get(name, version)

        # Render template
        try:
            rendered = template.render(**variables)
            render_time = (time.time() - start_time) * 1000  # ms
            
            # Track render for metrics
            render_id = f"{name}_{id(rendered)}"
            self._active_renders[render_id] = {
                'name': name,
                'variant': variant_used or template.variant or 'default',
                'experiment_id': experiment_id,
                'start_time': start_time,
                'render_time_ms': render_time,
            }
            
            logger.debug(f"Rendered prompt: {name} (variant: {template.variant or 'default'}, time: {render_time:.0f}ms)")
            return rendered
        except Exception as e:
            logger.error(f"Failed to render prompt {name}: {e}")
            # Record failure if experiment is active
            if self._enable_experiments and experiment_id and variant_used:
                em = self._get_experiment_manager()
                if em:
                    try:
                        em.record_result(experiment_id, variant_used, success=False, error=str(e))
                    except Exception:
                        pass
            raise

    def record_success(
        self,
        name: str,
        version: Optional[str] = None,
        variant: Optional[str] = None,
        experiment_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
        tokens: Optional[int] = None,
    ) -> None:
        """
        Record a successful use of a prompt.

        Args:
            name: Template name
            version: Template version
            variant: Variant identifier
            experiment_id: Optional experiment ID for tracking
            latency_ms: Latency in milliseconds
            tokens: Token count
        """
        template = self.registry.get(name, version, variant)
        template.record_success()
        
        # Record to experiment if active
        if self._enable_experiments and experiment_id and variant:
            em = self._get_experiment_manager()
            if em:
                try:
                    em.record_result(
                        experiment_id,
                        variant,
                        success=True,
                        latency_ms=latency_ms,
                        tokens=tokens,
                    )
                    logger.debug(f"Recorded experiment result: {experiment_id}/{variant} (success)")
                except Exception as e:
                    logger.debug(f"Failed to record experiment result: {e}")
        
        logger.debug(
            f"Recorded success for {name} "
            f"(success rate: {template.get_success_rate():.2%})"
        )

    def create_ab_test(
        self,
        template_name: str,
        variants: List[PromptTemplate],
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Create an A/B test for a template.

        Args:
            template_name: Base template name
            variants: List of variant templates
            weights: Optional weights for variant selection (default: equal)
        """
        # Register variants
        variant_ids = []
        for variant in variants:
            self.registry.register(variant)
            variant_id = variant.variant or variant.name
            variant_ids.append(variant_id)

        self.ab_tests[template_name] = variant_ids

        # Set weights (equal by default)
        if weights:
            self.ab_weights[template_name] = weights
        else:
            equal_weight = 1.0 / len(variants)
            self.ab_weights[template_name] = {vid: equal_weight for vid in variant_ids}

        logger.info(
            f"Created A/B test for {template_name} with {len(variants)} variants"
        )
    
    def _get_experiment_manager(self):
        """Lazy-load ExperimentManager."""
        if not self._enable_experiments:
            return None
        if self._experiment_manager is None and EXPERIMENTS_AVAILABLE:
            try:
                self._experiment_manager = get_experiment_manager()
            except Exception as e:
                logger.debug(f"Could not initialize ExperimentManager: {e}")
                self._enable_experiments = False
        return self._experiment_manager

    def _select_variant(self, template_name: str) -> PromptTemplate:
        """
        Select a variant for A/B testing.

        Args:
            template_name: Template name

        Returns:
            Selected template variant
        """
        variants = self.ab_tests[template_name]
        weights = self.ab_weights[template_name]

        # Weighted random selection
        variant_id = random.choices(
            list(weights.keys()), weights=list(weights.values()), k=1
        )[0]

        return self.registry.get(template_name, variant=variant_id)

    def get_ab_test_results(self, template_name: str) -> Dict[str, Any]:
        """
        Get A/B test results for a template.

        Args:
            template_name: Template name

        Returns:
            Dictionary with test results
        """
        if template_name not in self.ab_tests:
            return {"error": "No A/B test found for this template"}

        results = {
            "template": template_name,
            "variants": [],
        }

        for variant_id in self.ab_tests[template_name]:
            template = self.registry.get(template_name, variant=variant_id)
            results["variants"].append(
                {
                    "variant_id": variant_id,
                    "usage_count": template.usage_count,
                    "success_count": template.success_count,
                    "success_rate": template.get_success_rate(),
                    "weight": self.ab_weights[template_name].get(variant_id, 0.0),
                }
            )

        # Sort by success rate
        results["variants"].sort(key=lambda x: x["success_rate"], reverse=True)

        return results

    def optimize_weights(self, template_name: str, method: str = "thompson") -> None:
        """
        Optimize variant weights based on performance.

        Args:
            template_name: Template name
            method: Optimization method ('thompson', 'epsilon_greedy', 'ucb')
        """
        if template_name not in self.ab_tests:
            logger.warning(f"No A/B test found for {template_name}")
            return

        variants = self.ab_tests[template_name]
        
        if method == "thompson":
            # Thompson sampling
            weights = {}
            for variant_id in variants:
                template = self.registry.get(template_name, variant=variant_id)
                # Beta distribution parameters
                alpha = template.success_count + 1
                beta = (template.usage_count - template.success_count) + 1
                # Sample from beta distribution
                weights[variant_id] = random.betavariate(alpha, beta)
            
            # Normalize weights
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            
            self.ab_weights[template_name] = weights
            logger.info(f"Optimized weights for {template_name} using Thompson sampling")

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive performance report.

        Returns:
            Dictionary with performance metrics
        """
        stats = self.registry.get_stats()
        
        # Add A/B test information
        stats["ab_tests"] = {
            name: self.get_ab_test_results(name) for name in self.ab_tests.keys()
        }
        
        return stats

