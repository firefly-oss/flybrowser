# Copyright 2026 Firefly Software Solutions Inc

"""
A/B Testing Infrastructure for Prompt Templates

This module provides functionality for:
- Creating and managing prompt variants
- Running A/B experiments
- Tracking experiment results
- Statistical analysis of variant performance
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

from flybrowser.prompts.template import PromptTemplate
from flybrowser.utils.logger import logger


class ExperimentStatus(Enum):
    """Experiment lifecycle status"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class VariantMetrics:
    """Metrics for a single variant"""
    variant_id: str
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    error_messages: List[str] = field(default_factory=list)
    
    def add_result(
        self, 
        success: bool, 
        latency_ms: float = 0.0, 
        tokens: int = 0,
        error: Optional[str] = None
    ):
        """Record a single usage result"""
        self.usage_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
            if error:
                self.error_messages.append(error[:200])  # Limit error message length
        
        self.total_latency_ms += latency_ms
        self.total_tokens += tokens
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0-1.0)"""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency"""
        if self.usage_count == 0:
            return 0.0
        return self.total_latency_ms / self.usage_count
    
    @property
    def avg_tokens(self) -> float:
        """Calculate average token usage"""
        if self.usage_count == 0:
            return 0.0
        return self.total_tokens / self.usage_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'variant_id': self.variant_id,
            'usage_count': self.usage_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': self.success_rate,
            'avg_latency_ms': self.avg_latency_ms,
            'avg_tokens': self.avg_tokens,
            'recent_errors': self.error_messages[-5:],  # Last 5 errors
        }


@dataclass
class Experiment:
    """A/B testing experiment configuration"""
    experiment_id: str
    name: str
    template_name: str
    description: str
    variants: List[str]  # List of variant IDs
    traffic_split: Dict[str, float]  # variant_id -> percentage (0.0-1.0)
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    min_sample_size: int = 100  # Minimum samples before analysis
    confidence_level: float = 0.95  # Statistical confidence level
    
    def start(self):
        """Start the experiment"""
        self.status = ExperimentStatus.RUNNING
        self.started_at = time.time()
        logger.info(f"Started experiment: {self.experiment_id}")
    
    def pause(self):
        """Pause the experiment"""
        self.status = ExperimentStatus.PAUSED
        logger.info(f"Paused experiment: {self.experiment_id}")
    
    def complete(self):
        """Mark experiment as completed"""
        self.status = ExperimentStatus.COMPLETED
        self.completed_at = time.time()
        logger.info(f"Completed experiment: {self.experiment_id}")
    
    def get_variant(self, random_value: float) -> str:
        """
        Select variant based on traffic split
        
        Args:
            random_value: Random float between 0.0 and 1.0
            
        Returns:
            Selected variant ID
        """
        cumulative = 0.0
        for variant_id, percentage in self.traffic_split.items():
            cumulative += percentage
            if random_value <= cumulative:
                return variant_id
        
        # Fallback to first variant
        return self.variants[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'experiment_id': self.experiment_id,
            'name': self.name,
            'template_name': self.template_name,
            'description': self.description,
            'variants': self.variants,
            'traffic_split': self.traffic_split,
            'status': self.status.value,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'min_sample_size': self.min_sample_size,
            'confidence_level': self.confidence_level,
        }


class ExperimentManager:
    """Manages A/B testing experiments and tracks results"""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize experiment manager
        
        Args:
            storage_dir: Directory for storing experiment data
        """
        self.storage_dir = storage_dir or Path.home() / ".flybrowser" / "experiments"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments: Dict[str, Experiment] = {}
        self.metrics: Dict[str, Dict[str, VariantMetrics]] = {}  # experiment_id -> variant_id -> metrics
        
        self._load_experiments()
    
    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        template_name: str,
        variants: List[str],
        traffic_split: Optional[Dict[str, float]] = None,
        description: str = "",
        min_sample_size: int = 100,
    ) -> Experiment:
        """
        Create a new A/B test experiment
        
        Args:
            experiment_id: Unique experiment identifier
            name: Human-readable experiment name
            template_name: Template to test variants for
            variants: List of variant IDs to test
            traffic_split: Custom traffic distribution (defaults to equal split)
            description: Experiment description
            min_sample_size: Minimum samples before statistical analysis
            
        Returns:
            Created Experiment object
        """
        if not variants:
            raise ValueError("At least one variant required")
        
        # Default to equal traffic split
        if traffic_split is None:
            split_percentage = 1.0 / len(variants)
            traffic_split = {v: split_percentage for v in variants}
        
        # Validate traffic split sums to 1.0
        total = sum(traffic_split.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Traffic split must sum to 1.0, got {total}")
        
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            template_name=template_name,
            description=description,
            variants=variants,
            traffic_split=traffic_split,
            min_sample_size=min_sample_size,
        )
        
        self.experiments[experiment_id] = experiment
        self.metrics[experiment_id] = {
            variant_id: VariantMetrics(variant_id)
            for variant_id in variants
        }
        
        self._save_experiment(experiment)
        logger.info(f"Created experiment: {experiment_id}")
        
        return experiment
    
    def start_experiment(self, experiment_id: str):
        """Start an experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment = self.experiments[experiment_id]
        experiment.start()
        self._save_experiment(experiment)
    
    def record_result(
        self,
        experiment_id: str,
        variant_id: str,
        success: bool,
        latency_ms: float = 0.0,
        tokens: int = 0,
        error: Optional[str] = None,
    ):
        """
        Record a single experiment result
        
        Args:
            experiment_id: Experiment identifier
            variant_id: Variant that was used
            success: Whether the operation succeeded
            latency_ms: Latency in milliseconds
            tokens: Number of tokens used
            error: Error message if failed
        """
        if experiment_id not in self.metrics:
            logger.warning(f"Experiment not found: {experiment_id}")
            return
        
        if variant_id not in self.metrics[experiment_id]:
            logger.warning(f"Variant not found: {variant_id} in {experiment_id}")
            return
        
        metrics = self.metrics[experiment_id][variant_id]
        metrics.add_result(success, latency_ms, tokens, error)
        
        # Periodically save metrics
        if metrics.usage_count % 10 == 0:
            self._save_metrics(experiment_id)
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get comprehensive experiment results
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dictionary with experiment results and analysis
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment = self.experiments[experiment_id]
        variant_metrics = self.metrics.get(experiment_id, {})
        
        # Calculate overall metrics
        total_samples = sum(m.usage_count for m in variant_metrics.values())
        
        results = {
            'experiment': experiment.to_dict(),
            'total_samples': total_samples,
            'variants': {
                vid: metrics.to_dict()
                for vid, metrics in variant_metrics.items()
            },
            'sample_size_met': total_samples >= experiment.min_sample_size,
        }
        
        # Statistical analysis if enough samples
        if total_samples >= experiment.min_sample_size:
            results['analysis'] = self._analyze_results(experiment_id)
        
        return results
    
    def _analyze_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Perform statistical analysis on experiment results
        
        Returns:
            Dictionary with winner, confidence, and recommendations
        """
        variant_metrics = self.metrics.get(experiment_id, {})
        
        # Find best variant by success rate
        best_variant = max(
            variant_metrics.items(),
            key=lambda x: x[1].success_rate
        )
        
        # Simple analysis (proper statistical testing would use Chi-square, etc.)
        all_success_rates = [m.success_rate for m in variant_metrics.values()]
        avg_success_rate = sum(all_success_rates) / len(all_success_rates)
        
        improvement = (best_variant[1].success_rate - avg_success_rate) / avg_success_rate if avg_success_rate > 0 else 0
        
        return {
            'winner': best_variant[0],
            'winner_success_rate': best_variant[1].success_rate,
            'winner_avg_latency_ms': best_variant[1].avg_latency_ms,
            'improvement_percent': improvement * 100,
            'recommendation': (
                f"Variant '{best_variant[0]}' shows {improvement*100:.1f}% improvement. "
                f"Consider promoting to production."
                if improvement > 0.05 else
                "No significant improvement detected. Continue testing or adjust variants."
            ),
        }
    
    def list_experiments(
        self, 
        status: Optional[ExperimentStatus] = None
    ) -> List[Experiment]:
        """
        List all experiments, optionally filtered by status
        
        Args:
            status: Optional status filter
            
        Returns:
            List of experiments
        """
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        return experiments
    
    def _load_experiments(self):
        """Load experiments from storage"""
        experiments_file = self.storage_dir / "experiments.json"
        
        if not experiments_file.exists():
            return
        
        try:
            with open(experiments_file, 'r') as f:
                data = json.load(f)
            
            for exp_data in data.get('experiments', []):
                exp = Experiment(
                    experiment_id=exp_data['experiment_id'],
                    name=exp_data['name'],
                    template_name=exp_data['template_name'],
                    description=exp_data.get('description', ''),
                    variants=exp_data['variants'],
                    traffic_split=exp_data['traffic_split'],
                    status=ExperimentStatus(exp_data['status']),
                    created_at=exp_data['created_at'],
                    started_at=exp_data.get('started_at'),
                    completed_at=exp_data.get('completed_at'),
                    min_sample_size=exp_data.get('min_sample_size', 100),
                )
                self.experiments[exp.experiment_id] = exp
            
            logger.info(f"Loaded {len(self.experiments)} experiments")
            
        except Exception as e:
            logger.error(f"Failed to load experiments: {e}")
    
    def _save_experiment(self, experiment: Experiment):
        """Save single experiment to storage"""
        experiments_file = self.storage_dir / "experiments.json"
        
        # Load all experiments
        all_experiments = []
        if experiments_file.exists():
            with open(experiments_file, 'r') as f:
                data = json.load(f)
                all_experiments = data.get('experiments', [])
        
        # Update or add this experiment
        exp_dict = experiment.to_dict()
        found = False
        for i, exp in enumerate(all_experiments):
            if exp['experiment_id'] == experiment.experiment_id:
                all_experiments[i] = exp_dict
                found = True
                break
        
        if not found:
            all_experiments.append(exp_dict)
        
        # Save back
        with open(experiments_file, 'w') as f:
            json.dump({'experiments': all_experiments}, f, indent=2)
    
    def _save_metrics(self, experiment_id: str):
        """Save metrics for an experiment"""
        metrics_file = self.storage_dir / f"{experiment_id}_metrics.json"
        
        variant_metrics = self.metrics.get(experiment_id, {})
        metrics_data = {
            vid: metrics.to_dict()
            for vid, metrics in variant_metrics.items()
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)


# Singleton instance
_experiment_manager: Optional[ExperimentManager] = None


def get_experiment_manager() -> ExperimentManager:
    """Get or create the global experiment manager singleton"""
    global _experiment_manager
    if _experiment_manager is None:
        _experiment_manager = ExperimentManager()
    return _experiment_manager
