"""
Consolidator - Meta-system for compressing, pruning, and merging learned knowledge.

This module implements the "Parser" that learns how to safely consolidate
the brain's knowledge while preserving performance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from brain import Brain, BrainState
from skill_memory import SkillMemory


@dataclass
class ConsolidationProposal:
    """Represents a proposed consolidation operation."""
    
    operation_type: str  # 'prune', 'merge', 'compress', 'distill'
    target_parameters: List[str]
    delta_theta: Dict[str, np.ndarray]  # Proposed parameter changes
    expected_compression: float
    expected_behavioral_change: float
    rationale: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation_type': self.operation_type,
            'target_parameters': self.target_parameters,
            'expected_compression': self.expected_compression,
            'expected_behavioral_change': self.expected_behavioral_change,
            'rationale': self.rationale
        }


@dataclass
class ConsolidationResult:
    """Results from a consolidation operation."""
    
    proposal: ConsolidationProposal
    accepted: bool
    actual_compression: float
    actual_behavioral_change: float
    regression_metrics: Dict[str, float]
    restoration_needed: bool = False
    restored_parameters: List[str] = None
    
    def __post_init__(self):
        if self.restored_parameters is None:
            self.restored_parameters = []


class Consolidator:
    """
    The meta-learning system that learns how to safely consolidate knowledge.
    
    Core responsibilities:
    1. Generate consolidation proposals
    2. Evaluate proposals using importance metrics
    3. Learn from consolidation successes and failures
    4. Improve future consolidation decisions
    """
    
    def __init__(self):
        self.consolidation_history: List[ConsolidationResult] = []
        self.consolidation_policy: Dict[str, float] = {
            'pruning_threshold': 0.1,
            'merge_similarity_threshold': 0.8,
            'max_behavioral_change': 0.05,
            'compression_aggressiveness': 0.5
        }
        
        # Learning: Track which consolidation strategies work
        self.strategy_success_rates: Dict[str, float] = {
            'prune': 0.5,
            'merge': 0.5,
            'compress': 0.5,
            'distill': 0.5
        }
    
    def propose_consolidation(
        self,
        brain: Brain,
        target_compression: float = 0.3,
        strategy: Optional[str] = None,
    ) -> ConsolidationProposal:
        """
        Generate a consolidation proposal based on current brain state.
        
        This is where the consolidator uses its learned knowledge to
        decide what to consolidate.
        """
        if brain.state.importance_matrix is None:
            brain.estimate_importance()
        
        # Choose consolidation strategy based on learned success rates
        if strategy is None:
            strategy = self._select_strategy()
        elif strategy not in self.strategy_success_rates:
            raise ValueError(f"Unknown consolidation strategy: {strategy}")
        
        if strategy == 'prune':
            return self._propose_pruning(brain, target_compression)
        elif strategy == 'merge':
            return self._propose_merging(brain, target_compression)
        elif strategy == 'compress':
            return self._propose_compression(brain, target_compression)
        else:  # distill
            return self._propose_distillation(brain, target_compression)
    
    def _select_strategy(self) -> str:
        """Select consolidation strategy based on learned success rates."""
        strategies = list(self.strategy_success_rates.keys())
        success_rates = [self.strategy_success_rates[s] for s in strategies]
        
        # Softmax selection with temperature
        temperature = 0.5
        exp_rates = np.exp(np.array(success_rates) / temperature)
        probabilities = exp_rates / exp_rates.sum()
        
        return np.random.choice(strategies, p=probabilities)
    
    def _propose_pruning(self, brain: Brain, target_compression: float) -> ConsolidationProposal:
        """
        Propose pruning low-importance parameters.
        
        Pruning sets parameters to zero based on importance scores.
        """
        # Find low-importance parameters
        importance = brain.state.importance_matrix
        threshold = self.consolidation_policy['pruning_threshold']
        
        # Simplified: propose zeroing out random parameters
        # Real implementation would use importance matrix to select
        target_params = ['policy_weights']
        delta_theta = {}
        
        for param_name in target_params:
            param = brain.state.parameters[param_name]
            # Create mask for low-importance elements
            mask = np.random.random(param.shape) > (1 - target_compression)
            delta_theta[param_name] = -param * mask
        
        expected_change = target_compression * 0.1  # Rough estimate
        
        return ConsolidationProposal(
            operation_type='prune',
            target_parameters=target_params,
            delta_theta=delta_theta,
            expected_compression=target_compression,
            expected_behavioral_change=expected_change,
            rationale=f"Pruning {target_compression*100:.1f}% of parameters below importance threshold"
        )
    
    def _propose_merging(self, brain: Brain, target_compression: float) -> ConsolidationProposal:
        """Propose merging similar parameter groups."""
        # Placeholder: Real implementation would identify similar neurons/features
        return ConsolidationProposal(
            operation_type='merge',
            target_parameters=[],
            delta_theta={},
            expected_compression=target_compression * 0.5,
            expected_behavioral_change=0.03,
            rationale="Merging similar feature representations"
        )
    
    def _propose_compression(self, brain: Brain, target_compression: float) -> ConsolidationProposal:
        """Propose low-rank compression of parameter matrices."""
        # Placeholder: SVD-based compression
        return ConsolidationProposal(
            operation_type='compress',
            target_parameters=['policy_weights'],
            delta_theta={},
            expected_compression=target_compression,
            expected_behavioral_change=0.02,
            rationale="Low-rank matrix factorization compression"
        )
    
    def _propose_distillation(self, brain: Brain, target_compression: float) -> ConsolidationProposal:
        """Propose knowledge distillation into smaller model."""
        return ConsolidationProposal(
            operation_type='distill',
            target_parameters=['embedding'],
            delta_theta={},
            expected_compression=target_compression * 0.7,
            expected_behavioral_change=0.04,
            rationale="Distilling knowledge into more compact representation"
        )
    
    def apply_consolidation(
        self, 
        brain: Brain, 
        proposal: ConsolidationProposal,
        test_states: List[Any] = None
    ) -> Tuple[Brain, float]:
        """
        Apply a consolidation proposal and measure behavioral change.
        
        Returns:
            consolidated_brain: New brain with consolidation applied
            behavioral_change: Measured change in behavior
        """
        # Clone brain for testing
        consolidated = brain.clone()
        
        # Apply delta theta
        for param_name, delta in proposal.delta_theta.items():
            if param_name in consolidated.state.parameters:
                consolidated.state.parameters[param_name] += delta
        
        # Measure behavioral change
        if test_states:
            behavioral_change = brain.compute_behavioral_distance(consolidated, test_states)
        else:
            behavioral_change = proposal.expected_behavioral_change
        
        consolidated.state.consolidation_count += 1
        
        return consolidated, behavioral_change
    
    def evaluate_consolidation(
        self,
        original_brain: Brain,
        consolidated_brain: Brain,
        regression_suite: List[Any],
        proposal: ConsolidationProposal
    ) -> ConsolidationResult:
        """
        Evaluate a consolidation by testing on regression suite.
        
        Measures:
        - Performance on past games
        - Worst-case regression
        - Compression achieved
        - Behavioral change
        """
        # Placeholder: Test on regression suite
        # Real implementation would run both brains on test games
        
        regression_metrics = {
            'avg_performance_drop': 0.02,
            'worst_case_drop': 0.05,
            'games_regressed': 0,
            'total_games_tested': len(regression_suite) if regression_suite else 0
        }
        
        # Decide if consolidation is acceptable
        max_allowed_regression = self.consolidation_policy['max_behavioral_change']
        accepted = regression_metrics['worst_case_drop'] <= max_allowed_regression
        
        # Check if restoration needed
        restoration_needed = regression_metrics['worst_case_drop'] > max_allowed_regression * 2
        
        result = ConsolidationResult(
            proposal=proposal,
            accepted=accepted,
            actual_compression=proposal.expected_compression,  # Simplified
            actual_behavioral_change=regression_metrics['worst_case_drop'],
            regression_metrics=regression_metrics,
            restoration_needed=restoration_needed
        )
        
        self.consolidation_history.append(result)
        return result
    
    def learn_from_consolidation(self, result: ConsolidationResult) -> None:
        """
        Learn from consolidation result to improve future decisions.
        
        This is the meta-learning component: the consolidator learns
        which consolidation strategies work well.
        """
        operation_type = result.proposal.operation_type
        
        # Update strategy success rates
        learning_rate = 0.1
        success = 1.0 if result.accepted and not result.restoration_needed else 0.0
        
        current_rate = self.strategy_success_rates[operation_type]
        self.strategy_success_rates[operation_type] = (
            (1 - learning_rate) * current_rate + learning_rate * success
        )
        
        # Adapt consolidation policy
        if result.accepted:
            # Can be more aggressive
            self.consolidation_policy['compression_aggressiveness'] *= 1.05
            self.consolidation_policy['max_behavioral_change'] *= 1.02
        else:
            # Be more conservative
            self.consolidation_policy['compression_aggressiveness'] *= 0.95
            self.consolidation_policy['max_behavioral_change'] *= 0.98
        
        # Clamp values
        self.consolidation_policy['compression_aggressiveness'] = np.clip(
            self.consolidation_policy['compression_aggressiveness'], 0.1, 1.0
        )
        self.consolidation_policy['max_behavioral_change'] = np.clip(
            self.consolidation_policy['max_behavioral_change'], 0.01, 0.2
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get consolidation statistics."""
        if not self.consolidation_history:
            return {'total_consolidations': 0}
        
        accepted = sum(1 for r in self.consolidation_history if r.accepted)
        restorations = sum(1 for r in self.consolidation_history if r.restoration_needed)
        
        return {
            'total_consolidations': len(self.consolidation_history),
            'accepted_consolidations': accepted,
            'acceptance_rate': accepted / len(self.consolidation_history),
            'restorations_needed': restorations,
            'strategy_success_rates': self.strategy_success_rates.copy(),
            'current_policy': self.consolidation_policy.copy()
        }
