"""
Deterministic inference transforms for belief state updates.

These are NOT trial division or candidate enumeration.
They are constraint-based updates to the belief state.
"""
from __future__ import annotations

import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List

from domains.arithmetic.state import BeliefState


@dataclass
class TransformResult:
    """Result of applying a transform."""
    new_state: BeliefState
    info: Dict[str, Any]


class Transform(ABC):
    """Base class for belief state transforms."""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def apply(self, state: BeliefState, N: int) -> TransformResult:
        """Apply transform to belief state given target N."""
        pass
    
    def get_description(self) -> str:
        """Human-readable description."""
        return f"{self.name}({self.params})"


class ResidueConsistencyUpdate(Transform):
    """
    Update residue weights based on N mod m consistency.
    
    Uses fixed moduli (30, 210) to narrow down possible factor patterns.
    NOT trial division - just constraint propagation.
    """
    
    def apply(self, state: BeliefState, N: int) -> TransformResult:
        new_state = BeliefState(
            size_ratio_estimate=state.size_ratio_estimate,
            near_square_score=state.near_square_score,
            residue_weights_mod30=state.residue_weights_mod30.copy(),
            smoothness_score=state.smoothness_score,
            entropy_estimate=state.entropy_estimate,
            confidence=state.confidence,
            step_count=state.step_count + 1
        )
        
        # Valid residues mod 30: 1,7,11,13,17,19,23,29
        valid_residues = [1, 7, 11, 13, 17, 19, 23, 29]
        n_mod_30 = N % 30
        
        # Update weights based on which pairings of residues can produce n_mod_30
        new_weights = np.zeros(8)
        for i, r1 in enumerate(valid_residues):
            for j, r2 in enumerate(valid_residues):
                if (r1 * r2) % 30 == n_mod_30:
                    new_weights[i] += state.residue_weights_mod30[j]
        
        # Normalize
        if np.sum(new_weights) > 0:
            new_weights /= np.sum(new_weights)
            new_state.residue_weights_mod30 = new_weights
        
        new_state.update_entropy()
        
        return TransformResult(
            new_state=new_state,
            info={'n_mod_30': n_mod_30, 'weights_updated': True}
        )


class ExactSquareEvidenceUpdate(Transform):
    """
    Record exact perfect-square evidence without interpreting proximity.

    Under a semiprime-square assumption an exact square supports p=q. A merely
    nearby integer square carries no validated information about factor ratio.
    """
    
    def apply(self, state: BeliefState, N: int) -> TransformResult:
        new_state = BeliefState(
            size_ratio_estimate=state.size_ratio_estimate,
            near_square_score=state.near_square_score,
            residue_weights_mod30=state.residue_weights_mod30.copy(),
            smoothness_score=state.smoothness_score,
            entropy_estimate=state.entropy_estimate,
            confidence=state.confidence,
            step_count=state.step_count + 1
        )
        
        # Compute distance from nearest perfect square
        sqrt_floor = math.isqrt(N)
        sqrt_n = sqrt_floor if sqrt_floor * sqrt_floor == N else sqrt_floor + 1
        distance = abs(sqrt_n * sqrt_n - N)
        is_exact_square = distance == 0
        new_state.near_square_score = 1.0 if is_exact_square else 0.0
        if is_exact_square:
            new_state.size_ratio_estimate = 1.0
        
        new_state.update_entropy()
        
        return TransformResult(
            new_state=new_state,
            info={
                'sqrt_n': sqrt_n,
                'integer_square_gap': distance,
                'is_exact_square': is_exact_square,
            }
        )


# Backward-compatible import alias. Runtime descriptions use the honest class
# name because the transform registry instantiates ExactSquareEvidenceUpdate.
NearSquareUpdate = ExactSquareEvidenceUpdate


class SmoothnessHeuristicUpdate(Transform):
    """
    Update smoothness score based on N mod small primes.
    
    Does NOT enumerate candidates or perform trial division.
    Just uses N mod 2,3,5,7,11,13 as features.
    """
    
    def apply(self, state: BeliefState, N: int) -> TransformResult:
        new_state = BeliefState(
            size_ratio_estimate=state.size_ratio_estimate,
            near_square_score=state.near_square_score,
            residue_weights_mod30=state.residue_weights_mod30.copy(),
            smoothness_score=state.smoothness_score,
            entropy_estimate=state.entropy_estimate,
            confidence=state.confidence,
            step_count=state.step_count + 1
        )
        
        # Check residues mod small primes (NOT trial division)
        small_primes = [2, 3, 5, 7, 11, 13]
        residues = [N % p for p in small_primes]
        
        # Smoothness heuristic: count how many small residues
        small_residue_count = sum(1 for r in residues if r < 3)
        smoothness = small_residue_count / len(small_primes)
        
        # Update smoothness score (exponential moving average)
        alpha = 0.3
        new_state.smoothness_score = (1 - alpha) * state.smoothness_score + alpha * smoothness
        
        new_state.update_entropy()
        
        return TransformResult(
            new_state=new_state,
            info={'residues': residues, 'smoothness': smoothness}
        )


class ConstraintFusionUpdate(Transform):
    """
    Fuse multiple scores to reduce entropy and increase confidence.
    
    Combines: near_square_score, smoothness_score, residue_weights
    to produce a tighter belief estimate.
    """
    
    def apply(self, state: BeliefState, N: int) -> TransformResult:
        new_state = BeliefState(
            size_ratio_estimate=state.size_ratio_estimate,
            near_square_score=state.near_square_score,
            residue_weights_mod30=state.residue_weights_mod30.copy(),
            smoothness_score=state.smoothness_score,
            entropy_estimate=state.entropy_estimate,
            confidence=state.confidence,
            step_count=state.step_count + 1
        )
        
        # Sharpen residue distribution based on other scores
        weights = new_state.residue_weights_mod30
        
        # If near-square, favor middle residues
        if state.near_square_score > 0.7:
            middle_boost = np.array([0.8, 1.2, 1.2, 1.3, 1.3, 1.2, 1.2, 0.8])
            weights *= middle_boost
        
        # If smooth, favor smaller residues
        if state.smoothness_score > 0.5:
            smoothness_boost = np.array([1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6])
            weights *= smoothness_boost
        
        # Normalize
        weights = np.clip(weights, 0.01, 10.0)
        weights /= np.sum(weights)
        new_state.residue_weights_mod30 = weights
        
        # Update confidence based on entropy reduction
        new_state.update_entropy()
        entropy_reduction = max(0, state.entropy_estimate - new_state.entropy_estimate)
        new_state.confidence = min(1.0, state.confidence + entropy_reduction * 0.5)
        
        return TransformResult(
            new_state=new_state,
            info={'entropy_reduction': entropy_reduction, 'confidence_gain': entropy_reduction * 0.5}
        )


class TransformLibrary:
    """Registry of available transforms."""
    
    @staticmethod
    def get_all_transforms() -> List[Transform]:
        """Return list of all available transforms."""
        return [
            ResidueConsistencyUpdate(),
            ExactSquareEvidenceUpdate(),
            SmoothnessHeuristicUpdate(),
            ConstraintFusionUpdate()
        ]
    
    @staticmethod
    def get_transform_by_id(transform_id: int) -> Transform:
        """Get transform by index."""
        transforms = TransformLibrary.get_all_transforms()
        if 0 <= transform_id < len(transforms):
            return transforms[transform_id]
        return transforms[0]  # Default to first
    
    @staticmethod
    def get_action_space_size() -> int:
        """Return number of available transforms."""
        return len(TransformLibrary.get_all_transforms())
