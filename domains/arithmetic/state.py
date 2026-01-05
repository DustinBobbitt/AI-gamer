"""
BeliefState - Fixed-size token representation for factor inference.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class BeliefState:
    """
    Fixed-size belief state for semiprime factorization inference.
    
    This represents our current belief about the factors without
    explicitly storing candidates or performing trial division.
    """
    
    # Size ratio estimate: how skewed are p and q? (0=very skewed, 1=balanced/near-square)
    size_ratio_estimate: float = 0.5
    
    # Near-square score: how close is N to a perfect square?
    near_square_score: float = 0.0
    
    # Residue weights for N mod 30 (8 valid residues: 1,7,11,13,17,19,23,29)
    residue_weights_mod30: np.ndarray = None
    
    # Smoothness score: does N have small factors? (based on N mod small primes only)
    smoothness_score: float = 0.5
    
    # Entropy estimate: uncertainty in our belief
    entropy_estimate: float = 1.0
    
    # Confidence: how confident are we in our estimate?
    confidence: float = 0.0
    
    # Step count in this episode
    step_count: int = 0
    
    def __post_init__(self):
        if self.residue_weights_mod30 is None:
            # Initialize uniform distribution over valid residues mod 30
            self.residue_weights_mod30 = np.ones(8) / 8.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to flat numpy vector for neural network input."""
        vec = np.array([
            self.size_ratio_estimate,
            self.near_square_score,
            *self.residue_weights_mod30,
            self.smoothness_score,
            self.entropy_estimate,
            self.confidence,
            float(self.step_count) / 100.0  # Normalized
        ])
        return vec
    
    @classmethod
    def from_vector(cls, vec: np.ndarray) -> BeliefState:
        """Reconstruct from flat vector."""
        return cls(
            size_ratio_estimate=float(vec[0]),
            near_square_score=float(vec[1]),
            residue_weights_mod30=vec[2:10].copy(),
            smoothness_score=float(vec[10]),
            entropy_estimate=float(vec[11]),
            confidence=float(vec[12]),
            step_count=int(vec[13] * 100.0)
        )
    
    def compute_entropy(self) -> float:
        """
        Compute Shannon entropy of residue distribution.
        
        H = -Σ p_i log(p_i)
        """
        weights = self.residue_weights_mod30
        weights = np.clip(weights, 1e-10, 1.0)
        weights = weights / np.sum(weights)
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        return float(entropy)
    
    def update_entropy(self):
        """Update entropy_estimate based on current state."""
        self.entropy_estimate = self.compute_entropy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'size_ratio_estimate': float(self.size_ratio_estimate),
            'near_square_score': float(self.near_square_score),
            'residue_weights_mod30': self.residue_weights_mod30.tolist(),
            'smoothness_score': float(self.smoothness_score),
            'entropy_estimate': float(self.entropy_estimate),
            'confidence': float(self.confidence),
            'step_count': self.step_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BeliefState:
        """Deserialize from dictionary."""
        return cls(
            size_ratio_estimate=data['size_ratio_estimate'],
            near_square_score=data['near_square_score'],
            residue_weights_mod30=np.array(data['residue_weights_mod30']),
            smoothness_score=data['smoothness_score'],
            entropy_estimate=data['entropy_estimate'],
            confidence=data['confidence'],
            step_count=data['step_count']
        )
