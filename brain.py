"""
Brain Module - The shared neural policy/value model trained across many games.

The Brain (θ) is a parameter vector representing learned knowledge that can be
applied across multiple games. It learns transferable reasoning skills and 
maintains importance metrics for safe knowledge consolidation.
"""
from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class BrainState:
    """Represents the current state of the brain's parameters."""
    
    parameters: Dict[str, np.ndarray]  # θ - the main parameter vector
    importance_matrix: Optional[np.ndarray] = None  # G - importance metric
    version: int = 0
    total_games_seen: int = 0
    consolidation_count: int = 0
    
    def save(self, path: Path) -> None:
        """Save brain state to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            version=self.version,
            total_games_seen=self.total_games_seen,
            consolidation_count=self.consolidation_count,
            **self.parameters
        )
        if self.importance_matrix is not None:
            np.save(path.parent / f"{path.stem}_importance.npy", self.importance_matrix)
    
    @classmethod
    def load(cls, path: Path) -> "BrainState":
        """Load brain state from disk."""
        data = np.load(path, allow_pickle=True)
        parameters = {k: data[k] for k in data.files if k not in ['version', 'total_games_seen', 'consolidation_count']}
        
        importance_path = path.parent / f"{path.stem}_importance.npy"
        importance_matrix = np.load(importance_path) if importance_path.exists() else None
        
        return cls(
            parameters=parameters,
            importance_matrix=importance_matrix,
            version=int(data.get('version', 0)),
            total_games_seen=int(data.get('total_games_seen', 0)),
            consolidation_count=int(data.get('consolidation_count', 0))
        )


class Brain:
    """
    The core learning agent that maintains transferable knowledge across games.
    
    Architecture:
    - Shared parameter vector θ
    - Importance matrix G for measuring behavioral change
    - Learning history and consolidation state
    """
    
    def __init__(self, hidden_size: int = 256):
        self.hidden_size = hidden_size
        self.state = BrainState(
            parameters={
                'policy_weights': np.random.randn(hidden_size, hidden_size) * 0.01,
                'value_weights': np.random.randn(hidden_size, 1) * 0.01,
                'embedding': np.random.randn(hidden_size, hidden_size) * 0.01
            }
        )
        self.learning_rate = 0.001
        self.fisher_samples = []
    
    def predict_action(self, game_state: Any) -> int:
        """
        Predict action for a given game state.
        
        Returns action index (to be mapped to game-specific actions).
        """
        # Placeholder: Simple forward pass
        # In real implementation, this would be a full neural network forward pass
        embedding = self.state.parameters['embedding']
        policy_weights = self.state.parameters['policy_weights']
        
        # Simple random policy for now
        num_actions = policy_weights.shape[0]
        return np.random.randint(0, num_actions)
    
    def predict_value(self, game_state: Any) -> float:
        """Predict value of a game state."""
        # Placeholder: Random value estimation
        return np.random.randn()
    
    def update(self, experience: Dict[str, Any]) -> float:
        """
        Update brain based on game experience.
        
        Args:
            experience: Dictionary containing state, action, reward, next_state, done
        
        Returns:
            Loss value
        """
        # Placeholder: Simple gradient update
        # In real implementation, this would compute gradients and update parameters
        
        # Store fisher information samples for importance estimation
        self.fisher_samples.append(experience)
        if len(self.fisher_samples) > 1000:
            self.fisher_samples.pop(0)
        
        self.state.total_games_seen += 1
        return 0.0  # Placeholder loss
    
    def estimate_importance(self) -> np.ndarray:
        """
        Estimate importance of parameters using Fisher information.
        
        This creates the G matrix that defines geometry on parameter space.
        High importance means the parameter is critical for performance.
        """
        # Placeholder: Initialize importance matrix
        total_params = sum(p.size for p in self.state.parameters.values())
        importance = np.eye(total_params) * 0.01
        
        # In real implementation:
        # - Compute gradients for each fisher sample
        # - Average squared gradients to get diagonal Fisher approximation
        # - Or use full Fisher matrix for better geometry
        
        self.state.importance_matrix = importance
        return importance
    
    def consolidate(self, compression_target: float = 0.5) -> Dict[str, Any]:
        """
        Consolidate knowledge by pruning, merging, or compressing parameters.
        
        This is the "parsing" or "learned forgetting" phase.
        
        Args:
            compression_target: Target compression ratio (0.5 = 50% reduction)
        
        Returns:
            Dictionary with consolidation metrics
        """
        if self.state.importance_matrix is None:
            self.estimate_importance()
        
        # Placeholder consolidation logic
        # Real implementation would:
        # 1. Identify low-importance parameters
        # 2. Propose pruning/compression: θ' = θ + Δθ
        # 3. Measure behavioral change: Δθᵀ G Δθ
        # 4. Accept if change is below threshold
        
        self.state.consolidation_count += 1
        
        return {
            'compression_ratio': compression_target,
            'parameters_pruned': 0,
            'behavioral_change': 0.0,
            'consolidation_id': self.state.consolidation_count
        }
    
    def clone(self) -> "Brain":
        """Create a copy of this brain for testing consolidation."""
        new_brain = Brain(self.hidden_size)
        new_brain.state = BrainState(
            parameters={k: v.copy() for k, v in self.state.parameters.items()},
            importance_matrix=self.state.importance_matrix.copy() if self.state.importance_matrix is not None else None,
            version=self.state.version,
            total_games_seen=self.state.total_games_seen,
            consolidation_count=self.state.consolidation_count
        )
        return new_brain
    
    def compute_behavioral_distance(self, other: "Brain", test_states: List[Any]) -> float:
        """
        Compute behavioral distance between two brains on test states.
        
        This measures Δθᵀ G Δθ in the importance metric.
        """
        # Placeholder: Simple parameter distance
        distance = 0.0
        for key in self.state.parameters:
            diff = self.state.parameters[key] - other.state.parameters[key]
            distance += np.sum(diff ** 2)
        return float(distance)
