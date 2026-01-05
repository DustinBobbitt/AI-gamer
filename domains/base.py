"""
Base classes for domain abstraction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class DomainDifficulty(Enum):
    """Difficulty levels for scenarios."""
    TRIVIAL = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5


@dataclass
class ScenarioConfig:
    """Configuration for a specific scenario instance."""
    
    domain_name: str
    difficulty: DomainDifficulty
    scenario_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainMetrics:
    """Performance metrics for a domain."""
    
    domain_name: str
    scenarios_completed: int
    total_episodes: int
    avg_reward: float
    best_reward: float
    worst_reward: float
    success_rate: float
    domain_specific: Dict[str, Any] = field(default_factory=dict)


class DomainTask(ABC):
    """
    Abstract base class for all learning domains.
    
    A domain provides:
    - Scenario generation
    - State/action space
    - Reward computation
    - Evaluation metrics
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.domain_name = self.__class__.__name__
        self.current_scenario: Optional[ScenarioConfig] = None
        self.state = None
    
    @abstractmethod
    def make_scenario(self, rng: Any, difficulty: DomainDifficulty) -> ScenarioConfig:
        """Generate a new scenario configuration."""
        pass
    
    @abstractmethod
    def reset(self, scenario_config: ScenarioConfig) -> Any:
        """Reset to initial state for a scenario."""
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        Execute one step.
        
        Returns:
            observation: Current state observation
            reward: Reward for this step
            done: Whether scenario is complete
            info: Additional information
        """
        pass
    
    @abstractmethod
    def get_action_space(self) -> List[int]:
        """Return list of available actions."""
        pass
    
    @abstractmethod
    def list_actions(self) -> List[str]:
        """Return human-readable action descriptions."""
        pass
    
    @abstractmethod
    def evaluate_policy(self, policy: Any, scenarios: List[ScenarioConfig]) -> DomainMetrics:
        """
        Evaluate a policy on multiple scenarios.
        
        Args:
            policy: The policy to evaluate (Brain or callable)
            scenarios: List of scenario configurations to test
        
        Returns:
            DomainMetrics with evaluation results
        """
        pass
    
    def render(self) -> str:
        """Return string representation of current state (optional)."""
        return f"{self.domain_name} - {self.current_scenario}"
