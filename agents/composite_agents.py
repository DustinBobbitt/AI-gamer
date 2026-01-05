"""Composite agents combining residue filter + factor cleanup strategies."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from env import PrimeIceObservation
from rules import KeepResiduesRule, RuleLike
from utils.rng import RandomSource
from agents.residue_filter_agent import (
    DEFAULT_ALLOWED_MOD30,
    ResidueFilterAgent,
    ResidueFilterPolicy,
    generate_filter_candidates,
)

if TYPE_CHECKING:
    from agents import HeuristicSieveAgent


@dataclass
class FixedFilterCleanupAgent:
    """Fixed first-cut filter + deterministic factor cleanup.
    
    Strategy:
    1. Apply KeepResiduesRule(m=30, allowed=[1,7,11,13,17,19,23,29]) once
    2. Then run deterministic factor cleanup (heuristic sieve)
    """
    
    cleanup_agent: HeuristicSieveAgent
    filter_applied: bool = False
    m: int = 30
    allowed_residues: frozenset = field(default_factory=lambda: DEFAULT_ALLOWED_MOD30)
    
    name: str = "fixed_filter_then_cleanup"
    
    def reset_for_episode(self) -> None:
        self.filter_applied = False
        self.cleanup_agent.reset_for_episode()
    
    def select_rule(self, obs: PrimeIceObservation) -> Optional[RuleLike]:
        """First return filter, then delegate to cleanup agent."""
        if not self.filter_applied:
            self.filter_applied = True
            return KeepResiduesRule(m=self.m, allowed_residues=set(self.allowed_residues))
        
        # Delegate to cleanup agent
        return self.cleanup_agent.select_rule(obs)
    
    def observe(self, info: Dict[str, object]) -> None:
        if self.filter_applied:
            # After filter applied, pass observations to cleanup agent
            self.cleanup_agent.observe(info)
    
    def on_run_end(self) -> None:
        # Cleanup agent doesn't persist anything for deterministic mode
        pass


@dataclass
class LearningFilterCleanupAgent:
    """Learning first-cut filter + deterministic factor cleanup.
    
    Strategy:
    1. Learn and apply optimal KeepResiduesRule(m=30, allowed=?)
    2. Then run deterministic factor cleanup (heuristic sieve)
    """
    
    filter_agent: ResidueFilterAgent
    cleanup_agent: HeuristicSieveAgent
    filter_applied: bool = False
    
    name: str = "learning_filter_then_cleanup"
    
    def reset_for_episode(self) -> None:
        self.filter_applied = False
        self.filter_agent.reset_for_episode()
        self.cleanup_agent.reset_for_episode()
    
    def select_rule(self, obs: PrimeIceObservation) -> Optional[RuleLike]:
        """First return learned filter, then delegate to cleanup agent."""
        if not self.filter_applied:
            rule = self.filter_agent.select_rule(obs)
            if rule is not None:
                self.filter_applied = True
                return rule
        
        # Delegate to cleanup agent
        return self.cleanup_agent.select_rule(obs)
    
    def observe(self, info: Dict[str, object]) -> None:
        if not self.filter_applied:
            # Still in filter selection phase
            self.filter_agent.observe(info)
        else:
            # After filter applied, pass observations to cleanup agent
            self.cleanup_agent.observe(info)
    
    def on_run_end(self) -> None:
        self.filter_agent.on_run_end()
        # Cleanup agent doesn't persist anything
