"""Residue filter agent - learns optimal mod-30 filters for first-cut filtering."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from env import PrimeIceObservation
from rules import KeepResiduesRule, RuleLike
from utils.rng import RandomSource


# Default baseline: residues mod 30 that are coprime to 2, 3, 5
DEFAULT_ALLOWED_MOD30 = frozenset([1, 7, 11, 13, 17, 19, 23, 29])


@dataclass
class ResidueFilterPolicy:
    """Policy for learning good residue filters."""
    
    q_values: Dict[Tuple[int, ...], float]  # (sorted residues tuple) -> Q-value
    counts: Dict[Tuple[int, ...], int]
    epsilon: float
    alpha: float
    total_episodes: int
    
    @classmethod
    def new(cls, candidates: List[Tuple[int, ...]]) -> "ResidueFilterPolicy":
        return cls(
            q_values={c: 0.0 for c in candidates},
            counts={c: 0 for c in candidates},
            epsilon=0.2,
            alpha=0.2,
            total_episodes=0,
        )
    
    def to_dict(self) -> Dict[str, object]:
        # Convert tuple keys to strings for JSON
        q_str = {str(k): v for k, v in self.q_values.items()}
        c_str = {str(k): v for k, v in self.counts.items()}
        return {
            "q_values": q_str,
            "counts": c_str,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "total_episodes": self.total_episodes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, object], candidates: List[Tuple[int, ...]]) -> "ResidueFilterPolicy":
        q_raw: Dict[str, float] = data.get("q_values", {})  # type: ignore
        c_raw: Dict[str, int] = data.get("counts", {})  # type: ignore
        
        # Convert string keys back to tuples
        q = {}
        for k_str, v in q_raw.items():
            # Parse "(...)" string back to tuple
            nums = eval(k_str)
            if isinstance(nums, tuple):
                q[nums] = float(v)
        
        c = {}
        for k_str, v in c_raw.items():
            nums = eval(k_str)
            if isinstance(nums, tuple):
                c[nums] = int(v)
        
        # Ensure all candidates are present
        for cand in candidates:
            q.setdefault(cand, 0.0)
            c.setdefault(cand, 0)
        
        return cls(
            q_values=q,
            counts=c,
            epsilon=float(data.get("epsilon", 0.2)),
            alpha=float(data.get("alpha", 0.2)),
            total_episodes=int(data.get("total_episodes", 0)),
        )


def generate_filter_candidates(m: int = 30) -> List[Tuple[int, ...]]:
    """Generate candidate filter sets for mod m.
    
    Start with baseline (coprime to 2,3,5 for m=30) and generate variants.
    """
    baseline = tuple(sorted(DEFAULT_ALLOWED_MOD30))
    candidates = [baseline]
    
    # Generate mutations: toggle 1-2 residues
    for i in range(m):
        if i in DEFAULT_ALLOWED_MOD30:
            # Try removing one
            variant = tuple(sorted(r for r in DEFAULT_ALLOWED_MOD30 if r != i))
            if len(variant) >= 4:  # Keep at least 4 residues
                candidates.append(variant)
        else:
            # Try adding one (if coprime to 2,3,5)
            if i % 2 != 0 and i % 3 != 0 and i % 5 != 0:
                variant = tuple(sorted(list(DEFAULT_ALLOWED_MOD30) + [i]))
                candidates.append(variant)
    
    return list(set(candidates))  # Deduplicate


@dataclass
class ResidueFilterAgent:
    """Learning agent for first-cut residue filters."""
    
    rng: RandomSource
    policy: ResidueFilterPolicy
    policy_path: Path
    m: int = 30
    candidates: List[Tuple[int, ...]] = None  # type: ignore
    
    # State tracking
    _current_filter: Optional[Tuple[int, ...]] = None
    _episode_start_info: Dict[str, object] = None  # type: ignore
    
    name: str = "residue_filter"
    
    def __post_init__(self):
        if self.candidates is None:
            self.candidates = generate_filter_candidates(self.m)
    
    def reset_for_episode(self) -> None:
        self._current_filter = None
        self._episode_start_info = {}
    
    def select_rule(self, obs: PrimeIceObservation) -> Optional[RuleLike]:
        """Select a filter rule (only on first call per episode)."""
        if self._current_filter is not None:
            # Already selected filter this episode
            return None
        
        # Epsilon-greedy selection
        if self.rng.random() < self.policy.epsilon:
            # Explore: random candidate
            self._current_filter = self.rng.choice(self.candidates)
        else:
            # Exploit: best Q-value
            best = max(self.candidates, key=lambda c: self.policy.q_values.get(c, 0.0))
            self._current_filter = best
        
        # Store initial state for reward calculation
        self._episode_start_info = {
            "remaining_before": obs.remaining_count,
        }
        
        return KeepResiduesRule(m=self.m, allowed_residues=set(self._current_filter))
    
    def observe(self, info: Dict[str, object]) -> None:
        """Update Q-values after filter application."""
        if self._current_filter is None:
            return
        
        # Extract metrics
        remaining_after = int(info.get("remaining_count", 0))
        primes_removed = int(info.get("primes_removed", 0))
        composites_removed = int(info.get("composites_removed", 0))
        
        # Compute reward
        # CRITICAL: primes removed = huge penalty
        # Want: max composites removed, min remaining
        reward = float(composites_removed)
        reward -= 1_000_000.0 * float(primes_removed)
        reward -= 0.05 * float(remaining_after)
        
        # Update Q-value
        old_q = self.policy.q_values.get(self._current_filter, 0.0)
        self.policy.counts[self._current_filter] = self.policy.counts.get(self._current_filter, 0) + 1
        new_q = old_q + self.policy.alpha * (reward - old_q)
        self.policy.q_values[self._current_filter] = new_q
    
    def on_run_end(self) -> None:
        """Save policy to disk."""
        self.policy.total_episodes += 1
        self.policy_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.policy_path, "w", encoding="utf-8") as f:
            json.dump(self.policy.to_dict(), f, indent=2)
    
    @classmethod
    def load_or_create(
        cls,
        rng: RandomSource,
        policy_path: Path,
        m: int = 30,
    ) -> "ResidueFilterAgent":
        """Load existing policy or create new one."""
        candidates = generate_filter_candidates(m)
        
        if policy_path.exists():
            with open(policy_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            policy = ResidueFilterPolicy.from_dict(data, candidates)
        else:
            policy = ResidueFilterPolicy.new(candidates)
        
        return cls(
            rng=rng,
            policy=policy,
            policy_path=policy_path,
            m=m,
            candidates=candidates,
        )
