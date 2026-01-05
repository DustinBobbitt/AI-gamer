from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from env import PrimeIceObservation
from rules import DivisibilityRule, RuleLike
from utils.primes import primes_up_to
from utils.rng import RandomSource

# Import new agent types
from agents.residue_filter_agent import ResidueFilterAgent, ResidueFilterPolicy, generate_filter_candidates
from agents.composite_agents import FixedFilterCleanupAgent, LearningFilterCleanupAgent


AgentId = str

# Prime divisors used by sieve-like strategies (configurable upper bound).
D_MAX_DEFAULT = 199


class Agent:
    """Base interface for agents."""

    name: str

    def reset_for_episode(self) -> None:  # pragma: no cover - interface
        ...

    def select_rule(self, obs: PrimeIceObservation) -> Optional[RuleLike]:  # pragma: no cover
        """Select a rule to apply. Returns None to signal termination."""
        ...

    def observe(self, info: Dict[str, object]) -> None:  # pragma: no cover
        ...

    def on_run_end(self) -> None:  # pragma: no cover
        """Hook called after a full training run."""
        ...


def _prime_candidates(d_max: int = D_MAX_DEFAULT) -> List[int]:
    return [p for p in primes_up_to(d_max) if p >= 3]


@dataclass
class RandomRulesAgent(Agent):
    """Random baseline that samples prime divisors."""

    rng: RandomSource
    d_candidates: List[int] = field(default_factory=lambda: _prime_candidates())

    name: str = "random_rules"

    def reset_for_episode(self) -> None:
        return

    def select_rule(self, obs: PrimeIceObservation) -> Optional[RuleLike]:
        d = self.rng.choice(self.d_candidates)
        return DivisibilityRule(d=d, keep_self=True)

    def observe(self, info: Dict[str, object]) -> None:
        return


@dataclass
class HeuristicSieveAgent(Agent):
    """Deterministic sieve using ascending prime divisors."""

    divisors: List[int]
    index: int = 0

    name: str = "heuristic_sieve"

    def reset_for_episode(self) -> None:
        self.index = 0

    def select_rule(self, obs: PrimeIceObservation) -> Optional[RuleLike]:
        # Classic sieve: apply primes in order while d^2 <= max_n.
        while self.index < len(self.divisors):
            d = self.divisors[self.index]
            self.index += 1
            if d * d > obs.max_n:
                # No more useful divisors; signal completion
                return None
            return DivisibilityRule(d=d, keep_self=True)
        # Exhausted all divisors
        return None

    def observe(self, info: Dict[str, object]) -> None:
        return


@dataclass
class LearningSievePolicy:
    """In-memory representation of the learning sieve policy."""

    q_values: Dict[int, float]
    counts: Dict[int, int]
    epsilon: float
    alpha: float
    total_episodes: int

    @classmethod
    def new(cls, d_candidates: List[int]) -> "LearningSievePolicy":
        return cls(
            q_values={d: 0.0 for d in d_candidates},
            counts={d: 0 for d in d_candidates},
            epsilon=0.2,
            alpha=0.2,
            total_episodes=0,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "q_values": self.q_values,
            "counts": self.counts,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "total_episodes": self.total_episodes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object], d_candidates: List[int]) -> "LearningSievePolicy":
        q_raw: Dict[str, float] = data.get("q_values", {})  # type: ignore[assignment]
        c_raw: Dict[str, int] = data.get("counts", {})  # type: ignore[assignment]
        q = {int(k): float(v) for k, v in q_raw.items()}
        c = {int(k): int(v) for k, v in c_raw.items()}
        # Ensure all candidates are present.
        for d in d_candidates:
            q.setdefault(d, 0.0)
            c.setdefault(d, 0)
        return cls(
            q_values=q,
            counts=c,
            epsilon=float(data.get("epsilon", 0.2)),
            alpha=float(data.get("alpha", 0.2)),
            total_episodes=int(data.get("total_episodes", 0)),
        )


@dataclass
class LearningSieveAgent(Agent):
    """Bandit-like learning sieve over prime divisors."""

    rng: RandomSource
    d_candidates: List[int]
    policy: LearningSievePolicy
    policy_path: Path
    step_cost_lambda: float = 0.1

    name: str = "learning_sieve"

    _used_this_episode: set[int] = field(default_factory=set)

    def reset_for_episode(self) -> None:
        self._used_this_episode.clear()
        self.policy.total_episodes += 1
        # Simple epsilon decay over episodes.
        self.policy.epsilon = max(0.01, self.policy.epsilon * 0.995)

    def _eligible_divisors(self, obs: PrimeIceObservation) -> List[int]:
        """Return divisors whose square is <= max_n and not already tried."""
        return [
            d for d in self.d_candidates if d * d <= obs.max_n and d not in self._used_this_episode
        ]

    def _step_cost(self, d: int) -> float:
        return 1.0 + 0.02 * d

    def select_rule(self, obs: PrimeIceObservation) -> Optional[RuleLike]:
        candidates = self._eligible_divisors(obs)
        if not candidates:
            # No more useful divisors; signal completion
            return None

        # ε-greedy over Q-values adjusted by step cost.
        if self.rng.random() < self.policy.epsilon:
            d = self.rng.choice(candidates)
        else:
            best_score: Optional[float] = None
            best_d = candidates[0]
            for d in candidates:
                q = self.policy.q_values.get(d, 0.0)
                cost = self._step_cost(d)
                score = q - self.step_cost_lambda * cost
                if best_score is None or score > best_score:
                    best_score = score
                    best_d = d
            d = best_d

        self._used_this_episode.add(d)
        return DivisibilityRule(d=d, keep_self=True)

    def observe(self, info: Dict[str, object]) -> None:
        """Update Q[d] based on composites_removed and applied divisor."""
        rule_name = info.get("rule_name", "")
        if not rule_name.startswith("Divisible by "):
            return
        try:
            d_str = rule_name.split(" ")[2]
            d = int(d_str)
        except (IndexError, ValueError):
            return

        composites_removed = int(info.get("composites_removed", 0))
        reward = float(composites_removed) - self.step_cost_lambda * self._step_cost(d)

        old_q = self.policy.q_values.get(d, 0.0)
        alpha = self.policy.alpha
        new_q = old_q + alpha * (reward - old_q)
        self.policy.q_values[d] = new_q
        self.policy.counts[d] = self.policy.counts.get(d, 0) + 1

    def on_run_end(self) -> None:
        self._save_policy()

    # Persistence helpers -------------------------------------------------
    def _save_policy(self) -> None:
        self.policy_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.policy_path, "w", encoding="utf-8") as f:
            json.dump(self.policy.to_dict(), f, indent=2, sort_keys=True)

    def export_policy(self) -> Dict[str, object]:
        return self.policy.to_dict()


def _policy_file_path() -> Path:
    base = Path(__file__).resolve().parent.parent
    return base / "models" / "learning_sieve_policy.json"


def load_learning_policy(d_candidates: List[int], reset: bool = False) -> LearningSievePolicy:
    path = _policy_file_path()
    if reset or not path.exists():
        return LearningSievePolicy.new(d_candidates)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return LearningSievePolicy.from_dict(data, d_candidates)


def make_heuristic_sieve_agent(max_n: int) -> HeuristicSieveAgent:
    """Construct a deterministic heuristic sieve agent using primes up to sqrt(max_n)."""
    limit = int(max_n ** 0.5) + 1
    divisors = [p for p in primes_up_to(limit) if p >= 3]
    return HeuristicSieveAgent(divisors=divisors)


def make_learning_sieve_agent(rng: RandomSource, reset_policy: bool = False) -> LearningSieveAgent:
    """Create a learning sieve agent with persistent policy."""
    d_candidates = _prime_candidates(D_MAX_DEFAULT)
    policy = load_learning_policy(d_candidates, reset=reset_policy)
    return LearningSieveAgent(
        rng=rng,
        d_candidates=d_candidates,
        policy=policy,
        policy_path=_policy_file_path(),
    )


def get_agent(agent_id: AgentId, rng: RandomSource, train_high: int, reset_policy: bool) -> Agent:
    """Factory for agents by logical ID."""
    agent_id = agent_id.lower()
    if agent_id == "random_rules":
        return RandomRulesAgent(rng=rng)
    if agent_id == "heuristic_sieve":
        return make_heuristic_sieve_agent(max_n=train_high)
    if agent_id == "learning_sieve":
        return make_learning_sieve_agent(rng=rng, reset_policy=reset_policy)
    if agent_id == "fixed_filter_then_cleanup":
        return make_fixed_filter_cleanup_agent(train_high=train_high)
    if agent_id == "learning_filter_then_cleanup":
        return make_learning_filter_cleanup_agent(rng=rng, train_high=train_high, reset_policy=reset_policy)
    raise ValueError(f"Unknown agent/strategy: {agent_id}")


def make_fixed_filter_cleanup_agent(train_high: int) -> FixedFilterCleanupAgent:
    """Create fixed filter + deterministic cleanup composite agent."""
    cleanup_agent = make_heuristic_sieve_agent(max_n=train_high)
    return FixedFilterCleanupAgent(cleanup_agent=cleanup_agent)


def make_learning_filter_cleanup_agent(
    rng: RandomSource,
    train_high: int,
    reset_policy: bool = False,
) -> LearningFilterCleanupAgent:
    """Create learning filter + deterministic cleanup composite agent."""
    # Filter agent
    filter_policy_path = Path(__file__).resolve().parent.parent / "models" / "residue_filter_policy.json"
    filter_agent = ResidueFilterAgent.load_or_create(
        rng=rng,
        policy_path=filter_policy_path,
    )
    if reset_policy and filter_policy_path.exists():
        filter_policy_path.unlink()
        filter_agent = ResidueFilterAgent.load_or_create(rng=rng, policy_path=filter_policy_path)
    
    # Cleanup agent
    cleanup_agent = make_heuristic_sieve_agent(max_n=train_high)
    
    return LearningFilterCleanupAgent(
        filter_agent=filter_agent,
        cleanup_agent=cleanup_agent,
    )


def get_agent_metadata(agent_id: AgentId) -> tuple[str, str]:
    """Return normalized agent_id and human-readable agent_name.

    Defaults to the learning sieve if the provided ID is missing or invalid.
    """
    normalized = (agent_id or "").lower()
    mapping = {
        "heuristic_sieve": "Factor Cleanup (Deterministic)",
        "learning_sieve": "Learning Factor Cleanup (Persistent)",
        "random_rules": "Random Factor Checks (Baseline)",
        "fixed_filter_then_cleanup": "First-Cut Filter + Factor Cleanup (Fixed)",
        "learning_filter_then_cleanup": "First-Cut Filter + Factor Cleanup (Learning)",
    }
    if normalized not in mapping:
        normalized = "learning_sieve"
    return normalized, mapping[normalized]
