from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from rules import RuleLike
from utils.primes import is_prime


# Aggregate residue histograms for these fixed moduli.
MODULI_FOR_HIST = (3, 5, 7, 11, 13, 17, 19, 23, 29)


@dataclass
class PrimeIceObservation:
    remaining_count: int
    min_n: int
    max_n: int
    residue_histograms: Dict[int, List[int]]
    density: float


@dataclass
class PrimeIceConfig:
    low: int = 3
    high: int = 20003
    sample_size: int = 0  # kept for compatibility; not used
    no_progress_limit: int = 3  # consecutive zero-removal steps before termination


@dataclass
class EpisodeSummary:
    steps: int
    total_primes: int
    total_composites: int
    primes_removed_count: int
    primes_removed_list: List[int]
    composites_remaining_count: int
    primes_remaining_count: int
    score: int
    terminated_reason: str = "unknown"
    success: bool = False


class PrimeIceEnv:
    """Gym-like environment for the Prime Ice game."""

    def __init__(self, config: Optional[PrimeIceConfig] = None) -> None:
        self.config = config or PrimeIceConfig()
        self._all_numbers: List[int] = []
        self._is_prime: Dict[int, bool] = {}
        self._remaining: Set[int] = set()
        self._steps: int = 0
        self._primes_removed: int = 0
        self._composites_removed: int = 0
        self._removed_primes: List[int] = []
        self._no_progress_count: int = 0
        self._max_no_progress_steps: int = self.config.no_progress_limit

    def reset(self, config: Optional[PrimeIceConfig] = None) -> PrimeIceObservation:
        if config is not None:
            self.config = config
            self._max_no_progress_steps = config.no_progress_limit
        low = self.config.low
        high = self.config.high
        if low % 2 == 0:
            low += 1
        if high % 2 == 0:
            high -= 1

        all_numbers = list(range(low, high + 1, 2))
        self._all_numbers = sorted(all_numbers)
        self._remaining = set(all_numbers)
        self._is_prime = {n: is_prime(n) for n in self._all_numbers}

        self._steps = 0
        self._primes_removed = 0
        self._composites_removed = 0
        self._removed_primes = []
        self._no_progress_count = 0

        return self._make_observation()

    def step(
        self,
        rule: RuleLike,
    ) -> Tuple[PrimeIceObservation, float, bool, Dict[str, object]]:
        to_remove = rule.apply(self._remaining)
        primes_removed = sum(1 for n in to_remove if self._is_prime.get(n, False))
        composites_removed = len(to_remove) - primes_removed
        removed_primes_list = sorted(n for n in to_remove if self._is_prime.get(n, False))

        self._remaining.difference_update(to_remove)
        self._steps += 1
        self._primes_removed += primes_removed
        self._composites_removed += composites_removed
        if removed_primes_list:
            self._removed_primes.extend(removed_primes_list)

        # Track no-progress steps
        if len(to_remove) == 0:
            self._no_progress_count += 1
        else:
            self._no_progress_count = 0

        done = self._composites_remaining() == 0
        terminated_reason = "success" if done else None
        
        # Any prime removal is treated as a catastrophic failure and ends the episode.
        if primes_removed > 0:
            done = True
            terminated_reason = "prime_removed"
        
        # Terminate if no progress for consecutive steps
        if self._no_progress_count >= self._max_no_progress_steps:
            done = True
            terminated_reason = "no_progress"

        score = self._current_score()

        info: Dict[str, object] = {
            "rule_name": getattr(rule, "name", type(rule).__name__),
            "removed_numbers_count": len(to_remove),
            "primes_removed": self._primes_removed,
            "primes_removed_this_step": primes_removed,
            "removed_primes_list": removed_primes_list,
            "composites_removed": self._composites_removed,
            "composites_removed_this_step": composites_removed,
            "remaining_primes": self._primes_remaining(),
            "remaining_composites": self._composites_remaining(),
            "steps": self._steps,
            "score": score,
            "terminated_reason": terminated_reason,
        }

        obs = self._make_observation()
        return obs, -score, done, info

    def _make_observation(self) -> PrimeIceObservation:
        if not self._remaining:
            return PrimeIceObservation(
                remaining_count=0,
                min_n=0,
                max_n=0,
                residue_histograms={m: [0] * m for m in MODULI_FOR_HIST},
                density=0.0,
            )

        remaining = self._remaining
        min_n = min(remaining)
        max_n = max(remaining)
        histograms: Dict[int, List[int]] = {}
        for m in MODULI_FOR_HIST:
            buckets = [0] * m
            for n in remaining:
                buckets[n % m] += 1
            histograms[m] = buckets

        total_range_size = max(self._all_numbers) - min(self._all_numbers) + 2
        density = len(remaining) / float(total_range_size) if total_range_size > 0 else 0.0

        return PrimeIceObservation(
            remaining_count=len(remaining),
            min_n=min_n,
            max_n=max_n,
            residue_histograms=histograms,
            density=density,
        )

    def _primes_remaining(self) -> int:
        return sum(1 for n in self._remaining if self._is_prime.get(n, False))

    def _composites_remaining(self) -> int:
        return len(self._remaining) - self._primes_remaining()

    def _current_score(self) -> int:
        return (
            self._steps
            + 1_000_000 * self._primes_removed
            + 100 * self._composites_remaining()
        )

    @property
    def total_blocks(self) -> int:
        return len(self._all_numbers)

    @property
    def total_primes(self) -> int:
        return sum(1 for n in self._all_numbers if self._is_prime.get(n, False))

    @property
    def total_composites(self) -> int:
        return self.total_blocks - self.total_primes

    @property
    def remaining_numbers(self) -> Set[int]:
        return set(self._remaining)

    @property
    def remaining_primes_count(self) -> int:
        return self._primes_remaining()

    @property
    def remaining_composites_count(self) -> int:
        return self._composites_remaining()

    @property
    def primes_removed_count(self) -> int:
        return self._primes_removed

    @property
    def removed_primes_list(self) -> List[int]:
        return sorted(self._removed_primes)

    @property
    def score(self) -> int:
        return self._current_score()

    def episode_summary(self, steps: int, terminated_reason: str = "unknown") -> EpisodeSummary:
        success = (self.primes_removed_count == 0 and self.remaining_composites_count == 0)
        return EpisodeSummary(
            steps=steps,
            total_primes=self.total_primes,
            total_composites=self.total_composites,
            primes_removed_count=self.primes_removed_count,
            primes_removed_list=self.removed_primes_list,
            composites_remaining_count=self.remaining_composites_count,
            primes_remaining_count=self.remaining_primes_count,
            score=self.score,
            terminated_reason=terminated_reason,
            success=success,
        )
