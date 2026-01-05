from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence, TypeVar

T = TypeVar("T")


@dataclass
class RandomSource:
    """Wrapper around random.Random for deterministic seeding."""

    seed: int

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def choice(self, seq: Sequence[T]) -> T:
        return self._rng.choice(seq)

    def random(self) -> float:
        return self._rng.random()

    def shuffle(self, seq: Sequence[T]) -> None:
        self._rng.shuffle(seq)

