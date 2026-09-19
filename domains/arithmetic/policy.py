"""Policies for choosing constraint transforms during arithmetic inference."""
from __future__ import annotations

from typing import Optional

from domains.arithmetic.state import BeliefState


class AdaptiveInferencePolicy:
    """Run only empirically accepted factor-blind evidence transforms."""

    def __init__(self, min_progress: float = 1e-8, max_action_repeats: int = 8) -> None:
        self.min_progress = max(0.0, float(min_progress))
        self.max_action_repeats = max(1, int(max_action_repeats))
        self._exact_square_checked = False

    def select_action(self, state: BeliefState, target_n: int) -> Optional[int]:
        """Return the exact-square evidence action once, then stop."""
        del state, target_n
        if not self._exact_square_checked:
            self._exact_square_checked = True
            return 1
        return None
