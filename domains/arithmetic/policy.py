"""Policies for choosing constraint transforms during arithmetic inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set

import numpy as np

from domains.arithmetic.state import BeliefState
from domains.arithmetic.transforms import TransformLibrary


@dataclass(frozen=True)
class ActionPreview:
    """Expected effect of applying one transform."""

    action: int
    entropy_gain: float
    confidence_gain: float
    state_change: float


class AdaptiveInferencePolicy:
    """State-aware policy that avoids reapplying transforms with no benefit.

    The first three actions extract independent features from ``N``. Afterwards,
    the policy previews every deterministic transform and only applies one when
    it is expected to improve entropy or confidence. Returning ``None`` means
    that the available transform library has reached a fixed point.
    """

    _FEATURE_ACTIONS = (0, 1, 2)

    def __init__(self, min_progress: float = 1e-8, max_action_repeats: int = 8) -> None:
        self.min_progress = max(0.0, float(min_progress))
        self.max_action_repeats = max(1, int(max_action_repeats))
        self._features_applied: Set[int] = set()
        self.action_counts: Dict[int, int] = {}

    def select_action(self, state: BeliefState, target_n: int) -> Optional[int]:
        """Choose the next useful transform, or ``None`` when work is complete."""
        for action in self._FEATURE_ACTIONS:
            if action not in self._features_applied:
                self._features_applied.add(action)
                return action

        previews = [
            self._preview(action, state, target_n)
            for action in range(TransformLibrary.get_action_space_size())
        ]
        useful = [
            preview
            for preview in previews
            if self.action_counts.get(preview.action, 0) < self.max_action_repeats
            and (
                preview.entropy_gain > self.min_progress
                or preview.confidence_gain > self.min_progress
            )
        ]
        if not useful:
            return None

        best = max(
            useful,
            key=lambda preview: (
                preview.entropy_gain,
                preview.confidence_gain,
                preview.state_change,
                -self.action_counts.get(preview.action, 0),
            ),
        )
        self.action_counts[best.action] = self.action_counts.get(best.action, 0) + 1
        return best.action

    @staticmethod
    def _preview(action: int, state: BeliefState, target_n: int) -> ActionPreview:
        candidate = TransformLibrary.get_transform_by_id(action).apply(state, target_n).new_state
        before = state.to_vector()
        after = candidate.to_vector()
        return ActionPreview(
            action=action,
            entropy_gain=max(0.0, state.entropy_estimate - candidate.entropy_estimate),
            confidence_gain=max(0.0, candidate.confidence - state.confidence),
            state_change=float(np.linalg.norm(after[:-1] - before[:-1])),
        )
