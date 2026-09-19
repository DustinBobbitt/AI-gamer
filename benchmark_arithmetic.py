"""Reproducible held-out benchmark for arithmetic inference policies.

This experiment keeps factor ground truth out of inference. Ground truth is
used only after each episode to evaluate calibration and verifier success.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

from domains.arithmetic.env import SemiprimeInferenceEnv
from domains.arithmetic.policy import AdaptiveInferencePolicy
from domains.arithmetic.scenarios import ScenarioGenerator, SemiprimeScenario
from domains.arithmetic.state import BeliefState
from domains.arithmetic.verifier import verify_factors_adaptively, verify_factors_from_belief
from domains.arithmetic.verifier_policy import VerifierBudgetPolicy


@dataclass(frozen=True)
class EpisodeResult:
    policy: str
    verifier: str
    distribution: str
    bit_length: int
    steps: int
    termination_reason: str
    entropy_reduction: float
    confidence: float
    near_square_score: float
    true_factor_ratio: float
    verifier_success: bool
    verifier_checks: int
    window_contains_factor: bool


class FixedOrderPolicy:
    """Non-learning baseline that repeatedly applies a fixed transform order."""

    def __init__(self) -> None:
        self._step = 0

    def select_action(self, state: BeliefState, target_n: int) -> int:
        del state, target_n
        action = self._step % 4
        self._step += 1
        return action


class RandomPolicy:
    """Seeded random baseline."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def select_action(self, state: BeliefState, target_n: int) -> int:
        del state, target_n
        return self._rng.randrange(4)


def _initialize_env(scenario: SemiprimeScenario, max_steps: int) -> SemiprimeInferenceEnv:
    env = SemiprimeInferenceEnv(
        {
            "max_steps": max_steps,
            "min_steps": 0,
            "disable_early_stop": False,
            "convergence_threshold": 1e-4,
            "bit_length": scenario.bit_length,
            "distribution_type": scenario.distribution_type.value,
        }
    )
    env.current_N = scenario.N
    env.true_p = None
    env.true_q = None
    env.belief_state = BeliefState()
    env.belief_state.update_entropy()
    env.entropy_history = [env.belief_state.entropy_estimate]
    env.entropy_start = env.belief_state.entropy_estimate
    return env


def run_episode(
    scenario: SemiprimeScenario,
    policy_name: str,
    seed: int,
    verifier_name: str = "window",
    max_steps: int = 50,
) -> EpisodeResult:
    """Run one factor-blind episode and evaluate it against hidden truth."""
    env = _initialize_env(scenario, max_steps)
    if policy_name == "adaptive":
        policy = AdaptiveInferencePolicy(min_progress=1e-4)
    elif policy_name == "fixed":
        policy = FixedOrderPolicy()
    elif policy_name == "random":
        policy = RandomPolicy(seed)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    done = False
    while not done and env.step_count < max_steps:
        action: Optional[int] = policy.select_action(env.belief_state, scenario.N)
        if action is None:
            env.finish("policy_complete")
            break
        _, _, done, _ = env.step(action)

    if verifier_name == "window":
        verification = verify_factors_from_belief(scenario.N, env.belief_state)
    elif verifier_name == "portfolio":
        verification = verify_factors_adaptively(scenario.N, env.belief_state)
    elif verifier_name == "learned":
        policy = VerifierBudgetPolicy.load()
        verification = verify_factors_adaptively(
            scenario.N,
            env.belief_state,
            fermat_budget=policy.budget_for(scenario.N),
        )
    else:
        raise ValueError(f"Unknown verifier: {verifier_name}")
    summary = env.generate_inference_summary()
    true_ratio = scenario.p / scenario.q
    return EpisodeResult(
        policy=policy_name,
        verifier=verifier_name,
        distribution=scenario.distribution_type.value,
        bit_length=scenario.bit_length,
        steps=env.step_count,
        termination_reason=env.termination_reason,
        entropy_reduction=env.entropy_start - env.belief_state.entropy_estimate,
        confidence=env.belief_state.confidence,
        near_square_score=env.belief_state.near_square_score,
        true_factor_ratio=true_ratio,
        verifier_success=verification.factors_found,
        verifier_checks=verification.checks_attempted,
        window_contains_factor=summary.size_window_low <= scenario.p <= summary.size_window_high,
    )


def generate_held_out_scenarios(
    seed: int,
    count_per_group: int,
    bit_lengths: Iterable[int],
) -> List[SemiprimeScenario]:
    generator = ScenarioGenerator(seed=seed)
    scenarios: List[SemiprimeScenario] = []
    for bit_length in bit_lengths:
        scenarios.extend(generator.generate_batch(count_per_group, bit_length, "balanced"))
        scenarios.extend(generator.generate_batch(count_per_group, bit_length, "skewed"))
    return scenarios


def summarize(results: List[EpisodeResult]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[EpisodeResult]] = {}
    for result in results:
        key = f"{result.policy}:{result.verifier}:{result.distribution}"
        groups.setdefault(key, []).append(result)
        groups.setdefault(f"{result.policy}:{result.verifier}:all", []).append(result)

    return {
        key: {
            "episodes": len(group),
            "verifier_success_rate": mean(item.verifier_success for item in group),
            "window_coverage_rate": mean(item.window_contains_factor for item in group),
            "mean_steps": mean(item.steps for item in group),
            "mean_entropy_reduction": mean(item.entropy_reduction for item in group),
            "mean_verifier_checks": mean(item.verifier_checks for item in group),
            "mean_near_square_score": mean(item.near_square_score for item in group),
            "mean_true_factor_ratio": mean(item.true_factor_ratio for item in group),
        }
        for key, group in sorted(groups.items())
    }


def run_benchmark(
    seed: int = 20260919,
    count_per_group: int = 20,
    bit_lengths: Iterable[int] = (16, 24, 32),
) -> Dict[str, object]:
    scenarios = generate_held_out_scenarios(seed, count_per_group, bit_lengths)
    results = [
        run_episode(scenario, policy_name, seed + index, verifier_name)
        for index, scenario in enumerate(scenarios)
        for policy_name in ("random", "fixed", "adaptive")
        for verifier_name in ("window", "portfolio", "learned")
    ]
    return {
        "hypothesis": (
            "The adaptive policy reduces redundant steps and verifier work without "
            "lowering held-out factor-verification success versus random and fixed order."
        ),
        "seed": seed,
        "scenario_count": len(scenarios),
        "summary": summarize(results),
        "episodes": [asdict(result) for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--count-per-group", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run_benchmark(seed=args.seed, count_per_group=args.count_per_group)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
