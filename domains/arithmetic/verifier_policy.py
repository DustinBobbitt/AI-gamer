"""Offline training and persistence for bounded verifier strategy budgets."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

from domains.arithmetic.scenarios import ScenarioGenerator, SemiprimeScenario
from domains.arithmetic.state import BeliefState
from domains.arithmetic.verifier import verify_factors_adaptively


SCHEMA_VERSION = 1
DEFAULT_POLICY_PATH = Path(__file__).with_name("verifier_policy.json")


@dataclass(frozen=True)
class BudgetMetrics:
    success_rate: float
    mean_checks: float
    p90_checks: float
    episodes: int


@dataclass(frozen=True)
class BitBudget:
    max_bits: int
    fermat_budget: int
    train_metrics: BudgetMetrics
    validation_metrics: BudgetMetrics


@dataclass(frozen=True)
class VerifierBudgetPolicy:
    """Frozen mapping from target size to a validated Fermat probe budget."""

    buckets: Tuple[BitBudget, ...]
    training_seed: int
    validation_seed: int
    train_manifest_hash: str
    validation_manifest_hash: str
    schema_version: int = SCHEMA_VERSION

    def budget_for(self, target_n: int) -> int:
        bits = target_n.bit_length()
        for bucket in sorted(self.buckets, key=lambda item: item.max_bits):
            if bits <= bucket.max_bits:
                return bucket.fermat_budget
        return self.buckets[-1].fermat_budget

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def save(self, path: Path = DEFAULT_POLICY_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "VerifierBudgetPolicy":
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported verifier policy schema: {data.get('schema_version')}")
        buckets = tuple(
            BitBudget(
                max_bits=int(bucket["max_bits"]),
                fermat_budget=int(bucket["fermat_budget"]),
                train_metrics=BudgetMetrics(**bucket["train_metrics"]),
                validation_metrics=BudgetMetrics(**bucket["validation_metrics"]),
            )
            for bucket in data["buckets"]
        )
        return cls(
            buckets=buckets,
            training_seed=int(data["training_seed"]),
            validation_seed=int(data["validation_seed"]),
            train_manifest_hash=str(data["train_manifest_hash"]),
            validation_manifest_hash=str(data["validation_manifest_hash"]),
            schema_version=int(data["schema_version"]),
        )

    @classmethod
    def load(cls, path: Path = DEFAULT_POLICY_PATH) -> "VerifierBudgetPolicy":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _manifest_hash(scenarios: Iterable[SemiprimeScenario]) -> str:
    digest = hashlib.sha256()
    for scenario in scenarios:
        record = {
            "N": scenario.N,
            "p": scenario.p,
            "q": scenario.q,
            "actual_bits": scenario.bit_length,
            "distribution": scenario.distribution_type.value,
        }
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _generate_corpus(
    seed: int,
    bit_lengths: Sequence[int],
    count_per_distribution: int,
) -> Dict[int, List[SemiprimeScenario]]:
    generator = ScenarioGenerator(seed)
    return {
        bits: (
            generator.generate_batch(count_per_distribution, bits, "balanced")
            + generator.generate_batch(count_per_distribution, bits, "skewed")
        )
        for bits in bit_lengths
    }


def evaluate_budget(
    scenarios: Sequence[SemiprimeScenario],
    fermat_budget: int,
    max_checks: int = 100000,
) -> BudgetMetrics:
    checks: List[int] = []
    successes: List[bool] = []
    for scenario in scenarios:
        result = verify_factors_adaptively(
            scenario.N,
            BeliefState(),
            max_checks=max_checks,
            fermat_budget=fermat_budget,
        )
        successes.append(result.factors_found)
        checks.append(result.checks_attempted if result.factors_found else max_checks)
    ordered = sorted(checks)
    p90_index = max(0, min(len(ordered) - 1, int(0.9 * len(ordered)) - 1))
    return BudgetMetrics(
        success_rate=mean(successes),
        mean_checks=mean(checks),
        p90_checks=float(ordered[p90_index]),
        episodes=len(scenarios),
    )


def _select_budget(
    scenarios: Sequence[SemiprimeScenario],
    candidate_budgets: Sequence[int],
    minimum_success: float,
) -> Tuple[int, BudgetMetrics]:
    evaluations = [
        (budget, evaluate_budget(scenarios, budget))
        for budget in candidate_budgets
    ]
    eligible = [
        item for item in evaluations
        if item[1].success_rate >= minimum_success
    ]
    pool = eligible or evaluations
    return min(
        pool,
        key=lambda item: (
            -item[1].success_rate,
            item[1].mean_checks,
            item[0],
        ),
    )


def train_verifier_budget_policy(
    training_seed: int = 20260919,
    validation_seed: int = 271828,
    bit_lengths: Sequence[int] = (16, 24, 32, 40, 48),
    count_per_distribution: int = 100,
    candidate_budgets: Sequence[int] = (
        0, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
    ),
    minimum_success: float = 1.0,
) -> VerifierBudgetPolicy:
    """Fit budgets on training records and report untouched validation metrics."""
    train = _generate_corpus(training_seed, bit_lengths, count_per_distribution)
    validation = _generate_corpus(validation_seed, bit_lengths, count_per_distribution)
    buckets: List[BitBudget] = []
    for bits in bit_lengths:
        budget, train_metrics = _select_budget(
            train[bits],
            candidate_budgets,
            minimum_success,
        )
        validation_metrics = evaluate_budget(validation[bits], budget)
        buckets.append(
            BitBudget(
                max_bits=bits,
                fermat_budget=budget,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
            )
        )

    train_records = [scenario for bits in bit_lengths for scenario in train[bits]]
    validation_records = [scenario for bits in bit_lengths for scenario in validation[bits]]
    return VerifierBudgetPolicy(
        buckets=tuple(buckets),
        training_seed=training_seed,
        validation_seed=validation_seed,
        train_manifest_hash=_manifest_hash(train_records),
        validation_manifest_hash=_manifest_hash(validation_records),
    )
