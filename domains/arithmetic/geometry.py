"""Calibrated factor-geometry beliefs trained without runtime factor access."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

from domains.arithmetic.scenarios import ScenarioGenerator, SemiprimeScenario


REGIMES = ("balanced", "intermediate", "skewed")
SCHEMA_VERSION = 1
DEFAULT_GEOMETRY_PATH = Path(__file__).with_name("geometry_policy.json")


def classify_geometry(p: int, q: int) -> str:
    """Classify hidden factor geometry for training and evaluation only."""
    z = math.log2(max(p, q) / min(p, q))
    if z < 0.5:
        return "balanced"
    if z < 4.0:
        return "intermediate"
    return "skewed"


@dataclass(frozen=True)
class CalibrationMetrics:
    negative_log_likelihood: float
    brier_score: float
    accuracy: float
    episodes: int


@dataclass(frozen=True)
class GeometryBucket:
    max_bits: int
    probabilities: Dict[str, float]
    validation_metrics: CalibrationMetrics
    legacy_metrics: CalibrationMetrics


@dataclass(frozen=True)
class FactorGeometryPrediction:
    probabilities: Dict[str, float]
    label: str
    confidence: float
    model_manifest_hash: str


@dataclass(frozen=True)
class GeometryPriorPolicy:
    """Frozen calibrated prior selected by target bit length."""

    buckets: Tuple[GeometryBucket, ...]
    training_seed: int
    validation_seed: int
    train_manifest_hash: str
    validation_manifest_hash: str
    schema_version: int = SCHEMA_VERSION

    def predict(self, target_n: int) -> FactorGeometryPrediction:
        bits = target_n.bit_length()
        bucket = next(
            (item for item in sorted(self.buckets, key=lambda item: item.max_bits) if bits <= item.max_bits),
            self.buckets[-1],
        )
        best = max(bucket.probabilities, key=bucket.probabilities.get)
        ordered = sorted(bucket.probabilities.values(), reverse=True)
        label = "indeterminate" if ordered[0] - ordered[1] < 0.10 else best
        return FactorGeometryPrediction(
            probabilities=dict(bucket.probabilities),
            label=label,
            confidence=ordered[0],
            model_manifest_hash=self.train_manifest_hash,
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def save(self, path: Path = DEFAULT_GEOMETRY_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "GeometryPriorPolicy":
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported geometry policy schema: {data.get('schema_version')}")
        buckets = tuple(
            GeometryBucket(
                max_bits=int(bucket["max_bits"]),
                probabilities={
                    regime: float(bucket["probabilities"][regime])
                    for regime in REGIMES
                },
                validation_metrics=CalibrationMetrics(**bucket["validation_metrics"]),
                legacy_metrics=CalibrationMetrics(**bucket["legacy_metrics"]),
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
    def load(cls, path: Path = DEFAULT_GEOMETRY_PATH) -> "GeometryPriorPolicy":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _generate_corpus(
    seed: int,
    bit_lengths: Sequence[int],
    count_per_regime: int,
) -> Dict[int, List[SemiprimeScenario]]:
    generator = ScenarioGenerator(seed)
    return {
        bits: [
            scenario
            for regime in REGIMES
            for scenario in generator.generate_batch(count_per_regime, bits, regime)
        ]
        for bits in bit_lengths
    }


def _manifest_hash(scenarios: Iterable[SemiprimeScenario]) -> str:
    digest = hashlib.sha256()
    for scenario in scenarios:
        record = {
            "N": scenario.N,
            "p": scenario.p,
            "q": scenario.q,
            "actual_bits": scenario.bit_length,
            "label": classify_geometry(scenario.p, scenario.q),
        }
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _legacy_probabilities(target_n: int) -> Dict[str, float]:
    """Convert the retired integer-square-gap score into testable probabilities."""
    root = math.isqrt(target_n)
    ceiling = root if root * root == target_n else root + 1
    relative_gap = (ceiling * ceiling - target_n) / target_n
    balanced = max(0.001, min(0.998, 1.0 - min(relative_gap * 10.0, 1.0)))
    remainder = 1.0 - balanced
    return {
        "balanced": balanced,
        "intermediate": remainder / 2.0,
        "skewed": remainder / 2.0,
    }


def evaluate_geometry_predictions(
    scenarios: Sequence[SemiprimeScenario],
    predictor,
) -> CalibrationMetrics:
    nll: List[float] = []
    brier: List[float] = []
    correct: List[bool] = []
    for scenario in scenarios:
        probabilities = predictor(scenario.N)
        truth = classify_geometry(scenario.p, scenario.q)
        nll.append(-math.log(max(1e-12, probabilities[truth])))
        brier.append(sum(
            (probabilities[regime] - float(regime == truth)) ** 2
            for regime in REGIMES
        ))
        correct.append(max(probabilities, key=probabilities.get) == truth)
    return CalibrationMetrics(
        negative_log_likelihood=mean(nll),
        brier_score=mean(brier),
        accuracy=mean(correct),
        episodes=len(scenarios),
    )


def train_geometry_prior_policy(
    training_seed: int = 20260919,
    validation_seed: int = 271828,
    bit_lengths: Sequence[int] = (16, 24, 32, 40, 48),
    count_per_regime: int = 100,
    smoothing: float = 1.0,
) -> GeometryPriorPolicy:
    """Fit class priors on training records and score untouched validation data."""
    train = _generate_corpus(training_seed, bit_lengths, count_per_regime)
    validation = _generate_corpus(validation_seed, bit_lengths, count_per_regime)
    buckets: List[GeometryBucket] = []

    for bits in bit_lengths:
        counts = {regime: smoothing for regime in REGIMES}
        for scenario in train[bits]:
            counts[classify_geometry(scenario.p, scenario.q)] += 1.0
        total = sum(counts.values())
        probabilities = {regime: counts[regime] / total for regime in REGIMES}
        calibrated = evaluate_geometry_predictions(
            validation[bits],
            lambda _target, values=probabilities: values,
        )
        legacy = evaluate_geometry_predictions(validation[bits], _legacy_probabilities)
        buckets.append(
            GeometryBucket(
                max_bits=bits,
                probabilities=probabilities,
                validation_metrics=calibrated,
                legacy_metrics=legacy,
            )
        )

    train_records = [scenario for bits in bit_lengths for scenario in train[bits]]
    validation_records = [scenario for bits in bit_lengths for scenario in validation[bits]]
    return GeometryPriorPolicy(
        buckets=tuple(buckets),
        training_seed=training_seed,
        validation_seed=validation_seed,
        train_manifest_hash=_manifest_hash(train_records),
        validation_manifest_hash=_manifest_hash(validation_records),
    )
