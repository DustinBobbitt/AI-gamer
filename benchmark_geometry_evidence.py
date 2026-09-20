"""Evaluate exact-square geometry evidence on held-out semiprime squares."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Dict, Sequence

from domains.arithmetic.geometry import (
    GeometryPriorPolicy,
    evaluate_geometry_predictions,
)
from domains.arithmetic.scenarios import ScenarioGenerator


def run_benchmark(
    seed: int = 424242,
    count_per_bit_length: int = 100,
    bit_lengths: Sequence[int] = (16, 24, 32, 40, 48),
) -> Dict[str, object]:
    generator = ScenarioGenerator(seed)
    squares = [
        scenario
        for bits in bit_lengths
        for scenario in generator.generate_batch(count_per_bit_length, bits, "square")
    ]
    ordinary = [
        scenario
        for bits in bit_lengths
        for regime in ("balanced", "intermediate", "skewed")
        for scenario in generator.generate_batch(count_per_bit_length, bits, regime)
    ]
    policy = GeometryPriorPolicy.load()

    square_prior = evaluate_geometry_predictions(
        squares,
        lambda target: policy.predict(
            target,
            assume_semiprime=True,
            allow_square=False,
        ).probabilities,
    )
    square_evidence = evaluate_geometry_predictions(
        squares,
        lambda target: policy.predict(
            target,
            assume_semiprime=True,
            allow_square=True,
        ).probabilities,
    )
    ordinary_prior = evaluate_geometry_predictions(
        ordinary,
        lambda target: policy.predict(target).probabilities,
    )
    ordinary_evidence = evaluate_geometry_predictions(
        ordinary,
        lambda target: policy.predict(
            target,
            assume_semiprime=True,
            allow_square=True,
        ).probabilities,
    )
    return {
        "hypothesis": (
            "Assumption-gated exact-square evidence improves held-out semiprime-square "
            "geometry NLL and Brier score without changing ordinary non-square calibration."
        ),
        "seed": seed,
        "square_episodes": len(squares),
        "ordinary_episodes": len(ordinary),
        "square_prior": asdict(square_prior),
        "square_with_evidence": asdict(square_evidence),
        "ordinary_prior": asdict(ordinary_prior),
        "ordinary_with_evidence": asdict(ordinary_evidence),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--count-per-bit-length", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.seed, args.count_per_bit_length), indent=2))


if __name__ == "__main__":
    main()
