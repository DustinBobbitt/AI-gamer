"""Out-of-distribution benchmark for learned geometry and verifier policies."""
from __future__ import annotations

import argparse
import json
from statistics import mean
from typing import Dict, List, Sequence

from domains.arithmetic.geometry import GeometryPriorPolicy, classify_geometry
from domains.arithmetic.scenarios import ScenarioGenerator
from domains.arithmetic.state import BeliefState
from domains.arithmetic.verifier import verify_factors_adaptively
from domains.arithmetic.verifier_policy import VerifierBudgetPolicy


def run_benchmark(
    seed: int = 8675309,
    count_per_group: int = 50,
    bit_lengths: Sequence[int] = (20, 28, 36, 44, 56),
) -> Dict[str, object]:
    generator = ScenarioGenerator(seed)
    verifier_policy = VerifierBudgetPolicy.load()
    geometry_policy = GeometryPriorPolicy.load()
    rows: List[Dict[str, object]] = []

    for bits in bit_lengths:
        for regime in ("balanced", "intermediate", "skewed", "square"):
            for scenario in generator.generate_batch(count_per_group, bits, regime):
                budget = verifier_policy.budget_for(scenario.N)
                verification = verify_factors_adaptively(
                    scenario.N,
                    BeliefState(),
                    fermat_budget=budget,
                )
                geometry = geometry_policy.predict(
                    scenario.N,
                    assume_semiprime=True,
                    allow_square=regime == "square",
                )
                rows.append(
                    {
                        "requested_bits": bits,
                        "actual_bits": scenario.bit_length,
                        "regime": regime,
                        "true_geometry": classify_geometry(scenario.p, scenario.q),
                        "geometry_label": geometry.label,
                        "geometry_truth_probability": geometry.probabilities[
                            classify_geometry(scenario.p, scenario.q)
                        ],
                        "fermat_budget": budget,
                        "verifier_success": verification.factors_found,
                        "checks": verification.checks_attempted,
                        "strategy": verification.strategy_used,
                    }
                )

    summary: Dict[str, Dict[str, float]] = {}
    for bits in bit_lengths:
        for regime in ("balanced", "intermediate", "skewed", "square"):
            group = [
                row for row in rows
                if row["requested_bits"] == bits and row["regime"] == regime
            ]
            summary[f"{bits}:{regime}"] = {
                "episodes": len(group),
                "verifier_success_rate": mean(row["verifier_success"] for row in group),
                "mean_checks": mean(row["checks"] for row in group),
                "mean_geometry_truth_probability": mean(
                    row["geometry_truth_probability"] for row in group
                ),
            }

    return {
        "seed": seed,
        "count_per_group": count_per_group,
        "bit_lengths": list(bit_lengths),
        "summary": summary,
        "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=8675309)
    parser.add_argument("--count-per-group", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.seed, args.count_per_group), indent=2))


if __name__ == "__main__":
    main()
