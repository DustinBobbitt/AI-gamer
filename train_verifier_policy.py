"""Train and validate the frozen arithmetic verifier budget policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from domains.arithmetic.verifier_policy import (
    DEFAULT_POLICY_PATH,
    train_verifier_budget_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--count-per-distribution", type=int, default=100)
    parser.add_argument("--training-seed", type=int, default=20260919)
    parser.add_argument("--validation-seed", type=int, default=271828)
    args = parser.parse_args()

    policy = train_verifier_budget_policy(
        training_seed=args.training_seed,
        validation_seed=args.validation_seed,
        count_per_distribution=args.count_per_distribution,
    )
    policy.save(args.output)
    print(json.dumps(policy.to_dict(), indent=2))


if __name__ == "__main__":
    main()
