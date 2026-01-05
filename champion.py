from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


ProgramDict = Dict[str, object]


@dataclass
class ProgramRecord:
    program_name: str
    source: str
    rules: List[Dict[str, object]]
    train_metrics: Dict[str, object]
    battery_pass: bool
    battery_summary: Dict[str, object]


def _battery_totals(rec: ProgramRecord) -> tuple[int, int]:
    """Return (total_steps, max_steps) across battery ranges for a record."""
    summary = rec.battery_summary or {}
    total = int(summary.get("battery_total_steps", 0))
    max_steps = int(summary.get("battery_max_steps", 0))
    return total, max_steps


def choose_champion_program(program_records: List[ProgramRecord]) -> ProgramRecord:
    """Select a champion program for a run.

    Preference order:
    1) Any program with battery_pass == True, ranked by:
       - Minimize battery_total_steps (sum steps over all ranges).
       - Then minimize max steps on any range.
       - Then minimize train_steps.
       - Then fewer rules.
    2) If none pass battery, fall back to best training program using:
       - primes_removed_count
       - composites_remaining_count
       - train_steps
       - fewer rules.
    """
    if not program_records:
        raise ValueError("No program records available for champion selection")

    # Step 1: filter to battery-passing programs.
    passing = [rec for rec in program_records if rec.battery_pass]

    def _score_battery(rec: ProgramRecord) -> tuple[int, int, int, int]:
        total_steps, max_steps = _battery_totals(rec)
        train_steps = int(rec.train_metrics.get("steps", 0))
        rule_count = len(rec.rules)
        return (total_steps, max_steps, train_steps, rule_count)

    def _score_training(rec: ProgramRecord) -> tuple[int, int, int, int]:
        primes_removed = int(rec.train_metrics.get("primes_removed_count", 0))
        comps_remaining = int(rec.train_metrics.get("composites_remaining_count", 0))
        train_steps = int(rec.train_metrics.get("steps", 0))
        rule_count = len(rec.rules)
        return (primes_removed, comps_remaining, train_steps, rule_count)

    if passing:
        # Prefer the battery-passing program with best aggregate metrics.
        champion = min(passing, key=_score_battery)
    else:
        # Fall back to the best program by training metrics alone.
        champion = min(program_records, key=_score_training)

    return champion

