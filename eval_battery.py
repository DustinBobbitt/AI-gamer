from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

from env import EpisodeSummary, PrimeIceConfig, PrimeIceEnv
from programs import expand_program_if_sieve
from rules import rule_from_dict
from utils.io import dump_json, dump_text


RuleDict = Dict[str, object]


@dataclass
class BatteryRange:
    name: str
    low: int
    high: int


DEFAULT_BATTERY: List[BatteryRange] = [
    BatteryRange(name="battery_small", low=3, high=2000),
    BatteryRange(name="battery_medium", low=3, high=20000),
    BatteryRange(name="battery_shifted", low=1_000_003, high=1_020_003),
    BatteryRange(name="battery_bigwin", low=50_000_000, high=50_020_000),
]


def evaluate_program_on_range(
    rules: List[RuleDict],
    low: int,
    high: int,
) -> Tuple[EpisodeSummary, int, str]:
    """Replay program on a fresh env for a single range.

    If rules form a divisibility sieve prefix, expands to full sieve for the range.
    
    Returns episode summary, steps taken, and replay_mode used.
    """
    # Expand if it's a sieve prefix
    expanded_rules, replay_mode = expand_program_if_sieve(rules, high)
    
    # For sieve_expand mode, disable no_progress termination (late sieve steps often remove 0)
    # and set max_steps based on number of primes to apply
    if replay_mode == "sieve_expand":
        config = PrimeIceConfig(low=low, high=high, no_progress_limit=10**9)
        # Max steps should be number of rules + small buffer
        # This prevents timeouts while allowing the full sieve to run
        max_steps_override = len(expanded_rules) + 10
    else:
        config = PrimeIceConfig(low=low, high=high)
        max_steps_override = None
    
    env = PrimeIceEnv(config)
    env.reset()
    steps = 0
    done = False
    terminated_reason = None
    
    for rule_dict in expanded_rules:
        if done:
            break
        # Check max_steps override for sieve_expand
        if max_steps_override is not None and steps >= max_steps_override:
            done = True
            terminated_reason = "max_steps"
            break
        rule = rule_from_dict(rule_dict)
        _, _, done, info = env.step(rule)
        steps += 1
        terminated_reason = info.get("terminated_reason", "unknown")
    
    # If we exhausted all rules without early termination, mark as finished
    if not done:
        terminated_reason = "rules_exhausted"
    
    summary = env.episode_summary(steps, terminated_reason or "unknown")
    return summary, steps, replay_mode


def evaluate_program_on_battery(
    rules: List[RuleDict],
    battery: List[BatteryRange] | None = None,
) -> Dict[str, object]:
    """Evaluate a program on a set of ranges and return structured summary."""
    from programs import detect_sieve_prefix
    from utils.primes import primes_up_to
    
    ranges = battery or DEFAULT_BATTERY
    per_range: Dict[str, Dict[str, object]] = {}
    all_pass = True
    total_steps = 0
    max_steps = 0
    replay_mode = "fixed"  # Will be set by first range evaluation
    
    # Compute sieve expansion info if this is a sieve program
    sieve_expand_info: Dict[str, object] = {}
    prefix = detect_sieve_prefix(rules)
    if prefix is not None and ranges:
        # Use max range to compute expansion limit
        max_range_high = max(spec.high for spec in ranges)
        limit = int(max_range_high ** 0.5)
        all_primes = [p for p in primes_up_to(limit) if p >= 3]
        sieve_expand_info = {
            "sieve_expand_limit": limit,
            "num_primes_applied": len(all_primes),
            "prefix_length": len(prefix),
        }

    for spec in ranges:
        summary, steps, mode = evaluate_program_on_range(rules, spec.low, spec.high)
        replay_mode = mode  # Capture replay mode (should be same for all ranges)
        metrics = asdict(summary)
        metrics["range_low"] = spec.low
        metrics["range_high"] = spec.high
        metrics["steps"] = steps

        per_range[spec.name] = metrics

        total_steps += steps
        max_steps = max(max_steps, steps)
        if summary.primes_removed_count != 0 or summary.composites_remaining_count != 0:
            all_pass = False

    result = {
        "battery_pass": all_pass,
        "battery_total_steps": total_steps,
        "battery_max_steps": max_steps,
        "replay_mode": replay_mode,
        "ranges": per_range,
    }
    if sieve_expand_info:
        result.update(sieve_expand_info)
    return result


def write_battery_report(
    json_path,
    txt_path,
    report: Dict[str, object],
) -> None:
    """Write JSON and text battery reports for a run."""
    dump_json(json_path, report)

    lines: List[str] = []
    lines.append("Prime Game — Generalization Battery Report")
    lines.append("")
    lines.append(f"Overall PASS: {bool(report.get('battery_pass', False))}")
    lines.append(f"Replay mode: {report.get('replay_mode', 'fixed')}")
    lines.append(
        f"Total steps across ranges: {int(report.get('battery_total_steps', 0))}, "
        f"max steps on any range: {int(report.get('battery_max_steps', 0))}"
    )
    lines.append("")

    ranges: Dict[str, Dict[str, object]] = report.get("ranges", {})  # type: ignore[assignment]
    for name, metrics in ranges.items():
        lines.append(f"Range {name}: {metrics['range_low']}..{metrics['range_high']}")
        lines.append(f"  steps={metrics['steps']}")
        lines.append(
            f"  primes_preserved={metrics['primes_remaining_count']}/"
            f"{metrics['total_primes']}, primes_removed={metrics['primes_removed_count']}"
        )
        lines.append(
            f"  composites_remaining={metrics['composites_remaining_count']} "
            f"(out of {metrics['total_composites']})"
        )
        lines.append(f"  terminated_reason={metrics.get('terminated_reason', 'unknown')}")
        lines.append("")
    dump_text(txt_path, "\n".join(lines))

