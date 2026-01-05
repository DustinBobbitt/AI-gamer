from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import traceback


def _get_train_range_string(config: Dict[str, Any]) -> str:
    low = config.get("train_low")
    high = config.get("train_high")
    if low is None or high is None:
        return "N/A"
    return f"{low}..{high}"


def _fmt_int_or_na(x: Any) -> str:
    try:
        return str(int(x))
    except Exception:
        return "N/A"


def _cap_list(items: Iterable[Any], head: int = 30, tail: int = 10) -> str:
    lst = list(items)
    n = len(lst)
    if n == 0:
        return "[]"
    if n <= head + tail:
        return "[" + ", ".join(str(x) for x in lst) + "]"
    head_items = ", ".join(str(x) for x in lst[:head])
    tail_items = ", ".join(str(x) for x in lst[-tail:])
    return f"[{head_items}, ..., {tail_items}] (N={n})"


def _safe_get(d: Optional[Dict[str, Any]], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        if key not in cur:
            return default
        cur = cur[key]
    return cur


def _rule_human(rule_dict: Dict[str, Any]) -> str:
    rtype = rule_dict.get("type")
    if rtype == "divisibility":
        d = int(rule_dict.get("d", 0))
        keep_self = bool(rule_dict.get("keep_self", True))
        if keep_self:
            return f"Remove multiples of {d} (keep {d})"
        return f"Remove multiples of {d}"
    if rtype == "mod_class":
        m = int(rule_dict.get("m", 0))
        banned = rule_dict.get("banned_residues", []) or []
        resid_str = ",".join(str(int(x)) for x in sorted({int(x) for x in banned}))
        return f"Remove residues {{{resid_str}}} mod {m}"
    if rtype == "keep_residues":
        m = int(rule_dict.get("m", 0))
        allowed = rule_dict.get("allowed_residues", []) or []
        resid_str = ",".join(str(int(x)) for x in sorted({int(x) for x in allowed}))
        return f"Keep residues {{{resid_str}}} mod {m}"
    return str(rule_dict)


def _extract_training_metrics(
    train_result: Optional[Dict[str, Any]],
    champion_program: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    # Priority 1: champion replay_metrics
    if champion_program:
        replay = champion_program.get("replay_metrics")
        if isinstance(replay, dict):
            return replay
        final = champion_program.get("final_metrics")
        if isinstance(final, dict):
            return final

    # Priority 2: best episode summary from train_result
    if train_result:
        best = train_result.get("best_episode")
        if isinstance(best, dict):
            summary = best.get("summary")
            if isinstance(summary, dict):
                return summary

    return {}


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Safely load JSON from path, return None if file missing or parse error."""
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def build_summary_text(
    run_dir: Path,
    config: Dict[str, Any],
    train_result: Optional[Dict[str, Any]],
    rules_report: Optional[Dict[str, Any]],
    champion_program: Optional[Dict[str, Any]],
    battery_report: Optional[Dict[str, Any]],
    eval_report: Optional[Dict[str, Any]],
    error: Optional[BaseException] = None,
) -> str:
    """Build the full human-readable summary block for a run.

    Sections (in order):
      A) Header
      B) Training (champion program)
      C) Rules used (champion program order)
      D) Primes preserved list (if small range)
      E) Generalization battery
      F) Learned rules (top frequency)
      G) Evaluation (per test range)
      H) Error (if any)

    This function NEVER returns only a header. If inputs are None, it attempts
    to load from disk artifacts. If data is still missing, prints N/A sections.
    """
    # Load from disk if inputs are None
    if champion_program is None:
        champion_program = _safe_load_json(run_dir / "champion_program.json")
        if champion_program is None:
            champion_program = _safe_load_json(run_dir / "best_training_program.json")
    if battery_report is None:
        battery_report = _safe_load_json(run_dir / "battery_report.json")
    if eval_report is None:
        eval_report = _safe_load_json(run_dir / "eval_report.json")
    if rules_report is None:
        rules_report = _safe_load_json(run_dir / "rules_report.json")

    lines: List[str] = []

    # DEBUG output to help diagnose data flow
    debug_flags = {
        "train": train_result is not None,
        "rules": rules_report is not None,
        "champion": champion_program is not None,
        "battery": battery_report is not None,
        "eval": eval_report is not None,
        "err": error is not None,
    }
    debug_str = ", ".join(f"{k}={v}" for k, v in debug_flags.items())
    lines.append(f"DEBUG summary inputs: {debug_str}")
    lines.append("")

    # A) Header -----------------------------------------------------------
    strategy = config.get("agent_name") or config.get("strategy") or "N/A"
    games = config.get("episodes", "N/A")

    lines.append(f"Run directory: {run_dir}")
    lines.append(f"Strategy: {strategy}")
    lines.append(f"Training range: {_get_train_range_string(config)}")
    lines.append(f"Games: {games}")
    lines.append("")

    # B) Training (champion program) -------------------------------------
    metrics = _extract_training_metrics(train_result, champion_program)
    lines.append("Training (champion program):")
    
    steps = metrics.get("steps")
    total_primes = metrics.get("total_primes")
    primes_preserved = metrics.get("primes_preserved") or metrics.get("primes_remaining_count")
    primes_removed = metrics.get("primes_removed_count", 0)
    comps_remaining = metrics.get("composites_remaining_count", 0)
    total_composites = metrics.get("total_composites")
    terminated_reason = metrics.get("terminated_reason", "unknown")
    
    # Success means zero mistakes, zero composites left
    try:
        success = int(primes_removed or 0) == 0 and int(comps_remaining or 0) == 0
        success_str = "YES" if success else "NO"
        # Fallback: if success but terminated_reason is unknown, display "success"
        if success and terminated_reason == "unknown":
            terminated_reason = "success"
    except Exception:
        success_str = "N/A"
    
    lines.append(f"- Success: {success_str}")
    lines.append(f"- Termination reason: {terminated_reason}")
    lines.append(f"- Steps: {_fmt_int_or_na(steps)}")
    lines.append(f"- Primes on board: {_fmt_int_or_na(total_primes)}")
    lines.append(f"- Primes preserved ('found'): {_fmt_int_or_na(primes_preserved)}")
    lines.append(f"- Primes removed (mistakes): {_fmt_int_or_na(primes_removed)}")
    lines.append(f"- Composites remaining: {_fmt_int_or_na(comps_remaining)}")
    
    if primes_removed and isinstance(metrics.get("primes_removed_list"), list):
        removed_list = metrics["primes_removed_list"]
        if removed_list:
            capped = _cap_list(removed_list, head=30, tail=10)
            lines.append(f"  Mistaken primes: {capped}")
    
    # Filter compression metrics if present
    filter_metrics = metrics.get("filter_metrics")
    if isinstance(filter_metrics, dict):
        lines.append("")
        lines.append("Filter effect:")
        remaining_after = filter_metrics.get("remaining_after_filter", 0)
        board_before = filter_metrics.get("board_size_before", 1)
        compression_pct = (remaining_after / board_before * 100) if board_before > 0 else 0.0
        lines.append(f"- Remaining after filter: {remaining_after} / {board_before} ({compression_pct:.1f}%)")
        
        primes_after = filter_metrics.get("primes_after", 0)
        primes_before = filter_metrics.get("primes_before", 0)
        prime_retention_pct = (primes_after / primes_before * 100) if primes_before > 0 else 0.0
        lines.append(f"- Prime retention: {primes_after} / {primes_before} ({prime_retention_pct:.1f}%)")
        
        comps_before = filter_metrics.get("composites_before", 0)
        comps_after = filter_metrics.get("composites_after", 0)
        comps_removed = comps_before - comps_after
        composite_reduction_pct = (comps_removed / comps_before * 100) if comps_before > 0 else 0.0
        lines.append(f"- Composite reduction: {comps_removed} / {comps_before} ({composite_reduction_pct:.1f}%)")
    
    lines.append("")

    # C) Rules used (champion program order) -----------------------------
    lines.append("Rules used (champion program order):")
    rules_list = None
    if champion_program and isinstance(champion_program.get("rules"), list):
        rules_list = champion_program["rules"]
    if not rules_list:
        lines.append("N/A (champion program missing)")
    else:
        max_rules = 60
        for idx, rule_dict in enumerate(rules_list[:max_rules], start=1):
            lines.append(f"{idx}) {_rule_human(rule_dict)}")
        total_rules = len(rules_list)
        if total_rules > max_rules:
            lines.append(f"... ({total_rules} total rules)")
    lines.append("")

    # D) Primes preserved list -------------------------------------------
    lines.append("Primes preserved (sorted):")
    primes_preserved_list = None
    if champion_program:
        primes_preserved_list = champion_program.get("primes_preserved_list")
    if isinstance(primes_preserved_list, list) and primes_preserved_list:
        capped = _cap_list(sorted(primes_preserved_list), head=30, tail=10)
        lines.append(f"{capped}")
    else:
        lines.append("N/A")
    lines.append("")

    # E) Generalization battery -----------------------------------------
    lines.append("Generalization battery:")
    if not battery_report:
        lines.append("Battery: N/A")
    else:
        replay_mode = battery_report.get("replay_mode", "fixed")
        # Map internal mode to layman terminology
        replay_display = "auto_factor_cleanup" if replay_mode == "sieve_expand" else replay_mode
        lines.append(f"Replay mode: {replay_display}")
        
        # Show factor expansion info if present
        if replay_mode == "sieve_expand":
            sieve_limit = battery_report.get("sieve_expand_limit")
            num_primes = battery_report.get("num_primes_applied")
            prefix_len = battery_report.get("prefix_length")
            if sieve_limit is not None and num_primes is not None:
                lines.append(f"Factor checks expanded to sqrt(max_n) = {sieve_limit}")
                lines.append(f"Total factor checks applied: {num_primes} (prefix length: {prefix_len})")
        
        lines.append("")
        
        ranges = battery_report.get("ranges", {}) or {}
        ordered_names = ["battery_small", "battery_medium", "battery_shifted", "battery_bigwin"]
        for name in ordered_names:
            m = ranges.get(name)
            if not isinstance(m, dict):
                lines.append(f"- {name}: N/A")
                continue
            steps_r = _fmt_int_or_na(m.get("steps"))
            primes_removed_r = _fmt_int_or_na(m.get("primes_removed_count"))
            comps_remaining_r = _fmt_int_or_na(m.get("composites_remaining_count"))
            # Per-range pass condition.
            passed = (
                int(m.get("primes_removed_count", 0)) == 0
                and int(m.get("composites_remaining_count", 0)) == 0
            )
            status = "PASS" if passed else "FAIL"
            terminated_reason = m.get("terminated_reason", "unknown")
            lines.append(
                f"- {name}: {status} "
                f"(steps={steps_r}, primes_removed={primes_removed_r}, "
                f"composites_remaining={comps_remaining_r}, reason={terminated_reason})"
            )
    lines.append("")

    # F) Learned rules (top frequency) -----------------------------------
    lines.append("Learned rules (top frequency during training):")
    rules_stats = None
    if rules_report and isinstance(rules_report.get("rules"), dict):
        rules_stats = rules_report["rules"]
    if not rules_stats:
        lines.append("Top rules: N/A")
    else:
        # Sort by count descending, then avg removed descending.
        ranked = sorted(
            rules_stats.values(),
            key=lambda v: (
                -int(v.get("count", 0)),
                -float(v.get("avg_composites_removed", 0.0)),
            ),
        )
        for entry in ranked[:10]:
            rule_dict = entry.get("rule", {}) or {}
            count = entry.get("count", 0)
            avg_removed = entry.get("avg_composites_removed", 0.0)
            lines.append(
                f"- {_rule_human(rule_dict)}  (used {int(count)}, avg removed {float(avg_removed):.2f})"
            )
    lines.append("")

    # G) Evaluation (per test range) -------------------------------------
    lines.append("Evaluation (per test range):")
    if not eval_report:
        lines.append("Evaluation: N/A")
    else:
        ranges_eval = eval_report.get("ranges", []) or []
        for r in ranges_eval:
            low = r.get("low")
            high = r.get("high")
            final_info = r.get("final_info", {}) or {}
            total_primes_eval = r.get("total_primes", 0)
            steps_eval = final_info.get("steps", "N/A")
            primes_remaining_eval = final_info.get("remaining_primes", 0)
            primes_removed_eval = final_info.get("primes_removed", 0)
            comps_remaining_eval = final_info.get("remaining_composites", 0)
            terminated_reason_eval = final_info.get("terminated_reason", "unknown")
            
            # Determine PASS/FAIL
            try:
                passed = int(primes_removed_eval) == 0 and int(comps_remaining_eval) == 0
                status = "PASS" if passed else "FAIL"
            except Exception:
                status = "N/A"
            
            lines.append(
                f"Range {low}..{high}: {status} "
                f"(steps={steps_eval}, "
                f"primes_preserved={primes_remaining_eval}/{total_primes_eval}, "
                f"primes_removed={primes_removed_eval}, "
                f"composites_remaining={comps_remaining_eval}, "
                f"reason={terminated_reason_eval})"
            )
    lines.append("")

    # H) Error section ---------------------------------------------------
    if error is not None:
        lines.append("ERROR:")
        lines.append(f"- {type(error).__name__}: {str(error)}")
        tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
        tb_tail = "".join(tb_lines[-8:])
        lines.append("- Traceback (last 8 lines):")
        lines.append(tb_tail.rstrip("\n"))
        lines.append("")

    lines.append(
        "Artifacts: summary.txt, champion_program.*, best_training_program.*, "
        "battery_report.*, rules_report.*, eval_report.json"
    )
    
    summary = "\n".join(lines)
    
    # Safety check: ensure we have required sections
    if error is None and "Training (champion program):" not in summary:
        raise RuntimeError(
            "Summary missing required sections - build_summary_text failed to produce complete output"
        )
    
    return summary

