from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import __init__ as package_init
from agents import Agent, LearningSieveAgent, get_agent, get_agent_metadata
from champion import ProgramRecord, choose_champion_program
from env import EpisodeSummary, PrimeIceConfig, PrimeIceEnv
from eval_battery import evaluate_program_on_battery, write_battery_report
from programs import canonicalize_program
from reporting import build_summary_text
from rules import RuleLike, rule_from_dict, rule_to_dict, unique_rule_key
from utils.io import dump_json, dump_jsonl, dump_text, ensure_dir, make_run_dir
from utils.rng import RandomSource

__version__ = package_init.__version__



def parse_range(s: str) -> Tuple[int, int]:
    parts = s.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid range: {s}")
    low, high = int(parts[0]), int(parts[1])
    if low > high:
        raise argparse.ArgumentTypeError(f"Range low>high: {s}")
    return low, high


def run_episodes(
    agent: Agent,
    env: PrimeIceEnv,
    episodes: int,
    progress_cb=None,
) -> List[Dict[str, object]]:
    logs: List[Dict[str, object]] = []
    for ep in range(episodes):
        agent.reset_for_episode()
        obs = env.reset(env.config)
        done = False
        episode_steps: int = 0
        rules_applied: List[Dict[str, object]] = []
        last_info: Dict[str, object] = {}
        total_reward: float = 0.0
        
        # Calculate max steps based on board size
        max_steps = min(5000, max(200, env.total_blocks))
        
        while not done:
            rule: Optional[RuleLike] = agent.select_rule(obs)
            if rule is None:
                # Agent signals completion
                done = True
                last_info["terminated_reason"] = "agent_stop"
                break
            
            obs, reward, done, info = env.step(rule)
            agent.observe(info)
            episode_steps += 1
            total_reward += reward
            last_info = info
            rules_applied.append(
                {
                    "rule": rule_to_dict(rule),
                    "step": episode_steps,
                    "reward": reward,
                    "info": info,
                }
            )
            if episode_steps >= max_steps:
                done = True
                last_info["terminated_reason"] = "max_steps"
                break
        terminated_reason = last_info.get("terminated_reason", "unknown")
        episode_summary: EpisodeSummary = env.episode_summary(episode_steps, terminated_reason)
        log_entry: Dict[str, object] = {
            "episode": ep,
            "steps": episode_steps,
            "total_reward": total_reward,
            "final_info": last_info,
            "rules": rules_applied,
            "episode_summary": episode_summary,
        }
        if progress_cb is not None:
            progress_cb(ep, log_entry)
        logs.append(log_entry)
    return logs


def evaluate_on_ranges(
    agent: Agent,
    ranges: Iterable[Tuple[int, int]],
    episodes: int,
    seed: int,
) -> Dict[str, object]:
    results: Dict[str, object] = {"ranges": [], "episodes": episodes, "seed": seed}
    for low, high in ranges:
        env = PrimeIceEnv(PrimeIceConfig(low=low, high=high))
        logs = run_episodes(agent, env, episodes=1)
        final = logs[-1]["final_info"]
        results["ranges"].append(
            {
                "low": low,
                "high": high,
                "final_info": final,
                "total_primes": env.total_primes,
                "total_composites": env.total_composites,
            }
        )
    return results


def _compute_rules_report(train_logs: List[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate statistics about rules used and their effect."""
    rule_stats: Dict[str, Dict[str, object]] = {}
    best_episode: Dict[str, object] | None = None

    for ep in train_logs:
        ep_summary: EpisodeSummary = ep.get("episode_summary")  # type: ignore[assignment]
        rules = ep.get("rules", [])
        for step_entry in rules:
            rule_dict = step_entry.get("rule", {})
            info = step_entry.get("info", {})
            key = unique_rule_key(rule_from_dict(rule_dict))
            stats = rule_stats.setdefault(
                key,
                {
                    "rule": rule_dict,
                    "count": 0,
                    "total_composites_removed": 0,
                },
            )
            stats["count"] = int(stats["count"]) + 1
            stats["total_composites_removed"] = int(stats["total_composites_removed"]) + int(
                info.get("composites_removed_this_step", info.get("composites_removed", 0))
            )

        if ep_summary is not None:
            if best_episode is None:
                best_episode = {"summary": ep_summary, "rules": rules}
            else:
                current = best_episode["summary"]  # type: ignore[index]
                # Prefer fewer primes removed, then fewer composites remaining, then fewer steps.
                if (
                    ep_summary.primes_removed_count < current.primes_removed_count
                    or (
                        ep_summary.primes_removed_count == current.primes_removed_count
                        and ep_summary.composites_remaining_count < current.composites_remaining_count
                    )
                    or (
                        ep_summary.primes_removed_count == current.primes_removed_count
                        and ep_summary.composites_remaining_count == current.composites_remaining_count
                        and ep_summary.steps < current.steps
                    )
                ):
                    best_episode = {"summary": ep_summary, "rules": rules}

    for key, stats in rule_stats.items():
        count = max(int(stats["count"]), 1)
        total_removed = int(stats["total_composites_removed"])
        stats["avg_composites_removed"] = total_removed / float(count)

    return {"rules": rule_stats, "best_episode": best_episode}


def _write_rules_report(
    path: Path,
    train_logs: List[Dict[str, object]],
    learning_policy: Dict[str, object] | None,
) -> Dict[str, object]:
    report = _compute_rules_report(train_logs)
    lines: List[str] = []
    lines.append("Prime Ice — Rules Report")
    lines.append("")

    rules = report["rules"]
    sorted_rules = sorted(
        rules.values(),
        key=lambda v: (-float(v.get("avg_composites_removed", 0.0)), -int(v.get("count", 0))),
    )

    def _rule_to_human(rule_dict: Dict[str, object]) -> str:
        r_type = rule_dict.get("type")
        if r_type == "divisibility":
            d = int(rule_dict.get("d", 0))
            keep_self = bool(rule_dict.get("keep_self", True))
            if keep_self:
                return f"Remove multiples of {d} (keep {d})"
            return f"Remove multiples of {d}"
        if r_type == "mod_class":
            m = int(rule_dict.get("m", 0))
            banned = rule_dict.get("banned_residues", [])
            resid_str = ",".join(str(int(x)) for x in banned)  # type: ignore[arg-type]
            return f"Keep only residues {{{resid_str}}} mod {m}"
        return str(rule_dict)

    lines.append("Top rules by frequency and composites removed:")
    for entry in sorted_rules[:20]:
        rule_dict = entry["rule"]  # type: ignore[index]
        count = int(entry.get("count", 0))
        avg_removed = float(entry.get("avg_composites_removed", 0.0))
        lines.append(
            f"- {_rule_to_human(rule_dict)}  (used {count} times, avg removed {avg_removed:.2f})"
        )

    best_episode = report.get("best_episode")
    if best_episode is not None:
        ep_summary: EpisodeSummary = best_episode["summary"]  # type: ignore[index]
        ep_rules: List[Dict[str, object]] = best_episode["rules"]  # type: ignore[index]
        lines.append("")
        lines.append("Champion episode (best run):")
        lines.append(f"- Steps: {ep_summary.steps}")
        lines.append(f"- Primes removed: {ep_summary.primes_removed_count}")
        lines.append(f"- Primes remaining: {ep_summary.primes_remaining_count}")
        lines.append(f"- Composites remaining: {ep_summary.composites_remaining_count}")
        lines.append("Rules used in order:")
        for idx, step_entry in enumerate(ep_rules, start=1):
            rule_dict = step_entry.get("rule", {})
            lines.append(f"  {idx}) {_rule_to_human(rule_dict)}")

    # Automatically surface a small-prime sieve pattern when present.
    small_primes = [3, 5, 7, 11, 13]
    available_small: List[int] = []
    for p in small_primes:
        key = f"type=divisibility|d={p}|keep_self=True"
        if key in rules:
            available_small.append(p)
    if available_small:
        lines.append("")
        lines.append("Discovered small-prime sieve pattern:")
        lines.append("  Combining these rules approximates a classical sieve")
        lines.append("  over odd numbers, rapidly stripping composites while")
        lines.append("  leaving primes on the board:")
        for p in available_small:
            lines.append(f"    - Remove multiples of {p} (keep {p})")

    if learning_policy is not None:
        q_values: Dict[str, float] = learning_policy.get("q_values", {})  # type: ignore[assignment]
        lines.append("")
        lines.append("Learning agent policy snapshot (top 20 divisors):")
        sorted_q = sorted(
            ((int(k), float(v)) for k, v in q_values.items()),
            key=lambda kv: (-kv[1], kv[0]),
        )
        counts: Dict[str, int] = learning_policy.get("counts", {})  # type: ignore[assignment]
        for d, q in sorted_q[:20]:
            uses = int(counts.get(str(d), 0))
            avg_removed = 0.0
            rule_key = f"type=divisibility|d={d}|keep_self=True"
            rule_info = rules.get(rule_key)
            if rule_info is not None:
                c = int(rule_info.get("count", 1))
                total = int(rule_info.get("total_composites_removed", 0))
                if c > 0:
                    avg_removed = total / float(c)
            cost = 1.0 + 0.02 * d
            lines.append(
                f"  d={d}: Q={q:.3f}, uses={uses}, avg_removed={avg_removed:.2f}, cost={cost:.2f}"
            )

        recommended = [str(d) for d, _ in sorted_q[:20]]
        lines.append("")
        lines.append("Recommended order (top 20): " + ", ".join(recommended))

    dump_text(path, "\n".join(lines))

    rules_report_json = {
        "rules": rules,
        "best_episode": best_episode,
        "learning_policy": learning_policy,
    }
    dump_json(path.with_suffix(".json"), rules_report_json)
    return report


def command_run(args: argparse.Namespace) -> None:
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    rng = RandomSource(seed=seed)
    train_config = PrimeIceConfig(low=args.train_low, high=args.train_high)
    env = PrimeIceEnv(train_config)

    agent_id, agent_name = get_agent_metadata(args.strategy)
    agent = get_agent(
        agent_id=agent_id,
        rng=rng,
        train_high=args.train_high,
        reset_policy=args.reset_policy,
    )

    out_base = ensure_dir(args.out)
    run_dir = make_run_dir(out_base)

    config = {
        "agent_id": agent_id,
        "agent_name": agent_name,
        "strategy": agent_id,
        "episodes": args.episodes,
        "seed": seed,
        "train_low": args.train_low,
        "train_high": args.train_high,
        "test_ranges": args.test_ranges,
        "version": __version__,
        "reset_policy": args.reset_policy,
    }
    dump_json(run_dir / "config.json", config)

    # Placeholders for structured results passed into reporting.
    train_result: Dict[str, object] | None = None
    rules_report: Dict[str, object] | None = None
    champion_program: Dict[str, object] | None = None
    battery_report: Dict[str, object] | None = None
    eval_report: Dict[str, object] | None = None
    err: BaseException | None = None

    try:
        # Training phase.
        train_logs = run_episodes(agent, env, args.episodes)
        dump_jsonl(run_dir / "train_log.jsonl", train_logs)

        # Build per-episode program records and best_training selection.
        program_records: List[ProgramRecord] = []
        success_records: List[ProgramRecord] = []
        best_training: ProgramRecord | None = None

        for idx, ep in enumerate(train_logs):
            summary: EpisodeSummary = ep.get("episode_summary")  # type: ignore[assignment]
            rules_for_ep: List[Dict[str, object]] = ep.get("rules", [])
            train_metrics: Dict[str, object] = {
                "steps": summary.steps,
                "total_primes": summary.total_primes,
                "total_composites": summary.total_composites,
                "primes_removed_count": summary.primes_removed_count,
                "primes_removed_list": list(summary.primes_removed_list),
                "composites_remaining_count": summary.composites_remaining_count,
                "primes_remaining_count": summary.primes_remaining_count,
                "score": summary.score,
                "episode_index": idx,
            }
            rec = ProgramRecord(
                program_name=f"episode_{idx}",
                source="training_episode",
                rules=[step["rule"] for step in rules_for_ep],
                train_metrics=train_metrics,
                battery_pass=False,
                battery_summary={},
            )
            program_records.append(rec)
            if best_training is None:
                best_training = rec
            else:
                a = rec.train_metrics
                b = best_training.train_metrics
                a_key = (
                    int(a.get("primes_removed_count", 0)),
                    int(a.get("composites_remaining_count", 0)),
                    int(a.get("steps", 0)),
                    len(rec.rules),
                )
                b_key = (
                    int(b.get("primes_removed_count", 0)),
                    int(b.get("composites_remaining_count", 0)),
                    int(b.get("steps", 0)),
                    len(best_training.rules),
                )
                if a_key < b_key:
                    best_training = rec

            if summary.primes_removed_count == 0 and summary.composites_remaining_count == 0:
                success_records.append(rec)

        if best_training is None and program_records:
            best_training = program_records[0]

        if best_training is not None:
            train_result = {
                "best_episode": {
                    "summary": best_training.train_metrics,
                    "rules": best_training.rules,
                }
            }

        # Evaluation on configured test ranges.
        test_ranges = [parse_range(r) for r in args.test_ranges]
        eval_report = evaluate_on_ranges(agent, test_ranges, episodes=1, seed=seed)
        dump_json(run_dir / "eval_report.json", eval_report)

        # Learning policy snapshot if applicable.
        learning_policy: Dict[str, object] | None = None
        if isinstance(agent, LearningSieveAgent):
            agent.on_run_end()
            learning_policy = agent.export_policy()
            dump_json(run_dir / "policy_snapshot.json", learning_policy)

        # Rules report (text + JSON).
        rules_report = _write_rules_report(run_dir / "rules_report.txt", train_logs, learning_policy)

        # Per-run program libraries for successful programs.
        programs_success_path = run_dir / "programs_success.jsonl"
        programs_unique_path = run_dir / "programs_unique.jsonl"
        seen_ordered: set[str] = set()
        best_by_unordered: Dict[str, Dict[str, object]] = {}

        with open(programs_success_path, "w", encoding="utf-8") as f_all:
            for rec in success_records:
                canon = canonicalize_program(rec.rules)
                battery = evaluate_program_on_battery(canon.rules_canonical)
                rec.battery_pass = bool(battery.get("battery_pass", False))
                rec.battery_summary = battery

                record_dict: Dict[str, object] = {
                    "program_name": rec.program_name,
                    "source": rec.source,
                    "rules": canon.rules_canonical,
                    "train_metrics": rec.train_metrics,
                    "battery_pass": rec.battery_pass,
                    "battery_summary": rec.battery_summary,
                    "signature_ordered": canon.signature_ordered,
                    "signature_unordered": canon.signature_unordered,
                    "signature_stagewise": canon.signature_stagewise,
                }

                if canon.signature_ordered in seen_ordered:
                    continue
                seen_ordered.add(canon.signature_ordered)

                f_all.write(json.dumps(record_dict) + "\n")

                existing = best_by_unordered.get(canon.signature_unordered)
                def score_for_unique(d: Dict[str, object]) -> tuple[int, int]:
                    b = d.get("battery_summary", {}) or {}
                    return (
                        int(b.get("battery_total_steps", 0)),
                        int(d.get("train_metrics", {}).get("steps", 0)),  # type: ignore[union-attr]
                    )

                if existing is None or score_for_unique(record_dict) < score_for_unique(existing):
                    best_by_unordered[canon.signature_unordered] = record_dict

        with open(programs_unique_path, "w", encoding="utf-8") as f_unique:
            for record_dict in best_by_unordered.values():
                f_unique.write(json.dumps(record_dict) + "\n")

        # Select champion program by generalization-first policy.
        champion = choose_champion_program(program_records)
        canon_champion = canonicalize_program(champion.rules)

        # Ensure champion has battery summary and write battery reports.
        if not champion.battery_summary:
            champion.battery_summary = evaluate_program_on_battery(canon_champion.rules_canonical)
            champion.battery_pass = bool(champion.battery_summary.get("battery_pass", False))
        battery_report = champion.battery_summary
        write_battery_report(
            run_dir / "battery_report.json",
            run_dir / "battery_report.txt",
            champion.battery_summary,
        )

        # Replay champion once on training range to compute replay_metrics.
        replay_env = PrimeIceEnv(PrimeIceConfig(low=args.train_low, high=args.train_high))
        replay_env.reset()
        steps_replay = 0
        done = False
        terminated_reason = "unknown"
        
        # Track filter compression if first rule is a filter
        filter_metrics: Dict[str, object] | None = None
        if canon_champion.rules_canonical:
            first_rule_dict = canon_champion.rules_canonical[0]
            first_rule_type = first_rule_dict.get("type")
            if first_rule_type in ("keep_residues", "mod_class"):
                # Capture before-filter state
                board_size_before = replay_env.remaining_count
                primes_before = replay_env.remaining_primes_count
                composites_before = replay_env.remaining_composites_count
                
                # Apply first rule (the filter)
                first_rule = rule_from_dict(first_rule_dict)
                _, _, done, info = replay_env.step(first_rule)
                steps_replay += 1
                terminated_reason = info.get("terminated_reason", "unknown")
                
                # Capture after-filter state
                remaining_after = replay_env.remaining_count
                primes_after = replay_env.remaining_primes_count
                composites_after = replay_env.remaining_composites_count
                
                # Compute compression metrics
                compression_ratio = remaining_after / board_size_before if board_size_before > 0 else 0.0
                prime_retention = primes_after / primes_before if primes_before > 0 else 1.0
                composite_reduction = (composites_before - composites_after) / composites_before if composites_before > 0 else 0.0
                
                filter_metrics = {
                    "board_size_before": board_size_before,
                    "remaining_after_filter": remaining_after,
                    "primes_before": primes_before,
                    "primes_after": primes_after,
                    "composites_before": composites_before,
                    "composites_after": composites_after,
                    "compression_ratio": compression_ratio,
                    "prime_retention": prime_retention,
                    "composite_reduction": composite_reduction,
                }
                
                # Continue with remaining rules
                for rdict in canon_champion.rules_canonical[1:]:
                    if done:
                        break
                    rule = rule_from_dict(rdict)
                    _, _, done, info = replay_env.step(rule)
                    steps_replay += 1
                    terminated_reason = info.get("terminated_reason", "unknown")
            else:
                # No filter, replay all rules normally
                for rdict in canon_champion.rules_canonical:
                    if done:
                        break
                    rule = rule_from_dict(rdict)
                    _, _, done, info = replay_env.step(rule)
                    steps_replay += 1
                    terminated_reason = info.get("terminated_reason", "unknown")
        
        if not done:
            terminated_reason = "rules_exhausted"
        replay_summary = replay_env.episode_summary(steps_replay, terminated_reason)
        
        # Get preserved primes list from environment
        primes_preserved_list = sorted(
            n for n in replay_env.remaining_numbers if replay_env._is_prime.get(n, False)
        )
        
        replay_metrics: Dict[str, object] = {
            "steps": replay_summary.steps,
            "total_primes": replay_summary.total_primes,
            "primes_preserved": replay_summary.primes_remaining_count,
            "primes_removed_count": replay_summary.primes_removed_count,
            "primes_removed_list": list(replay_summary.primes_removed_list),
            "total_composites": replay_summary.total_composites,
            "composites_remaining_count": replay_summary.composites_remaining_count,
            "score": replay_summary.score,
            "terminated_reason": replay_summary.terminated_reason,
            "success": replay_summary.success,
        }
        if filter_metrics:
            replay_metrics["filter_metrics"] = filter_metrics

        # Helper for human-readable rules.
        def _rule_to_human(rule_dict: Dict[str, object]) -> str:
            r_type = rule_dict.get("type")
            if r_type == "divisibility":
                d = int(rule_dict.get("d", 0))
                keep_self = bool(rule_dict.get("keep_self", True))
                if keep_self:
                    return f"Remove multiples of {d} (keep {d})"
                return f"Remove multiples of {d}"
            if r_type == "mod_class":
                m = int(rule_dict.get("m", 0))
                banned = rule_dict.get("banned_residues", [])
                resid_str = ",".join(str(int(x)) for x in banned)  # type: ignore[arg-type]
                return f"Keep only residues {{{resid_str}}} mod {m}"
            if r_type == "keep_residues":
                m = int(rule_dict.get("m", 0))
                allowed = rule_dict.get("allowed_residues", [])
                resid_str = ",".join(str(int(x)) for x in allowed)  # type: ignore[arg-type]
                return f"Keep residues {{{resid_str}}} mod {m}"
            return str(rule_dict)

        # Write best_training_program.*
        if best_training is not None:
            canon_best = canonicalize_program(best_training.rules)
            best_dict = {
                "program_name": best_training.program_name,
                "source": best_training.source,
                "rules": canon_best.rules_canonical,
                "train_metrics": best_training.train_metrics,
                "battery_pass": best_training.battery_pass,
                "battery_summary": best_training.battery_summary,
                "signature_ordered": canon_best.signature_ordered,
                "signature_unordered": canon_best.signature_unordered,
                "signature_stagewise": canon_best.signature_stagewise,
            }
            dump_json(run_dir / "best_training_program.json", best_dict)
            lines_best: List[str] = []
            lines_best.append(f"Best_training Program: {best_training.program_name}")
            lines_best.append(f"Source: {best_training.source}")
            lines_best.append(f"Signature (stagewise): {canon_best.signature_stagewise}")
            lines_best.append(f"Battery PASS: {best_training.battery_pass}")
            lines_best.append("Rules (in order):")
            for idx, rule_dict in enumerate(canon_best.rules_canonical, start=1):
                lines_best.append(f"  {idx}) {_rule_to_human(rule_dict)}")
            dump_text(run_dir / "best_training_program.txt", "\n".join(lines_best))

        # Write champion_program.* including replay_metrics and final_metrics.
        champion_program = {
            "program_name": champion.program_name,
            "source": champion.source,
            "rules": canon_champion.rules_canonical,
            "train_metrics": champion.train_metrics,
            "final_metrics": champion.train_metrics,
            "replay_metrics": replay_metrics,
            "primes_preserved_list": primes_preserved_list,
            "battery_pass": champion.battery_pass,
            "battery_summary": champion.battery_summary,
            "signature_ordered": canon_champion.signature_ordered,
            "signature_unordered": canon_champion.signature_unordered,
            "signature_stagewise": canon_champion.signature_stagewise,
        }
        dump_json(run_dir / "champion_program.json", champion_program)
        lines_champ: List[str] = []
        lines_champ.append(f"Champion Program: {champion.program_name}")
        lines_champ.append(f"Source: {champion.source}")
        lines_champ.append(f"Signature (stagewise): {canon_champion.signature_stagewise}")
        lines_champ.append(f"Battery PASS: {champion.battery_pass}")
        lines_champ.append("Rules (in order):")
        for idx, rule_dict in enumerate(canon_champion.rules_canonical, start=1):
            lines_champ.append(f"  {idx}) {_rule_to_human(rule_dict)}")
        dump_text(run_dir / "champion_program.txt", "\n".join(lines_champ))

    except BaseException as e:  # catch broad to ensure summary writing
        err = e
    finally:
        summary_text = build_summary_text(
            run_dir=run_dir,
            config=config,
            train_result=train_result,
            rules_report=rules_report,
            champion_program=champion_program,
            battery_report=battery_report,
            eval_report=eval_report,
            error=err,
        )
        dump_text(run_dir / "summary.txt", summary_text)
        print(summary_text)
        if err is not None:
            print("Run failed; see ERROR section in summary.txt.")


def command_eval(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"No config.json found in {run_dir}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    raw_agent_id = config.get("agent_id") or config.get("strategy") or "learning_sieve"
    agent_id, _agent_name = get_agent_metadata(str(raw_agent_id))
    seed = int(config["seed"])

    rng = RandomSource(seed=seed)
    agent = get_agent(
        agent_id=agent_id,
        rng=rng,
        train_high=int(config["train_high"]),
        reset_policy=args.reset_policy,
    )

    ranges = [parse_range(r) for r in args.test_ranges]
    eval_report = evaluate_on_ranges(agent, ranges, episodes=1, seed=seed)
    dump_json(run_dir / "eval_report_eval.json", eval_report)
    print(json.dumps(eval_report, indent=2))


def command_show(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    summary_path = run_dir / "summary.txt"
    rules_path = run_dir / "rules_report.txt"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            print(f.read())
    if rules_path.exists():
        print("\nRules report:")
        with open(rules_path, "r", encoding="utf-8") as f:
            print(f.read())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prime_ice", description="Prime Ice rule-discovery game")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Train and evaluate an agent")
    p_run.add_argument("--train-low", type=int, default=3)
    p_run.add_argument("--train-high", type=int, default=20003)
    p_run.add_argument(
        "--test-ranges",
        nargs="+",
        default=["3:20003", "3:200030"],
    )
    p_run.add_argument(
        "--strategy",
        type=str,
        choices=["heuristic_sieve", "learning_sieve", "random_rules"],
        default="learning_sieve",
    )
    p_run.add_argument("--episodes", type=int, default=200)
    p_run.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed; if omitted, a random seed is generated.",
    )
    p_run.add_argument(
        "--reset-policy",
        action="store_true",
        help="Reset the learning sieve policy instead of continuing from previous runs.",
    )
    p_run.add_argument("--out", type=str, default="runs")
    p_run.set_defaults(func=command_run)

    p_eval = sub.add_parser("eval", help="Evaluate a saved policy on new ranges")
    p_eval.add_argument("run_dir", type=str)
    p_eval.add_argument(
        "--test-ranges",
        nargs="+",
        required=True,
        help='Ranges like "3:200003"',
    )
    p_eval.add_argument(
        "--reset-policy",
        action="store_true",
        help="Reset learning policy instead of using persisted one.",
    )
    p_eval.set_defaults(func=command_eval)

    p_show = sub.add_parser("show", help="Show rules and summary for a run")
    p_show.add_argument("run_dir", type=str)
    p_show.set_defaults(func=command_show)

    return parser


def app(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def main() -> None:
    """Entry point for the CLI application."""
    app()
