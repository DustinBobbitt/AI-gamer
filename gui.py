from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional, Dict, List
from datetime import datetime

import __init__ as package_init
from agents import LearningSieveAgent, get_agent, get_agent_metadata
from champion import ProgramRecord, choose_champion_program
from cli import evaluate_on_ranges, parse_range, run_episodes, _write_rules_report
from env import EpisodeSummary, PrimeIceConfig, PrimeIceEnv
from eval_battery import evaluate_program_on_battery, write_battery_report
from programs import canonicalize_program
from reporting import build_summary_text
from rules import rule_from_dict
from utils.io import dump_json, dump_jsonl, dump_text, ensure_dir, make_run_dir
from utils.rng import RandomSource

__version__ = package_init.__version__


class GameLearningApp(tk.Tk):
    """Main GUI application window for Self-Improving Game-Learning AI."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Self-Improving Game-Learning AI – Meta-Learning System")
        self.geometry("1000x700")

        self._worker_thread: Optional[threading.Thread] = None
        self._results_window: Optional["ResultViewerWindow"] = None
        self._last_run_dir: Optional[Path] = None

        self._build_widgets()

    # UI construction -----------------------------------------------------
    def _build_widgets(self) -> None:
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Configuration area
        cfg_frame = ttk.LabelFrame(main, text="Meta-Learning Configuration")
        cfg_frame.pack(fill=tk.X, pady=5)

        # Total phases
        ttk.Label(cfg_frame, text="Training Phases:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        self.phases_var = tk.StringVar(value="10")
        ttk.Entry(cfg_frame, textvariable=self.phases_var, width=10).grid(
            row=0, column=1, sticky=tk.W, padx=2, pady=2
        )
        ttk.Label(cfg_frame, text="(Acquisition-Consolidation cycles)").grid(
            row=0, column=2, sticky=tk.W, padx=5, pady=2
        )

        # Games per phase
        ttk.Label(cfg_frame, text="Games per Phase:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        self.games_var = tk.StringVar(value="10")
        ttk.Entry(cfg_frame, textvariable=self.games_var, width=10).grid(
            row=1, column=1, sticky=tk.W, padx=2, pady=2
        )

        # Episodes per game
        ttk.Label(cfg_frame, text="Episodes per Game:").grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
        self.episodes_var = tk.StringVar(value="100")
        ttk.Entry(cfg_frame, textvariable=self.episodes_var, width=10).grid(
            row=2, column=1, sticky=tk.W, padx=2, pady=2
        )

        # Consolidation frequency
        ttk.Label(cfg_frame, text="Consolidation Frequency:").grid(row=3, column=0, sticky=tk.W, padx=2, pady=2)
        self.consolidation_freq_var = tk.StringVar(value="5")
        ttk.Entry(cfg_frame, textvariable=self.consolidation_freq_var, width=10).grid(
            row=3, column=1, sticky=tk.W, padx=2, pady=2
        )
        ttk.Label(cfg_frame, text="(Every N phases)").grid(
            row=3, column=2, sticky=tk.W, padx=5, pady=2
        )

        # Target compression
        ttk.Label(cfg_frame, text="Target Compression:").grid(row=4, column=0, sticky=tk.W, padx=2, pady=2)
        self.compression_var = tk.StringVar(value="0.3")
        ttk.Entry(cfg_frame, textvariable=self.compression_var, width=10).grid(
            row=4, column=1, sticky=tk.W, padx=2, pady=2
        )
        ttk.Label(cfg_frame, text="(0.3 = 30% reduction)").grid(
            row=4, column=2, sticky=tk.W, padx=5, pady=2
        )

        # Output directory
        ttk.Label(cfg_frame, text="Output Folder:").grid(row=5, column=0, sticky=tk.E, padx=2, pady=2)
        self.out_var = tk.StringVar(value="meta_learning_runs")
        out_entry = ttk.Entry(cfg_frame, textvariable=self.out_var, width=40)
        out_entry.grid(row=5, column=1, columnspan=4, sticky=tk.W, padx=2, pady=2)
        ttk.Button(cfg_frame, text="Browse...", command=self._browse_out_dir).grid(
            row=5, column=5, sticky=tk.W, padx=2, pady=2
        )

        # Control buttons and status
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=5)

        self.run_button = ttk.Button(btn_frame, text="▶ Start Meta-Learning", command=self._on_run_clicked)
        self.run_button.pack(side=tk.LEFT)

        ttk.Button(
            btn_frame,
            text="📊 View Results",
            command=self._open_results_window,
        ).pack(side=tk.LEFT, padx=10)

        ttk.Button(
            btn_frame,
            text="Open Run Folder",
            command=self._open_run_folder,
        ).pack(side=tk.LEFT, padx=10)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(btn_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=10)

        self.progress = ttk.Progressbar(
            btn_frame, orient="horizontal", length=250, mode="determinate"
        )
        self.progress.pack(side=tk.LEFT, padx=10)

        # Log / output area
        log_frame = ttk.LabelFrame(main, text="Run Summary (live)")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Footer
        footer = ttk.Frame(main)
        footer.pack(fill=tk.X, pady=5)
        ttk.Label(footer, text=f"Game-Learning AI v{__version__} – Learning to Learn").pack(side=tk.RIGHT)

    # Basic helpers -------------------------------------------------------
    def _browse_out_dir(self) -> None:
        selected = filedialog.askdirectory(title="Select output base directory")
        if selected:
            self.out_var.set(selected)

    def _append_log(self, text: str) -> None:
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def _open_run_folder(self) -> None:
        """Open the last run folder in the system file explorer."""
        import os

        if not self._last_run_dir:
            messagebox.showinfo("Prime Ice", "No run has been completed yet.")
            return
        try:
            os.startfile(self._last_run_dir)  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showerror("Prime Ice", f"Could not open folder:\n{exc}")

    # Results viewer handling --------------------------------------------
    def _open_results_window(self, initial_run_dir: Optional[Path] = None) -> None:
        if self._results_window is None or not self._results_window.winfo_exists():
            self._results_window = ResultViewerWindow(self, initial_run_dir)
        else:
            self._results_window.lift()
            if initial_run_dir is not None:
                self._results_window.load_run(initial_run_dir)

    # Run control ---------------------------------------------------------
    def _on_run_clicked(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showinfo("Game-Learning AI", "A training run is already in progress.")
            return
        try:
            phases = int(self.phases_var.get())
            games_per_phase = int(self.games_var.get())
            episodes_per_game = int(self.episodes_var.get())
            consolidation_freq = int(self.consolidation_freq_var.get())
            compression = float(self.compression_var.get())
            out_base = self.out_var.get()
        except ValueError as exc:
            messagebox.showerror("Invalid input", f"Please check numeric fields.\n\n{exc}")
            return

        self.run_button.config(state=tk.DISABLED)
        self.progress["value"] = 0
        self.progress["maximum"] = phases
        self.status_var.set("Initializing meta-learning...")
        self.log_text.delete("1.0", tk.END)
        self._append_log("=== Self-Improving Game-Learning AI ===")
        self._append_log(f"Phases: {phases}")
        self._append_log(f"Games per phase: {games_per_phase}")
        self._append_log(f"Episodes per game: {episodes_per_game}")
        self._append_log(f"Consolidation every: {consolidation_freq} phases")
        self._append_log(f"Target compression: {compression*100:.0f}%")
        self._append_log("")

        def worker() -> None:
            try:
                self._run_meta_learning(
                    phases=phases,
                    games_per_phase=games_per_phase,
                    episodes_per_game=episodes_per_game,
                    consolidation_freq=consolidation_freq,
                    compression=compression,
                    out_base=out_base
                )
            except Exception as exc:
                import traceback
                error_msg = f"{str(exc)}\n\n{traceback.format_exc()}"
                self.after(
                    0,
                    lambda: messagebox.showerror("Meta-Learning error", error_msg),
                )
            finally:
                self.after(
                    0,
                    lambda: (
                        self.run_button.config(state=tk.NORMAL),
                        self.status_var.set("Ready."),
                    ),
                )

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _run_meta_learning(
        self,
        phases: int,
        games_per_phase: int,
        episodes_per_game: int,
        consolidation_freq: int,
        compression: float,
        out_base: str,
    ) -> None:
        from pathlib import Path
        from meta_learning_loop import MetaLearningLoop, TrainingConfig
        
        # Create training configuration
        config = TrainingConfig(
            initial_games=games_per_phase * phases,
            episodes_per_game=episodes_per_game,
            consolidation_frequency=consolidation_freq,
            target_compression=compression,
            output_dir=Path(out_base)
        )
        
        # Initialize meta-learning loop
        loop = MetaLearningLoop(config)
        
        # Progress callback
        def progress_callback(current, total, message):
            self.after(0, lambda: self._update_progress(current, total, message))
        
        # Run meta-learning
        self._append_log("Starting meta-learning loop...")
        summary = loop.run(total_phases=phases, progress_callback=progress_callback)
        
        # Display results
        self._append_log("\n=== Training Complete ===")
        self._append_log(f"Total duration: {summary['run_duration_seconds']:.1f}s")
        self._append_log(f"Total episodes: {summary['total_episodes']}")
        self._append_log(f"Unique games seen: {summary['unique_games']}")
        self._append_log(f"Consolidations: {summary['consolidations_performed']}")
        self._append_log(f"\nSkills learned: {summary['skill_memory_stats'].get('total_skills', 0)}")
        self._append_log(f"Avg skill importance: {summary['skill_memory_stats'].get('avg_importance', 0):.3f}")
        
        # Save state
        run_dir = config.output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        loop.save_state(run_dir)
        self._last_run_dir = run_dir
        
        self._append_log(f"\nResults saved to: {run_dir}")
        self.after(0, lambda: self.status_var.set("Complete!"))

    def _update_progress(self, current: int, total: int, message: str):
        """Update progress bar and status from worker thread."""
        if total > 0:
            self.progress["value"] = current
            self.progress["maximum"] = total
        self.status_var.set(message)
        self._append_log(f"[{current}/{total}] {message}")

    def _run_prime_ice(
        self,
        strategy_label: str,
        train_low: int,
        train_high: int,
        episodes: int,
        test_ranges: List[str],
        out_base: str,
    ) -> None:
        import secrets

        # Map human-readable label to internal strategy ID and name.
        strategy_map = {
            "Factor Cleanup (Deterministic)": "heuristic_sieve",
            "Learning Factor Cleanup (Persistent)": "learning_sieve",
            "Random Factor Checks (Baseline)": "random_rules",
            "First-Cut Filter + Factor Cleanup (Fixed)": "fixed_filter_then_cleanup",
            "First-Cut Filter + Factor Cleanup (Learning)": "learning_filter_then_cleanup",
        }
        requested_id = strategy_map.get(strategy_label, "learning_sieve")
        strategy_id, strategy_name = get_agent_metadata(requested_id)

        seed = secrets.randbits(32)
        rng = RandomSource(seed=seed)
        env = PrimeIceEnv(PrimeIceConfig(low=train_low, high=train_high))
        agent = get_agent(
            agent_id=strategy_id,
            rng=rng,
            train_high=train_high,
            reset_policy=self.reset_policy_var.get(),
        )

        base_path = ensure_dir(out_base)
        run_dir = make_run_dir(base_path)

        config = {
            "agent_id": strategy_id,
            "agent_name": strategy_name,
            "strategy": strategy_id,
            "episodes": episodes,
            "seed": seed,
            "train_low": train_low,
            "train_high": train_high,
            "test_ranges": test_ranges,
            "version": __version__,
            "reset_policy": self.reset_policy_var.get(),
        }
        dump_json(run_dir / "config.json", config)

        def progress_cb(ep_idx: int, log_entry: Dict[str, object]) -> None:
            """Update GUI with per-episode progress."""

            def ui_update() -> None:
                self.progress["value"] = ep_idx + 1
                steps = log_entry.get("steps")
                final_info = log_entry.get("final_info", {})
                score = final_info.get("score")
                self.status_var.set(
                    f"Running... episode {ep_idx + 1}/{episodes}, "
                    f"steps={steps}, score={score}"
                )
                self._append_log(
                    f"Episode {ep_idx + 1}: steps={steps}, "
                    f"primes_removed={final_info.get('primes_removed')}, "
                    f"composites_remaining={final_info.get('remaining_composites')}, "
                    f"score={score}"
                )

            self.after(0, ui_update)

        # Run the entire pipeline in try/except/finally to ensure summary always written
        train_result: Optional[Dict[str, object]] = None
        rules_report: Optional[Dict[str, object]] = None
        champion_program: Optional[Dict[str, object]] = None
        battery_report: Optional[Dict[str, object]] = None
        eval_report: Optional[Dict[str, object]] = None
        err: Optional[BaseException] = None
        
        try:
            # Training
            train_logs = run_episodes(agent, env, episodes, progress_cb=progress_cb)
            dump_jsonl(run_dir / "train_log.jsonl", train_logs)

            # Build program records from training episodes
            program_records: List[ProgramRecord] = []
            for idx, ep in enumerate(train_logs):
                ep_summary: EpisodeSummary = ep.get("episode_summary")  # type: ignore
                rules_for_ep: List[Dict[str, object]] = ep.get("rules", [])
                
                train_metrics = {
                    "steps": ep_summary.steps,
                    "total_primes": ep_summary.total_primes,
                    "primes_preserved": ep_summary.primes_remaining_count,
                    "primes_removed_count": ep_summary.primes_removed_count,
                    "primes_removed_list": list(ep_summary.primes_removed_list),
                    "composites_remaining_count": ep_summary.composites_remaining_count,
                    "total_composites": ep_summary.total_composites,
                    "score": ep_summary.score,
                }
                
                rec = ProgramRecord(
                    program_name=f"episode_{idx}",
                    source="training",
                    rules=[step["rule"] for step in rules_for_ep],  # Extract just the rule dict
                    train_metrics=train_metrics,
                    battery_pass=False,
                    battery_summary={},
                )
                program_records.append(rec)

            # Select champion
            if program_records:
                champion_rec = choose_champion_program(program_records)
                canon_champion = canonicalize_program(champion_rec.rules)

                # Evaluate champion on battery
                champion_rec.battery_summary = evaluate_program_on_battery(canon_champion.rules_canonical)
                champion_rec.battery_pass = bool(champion_rec.battery_summary.get("battery_pass", False))
                battery_report = champion_rec.battery_summary
                write_battery_report(
                    run_dir / "battery_report.json",
                    run_dir / "battery_report.txt",
                    champion_rec.battery_summary,
                )

                # Replay champion on training range to get preserved primes list
                replay_env = PrimeIceEnv(PrimeIceConfig(low=train_low, high=train_high))
                replay_env.reset()
                steps_replay = 0
                done = False
                for rdict in canon_champion.rules_canonical:
                    if done:
                        break
                    rule = rule_from_dict(rdict)
                    _, _, done, _ = replay_env.step(rule)
                    steps_replay += 1
                replay_summary = replay_env.episode_summary(steps_replay)
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
                }

                champion_program = {
                    "program_name": champion_rec.program_name,
                    "source": champion_rec.source,
                    "rules": canon_champion.rules_canonical,
                    "train_metrics": champion_rec.train_metrics,
                    "final_metrics": champion_rec.train_metrics,
                    "replay_metrics": replay_metrics,
                    "primes_preserved_list": primes_preserved_list,
                    "battery_pass": champion_rec.battery_pass,
                    "battery_summary": champion_rec.battery_summary,
                    "signature_ordered": canon_champion.signature_ordered,
                    "signature_unordered": canon_champion.signature_unordered,
                    "signature_stagewise": canon_champion.signature_stagewise,
                }
                dump_json(run_dir / "champion_program.json", champion_program)

                train_result = {
                    "best_episode": {
                        "summary": champion_rec.train_metrics,
                        "rules": champion_rec.rules,
                    }
                }

            # Evaluation on test ranges
            if not test_ranges:
                default_ranges = [
                    f"{train_low}:{train_high}",
                    f"{train_low}:{train_high * 10}",
                ]
                ranges = [parse_range(r) for r in default_ranges]
            else:
                ranges = [parse_range(r) for r in test_ranges]
            eval_report = evaluate_on_ranges(agent, ranges, episodes=1, seed=seed)
            dump_json(run_dir / "eval_report.json", eval_report)

            # Learning policy snapshot if applicable
            learning_policy: Dict[str, object] | None = None
            if isinstance(agent, LearningSieveAgent):
                agent.on_run_end()
                learning_policy = agent.export_policy()
                dump_json(run_dir / "policy_snapshot.json", learning_policy)

            # Rules report
            rules_report = _write_rules_report(run_dir / "rules_report.txt", train_logs, learning_policy)

        except BaseException as e:
            err = e
        finally:
            # Always write summary.txt using unified reporting
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

            self._last_run_dir = run_dir

            # Update GUI log with summary.txt content
            def update_log() -> None:
                self.log_text.delete("1.0", tk.END)
                summary_path = run_dir / "summary.txt"
                if summary_path.exists():
                    with open(summary_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        self._append_log(content)
                else:
                    self._append_log(f"ERROR: summary.txt not found in {run_dir}")
                    self._append_log("This should never happen - please report this bug.")
                
                # Auto-open results viewer for this run.
                self._open_results_window(run_dir)

            self.after(0, update_log)
            
            if err is not None:
                def show_error() -> None:
                    messagebox.showerror("Prime Ice Error", f"Run failed: {type(err).__name__}: {str(err)}")
                self.after(0, show_error)


class ResultViewerWindow(tk.Toplevel):
    """Separate window for viewing and exporting run results."""

    def __init__(self, master: PrimeIceApp, initial_run_dir: Optional[Path] = None) -> None:
        super().__init__(master)
        self.master_app = master
        self.title("Prime Ice – Results Viewer")
        self.geometry("900x600")

        self.run_select_var = tk.StringVar()
        self._build_widgets()
        self.refresh_runs()

        if initial_run_dir is not None:
            self.load_run(initial_run_dir)
        elif self.run_select_var.get():
            # Load latest run by default.
            self._load_selected_run()

    def _build_widgets(self) -> None:
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        top_row = ttk.Frame(main)
        top_row.pack(fill=tk.X, pady=2)

        ttk.Label(top_row, text="Select run:").pack(side=tk.LEFT)
        self.run_select_combo = ttk.Combobox(
            top_row, textvariable=self.run_select_var, width=50, state="readonly"
        )
        self.run_select_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(top_row, text="Refresh", command=self.refresh_runs).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(top_row, text="Load", command=self._load_selected_run).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(
            top_row,
            text="Export primes.txt",
            command=self._export_primes_for_run,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            top_row,
            text="Prepare AI prompt",
            command=self._prepare_ai_prompt_for_run,
        ).pack(side=tk.LEFT, padx=2)

        self.results_text = tk.Text(main, wrap=tk.WORD, height=30)
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def _append_results(self, text: str) -> None:
        self.results_text.insert(tk.END, text + "\n")
        self.results_text.see(tk.END)

    def refresh_runs(self) -> None:
        base = Path(self.master_app.out_var.get())
        if not base.exists():
            self.run_select_combo["values"] = []
            self.run_select_var.set("")
            return
        runs = sorted(
            (p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")),
            key=lambda p: p.name,
        )
        run_paths = [str(p) for p in runs]
        self.run_select_combo["values"] = run_paths
        if run_paths:
            self.run_select_var.set(run_paths[-1])
        else:
            self.run_select_var.set("")

    def load_run(self, run_dir: Path) -> None:
        self.refresh_runs()
        self.run_select_var.set(str(run_dir))
        self._load_selected_run()

    def _load_selected_run(self) -> None:
        run_path = self.run_select_var.get()
        if not run_path:
            messagebox.showinfo("Prime Ice", "No run selected.")
            return
        run_dir = Path(run_path)
        cfg_path = run_dir / "config.json"
        summary_path = run_dir / "summary.txt"
        eval_path = run_dir / "eval_report.json"
        self.results_text.delete("1.0", tk.END)

        config: Dict[str, object] = {}
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        self._append_results(f"Run directory: {run_dir}")
        if config:
            agent_name = config.get("agent_name") or ""
            if not agent_name:
                agent_id = str(config.get("agent_id") or config.get("strategy") or "learning_sieve")
                _normalized_id, agent_name = get_agent_metadata(agent_id)
            self._append_results(f"Strategy: {agent_name}")
            self._append_results(
                f"Train range: {config.get('train_low')}..{config.get('train_high')}"
            )
            self._append_results(f"Episodes: {config.get('episodes')}")
            self._append_results("")

        if summary_path.exists():
            self._append_results("Summary.txt:")
            with open(summary_path, "r", encoding="utf-8") as f:
                self._append_results(f.read())

        # Show champion and battery reports in the Patterns-style section if present.
        champion_path = run_dir / "champion_program.txt"
        if champion_path.exists():
            self._append_results("\nChampion Program:")
            with open(champion_path, "r", encoding="utf-8") as f:
                self._append_results(f.read())
        battery_path = run_dir / "battery_report.txt"
        if battery_path.exists():
            self._append_results("\nBattery Report:")
            with open(battery_path, "r", encoding="utf-8") as f:
                self._append_results(f.read())

        if eval_path.exists():
            self._append_results("\nEvaluation report (ranges and final scores):")
            with open(eval_path, "r", encoding="utf-8") as f:
                self._append_results(f.read())

        # Simple pass/fail indication
        if config and summary_path.exists():
            last_line = ""
            with open(summary_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        last_line = line
            # Pass if zero primes removed and zero composites remaining.
            primes_removed = None
            composites_remaining = None
            for line in open(summary_path, "r", encoding="utf-8"):
                if line.startswith("Primes removed:"):
                    primes_removed = int(line.split(":")[1].strip())
                if line.startswith("Composites remaining:"):
                    composites_remaining = int(line.split(":")[1].strip())
            if primes_removed is not None and composites_remaining is not None:
                status = "PASS" if primes_removed == 0 and composites_remaining == 0 else "FAIL"
                self._append_results(f"\nOverall status: {status}")

    def _export_primes_for_run(self) -> None:
        run_path = self.run_select_var.get()
        if not run_path:
            messagebox.showinfo("Prime Ice", "No run selected.")
            return
        run_dir = Path(run_path)
        cfg_path = run_dir / "config.json"
        champion_prog_path = run_dir / "champion_program.json"
        if not cfg_path.exists():
            messagebox.showerror(
                "Prime Ice",
                "config.json not found in the selected run.",
            )
            return
        if not champion_prog_path.exists():
            messagebox.showerror(
                "Prime Ice",
                "champion_program.json not found; run training first.",
            )
            return
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        with open(champion_prog_path, "r", encoding="utf-8") as f:
            champion_prog = json.load(f)
        rules = champion_prog.get("rules", [])

        env = PrimeIceEnv(
            PrimeIceConfig(low=config["train_low"], high=config["train_high"])
        )
        env.reset()
        for rule_dict in rules:
            rule = rule_from_dict(rule_dict)
            _, _, done, _ = env.step(rule)
            if done:
                break

        remaining = sorted(env._remaining)  # type: ignore[attr-defined]
        primes_found = [n for n in remaining if env._is_prime.get(n, False)]  # type: ignore[attr-defined]

        out_path = run_dir / "primes_found.txt"
        dump_text(out_path, "\n".join(str(n) for n in primes_found))
        self.results_text.delete("1.0", tk.END)
        self._append_results(f"Exported {len(primes_found)} primes to {out_path}")
        if primes_found:
            sample_preview = ", ".join(str(n) for n in primes_found[:50])
            self._append_results(f"First primes: {sample_preview}")

    def _prepare_ai_prompt_for_run(self) -> None:
        run_path = self.run_select_var.get()
        if not run_path:
            messagebox.showinfo("Prime Ice", "No run selected.")
            return
        run_dir = Path(run_path)
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            messagebox.showerror("Prime Ice", "config.json not found for this run.")
            return
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        prompt_lines = [
            "You are an AI assistant helping analyze a Prime Ice run.",
            "",
            "Prime Ice is a rule-discovery game where an agent removes composite-number blocks",
            "while trying to keep all prime-number blocks. The environment knows which numbers",
            "are prime (for scoring) but the agent itself only sees aggregate observations and",
            "learns from penalties.",
            "",
            f"Run directory: {run_dir}",
            "Key files:",
            "- config.json: configuration of this run (ranges, agent, episodes, seed).",
            "- train_log.jsonl: JSONL with one line per episode, including the rules used.",
            "- eval_report.json: evaluation metrics on held-out ranges.",
            "- summary.txt: human-readable final summary.",
            "- primes_found.txt (if present): primes remaining after the last episode.",
            "",
            "Config for this run:",
            json.dumps(config, indent=2),
            "",
            "Tasks:",
            "1) Check whether the learned/applied rules are safely preserving primes and removing composites.",
            "2) Look for patterns in the rules that generalize well across evaluation ranges.",
            "3) Suggest improvements to the rule set, learning parameters, or observation features",
            "   to reduce steps while still never removing primes.",
            "4) If you see repeated mistakes (e.g., losing primes or leaving many composites),",
            "   point out which rules are responsible and how they might be refined.",
            "",
            "You may ask to inspect specific files from this run and then propose concrete",
            "code or configuration changes.",
        ]
        prompt_text = "\n".join(prompt_lines)
        out_path = run_dir / "ai_prompt.txt"
        dump_text(out_path, prompt_text)

        self.results_text.delete("1.0", tk.END)
        self._append_results("Prepared AI analysis prompt for this run:")
        self._append_results("")
        self._append_results(prompt_text)
        self._append_results("")
        self._append_results(f"(Saved to {out_path})")


def launch_gui() -> None:
    app = GameLearningApp()
    app.title("Self-Improving Game-Learning AI — Meta-Learning with Forgetting")
    try:
        icon_path = Path(__file__).resolve().parent / "assets" / "brain.ico"
        if icon_path.exists():
            app.iconbitmap(str(icon_path))
    except Exception:
        pass
    app.mainloop()
