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
        self.title("CDI — Constraint Discovery & Inference System")
        self.geometry("1100x750")

        self._worker_thread: Optional[threading.Thread] = None
        self._results_window: Optional["ResultViewerWindow"] = None
        self._last_run_dir: Optional[Path] = None
        self._last_mode: str = "meta_learning"  # or "inference"

        self._build_widgets()

    # UI construction -----------------------------------------------------
    def _build_widgets(self) -> None:
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabbed interface
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.tab_meta_learning = ttk.Frame(self.notebook)
        self.tab_inference = ttk.Frame(self.notebook)
        self.tab_results = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_meta_learning, text="1. Meta-Learning")
        self.notebook.add(self.tab_inference, text="2. Inference")
        self.notebook.add(self.tab_results, text="3. Results")

        # Build each tab
        self._build_meta_learning_tab()
        self._build_inference_tab()
        self._build_results_tab()

    # TAB 1: META-LEARNING -----------------------------------------------
    def _build_meta_learning_tab(self) -> None:
        tab = self.tab_meta_learning
        
        # Training configuration section
        cfg_frame = ttk.LabelFrame(tab, text="Training Configuration")
        cfg_frame.pack(fill=tk.X, pady=5, padx=5)

        # Number of games
        ttk.Label(cfg_frame, text="Number of games:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        self.n_games_var = tk.StringVar(value="10")
        ttk.Entry(cfg_frame, textvariable=self.n_games_var, width=10).grid(
            row=0, column=1, sticky=tk.W, padx=2, pady=2
        )
        ttk.Label(cfg_frame, text="(Different scenarios to learn from)").grid(
            row=0, column=2, sticky=tk.W, padx=5, pady=2
        )

        # Acquisition rounds
        ttk.Label(cfg_frame, text="Acquisition rounds:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        self.acq_rounds_var = tk.StringVar(value="5")
        ttk.Entry(cfg_frame, textvariable=self.acq_rounds_var, width=10).grid(
            row=1, column=1, sticky=tk.W, padx=2, pady=2
        )
        ttk.Label(cfg_frame, text="(Training iterations per game)").grid(
            row=1, column=2, sticky=tk.W, padx=5, pady=2
        )

        # Consolidation budget
        ttk.Label(cfg_frame, text="Consolidation budget:").grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
        self.cons_budget_var = tk.StringVar(value="100")
        ttk.Entry(cfg_frame, textvariable=self.cons_budget_var, width=10).grid(
            row=2, column=1, sticky=tk.W, padx=2, pady=2
        )
        ttk.Label(cfg_frame, text="(Resources for knowledge compression)").grid(
            row=2, column=2, sticky=tk.W, padx=5, pady=2
        )

        # Test games
        ttk.Label(cfg_frame, text="Test games:").grid(row=3, column=0, sticky=tk.W, padx=2, pady=2)
        self.test_games_var = tk.StringVar(value="3")
        ttk.Entry(cfg_frame, textvariable=self.test_games_var, width=10).grid(
            row=3, column=1, sticky=tk.W, padx=2, pady=2
        )
        ttk.Label(cfg_frame, text="(Validation games after training)").grid(
            row=3, column=2, sticky=tk.W, padx=5, pady=2
        )

        # Game family
        ttk.Label(cfg_frame, text="Game family:").grid(row=4, column=0, sticky=tk.W, padx=2, pady=2)
        self.game_family_var = tk.StringVar(value="TicTacToe")
        game_combo = ttk.Combobox(cfg_frame, textvariable=self.game_family_var,
                                  values=["TicTacToe", "NumberGuessing", "Mixed"],
                                  state="readonly", width=15)
        game_combo.grid(row=4, column=1, sticky=tk.W, padx=2, pady=2)

        # Consolidation strategy
        ttk.Label(cfg_frame, text="Consolidation strategy:").grid(row=5, column=0, sticky=tk.W, padx=2, pady=2)
        self.cons_strategy_var = tk.StringVar(value="prune")
        cons_combo = ttk.Combobox(cfg_frame, textvariable=self.cons_strategy_var,
                                  values=["prune", "merge", "compress", "distill"],
                                  state="readonly", width=15)
        cons_combo.grid(row=5, column=1, sticky=tk.W, padx=2, pady=2)

        # Options
        options_frame = ttk.LabelFrame(tab, text="Options")
        options_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.random_seed_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Use random seed each run (recommended)",
            variable=self.random_seed_var
        ).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.reset_policy_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame,
            text="Reset learned policy at start",
            variable=self.reset_policy_var
        ).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)

        # Output directory
        output_frame = ttk.Frame(tab)
        output_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(output_frame, text="Output Folder:").pack(side=tk.LEFT, padx=2)
        self.out_var = tk.StringVar(value="meta_learning_runs")
        out_entry = ttk.Entry(output_frame, textvariable=self.out_var, width=40)
        out_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(output_frame, text="Browse...", command=self._browse_out_dir).pack(side=tk.LEFT, padx=2)
        
        # Control buttons
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, pady=5, padx=5)

        self.meta_run_button = ttk.Button(btn_frame, text="▶ Start Meta-Learning", command=self._on_meta_learning_clicked)
        self.meta_run_button.pack(side=tk.LEFT, padx=2)

        ttk.Button(btn_frame, text="📊 View Results", command=self._switch_to_results_tab).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📁 Open Run Folder", command=self._open_run_folder).pack(side=tk.LEFT, padx=2)

        # Status and progress
        status_frame = ttk.Frame(tab)
        status_frame.pack(fill=tk.X, pady=2, padx=5)
        
        self.meta_status_var = tk.StringVar(value="Ready.")
        ttk.Label(status_frame, textvariable=self.meta_status_var).pack(side=tk.LEFT, padx=5)
        
        self.meta_progress = ttk.Progressbar(status_frame, orient="horizontal", length=300, mode="determinate")
        self.meta_progress.pack(side=tk.LEFT, padx=5)

        # Log area
        log_frame = ttk.LabelFrame(tab, text="Run Summary (live)")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        self.meta_log_text = tk.Text(log_frame, wrap=tk.WORD, height=15)
        self.meta_log_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        meta_scrollbar = ttk.Scrollbar(log_frame, command=self.meta_log_text.yview)
        meta_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.meta_log_text.config(yscrollcommand=meta_scrollbar.set)

    # TAB 2: INFERENCE ---------------------------------------------------
    def _build_inference_tab(self) -> None:
        tab = self.tab_inference
        
        # Target Number section
        target_frame = ttk.LabelFrame(tab, text="Target Number")
        target_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Label(target_frame, text="Target N:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.target_n_var = tk.StringVar(value="")
        target_entry = ttk.Entry(target_frame, textvariable=self.target_n_var, width=60)
        target_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        target_frame.columnconfigure(1, weight=1)
        
        ttk.Label(target_frame, text="(Supports large integers)").grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=2
        )
        
        # Assumptions section
        assumptions_frame = ttk.LabelFrame(tab, text="Assumptions / Scenario")
        assumptions_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.assume_odd_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(assumptions_frame, text="Assume N is odd", variable=self.assume_odd_var).grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2
        )
        
        self.assume_semiprime_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(assumptions_frame, text="Assume N has exactly two prime factors", variable=self.assume_semiprime_var).grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=2
        )
        
        self.allow_square_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(assumptions_frame, text="Allow p=q (semiprime square)", variable=self.allow_square_var).grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=2
        )
        
        self.use_learned_policy_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(assumptions_frame, text="Use learned policy", variable=self.use_learned_policy_var,
                       command=self._on_policy_toggle).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=2
        )
        
        self.use_baseline_policy_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(assumptions_frame, text="Use baseline policy", variable=self.use_baseline_policy_var,
                       command=self._on_policy_toggle).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=2
        )
        
        self.show_debug_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(assumptions_frame, text="Show debug details", variable=self.show_debug_var).grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=2
        )
        
        self.allow_update_policy_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(assumptions_frame, text="Allow learning during inference (experimental)", variable=self.allow_update_policy_var).grid(
            row=3, column=0, sticky=tk.W, padx=5, pady=2
        )
        
        # Verification options (new)
        self.verify_factors_var = tk.BooleanVar(value=True)  # Default ON for generated semiprimes
        ttk.Checkbutton(assumptions_frame, text="Verify factors using narrowed window (evaluation/optional)", 
                       variable=self.verify_factors_var).grid(
            row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2
        )
        
        self.compare_baseline_var = tk.BooleanVar(value=False)  # Default OFF
        ttk.Checkbutton(assumptions_frame, text="Compare with baseline (Fermat near-square, small-trial)", 
                       variable=self.compare_baseline_var).grid(
            row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2
        )
        
        # Inference Limits section
        limits_frame = ttk.LabelFrame(tab, text="Inference Limits")
        limits_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Label(limits_frame, text="Max inference steps:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.max_steps_var = tk.StringVar(value="200")
        ttk.Entry(limits_frame, textvariable=self.max_steps_var, width=10).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=2
        )
        
        ttk.Label(limits_frame, text="Min inference steps:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.min_steps_var = tk.StringVar(value="0")
        ttk.Entry(limits_frame, textvariable=self.min_steps_var, width=10).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=2
        )
        
        ttk.Label(limits_frame, text="Stop if entropy change <").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.epsilon_var = tk.StringVar(value="1e-4")
        ttk.Entry(limits_frame, textvariable=self.epsilon_var, width=10).grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=2
        )
        
        self.disable_early_stop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(limits_frame, text="Disable early-stop (debug)", variable=self.disable_early_stop_var).grid(
            row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2
        )
        
        ttk.Label(limits_frame, text="Display precision (digits):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.display_precision_var = tk.StringVar(value="20")
        ttk.Entry(limits_frame, textvariable=self.display_precision_var, width=10).grid(
            row=4, column=1, sticky=tk.W, padx=5, pady=2
        )
        
        # Test scenario presets
        preset_frame = ttk.Frame(tab)
        preset_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Label(preset_frame, text="Scale preset:").pack(side=tk.LEFT, padx=5)
        self.preset_var = tk.StringVar(value="Medium")
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                    values=["Small demo (fast)", "Medium", "Large"],
                                    state="readonly", width=20)
        preset_combo.pack(side=tk.LEFT, padx=5)
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)
        
        # Generate random semiprime
        gen_frame = ttk.LabelFrame(tab, text="Generate Random Semiprime")
        gen_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Label(gen_frame, text="p range:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.p_low_var = tk.StringVar(value="1000")
        ttk.Entry(gen_frame, textvariable=self.p_low_var, width=12).grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(gen_frame, text="to").grid(row=0, column=2, padx=2)
        self.p_high_var = tk.StringVar(value="10000")
        ttk.Entry(gen_frame, textvariable=self.p_high_var, width=12).grid(row=0, column=3, padx=2, pady=2)
        
        ttk.Label(gen_frame, text="q range:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.q_low_var = tk.StringVar(value="1000")
        ttk.Entry(gen_frame, textvariable=self.q_low_var, width=12).grid(row=1, column=1, padx=2, pady=2)
        ttk.Label(gen_frame, text="to").grid(row=1, column=2, padx=2)
        self.q_high_var = tk.StringVar(value="10000")
        ttk.Entry(gen_frame, textvariable=self.q_high_var, width=12).grid(row=1, column=3, padx=2, pady=2)
        
        self.gen_allow_equal_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(gen_frame, text="Allow p=q", variable=self.gen_allow_equal_var).grid(
            row=0, column=4, padx=10, pady=2
        )
        
        ttk.Button(gen_frame, text="🎲 Generate", command=self._generate_semiprime).grid(
            row=1, column=4, padx=10, pady=2
        )
        
        # Control buttons
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.inference_run_button = ttk.Button(btn_frame, text="▶ Run Inference Only", 
                                               command=self._on_inference_clicked)
        self.inference_run_button.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(btn_frame, text="📊 View Results", command=self._switch_to_results_tab).pack(side=tk.LEFT, padx=2)
        
        # Status and progress
        status_frame = ttk.Frame(tab)
        status_frame.pack(fill=tk.X, pady=2, padx=5)
        
        self.inference_status_var = tk.StringVar(value="Ready. Enter target N above.")
        ttk.Label(status_frame, textvariable=self.inference_status_var).pack(side=tk.LEFT, padx=5)
        
        self.inference_progress = ttk.Progressbar(status_frame, orient="horizontal", length=300, mode="determinate")
        self.inference_progress.pack(side=tk.LEFT, padx=5)
        
        # Belief Summary section (new - structured output after run)
        summary_frame = ttk.LabelFrame(tab, text="Belief Summary (after inference)")
        summary_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Create grid layout for summary fields
        summary_labels = [
            "Termination reason:",
            "Steps taken:",
            "Entropy:",
            "Confidence:",
            "Near-square score:",
            "Size window (smaller factor):",
            "Residue preferences (mod 30):",
            "Interpretation:"
        ]
        
        self.summary_vars = {}
        for i, label_text in enumerate(summary_labels):
            ttk.Label(summary_frame, text=label_text, font=("", 9, "bold")).grid(
                row=i, column=0, sticky=tk.W, padx=5, pady=2
            )
            var = tk.StringVar(value="—")
            label = ttk.Label(summary_frame, textvariable=var, font=("", 9))
            label.grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
            self.summary_vars[label_text] = var
        
        # Interpretation uses a Text widget for multi-line display
        self.summary_interpretation_text = tk.Text(summary_frame, wrap=tk.WORD, height=3, font=("", 9))
        self.summary_interpretation_text.grid(row=7, column=1, sticky=tk.EW, padx=5, pady=2)
        self.summary_interpretation_text.config(state=tk.DISABLED)
        summary_frame.columnconfigure(1, weight=1)
        
        # Log area
        log_frame = ttk.LabelFrame(tab, text="Inference Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        self.inference_log_text = tk.Text(log_frame, wrap=tk.WORD, height=10)
        self.inference_log_text.pack(fill=tk.BOTH, expand=True)
        
        inference_scrollbar = ttk.Scrollbar(log_frame, command=self.inference_log_text.yview)
        inference_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.inference_log_text.config(yscrollcommand=inference_scrollbar.set)

    # TAB 3: RESULTS -----------------------------------------------------
    def _build_results_tab(self) -> None:
        tab = self.tab_results
        
        # Header
        header_frame = ttk.Frame(tab)
        header_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.results_header_text = tk.StringVar(value="No results loaded.")
        ttk.Label(header_frame, textvariable=self.results_header_text, font=("", 11, "bold")).pack(anchor=tk.W)
        
        # Control buttons
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Button(btn_frame, text="🔄 Reload Latest", command=self._reload_latest_results).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📋 Copy Summary", command=self._copy_results_summary).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📁 Open Run Folder", command=self._open_run_folder).pack(side=tk.LEFT, padx=2)
        
        # Results display
        results_notebook = ttk.Notebook(tab)
        results_notebook.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        # Summary tab
        summary_tab = ttk.Frame(results_notebook)
        results_notebook.add(summary_tab, text="Summary")
        
        self.results_summary_text = tk.Text(summary_tab, wrap=tk.WORD)
        self.results_summary_text.pack(fill=tk.BOTH, expand=True)
        
        summary_scrollbar = ttk.Scrollbar(summary_tab, command=self.results_summary_text.yview)
        summary_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_summary_text.config(yscrollcommand=summary_scrollbar.set)
        
        # Belief State tab
        belief_tab = ttk.Frame(results_notebook)
        results_notebook.add(belief_tab, text="Belief State")
        
        self.belief_text = tk.Text(belief_tab, wrap=tk.WORD)
        self.belief_text.pack(fill=tk.BOTH, expand=True)
        
        belief_scrollbar = ttk.Scrollbar(belief_tab, command=self.belief_text.yview)
        belief_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.belief_text.config(yscrollcommand=belief_scrollbar.set)
        
        # Explanation panel
        explain_frame = ttk.LabelFrame(tab, text="Interpretation (Caution: Not Certainty)")
        explain_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.explanation_text = tk.Text(explain_frame, wrap=tk.WORD, height=4)
        self.explanation_text.pack(fill=tk.BOTH, padx=5, pady=5)

    # EVENT HANDLERS ======================================================
    
    # Meta-Learning Tab Handlers ------------------------------------------
    def _browse_out_dir(self) -> None:
        selected = filedialog.askdirectory(title="Select output base directory")
        if selected:
            self.out_var.set(selected)
    
    def _on_meta_learning_clicked(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showinfo("CDI", "A training run is already in progress.")
            return
        
        # Validate inputs
        try:
            n_games = int(self.n_games_var.get())
            acq_rounds = int(self.acq_rounds_var.get())
            cons_budget = int(self.cons_budget_var.get())
            test_games = int(self.test_games_var.get())
            
            if n_games < 1 or acq_rounds < 1 or cons_budget < 1 or test_games < 1:
                raise ValueError("All values must be positive integers")
        except ValueError as e:
            messagebox.showerror("CDI", f"Invalid configuration: {e}")
            return
        
        # Clear log
        self.meta_log_text.delete(1.0, tk.END)
        
        # Build config
        config = {
            "n_games": n_games,
            "acq_rounds": acq_rounds,
            "cons_budget": cons_budget,
            "test_games": test_games,
            "game_family": self.game_family_var.get(),
            "consolidation_strategies": self.cons_strategy_var.get(),
            "options": {
                "random_seed": self.random_seed_var.get(),
                "reset_policy_each_run": self.reset_policy_var.get()
            }
        }
        
        # Run in background thread
        self.meta_run_button.config(state=tk.DISABLED)
        self.meta_status_var.set("Meta-learning in progress...")
        self.meta_progress["value"] = 0
        
        def run_meta_learning():
            try:
                self._append_meta_log("Starting meta-learning run...")
                self._run_meta_learning(config)
                self.after(0, lambda: self.meta_status_var.set("✓ Meta-learning complete!"))
                self.after(0, lambda: self._switch_to_results_tab())
            except Exception as exc:
                self.after(0, lambda: self.meta_status_var.set(f"✗ Error: {exc}"))
                self.after(0, lambda: messagebox.showerror("CDI", f"Error during meta-learning:\n{exc}"))
            finally:
                self.after(0, lambda: self.meta_run_button.config(state=tk.NORMAL))
        
        self._worker_thread = threading.Thread(target=run_meta_learning, daemon=True)
        self._worker_thread.start()

    def _append_meta_log(self, text: str) -> None:
        self.meta_log_text.insert(tk.END, text + "\n")
        self.meta_log_text.see(tk.END)
    
    # Inference Tab Handlers ----------------------------------------------
    def _on_policy_toggle(self) -> None:
        # Ensure only one policy is selected
        if self.use_learned_policy_var.get():
            self.use_baseline_policy_var.set(False)
        elif self.use_baseline_policy_var.get():
            self.use_learned_policy_var.set(False)
        else:
            # At least one must be selected
            self.use_learned_policy_var.set(True)
    
    def _on_preset_selected(self, event=None) -> None:
        preset = self.preset_var.get()
        if preset == "Small demo (fast)":
            self.max_steps_var.set("50")
            self.min_steps_var.set("0")
            self.epsilon_var.set("1e-3")
        elif preset == "Medium":
            self.max_steps_var.set("200")
            self.min_steps_var.set("0")
            self.epsilon_var.set("1e-4")
        elif preset == "Large":
            self.max_steps_var.set("1000")
            self.epsilon_var.set("1e-5")
    
    def _generate_semiprime(self) -> None:
        try:
            import random
            import time
            from utils.primes import is_prime, generate_prime
            
            p_low = int(self.p_low_var.get())
            p_high = int(self.p_high_var.get())
            q_low = int(self.q_low_var.get())
            q_high = int(self.q_high_var.get())
            
            timeout = 8.0  # seconds
            max_attempts = 10000
            
            # Generate p with timeout
            start_time = time.time()
            p = random.randint(p_low, p_high)
            attempts = 0
            while not is_prime(p):
                if time.time() - start_time > timeout:
                    messagebox.showerror(
                        "CDI",
                        f"Could not find prime in range [{p_low}, {p_high}] within {timeout} seconds.\n" +
                        "Try a larger range or smaller numbers."
                    )
                    return
                attempts += 1
                if attempts > max_attempts:
                    messagebox.showerror(
                        "CDI",
                        f"Could not find prime in range [{p_low}, {p_high}] after {max_attempts} attempts.\n" +
                        "This range may have very few primes. Try a larger range."
                    )
                    return
                p = random.randint(p_low, p_high)
            
            # Generate q with timeout
            start_time = time.time()
            q = random.randint(q_low, q_high)
            attempts = 0
            while not is_prime(q):
                if time.time() - start_time > timeout:
                    messagebox.showerror(
                        "CDI",
                        f"Could not find prime in range [{q_low}, {q_high}] within {timeout} seconds.\n" +
                        "Try a larger range or smaller numbers."
                    )
                    return
                attempts += 1
                if attempts > max_attempts:
                    messagebox.showerror(
                        "CDI",
                        f"Could not find prime in range [{q_low}, {q_high}] after {max_attempts} attempts.\n" +
                        "This range may have very few primes. Try a larger range."
                    )
                    return
                q = random.randint(q_low, q_high)
            
            # Ensure p != q if not allowed
            if not self.gen_allow_equal_var.get() and p == q:
                start_time = time.time()
                attempts = 0
                q = random.randint(q_low, q_high)
                while not is_prime(q) or q == p:
                    if time.time() - start_time > timeout:
                        messagebox.showerror(
                            "CDI",
                            f"Could not find distinct prime in range [{q_low}, {q_high}] within {timeout} seconds.\n" +
                            "Try a larger range or allow p=q."
                        )
                        return
                    attempts += 1
                    if attempts > max_attempts:
                        messagebox.showerror(
                            "CDI",
                            f"Could not find distinct prime after {max_attempts} attempts.\n" +
                            "Try a larger range or allow p=q."
                        )
                        return
                    q = random.randint(q_low, q_high)
            
            n = p * q
            self.target_n_var.set(str(n))
            self._append_inference_log(f"Generated: N = {p} × {q} = {n}")
            
        except ValueError as e:
            messagebox.showerror("CDI", f"Invalid range values:\n{e}")
        except Exception as e:
            messagebox.showerror("CDI", f"Error generating semiprime:\n{e}")
    
    def _on_inference_clicked(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showinfo("CDI", "An inference run is already in progress.")
            return
        
        # Validate Target N
        try:
            n_str = self.target_n_var.get().strip()
            if not n_str:
                raise ValueError("Please enter a target N")
            target_n = int(n_str)
            if target_n < 2:
                raise ValueError("Target N must be >= 2")
        except ValueError as e:
            messagebox.showerror("CDI", f"Invalid target N: {e}")
            return
        
        # Clear log
        self.inference_log_text.delete(1.0, tk.END)
        
        # Build config
        config = {
            "target_n": target_n,
            "assumptions": {
                "is_odd": self.assume_odd_var.get(),
                "is_semiprime": self.assume_semiprime_var.get(),
                "allow_square": self.allow_square_var.get()
            },
            "policy": "learned" if self.use_learned_policy_var.get() else "baseline",
            "limits": {
                "max_steps": int(self.max_steps_var.get()),
                "min_steps": int(self.min_steps_var.get()),
                "epsilon": float(self.epsilon_var.get()),
                "disable_early_stop": self.disable_early_stop_var.get(),
                "display_precision": int(self.display_precision_var.get())
            },
            "options": {
                "show_debug": self.show_debug_var.get(),
                "allow_update_policy": self.allow_update_policy_var.get(),
                "verify_factors": self.verify_factors_var.get(),
                "compare_baseline": self.compare_baseline_var.get()
            }
        }
        
        # Run in background thread
        self.inference_run_button.config(state=tk.DISABLED)
        self.inference_status_var.set("Inference in progress...")
        self.inference_progress["value"] = 0
        
        def run_inference():
            try:
                self._append_inference_log(f"Starting inference on N = {target_n}...")
                self._run_inference(config)
                self.after(0, lambda: self.inference_status_var.set("✓ Inference complete!"))
                self.after(0, lambda: self._switch_to_results_tab())
            except Exception as exc:
                error_msg = str(exc)
                import traceback
                full_trace = traceback.format_exc()
                self.after(0, lambda msg=error_msg: self.inference_status_var.set(f"✗ Error: {msg}"))
                self.after(0, lambda msg=error_msg, trace=full_trace: messagebox.showerror("CDI", f"Error during inference:\n{msg}\n\nFull trace:\n{trace}"))
            finally:
                self.after(0, lambda: self.inference_run_button.config(state=tk.NORMAL))
        
        self._worker_thread = threading.Thread(target=run_inference, daemon=True)
        self._worker_thread.start()
    
    def _append_inference_log(self, text: str) -> None:
        self.inference_log_text.insert(tk.END, text + "\n")
        self.inference_log_text.see(tk.END)
    
    # Results Tab Handlers ------------------------------------------------
    def _switch_to_results_tab(self) -> None:
        self.notebook.select(self.tab_results)
        self._reload_latest_results()
    
    def _reload_latest_results(self) -> None:
        if not self._last_run_dir:
            self.results_header_text.set("No results available yet.")
            return
        
        try:
            import json
            from domains.arithmetic.reporting import InferenceSummary, VerificationResult, BaselineComparison
            
            # Load summary.txt
            summary_file = self._last_run_dir / "summary.txt"
            if summary_file.exists():
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary_text = f.read()
                self.results_summary_text.delete(1.0, tk.END)
                self.results_summary_text.insert(1.0, summary_text)
            
            # Load and display InferenceSummary if available
            summary_json_file = self._last_run_dir / "inference_summary.json"
            if summary_json_file.exists():
                with open(summary_json_file, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
                    summary = InferenceSummary.from_dict(summary_data)
                
                # Build structured display
                belief_display = "INFERENCE RUN CARD\n"
                belief_display += "=" * 70 + "\n\n"
                belief_display += f"Target N: {summary.target_n}\n"
                belief_display += f"Bit length: {summary.bit_length}\n"
                belief_display += f"Assumptions: odd, semiprime (see config.json)\n\n"
                
                belief_display += "BELIEF STATE SUMMARY\n"
                belief_display += "-" * 70 + "\n"
                belief_display += f"Termination reason: {summary.termination_reason}\n"
                belief_display += f"Steps taken: {summary.steps_taken}\n"
                belief_display += f"Entropy: {summary.format_entropy_change()}\n"
                belief_display += f"Confidence: {summary.confidence:.3f}\n" if summary.confidence else "Confidence: n/a\n"
                belief_display += f"Near-square score: {summary.near_square_score:.3f} ({summary.get_near_square_label()})\n"
                belief_display += f"Estimated smaller factor magnitude:\n"
                belief_display += f"{summary.format_size_window(summary.target_n)}\n"
                belief_display += f"Top residues (mod 30): {summary.format_residues()}\n\n"
                
                belief_display += "INTERPRETATION\n"
                belief_display += "-" * 70 + "\n"
                for line in summary.interpretation_lines:
                    belief_display += f"• {line}\n"
                belief_display += "\n"
                
                # Load verification result if available
                verification_file = self._last_run_dir / "verification.json"
                if verification_file.exists():
                    with open(verification_file, "r", encoding="utf-8") as f:
                        verification_data = json.load(f)
                        verification = VerificationResult.from_dict(verification_data)
                    
                    belief_display += "VERIFICATION RESULT\n"
                    belief_display += "-" * 70 + "\n"
                    if verification.verifier_skipped:
                        belief_display += f"Verifier skipped: {verification.skip_reason}\n\n"
                    else:
                        belief_display += f"Factors found: {'YES' if verification.factors_found else 'NO'}\n"
                        if verification.factors_found:
                            belief_display += f"p = {verification.p}\n"
                            belief_display += f"q = {verification.q}\n"
                        elif not verification.factors_found and verification.failure_reason:
                            belief_display += f"Reason: {verification.failure_reason}\n"
                        belief_display += f"Window width: {verification.window_width}\n"
                        belief_display += f"Checks attempted: {verification.checks_attempted}\n"
                        belief_display += f"Time: {verification.time_ms:.2f} ms\n\n"
                
                # Load baseline comparison if available
                baseline_file = self._last_run_dir / "baseline.json"
                if baseline_file.exists():
                    with open(baseline_file, "r", encoding="utf-8") as f:
                        baseline_data = json.load(f)
                        baseline = BaselineComparison.from_dict(baseline_data)
                    
                    belief_display += "BASELINE COMPARISON\n"
                    belief_display += "-" * 70 + "\n"
                    belief_display += f"Method: {baseline.method_name}\n"
                    belief_display += f"Factors found: {'YES' if baseline.factors_found else 'NO'}\n"
                    if baseline.factors_found:
                        belief_display += f"p = {baseline.p}\n"
                        belief_display += f"q = {baseline.q}\n"
                    belief_display += f"Checks attempted: {baseline.checks_attempted}\n"
                    belief_display += f"Time: {baseline.time_ms:.2f} ms\n\n"
                
                belief_display += "=" * 70 + "\n"
                belief_display += "See summary.txt and config.json for full details.\n"
                
                self.belief_text.delete(1.0, tk.END)
                self.belief_text.insert(1.0, belief_display)
            else:
                # Fallback to old belief_final.json if inference_summary.json not available
                belief_file = self._last_run_dir / "belief_final.json"
                if belief_file.exists():
                    with open(belief_file, "r", encoding="utf-8") as f:
                        belief_data = json.load(f)
                    
                    belief_display = "Belief State:\n" + "="*60 + "\n"
                    belief_display += json.dumps(belief_data, indent=2)
                    
                    self.belief_text.delete(1.0, tk.END)
                    self.belief_text.insert(1.0, belief_display)
            
            # Update header
            self.results_header_text.set(f"Results from: {self._last_run_dir.name}")
            
            # Generate cautious explanation
            self._generate_explanation()
            
        except Exception as e:
            messagebox.showerror("CDI", f"Error loading results:\n{e}")
    
    def _generate_explanation(self) -> None:
        self.explanation_text.delete(1.0, tk.END)
        self.explanation_text.insert(1.0, 
            "This shows the system's belief state after inference/learning. "
            "Higher confidence values suggest the system has converged on constraints. "
            "This is NOT a guarantee of correctness—always verify independently.")
    
    def _copy_results_summary(self) -> None:
        summary = self.results_summary_text.get(1.0, tk.END)
        self.clipboard_clear()
        self.clipboard_append(summary)
        messagebox.showinfo("CDI", "Summary copied to clipboard.")
    
    def _open_run_folder(self) -> None:
        """Open the last run folder in the system file explorer."""
        if not self._last_run_dir:
            messagebox.showinfo("CDI", "No run has been completed yet.")
            return
        try:
            import os
            os.startfile(str(self._last_run_dir))  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showerror("CDI", f"Could not open folder:\n{exc}")

    # MAIN RUN METHODS ====================================================
    
    def _run_meta_learning(self, config: dict) -> None:
        """Execute a meta-learning run."""
        from meta_learning_loop import MetaLearningLoop
        import json
        from datetime import datetime
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(self.out_var.get()) / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._last_run_dir = run_dir
        
        # Save config
        config_file = run_dir / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        self._append_meta_log(f"Created run directory: {run_dir}")
        
        # Initialize and run
        loop = MetaLearningLoop(
            n_games=config["n_games"],
            acquisition_rounds=config["acq_rounds"],
            consolidation_budget=config["cons_budget"],
            test_games=config["test_games"],
            run_dir=run_dir
        )
        
        # Run with progress callbacks
        def progress_callback(phase: str, pct: float):
            self.after(0, lambda: self.meta_progress.configure(value=pct))
            self.after(0, lambda: self._append_meta_log(f"[{phase}] {pct:.0f}%"))
        
        loop.run(progress_callback=progress_callback)
        
        # Write summary
        summary_file = run_dir / "summary.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Meta-Learning Run Summary\n")
            f.write(f"{'='*60}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Games trained: {config['n_games']}\n")
            f.write(f"Acquisition rounds: {config['acq_rounds']}\n")
            f.write(f"Consolidation budget: {config['cons_budget']}\n")
            f.write(f"Test games: {config['test_games']}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Run completed successfully.\n")
        
        self._append_meta_log(f"✓ Run complete. Summary saved to {summary_file}")
    
    def _run_inference(self, config: dict) -> None:
        """Execute an inference run on a specific target N."""
        import json
        import random
        from datetime import datetime
        
        # Import numpy here to catch missing module error gracefully
        try:
            import numpy as np
        except ImportError:
            messagebox.showerror(
                "CDI",
                "NumPy is required for inference but not installed.\n\n" +
                "Please install it with:\n" +
                "pip install numpy"
            )
            return
        
        from domains.arithmetic.env import SemiprimeInferenceEnv
        from domains.arithmetic.verifier import verify_factors_from_belief, run_baseline_fermat, run_baseline_trial_division
        from domains.arithmetic.reporting import write_summary_txt, VerificationResult, BaselineComparison
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(self.out_var.get()) / f"inference_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._last_run_dir = run_dir
        
        # Save config
        config_file = run_dir / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        self._append_inference_log(f"Created run directory: {run_dir}")
        
        # Initialize environment
        target_n = config["target_n"]
        max_steps = config["limits"]["max_steps"]
        min_steps = config["limits"]["min_steps"]
        disable_early_stop = config["limits"]["disable_early_stop"]
        
        env = SemiprimeInferenceEnv(config={
            'max_steps': max_steps,
            'min_steps': min_steps,
            'disable_early_stop': disable_early_stop,
            'bit_length': target_n.bit_length(),
            'distribution_type': 'unknown'  # User-provided N
        })
        
        self._append_inference_log(f"Running inference on N = {target_n}")
        self._append_inference_log(f"Max steps: {max_steps}, Min steps: {min_steps}")
        early_stop_status = "disabled" if disable_early_stop else "enabled"
        self._append_inference_log(f"Early-stop: {early_stop_status}")
        self._append_inference_log(f"Policy: {config['policy']}")
        
        # Initialize environment with target N (manually set for user-provided input)
        env.current_N = target_n
        env.true_p = None  # Unknown
        env.true_q = None  # Unknown
        from domains.arithmetic.state import BeliefState
        env.belief_state = BeliefState()
        env.belief_state.update_entropy()
        env.step_count = 0
        env.entropy_history = [env.belief_state.entropy_estimate]
        env.reward_history = []
        env.transform_sequence = []
        env.termination_reason = ""
        env.entropy_start = env.belief_state.entropy_estimate
        
        # Run inference loop
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Simple random policy (TODO: integrate with Brain)
            action = random.randint(0, len(env.get_action_space()) - 1)
            
            obs, reward, done, info = env.step(action)
            step += 1
            
            # Update progress
            progress_pct = int((step / max_steps) * 100)
            self.after(0, lambda p=progress_pct: self.inference_progress.configure(value=p))
            
            # Log step (if debug enabled)
            if config['options'].get('show_debug'):
                entropy = info.get('entropy', 0)
                transform = info.get('transform_applied', 'unknown')
                self.after(0, lambda s=step, e=entropy, t=transform: 
                          self._append_inference_log(f"Step {s}: {t} | Entropy: {e:.4f}"))
        
        # Generate inference summary
        summary = env.generate_inference_summary()
        self._append_inference_log(f"✓ Inference completed. Reason: {summary.termination_reason}")
        self._append_inference_log(f"  Entropy: {summary.format_entropy_change()}")
        self._append_inference_log(f"  Confidence: {summary.confidence:.3f}")
        
        # Update GUI Belief Summary
        self.after(0, lambda: self._update_belief_summary(summary))
        
        # Write transforms trace
        transforms_file = run_dir / "transforms_trace.jsonl"
        with open(transforms_file, "w", encoding="utf-8") as f:
            for i, (transform_name, entropy_val) in enumerate(zip(env.transform_sequence, env.entropy_history[1:])):
                record = {
                    "step": i + 1,
                    "transform": transform_name,
                    "entropy": entropy_val
                }
                f.write(json.dumps(record) + "\n")
        
        # Write metrics
        metrics_file = run_dir / "metrics.json"
        metrics_data = {
            "entropy_series": env.entropy_history,
            "reward_series": env.reward_history,
            "transform_sequence": env.transform_sequence,
            "termination_reason": summary.termination_reason
        }
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)
        
        # Write inference summary
        summary_json_file = run_dir / "inference_summary.json"
        with open(summary_json_file, "w", encoding="utf-8") as f:
            f.write(summary.to_json())
        
        # Optional verification
        verification = None
        if config['options'].get('verify_factors', True):
            self._append_inference_log("Running post-inference verification...")
            verification = verify_factors_from_belief(target_n, env.belief_state, max_checks=100000)
            
            if verification.verifier_skipped:
                self._append_inference_log(f"  Verifier skipped: {verification.skip_reason}")
            elif verification.factors_found:
                self._append_inference_log(f"  ✓ Factors found: {verification.p} × {verification.q}")
                self._append_inference_log(f"  Checks attempted: {verification.checks_attempted}")
                self._append_inference_log(f"  Time: {verification.time_ms:.2f} ms")
            else:
                self._append_inference_log(f"  No factors found in window (width: {verification.window_width})")
                self._append_inference_log(f"  Checks attempted: {verification.checks_attempted}")
            
            # Write verification result
            verification_file = run_dir / "verification.json"
            with open(verification_file, "w", encoding="utf-8") as f:
                f.write(verification.to_json())
        
        # Optional baseline comparison
        baseline = None
        if config['options'].get('compare_baseline', False):
            self._append_inference_log("Running baseline comparison...")
            
            # Try Fermat first (good for near-square)
            found, p, q, iters, time_ms = run_baseline_fermat(target_n, max_iterations=10000)
            
            if found:
                baseline = BaselineComparison(
                    method_name="Fermat near-square",
                    checks_attempted=iters,
                    time_ms=time_ms,
                    factors_found=True,
                    p=p,
                    q=q
                )
                self._append_inference_log(f"  Fermat: Found {p} × {q} in {iters} iterations ({time_ms:.2f} ms)")
            else:
                # Try trial division
                found, p, q, checks, time_ms = run_baseline_trial_division(target_n, B=10000)
                baseline = BaselineComparison(
                    method_name="Trial division B=10000",
                    checks_attempted=checks,
                    time_ms=time_ms,
                    factors_found=found,
                    p=p,
                    q=q
                )
                if found:
                    self._append_inference_log(f"  Trial division: Found {p} × {q} in {checks} checks ({time_ms:.2f} ms)")
                else:
                    self._append_inference_log(f"  Trial division: No factors found (B=10000)")
            
            # Write baseline result
            baseline_file = run_dir / "baseline.json"
            with open(baseline_file, "w", encoding="utf-8") as f:
                f.write(baseline.to_json())
        
        # Write summary.txt (human-readable)
        summary_txt_file = run_dir / "summary.txt"
        write_summary_txt(summary, str(summary_txt_file), verification, baseline)
        
        self._append_inference_log(f"✓ All artifacts saved to {run_dir}")
    
    def _update_belief_summary(self, summary) -> None:
        """Update the GUI Belief Summary section with InferenceSummary data."""
        self.summary_vars["Termination reason:"].set(summary.termination_reason)
        self.summary_vars["Steps taken:"].set(str(summary.steps_taken))
        self.summary_vars["Entropy:"].set(summary.format_entropy_change())
        self.summary_vars["Confidence:"].set(f"{summary.confidence:.3f}" if summary.confidence is not None else "n/a")
        self.summary_vars["Near-square score:"].set(f"{summary.near_square_score:.3f} ({summary.get_near_square_label()})")
        self.summary_vars["Size window (smaller factor):"].set(summary.format_size_window(summary.target_n))
        self.summary_vars["Residue preferences (mod 30):"].set(summary.format_residues())
        
        # Update interpretation text
        self.summary_interpretation_text.config(state=tk.NORMAL)
        self.summary_interpretation_text.delete(1.0, tk.END)
        for line in summary.interpretation_lines:
            self.summary_interpretation_text.insert(tk.END, "• " + line + "\n")
        self.summary_interpretation_text.config(state=tk.DISABLED)
        
        self._append_inference_log(f"✓ Inference complete. Results saved to {run_dir}")


def main() -> None:
    """Launch the CDI GUI."""
    app = GameLearningApp()
    app.mainloop()


def launch_gui() -> None:
    """Launch the AI Gamer GUI (alias for main)."""
    main()


if __name__ == "__main__":
    main()
