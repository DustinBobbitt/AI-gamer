# Semiprime Conjecture Engine — Handoff Overview

## What is this project?

This is a meta-learning system designed to infer factors of semiprime numbers (N = p × q, where p and q are prime) through **constraint-based belief state updates**, not trial division. The system learns to apply transforms that narrow uncertainty about factor properties (size ratio, residue classes, near-squareness) and uses entropy reduction as the reward signal. The inference core **never enumerates prime lists or performs divisibility checks as actions**—that would be cheating. An optional post-inference verifier can use the narrowed belief window to attempt factor extraction, but verification is separate from the inference process itself.

---

## Current State (January 5, 2026)

### UI Overview (3-Tab GUI)

**Tab 1: Meta-Learning**
- Purpose: Train policies across game families (currently disabled/unused)
- Legacy from toy games architecture (TicTacToe, etc.)
- Keep intact but not primary focus

**Tab 2: Inference**
- **Primary interface** for arithmetic factor inference
- Inputs:
  - Target N (text field, supports large integers)
  - Assumptions: odd, semiprime, allow p=q
  - Policy selection: learned vs baseline (random)
  - Max steps, **Min steps** (v1.3), **Disable early-stop** (v1.3)
  - Epsilon threshold for convergence
  - Verification toggle (optional post-inference)
- Generate random semiprime: p range, q range, with timeout protection (8 seconds)
- Run button → background thread → updates Belief Summary section
- Belief Summary shows:
  - Termination reason
  - Steps taken
  - Entropy change (start → end, Δ, % reduction)
  - Confidence (0-1 or n/a)
  - Near-square score (LOW/MED/HIGH)
  - **Adaptive size window** (v1.3): ~ [low, high] (≈ ratio × sqrt(N))
  - Top residues mod 30
  - Interpretation (cautious language, confidence threshold)

**Tab 3: Results**
- Loads latest run folder
- Displays summary.txt in Summary tab
- Displays **run_card.txt** in Belief State tab (single source of truth)
- Shows interpretation panel
- Buttons: Reload, Copy, Open Folder

### Inference Pipeline

```
User Input (Target N)
  ↓
Initialize BeliefState (13-token fixed representation)
  - size_ratio_estimate (0-1)
  - near_square_score (0-1)
  - residue_weights_mod30 (8 valid residues)
  - confidence (0-1)
  - entropy_estimate (computed from distribution)
  ↓
Episode Loop (max max_steps, respects min_steps/disable_early_stop)
  ↓
  Select Transform (random policy or learned policy)
    - 4 transforms available:
      1. RefineRatioEstimate
      2. UpdateResidueWeights
      3. RefineNearSquareScore
      4. RecomputeEntropy
  ↓
  Apply Transform → update BeliefState
  ↓
  Compute reward = max(0, entropy_reduction)
  ↓
  Check termination:
    - invalid_input (N <= 1)
    - max_steps reached
    - confidence_reached (>0.95)
    - entropy_converged (range < epsilon over window) — gated by min_steps/disable_early_stop
    - policy_stalled (no change for 10 steps) — gated by min_steps/disable_early_stop
  ↓
Generate InferenceSummary
  - Termination reason
  - Entropy metrics (start, end, delta, % reduction)
  - Belief state snapshot
  - **Adaptive window computation** (v1.3): based on near_square_score
  - Interpretation (with confidence threshold ≥ 0.05)
  ↓
Optional: Post-Inference Verification
  - Uses narrowed size window from belief state
  - Checks candidates in window for divisibility
  - **Enforces primality** (v1.2): rejects p < 2, q < 2, or non-prime factors
  - Returns VerificationResult with factors or failure reason
  - Failure reason: "no valid factors found within inferred window" (v1.3)
  ↓
Write Artifacts (all UTF-8):
  - config.json
  - inference_summary.json
  - verification.json (if enabled)
  - baseline.json (if enabled)
  - summary.txt (full detail)
  - **run_card.txt** (compact, single source of truth v1.0)
  - transforms_trace.jsonl
  - metrics.json
```

---

## Explicit Constraints (MUST MAINTAIN)

1. **No Cheating in Inference Core**
   - Inference cannot enumerate prime lists
   - Inference cannot perform trial division as actions
   - Inference cannot probe N % k as primary mechanism
   - Only constraint-based updates allowed

2. **Verifier is Separate**
   - Runs ONLY after inference completes
   - Labeled as optional evaluation
   - Cost metrics tracked (checks attempted, time)
   - Primality checks allowed ONLY in verifier

3. **BeliefState is Fixed 13-Token Representation**
   - Cannot expand dynamically
   - No candidate lists stored
   - Represents distributions/constraints, not factor values

4. **Transforms are Constraint-Based**
   - Cannot return "try factor k"
   - Cannot reveal ground truth during inference
   - Must update distributions/scores, not explicit values

5. **Reporting Honesty**
   - No fake certainty (confidence 0.0 is valid)
   - Window is an estimate, not a guarantee
   - Verifier failure messages must be honest
   - Interpretation must be cautious

---

## Reporting Rules (v1.3 + Run Card Parity v1.0)

### Summary.txt and Run Card Parity
- Both generated from **single source of truth functions**:
  - `build_inference_summary_text()` → summary.txt
  - `build_inference_run_card_text()` → run_card.txt
- Located in: `domains/arithmetic/reporting.py`
- **No duplicate logic** outside these functions
- GUI loads run_card.txt (preferred) or generates from JSON (fallback)

### UTF-8 Encoding
- All file writes: `open(path, 'w', encoding='utf-8')`
- All file reads: `open(path, 'r', encoding='utf-8')`
- Applies to: gui.py (13 operations), reporting.py (all writes)

### Termination Reasons
- `invalid_input`: N <= 1
- `max_steps`: Hit step limit
- `confidence_reached`: Confidence > 0.95
- `entropy_converged`: Entropy range < epsilon over window (gated by min_steps)
- `policy_stalled`: No entropy change for 10 steps (gated by min_steps)

### Window Semantics (v1.3 Adaptive Window)
- **NOT** fixed 0.40-0.60 × sqrt(N)
- Adaptive based on `near_square_score`:
  - **Near-square** (≥0.70): center=0.95√N, half_width=0.20√N
  - **Skewed** (≤0.30): center=0.35√N, half_width=0.35√N (wider)
  - **Balanced** (else): center=0.60√N, half_width=0.30√N
- Window clamped to [2, √N], force odd if `assume_odd=True`
- Computed by: `compute_adaptive_window()` in reporting.py
- Format: `~ [low, high] (≈ ratio_low–ratio_high × sqrt(N))`

### Verifier Messaging (v1.3)
- Success: `Factors found: YES`, show p and q
- Failure: `Factors found: NO`, reason:
  - **"no valid factors found within inferred window"** (honest)
  - NOT "only trivial factorization (1 × N) possible" (misleading)
- Skipped: `Verifier skipped: {reason}`

### Entropy Delta Sign (v1.3)
- Convention: **Delta = end - start** (negative for reductions)
- Label: `Delta (end-start): -0.0296`
- Reduction: Always positive percentage for clarity

### Interpretation Confidence Threshold (v1.3)
- Only make near-square/skew claims if `confidence >= 0.05`
- Wording: "Under current constraints, target may..."
- Avoids overconfident claims on low-confidence runs

### Small-N Hygiene (v1.2)
- Threshold: 8 bits (N < 256)
- Add note: "N is very small; limited structural inference expected at this scale"

---

## What We Changed and Why

### The Pivot (Late 2025)
- **Original**: Meta-learning system for toy games (TicTacToe, Nim, Mastermind)
- **Pivot**: Focus on Arithmetic Factor Inference (semiprime factorization)
- **Why**: More interesting RL problem, clearer win condition, real-world relevance
- **TicTacToe status**: Keep intact as optional domain, do not delete

### Major Implementations

**v1.1: Inference Tab (Dec 2025)**
- Added standalone inference interface
- Belief Summary display
- Generate random semiprime with timeout protection
- Inference log area

**v1.2: Verifier Guardrails, Size Window Clarity, Small-N Hygiene (Jan 2026)**
- Primality checks in verifier (reject p=1, q=N)
- Size window labeled as "Estimated smaller factor magnitude"
- Small-N threshold (8 bits) with interpretation note
- failure_reason field in VerificationResult

**v1.3: Early-Stop Control, Verifier Fix, Adaptive Window, UTF-8, Metric Hygiene (Jan 5, 2026)**
- Min inference steps parameter (default 0)
- Disable early-stop checkbox (debug mode)
- Gate entropy_converged/policy_stalled behind min_steps
- UTF-8 encoding for all file operations
- Adaptive window estimation (3 ranges based on near_square_score)
- Honest verifier failure messaging
- Entropy delta sign consistency (Delta = end - start)
- Interpretation confidence threshold (≥0.05)
- Config tracking in summary (max_steps, min_steps, early_stop_enabled)

**Run Card Parity v1.0 (Jan 5, 2026)**
- Single source of truth for summary and run card generation
- Centralized formatting in reporting.py
- Eliminates duplicate logic and drift
- GUI loads run_card.txt or generates from JSON

---

## Known Issues (Mostly Fixed in v1.3)

### Fixed
- ✅ Early stop vs max steps: User sets max=50 but stops at 8 → **Fixed with min_steps control**
- ✅ Unicode mojibake (•, ×, ≈) → **Fixed with UTF-8 encoding**
- ✅ Verifier messaging misleading ("only trivial...") → **Fixed with honest failure reason**
- ✅ Window too fixed (0.40-0.60 × sqrt(N)) → **Fixed with adaptive window**
- ✅ Entropy delta sign confusing → **Fixed with Delta = end - start convention**
- ✅ Run card drift from summary.txt → **Fixed with single source of truth**
- ✅ ValueError on empty input fields → **Fixed with default value handling**

### Remaining
- ⚠️ Learned policy not integrated yet (uses random policy)
- ⚠️ Brain training not connected to inference tab
- ⚠️ No actual learning happening (all inference uses random transforms)
- ⚠️ Window may still miss factors for some N (adaptive helps but not perfect)

---

## File/Artifact Contract Per Run

Every inference run creates a folder: `meta_learning_runs/inference_YYYYMMDD_HHMMSS/`

**Required files:**
- `config.json` — Full configuration (target_n, limits, options, assumptions)
- `inference_summary.json` — Serialized InferenceSummary object
- `summary.txt` — Human-readable full summary (UTF-8)
- `run_card.txt` — Compact run card (UTF-8, single source of truth)
- `transforms_trace.jsonl` — Step-by-step transform applications
- `metrics.json` — Entropy/reward series, termination reason

**Optional files:**
- `verification.json` — VerificationResult if verifier enabled
- `baseline.json` — BaselineComparison if baseline comparison enabled

**File format rules:**
- All JSON: standard serialization, indent=2
- All text: UTF-8 encoding, Unix LF or Windows CRLF (Git handles)
- JSONL: one JSON object per line (transforms_trace.jsonl)

---

## How to Run

### Launch GUI
```powershell
cd "C:\Users\ksima\semiprime Conjecture Engine"
python __main__.py
```
- Opens 3-tab GUI
- Use **Inference tab** for factor inference
- Results tab auto-loads after run

### Launch CLI (if present)
```powershell
python prime_ice/cli.py --help
```
- Not primary interface
- May be outdated

### Where Run Folders Appear
- Default: `meta_learning_runs/`
- Configurable in GUI (Meta-Learning tab, though not used for inference)
- Each run creates: `inference_YYYYMMDD_HHMMSS/`

---

## Glossary

**Domain vs Toy Games**
- Domain: A problem space (e.g., Arithmetic Factor Inference, TicTacToe)
- Toy Games: Original focus (TicTacToe, Nim) — still present, keep intact
- Arithmetic: Current primary domain

**Inference vs Verification**
- Inference: Core RL process, updates belief state via transforms, NO trial division
- Verification: Optional post-step, uses narrowed window to attempt factor extraction
- Verification is evaluation, NOT part of inference

**Entropy vs Confidence**
- Entropy: Measure of uncertainty in belief distribution (high = uncertain)
- Confidence: Agent's self-assessment of belief state quality (0-1)
- Reward = entropy reduction (want low entropy)

**Transform vs Action**
- Transform: Constraint-based update to belief state (e.g., RefineRatioEstimate)
- Action: General RL term (we use "transform" to emphasize no trial division)
- Transforms update distributions, not candidate lists

**BeliefState**
- 13-token fixed representation of inference state
- Tokens: size_ratio_estimate, near_square_score, 8 residue weights, confidence
- Does NOT store candidate factors

**Near-Square Score**
- Estimate of how close N is to a perfect square (0-1)
- HIGH (>0.7): factors close in magnitude (p ≈ q ≈ √N)
- LOW (<0.3): skewed (p << q or vice versa)

**Size Window**
- Estimated range for smaller factor: [low, high]
- Adaptive based on near_square_score (v1.3)
- Used by verifier for candidate search

---

## Next Steps Checklist (Priority Order)

### High Priority
- [ ] **Test N=15, N=77 cases** with v1.3 adaptive window
  - Verify N=15 shows small-N hygiene note
  - Verify N=77 (7×11) adaptive window includes 7 or 11
  - Verify honest verifier failure messaging
  - Verify min_steps enforcement works

- [ ] **Integrate Brain with Inference Tab**
  - Currently uses random policy
  - Need to load learned policy from `models/learning_sieve_policy.json`
  - Connect policy selection to actual learned weights
  - Ensure transforms chosen by learned policy, not random

- [ ] **Test early-stop controls**
  - Set Max=50, Min=25, verify steps >= 25
  - Enable "Disable early-stop", verify runs to max_steps
  - Test with various epsilon thresholds

### Medium Priority
- [ ] **Add domain selector to GUI**
  - Dropdown: Arithmetic / TicTacToe / Nim / etc.
  - Currently hardcoded to Arithmetic
  - Keep toy games accessible but not default

- [ ] **Add scenario presets for Inference Tab**
  - Small (8-bit), Medium (16-bit), Large (32-bit)
  - Preset bit ranges for p and q generation
  - Currently has "Scale preset" but minimal effect

- [ ] **Improve adaptive window heuristic**
  - Current 3-range approach is simple
  - Could use more sophisticated logic
  - Consider adding "skew_score" token to BeliefState

- [ ] **Add entropy curve plotting**
  - Show entropy reduction over steps
  - Interactive plot in Results tab
  - Currently only metrics.json has raw data

### Low Priority
- [ ] **Optimize residue ordering**
  - Verifier checks preferred residues first
  - Could order by belief weights (highest first)
  - Currently uses top 3 residues from belief state

- [ ] **Add run comparison view**
  - Compare multiple runs side-by-side
  - Show which configs worked better
  - Currently only view one run at a time

- [ ] **Implement baseline auto-comparison**
  - Currently checkbox exists but rarely used
  - Auto-run Fermat + trial division for comparison
  - Show speedup/checks saved

---

## Important Reminders for Next Agent

1. **Do NOT delete TicTacToe or toy games code**
   - They are optional domains, kept for backward compatibility
   - User may want to switch back
   - Just de-emphasize in UI, don't remove

2. **Maintain constraint discipline**
   - Inference core = no cheating
   - Verifier = separate, clearly labeled
   - If adding features, ensure no trial division leaks into inference

3. **Keep reporting parity**
   - All formatting changes go in reporting.py
   - Never duplicate logic in GUI
   - summary.txt and run_card.txt must always match

4. **UTF-8 everywhere**
   - All new file operations: explicit `encoding='utf-8'`
   - Windows default (cp1252) breaks Unicode symbols

5. **Test with small-N and edge cases**
   - N=15, N=21, N=35 (small)
   - N=77 (7×11, skewed, tests adaptive window)
   - N=143 (11×13, near-square)
   - Large N (1000+ bits) for performance

6. **Git workflow**
   - Two remotes: origin (AI-gamer), semiprime (Semiprime-Conjecture-Engine)
   - Push to both: `git push; git push semiprime master`
   - Commit messages should be detailed

---

## Architecture Notes

### Key Files
- `__main__.py` — Entry point, launches GUI
- `gui.py` — Main GUI implementation (1100+ lines)
- `domains/arithmetic/env.py` — SemiprimeInferenceEnv (RL environment)
- `domains/arithmetic/state.py` — BeliefState class
- `domains/arithmetic/transforms.py` — Transform library (4 transforms)
- `domains/arithmetic/reporting.py` — Single source of truth for formatting
- `domains/arithmetic/verifier.py` — Post-inference factor verification
- `utils/primes.py` — Primality testing (is_prime, generate_prime)

### Transform Library
1. **RefineRatioEstimate**: Update size_ratio_estimate based on constraints
2. **UpdateResidueWeights**: Adjust residue_weights_mod30 distribution
3. **RefineNearSquareScore**: Update near_square_score based on ratio
4. **RecomputeEntropy**: Recalculate entropy_estimate from distribution

### Termination Logic
Located in: `domains/arithmetic/env.py`, method `_check_termination()`
- Checks conditions in order
- Early-stop checks gated by `min_steps` and `disable_early_stop`
- Returns `(done: bool, reason: str)`

### Belief State Vector
13 tokens total:
- size_ratio_estimate (1)
- near_square_score (1)
- confidence (1)
- residue_weights_mod30 (8, for residues 1,7,11,13,17,19,23,29)
- entropy_estimate (1)
- importance_weights (1, for Fisher-like drift)

---

## Version History

- **v1.0** (Dec 2025): Initial toy games implementation
- **v1.1** (Dec 2025): Inference Tab added
- **v1.2** (Jan 2026): Verifier guardrails, size window clarity, small-N hygiene
- **v1.3** (Jan 5, 2026): Early-stop control, adaptive window, UTF-8, metric hygiene
- **Run Card Parity v1.0** (Jan 5, 2026): Single source of truth for summary/run card

---

**Last Updated**: January 5, 2026  
**Current Commit**: b2661d8  
**Status**: All core features implemented, ready for learned policy integration and testing
