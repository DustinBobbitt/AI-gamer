# Output Specification: Summary and Run Card Fields

## Purpose
This document defines the expected fields and wording rules for `summary.txt` and `run_card.txt`.
Both files are generated from **single source of truth functions** in `domains/arithmetic/reporting.py`.

---

## Required Fields

### Header Section
```
==============================================================
ARITHMETIC FACTOR INFERENCE RUN SUMMARY  (or "INFERENCE RUN CARD")
==============================================================

Target N: {N}
Bit length: {N.bit_length()}
```

### Inference Limits Section (v1.3+)
```
Max steps: {max_steps}, Min steps: {min_steps}, Early-stop: {enabled|disabled}
```
- Show only if `InferenceSummary.max_steps` is not None
- Format: Comma-separated values on single line

### Termination Section
```
Termination reason: {reason}
Steps taken: {steps_taken}
```
- Reason must be one of:
  - `invalid_input`
  - `max_steps`
  - `confidence_reached`
  - `entropy_converged`
  - `policy_stalled`

### Entropy Metrics Section
```
ENTROPY METRICS
--------------------------------------------------------------
Start: {entropy_start:.4f}
End: {entropy_end:.4f}
Delta (end-start): {entropy_delta:+.4f}
Reduction: {entropy_pct_reduction:.2f}%
```
- **Delta sign**: Must be `end - start` (negative for reductions)
- **Label**: Must show `Delta (end-start):` to clarify sign convention
- **Reduction**: Always positive percentage, represents `(start - end) / start * 100`

### Belief State Section
```
BELIEF STATE
--------------------------------------------------------------
Confidence: {confidence:.3f} (or 'n/a' if None)
Near-square score: {near_square_score:.3f} ({label})
Estimated smaller factor magnitude:
  ~ [{low:.1f}, {high:.1f}] (≈ {low_ratio:.2f}–{high_ratio:.2f} × sqrt(N))
Top residues (mod 30): {residue:weight, residue:weight, residue:weight}
```

#### Near-Square Label Rules
- `score < 0.3`: **LOW** (skewed)
- `0.3 <= score < 0.7`: **MED** (balanced)
- `score >= 0.7`: **HIGH** (near-square)

#### Size Window Rules (v1.3 Adaptive)
- **Label**: "Estimated smaller factor magnitude"
- **Format**: Absolute range with optional relative-to-sqrt
- **Source**: From `InferenceSummary.size_window_low/high`
- **NOT** recomputed (uses adaptive window from summary object)
- If `target_n` provided: Show relative ratios `(≈ X–Y × sqrt(N))`
- If `target_n` None: Show absolute only `~ [low, high]`

#### Residue Format
- Show top 3 residues from `InferenceSummary.top_residues_mod30`
- Format: `residue:weight` pairs, comma-separated
- Example: `1:0.23, 7:0.19, 13:0.15`
- If empty: `n/a`

### Interpretation Section
```
INTERPRETATION
--------------------------------------------------------------
• {line1}
• {line2}
• {line3}
```
- Bullet points with `•` (Unicode U+2022)
- Lines from `InferenceSummary.interpretation_lines`
- Max 3 lines
- Wording rules:
  - Use cautious language: "may", "appears", "likely", "Under current constraints"
  - Never claim certainty unless confidence very high
  - Include small-N note if `bit_length <= 8`
  - Only make near-square/skew claims if `confidence >= 0.05`

### Verification Section (Optional)
```
VERIFICATION RESULT
--------------------------------------------------------------
Factors found: {YES|NO}
```

#### If Factors Found
```
p = {p}
q = {q}
Window width: {window_width}, Checks: {checks_attempted}, Time: {time_ms:.2f} ms
```

#### If Factors NOT Found
```
Reason: {failure_reason}
Window width: {window_width}, Checks: {checks_attempted}, Time: {time_ms:.2f} ms
```

**Failure Reason Rules (v1.3)**:
- Primary: `"no valid factors found within inferred window"`
- **NEVER** use: `"only trivial factorization (1 × N) possible under inferred window"`
- Rationale: Honest about window search failure, not mathematical impossibility

#### If Verifier Skipped
```
Verifier skipped: {skip_reason}
```

### Baseline Comparison Section (Optional)
```
BASELINE COMPARISON
--------------------------------------------------------------
Method: {method_name}
Factors found: {YES|NO}
p = {p} (if found)
q = {q} (if found)
Checks attempted: {checks_attempted}
Time: {time_ms:.2f} ms
```

---

## Wording Rules

### Termination Reasons (User-Facing)
- `invalid_input` → "Invalid input"
- `max_steps` → "Maximum steps reached"
- `confidence_reached` → "High confidence threshold reached"
- `entropy_converged` → "Entropy converged"
- `policy_stalled` → "Policy stalled (no progress)"

### Interpretation Phrases (Approved)
- ✅ "Inference significantly narrowed the search space (>50% entropy reduction)"
- ✅ "Inference moderately narrowed the search space (20-50% entropy reduction)"
- ✅ "Inference made minimal progress (< 20% entropy reduction)"
- ✅ "Under current constraints, target may have factors close in magnitude (near-square)"
- ✅ "Under current constraints, target may be skewed (factors differ significantly in size)"
- ✅ "System expressed high confidence ({confidence:.2f}) in belief state"
- ✅ "System expressed low confidence ({confidence:.2f})"
- ✅ "N is very small; limited structural inference expected at this scale" (if bit_length <= 8)

### Interpretation Phrases (Prohibited)
- ❌ "Target IS near-square" (too certain)
- ❌ "Factors are definitely..." (too certain)
- ❌ "This proves..." (never claim proof)
- ❌ "The system knows..." (system doesn't "know" factors)
- ❌ Any wording that implies trial division occurred in inference

---

## Format Differences: Summary vs Run Card

### Summary.txt (Full Detail)
- All sections included
- More whitespace for readability
- Verbose labels
- Separate sections for entropy, belief state, interpretation
- Full verification details with all metrics
- Includes baseline comparison if present
- Footer: `See config.json for full configuration details`

### Run Card.txt (Compact)
- Header: "INFERENCE RUN CARD"
- Config limits on single line
- Combined belief state summary (no separate entropy section)
- Interpretation included
- Verification condensed to single line if possible
- Baseline very condensed (`Method: result (checks, time)`)
- Footer: None or minimal

### Shared Formatting
- Both use same entropy delta convention
- Both use same size window format
- Both use same verifier failure reason
- Both use same Unicode symbols (UTF-8)
- Both generated from same functions

---

## Unicode Symbols (Must Be UTF-8)

- Bullet: `•` (U+2022)
- Multiplication: `×` (U+00D7)
- Approximately: `≈` (U+2248)
- En dash: `–` (U+2013)
- Square root: `√` (U+221A)
- Delta: `Δ` (U+0394)
- Arrow: `→` (U+2192)
- Checkmark: `✓` (U+2713)
- X mark: `✗` (U+2717)

**Critical**: All file writes must use `encoding='utf-8'` or symbols will render as mojibake on Windows.

---

## Examples

### Example: Summary.txt Header
```
======================================================================
ARITHMETIC FACTOR INFERENCE RUN SUMMARY
======================================================================

Target N: 77
Bit length: 7

Max steps: 50, Min steps: 0, Early-stop: enabled

Termination reason: entropy_converged
Steps taken: 12
```

### Example: Entropy Section
```
ENTROPY METRICS
----------------------------------------------------------------------
Start: 3.4532
End: 3.3891
Delta (end-start): -0.0641
Reduction: 1.86%
```

### Example: Size Window (Adaptive, Skewed)
```
Estimated smaller factor magnitude:
  ~ [3.1, 6.4] (≈ 0.35–0.73 × sqrt(N))
```

### Example: Verification Failure (Honest Messaging)
```
VERIFICATION RESULT
----------------------------------------------------------------------
Factors found: NO
Reason: no valid factors found within inferred window
Window width: 58, Checks: 29, Time: 0.42 ms
```

### Example: Run Card Compact Format
```
INFERENCE RUN CARD
======================================================================

Target N: 77
Bit length: 7
Assumptions: odd=True, semiprime=True

Max steps: 50, Min steps: 0, Early-stop: enabled

BELIEF STATE SUMMARY
----------------------------------------------------------------------
Termination reason: entropy_converged
Steps taken: 12
Entropy: 3.453 → 3.389 (Δ-0.064, 1.9%)
Confidence: 0.123
Near-square score: 0.234 (LOW)
Estimated smaller factor magnitude:
  ~ [3.1, 6.4] (≈ 0.35–0.73 × sqrt(N))
Top residues (mod 30): 7:0.19, 1:0.17, 13:0.15

INTERPRETATION
----------------------------------------------------------------------
• Inference made minimal progress (< 20% entropy reduction).
• Under current constraints, target may be skewed (factors differ significantly in size).

VERIFICATION RESULT
----------------------------------------------------------------------
Factors found: YES
p = 7
q = 11
Window width: 58, Checks: 3, Time: 0.18 ms
```

---

## Validation Checklist

Before committing changes to reporting.py, verify:

- ✅ Both summary.txt and run_card.txt use same functions
- ✅ No duplicate formatting logic in GUI or elsewhere
- ✅ All file writes have `encoding='utf-8'`
- ✅ Entropy delta uses `end - start` convention
- ✅ Delta label shows `(end-start):` for clarity
- ✅ Size window from summary object, not recomputed
- ✅ Verifier failure reason is honest, not misleading
- ✅ Interpretation respects confidence threshold
- ✅ Small-N note appears for bit_length <= 8
- ✅ Unicode symbols render correctly
- ✅ Termination reasons match spec
- ✅ Near-square labels use correct thresholds

---

**Last Updated**: January 5, 2026  
**Source**: domains/arithmetic/reporting.py  
**Functions**: `build_inference_summary_text()`, `build_inference_run_card_text()`
