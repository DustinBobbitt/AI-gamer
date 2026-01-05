# Major Decisions and Pivots

## Architecture Decisions

### Pivot from Toy Games to Arithmetic Focus (Late 2025)
- **Decision**: Shift primary focus to arithmetic factor inference
- **Rationale**: More interesting RL problem, clearer success metrics, real-world relevance
- **Impact**: Toy games remain but de-emphasized; TicTacToe kept as optional domain
- **Status**: Complete

### No Cheating in Inference Core (Foundational)
- **Decision**: Inference cannot enumerate primes or do trial division as actions
- **Rationale**: System must learn constraint-based inference, not brute-force
- **Implementation**: Transforms update distributions only, verifier is separate
- **Status**: Enforced

### Fixed 13-Token BeliefState Representation (Foundational)
- **Decision**: BeliefState is fixed-size vector, not expandable
- **Rationale**: Scalability to large N, forces abstraction, prevents candidate list storage
- **Trade-off**: Less precise than candidate tracking, but more general
- **Status**: Implemented

### Verifier as Separate Evaluation Step (Foundational)
- **Decision**: Post-inference verification is optional and clearly labeled
- **Rationale**: Allows honesty about inference limitations, separates learning from validation
- **Implementation**: Runs after inference, uses narrowed window, tracks cost
- **Status**: Implemented with v1.2 guardrails

---

## UI/UX Decisions

### 3-Tab GUI Design (v1.0)
- **Decision**: Meta-Learning / Inference / Results tabs
- **Rationale**: Separates training, testing, and analysis workflows
- **Status**: Implemented, Inference tab is primary

### Belief Summary in Inference Tab (v1.1)
- **Decision**: Show structured belief state metrics after inference
- **Rationale**: Users need visibility into what the system learned
- **Implementation**: Termination, entropy, confidence, window, residues, interpretation
- **Status**: Implemented

### Target N as Text Field (v1.1)
- **Decision**: Allow arbitrary-length integer input
- **Rationale**: Support large N (1000+ bits) without dropdown limits
- **Implementation**: Python's unlimited int precision
- **Status**: Implemented

---

## Reporting Decisions

### Summary.txt and Run Card Parity (v1.0, Jan 2026)
- **Decision**: Generate both from single source of truth functions
- **Rationale**: Eliminate drift, ensure consistency, reduce maintenance
- **Implementation**: `build_inference_summary_text()` and `build_inference_run_card_text()`
- **Status**: Implemented

### UTF-8 Encoding Everywhere (v1.2-v1.3, Jan 2026)
- **Decision**: Explicit `encoding='utf-8'` for all file operations
- **Rationale**: Windows default (cp1252) breaks Unicode symbols (•, ×, ≈)
- **Implementation**: 13 file operations in gui.py, all reporting.py writes
- **Status**: Implemented

### Adaptive Window Estimation (v1.3, Jan 2026)
- **Decision**: Replace fixed 0.40-0.60 × sqrt(N) with adaptive 3-range approach
- **Rationale**: Fixed window missed factors for skewed semiprimes (e.g., N=77)
- **Implementation**: Based on near_square_score (high/low/balanced)
- **Trade-off**: More complex, but better coverage
- **Status**: Implemented

### Honest Verifier Failure Messaging (v1.3, Jan 2026)
- **Decision**: Use "no valid factors found within inferred window" instead of "only trivial factorization possible"
- **Rationale**: Old message was misleading (implied mathematical impossibility, not search failure)
- **Implementation**: Single failure reason, honest about window search
- **Status**: Implemented

### Entropy Delta Sign Convention (v1.3, Jan 2026)
- **Decision**: Delta = end - start (negative for reductions)
- **Rationale**: Standard convention, clearer labeling
- **Implementation**: Updated `format_entropy_change()`, label shows "Delta (end-start)"
- **Trade-off**: Negative deltas may confuse users; mitigated with positive reduction %
- **Status**: Implemented

---

## Control Flow Decisions

### Early-Stop Control (v1.3, Jan 2026)
- **Decision**: Add min_steps parameter and disable_early_stop checkbox
- **Rationale**: User sets max=50 but run stopped at 8 due to entropy_converged
- **Implementation**: Gate entropy_converged/policy_stalled behind min_steps check
- **Use case**: Debugging, forcing longer runs, ensuring minimum data collection
- **Status**: Implemented

### Termination Reason Priority (v1.0)
- **Decision**: Check termination conditions in specific order
- **Rationale**: invalid_input highest priority, then max_steps, then heuristics
- **Order**: invalid_input → max_steps → confidence_reached → entropy_converged → policy_stalled
- **Status**: Implemented

---

## Interpretation Decisions

### Confidence Threshold for Claims (v1.3, Jan 2026)
- **Decision**: Only make near-square/skew claims if confidence >= 0.05
- **Rationale**: Low-confidence runs should not make strong claims
- **Implementation**: Added CONFIDENCE_THRESHOLD constant, guard in generate_interpretation()
- **Wording**: Changed to "Under current constraints, target may..." (tentative)
- **Status**: Implemented

### Small-N Hygiene Note (v1.2, Jan 2026)
- **Decision**: Add interpretation note for N < 256 (8 bits)
- **Rationale**: Users expect inference on N=15, but system has no signal at that scale
- **Implementation**: SMALL_N_THRESHOLD = 8, add note to interpretation
- **Wording**: "N is very small; limited structural inference expected at this scale"
- **Status**: Implemented

---

## Verification Decisions

### Primality Enforcement in Verifier (v1.2, Jan 2026)
- **Decision**: Verifier must check p >= 2, q >= 2, is_prime(p), is_prime(q)
- **Rationale**: Verifier was reporting p=1, q=N as success (trivial factorization)
- **Implementation**: Added is_prime() checks in both search passes
- **Trade-off**: More checks per candidate, but correctness guaranteed
- **Status**: Implemented

### Verifier as Optional (Foundational)
- **Decision**: Verification is opt-in, not automatic
- **Rationale**: For very large N, verification may be infeasible; user chooses
- **Implementation**: Checkbox in GUI, config['options']['verify_factors']
- **Status**: Implemented

---

## Performance Decisions

### Prime Generation Timeout (v1.1, Dec 2025)
- **Decision**: 8-second timeout and 10,000 attempt limit for prime generation
- **Rationale**: Ranges with few primes (e.g., [1000000, 1000010]) could hang forever
- **Implementation**: Rejection sampling with Miller-Rabin, time.time() check
- **Status**: Implemented

### Max Checks in Verifier (v1.0)
- **Decision**: Default 100,000 divisibility checks
- **Rationale**: Balance thoroughness vs. time for large windows
- **Configurable**: Yes (max_checks parameter)
- **Status**: Implemented

---

## Rejected Alternatives

### Dynamic BeliefState Size
- **Rejected**: Allow BeliefState to grow with more tokens
- **Reason**: Breaks fixed RL observation space, complicates training
- **Alternative Chosen**: Fixed 13-token representation

### Trial Division as Transform
- **Rejected**: Add "TryFactor(k)" transform
- **Reason**: Would be cheating, defeats purpose of constraint-based inference
- **Alternative Chosen**: Verifier is separate optional step

### Multiple Window Estimates
- **Rejected**: Store multiple size ratio hypotheses
- **Reason**: Complicates belief state, unclear how to merge
- **Alternative Chosen**: Single size_ratio_estimate with confidence

### Enum-Based Termination Reasons
- **Rejected**: Use Python Enum for termination reasons
- **Reason**: String literals are simpler, easier to serialize
- **Alternative Chosen**: String constants ("max_steps", "entropy_converged", etc.)

---

## Future Decisions Needed

### Policy Integration
- **Question**: How to load learned policy in Inference Tab?
- **Options**: Load from JSON, retrain on demand, use pre-trained checkpoint
- **Status**: Not yet decided

### Domain Selector UI
- **Question**: Dropdown vs tabs vs separate apps?
- **Options**: Combobox, multi-tab with domain tabs, launcher menu
- **Status**: Not yet decided

### Residue Ordering in Verifier
- **Question**: Should verifier prioritize residues by belief weights?
- **Options**: Use belief state weights, fixed ordering, random
- **Status**: Currently uses top 3 from belief state

---

**Last Updated**: January 5, 2026
