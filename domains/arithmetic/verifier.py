"""
Post-inference factor verifier - SEPARATE from inference core.

This module provides optional verification that uses the narrowed belief window
from inference to attempt factor extraction. This is NOT part of the inference
process itself - it's a separate evaluation step.

Key principles:
- Runs ONLY after inference completes
- Uses only the size window and residue preferences from belief state
- Reports cost metrics (checks attempted, time)
- Clearly labeled as optional evaluation, not core inference
"""

import time
import math
from typing import Optional, Tuple
from domains.arithmetic.state import BeliefState
from domains.arithmetic.reporting import VerificationResult, compute_adaptive_window


def verify_factors_from_belief(
    N: int,
    belief: BeliefState,
    max_checks: int = 100000
) -> VerificationResult:
    """
    Attempt to extract factors using narrowed window from belief state.
    
    This is NOT part of inference - it's a separate verification step that
    runs after inference completes.
    
    Args:
        N: Target semiprime
        belief: Final belief state from inference
        max_checks: Maximum divisibility checks to attempt
    
    Returns:
        VerificationResult with factors (if found) and cost metrics
    """
    start_time = time.perf_counter()
    
    # Check if we have a usable size window
    if belief.size_ratio_estimate <= 0 or belief.size_ratio_estimate > 1:
        return VerificationResult(
            factors_found=False,
            verifier_skipped=True,
            skip_reason="Invalid size ratio estimate from belief state"
        )
    
    # Use the same window shown in summaries and run cards. Keeping one source
    # of truth prevents the verifier from searching a narrower, hidden range.
    window_low, window_high = compute_adaptive_window(
        N,
        belief.near_square_score,
        assume_odd=True,
    )
    
    window_width = (window_high - window_low) // 2 + 1
    
    # Check if window is too large
    if window_width > max_checks:
        return VerificationResult(
            factors_found=False,
            verifier_skipped=True,
            skip_reason=f"Window too large ({window_width} > {max_checks} max checks)",
            window_width=window_width
        )
    
    # Get residue preferences from belief
    valid_residues = [1, 7, 11, 13, 17, 19, 23, 29]
    residue_weights = [(valid_residues[i], belief.residue_weights_mod30[i]) 
                       for i in range(len(valid_residues))]
    residue_weights.sort(key=lambda x: x[1], reverse=True)
    preferred_residues = [r for r, w in residue_weights if w > 0.05]  # Filter low-weight residues
    
    # Search for factors in window, prioritizing preferred residues
    checks_attempted = 0
    
    # First pass: check candidates matching preferred residues
    if preferred_residues:
        for candidate in range(window_low, window_high + 1, 2):
            if checks_attempted >= max_checks:
                break
            
            # Check if candidate matches preferred residue
            if (candidate % 30) not in preferred_residues:
                continue
            
            checks_attempted += 1
            if N % candidate == 0:
                p = candidate
                q = N // candidate
                
                # Verify both factors are >= 2 and prime (reject trivial factorizations)
                from utils.primes import is_prime
                if p < 2 or q < 2:
                    continue  # Skip trivial factorization
                if not is_prime(p) or not is_prime(q):
                    continue  # Skip non-prime factors
                
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return VerificationResult(
                    factors_found=True,
                    p=min(p, q),
                    q=max(p, q),
                    window_width=window_width,
                    checks_attempted=checks_attempted,
                    time_ms=elapsed_ms
                )
    
    # Second pass: check all candidates if preferred residues didn't work
    for candidate in range(window_low, window_high + 1, 2):
        if checks_attempted >= max_checks:
            break
        
        # Skip if already checked in first pass
        if preferred_residues and (candidate % 30) in preferred_residues:
            continue
        
        checks_attempted += 1
        if N % candidate == 0:
            p = candidate
            q = N // candidate
            
            # Verify both factors are >= 2 and prime (reject trivial factorizations)
            from utils.primes import is_prime
            if p < 2 or q < 2:
                continue  # Skip trivial factorization
            if not is_prime(p) or not is_prime(q):
                continue  # Skip non-prime factors
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return VerificationResult(
                factors_found=True,
                p=min(p, q),
                q=max(p, q),
                window_width=window_width,
                checks_attempted=checks_attempted,
                time_ms=elapsed_ms
            )
    
    # No valid prime factors found
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    # Honest failure reason - we searched the window and didn't find valid factors
    failure_reason = "no valid factors found within inferred window"
    
    return VerificationResult(
        factors_found=False,
        window_width=window_width,
        checks_attempted=checks_attempted,
        time_ms=elapsed_ms,
        failure_reason=failure_reason
    )


def verify_factors_adaptively(
    N: int,
    belief: BeliefState,
    max_checks: int = 100000,
    fermat_budget: int = 512,
) -> VerificationResult:
    """Verify with a cost-aware portfolio of complementary strategies.

    The portfolio is deliberately outside the inference environment. It first
    tries bounded Fermat checks for balanced factors, then a bounded low-factor
    search for skewed factors, and finally the inferred belief window. The
    trace makes every search decision and its cost visible.
    """
    start_time = time.perf_counter()
    checks = 0
    trace = []

    if N <= 1 or N % 2 == 0:
        return VerificationResult(
            factors_found=False,
            verifier_skipped=True,
            skip_reason="Adaptive verifier currently requires odd N > 1",
            strategy_used="adaptive_portfolio",
            strategy_trace=trace,
        )

    def success(candidate: int, strategy: str) -> Optional[VerificationResult]:
        from utils.primes import is_prime

        other = N // candidate
        if candidate < 2 or other < 2 or not is_prime(candidate) or not is_prime(other):
            return None
        return VerificationResult(
            factors_found=True,
            p=min(candidate, other),
            q=max(candidate, other),
            checks_attempted=checks,
            time_ms=(time.perf_counter() - start_time) * 1000,
            strategy_used=strategy,
            strategy_trace=trace.copy(),
        )

    # Balanced-factor hypothesis: bounded Fermat probes are cheap when p ≈ q.
    trace.append("fermat_probe")
    a = math.isqrt(N)
    if a * a < N:
        a += 1
    for _ in range(min(fermat_budget, max_checks - checks)):
        checks += 1
        b_squared = a * a - N
        b = math.isqrt(b_squared)
        if b * b == b_squared:
            result = success(a - b, "fermat_probe")
            if result:
                return result
        a += 1

    # Skewed-factor hypothesis: search only the generator-independent low
    # quarter-bit band, rather than pretending the near-square score identifies
    # factor balance.
    trace.append("low_factor_band")
    sqrt_n = math.isqrt(N)
    low_factor_high = min(sqrt_n, 1 << min(16, max(3, N.bit_length() // 4 + 1)))
    for candidate in range(3, low_factor_high + 1, 2):
        if checks >= max_checks:
            break
        checks += 1
        if N % candidate == 0:
            result = success(candidate, "low_factor_band")
            if result:
                return result

    # Preserve the belief-guided path for medium cases not covered above.
    trace.append("belief_window")
    if checks < max_checks:
        window_result = verify_factors_from_belief(
            N,
            belief,
            max_checks=max_checks - checks,
        )
        checks += window_result.checks_attempted
        if window_result.factors_found:
            window_result.checks_attempted = checks
            window_result.time_ms = (time.perf_counter() - start_time) * 1000
            window_result.strategy_used = "belief_window"
            window_result.strategy_trace = trace.copy()
            return window_result

    return VerificationResult(
        factors_found=False,
        checks_attempted=checks,
        time_ms=(time.perf_counter() - start_time) * 1000,
        failure_reason="adaptive verifier exhausted its bounded strategy portfolio",
        strategy_used="adaptive_portfolio",
        strategy_trace=trace,
    )


def run_baseline_fermat(N: int, max_iterations: int = 10000) -> Tuple[bool, Optional[int], Optional[int], int, float]:
    """
    Baseline comparison: Fermat's method for near-square semiprimes.
    
    Returns:
        (found, p, q, iterations, time_ms)
    """
    start_time = time.perf_counter()
    
    a = math.isqrt(N)
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
        a += 1
        b_squared = a * a - N
        
        if b_squared < 0:
            continue
        
        b = math.isqrt(b_squared)
        if b * b == b_squared:
            # Found factors
            p = a - b
            q = a + b
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return True, min(p, q), max(p, q), iterations, elapsed_ms
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return False, None, None, iterations, elapsed_ms


def run_baseline_trial_division(N: int, B: int = 10000) -> Tuple[bool, Optional[int], Optional[int], int, float]:
    """
    Baseline comparison: Small trial division up to bound B.
    
    Returns:
        (found, p, q, checks, time_ms)
    """
    start_time = time.perf_counter()
    
    checks = 0
    for candidate in range(3, min(B, int(math.sqrt(N)) + 1), 2):
        checks += 1
        if N % candidate == 0:
            p = candidate
            q = N // candidate
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return True, min(p, q), max(p, q), checks, elapsed_ms
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return False, None, None, checks, elapsed_ms
