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
from domains.arithmetic.reporting import VerificationResult


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
    
    # Compute search window based on belief
    sqrt_n = math.sqrt(N)
    
    if belief.near_square_score > 0.7:  # High near-square score
        # Search around sqrt(N)
        window_low = int(sqrt_n * 0.95)
        window_high = int(sqrt_n * 1.05)
    else:  # Skewed factors
        # Use size ratio to estimate smaller factor range
        estimated_small = sqrt_n * belief.size_ratio_estimate
        window_low = int(estimated_small * 0.8)
        window_high = int(estimated_small * 1.2)
    
    # Ensure odd candidates only
    if window_low % 2 == 0:
        window_low += 1
    if window_high % 2 == 0:
        window_high -= 1
    
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
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return VerificationResult(
                factors_found=True,
                p=min(p, q),
                q=max(p, q),
                window_width=window_width,
                checks_attempted=checks_attempted,
                time_ms=elapsed_ms
            )
    
    # No factors found
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return VerificationResult(
        factors_found=False,
        window_width=window_width,
        checks_attempted=checks_attempted,
        time_ms=elapsed_ms
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
