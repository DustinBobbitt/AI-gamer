"""
Reporting and summary structures for arithmetic factor inference runs.
"""

from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import json


@dataclass
class InferenceSummary:
    """
    Structured summary of an inference run, capturing belief state,
    termination reason, and key metrics.
    """
    # Termination
    termination_reason: str  # entropy_converged, max_steps, confidence_reached, policy_stalled, invalid_input
    steps_taken: int
    
    # Entropy metrics
    entropy_start: float
    entropy_end: float
    entropy_delta: float
    entropy_pct_reduction: float
    
    # Belief state metrics
    confidence: Optional[float]  # 0..1 or None
    near_square_score: float  # 0..1
    
    # Size window (smaller factor estimate)
    size_window_low: Optional[float]  # Lower bound estimate, or None
    size_window_high: Optional[float]  # Upper bound estimate, or None
    
    # Residue preferences (mod 30)
    top_residues_mod30: List[Tuple[int, float]]  # [(residue, weight), ...] top 3
    
    # Interpretation
    interpretation_lines: List[str]  # 1-3 lines of cautious interpretation
    
    # Optional target info
    target_n: Optional[int] = None
    bit_length: Optional[int] = None
    
    # Optional config info (for display)
    max_steps: Optional[int] = None
    min_steps: Optional[int] = None
    early_stop_enabled: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InferenceSummary':
        """Deserialize from dictionary."""
        return cls(**data)
    
    def get_near_square_label(self) -> str:
        """Convert near_square_score to human label."""
        if self.near_square_score is None:
            return "UNKNOWN"
        if self.near_square_score < 0.3:
            return "LOW"
        elif self.near_square_score < 0.7:
            return "MED"
        else:
            return "HIGH"
    
    def format_entropy_change(self) -> str:
        """Format entropy change as readable string with consistent sign convention."""
        if self.entropy_start is None or self.entropy_end is None:
            return "n/a"
        # Delta = end - start (negative for reductions)
        delta_val = self.entropy_end - self.entropy_start
        delta_str = f"{delta_val:+.3f}"
        pct_str = f"{self.entropy_pct_reduction:.1f}%" if self.entropy_pct_reduction is not None else "0.0%"
        return f"{self.entropy_start:.3f} → {self.entropy_end:.3f} (Δ{delta_str}, {pct_str})"
    
    def format_size_window(self, target_n: Optional[int] = None) -> str:
        """Format size window as readable string with clear labeling."""
        if self.size_window_low is None or self.size_window_high is None:
            return "n/a"
        
        # Format absolute range
        result = f"~ [{self.size_window_low:.1f}, {self.size_window_high:.1f}]"
        
        # Add relative to sqrt(N) if target_n provided
        if target_n is not None and target_n > 0:
            import math
            sqrt_n = math.sqrt(target_n)
            low_ratio = self.size_window_low / sqrt_n
            high_ratio = self.size_window_high / sqrt_n
            result += f" (≈ {low_ratio:.2f}–{high_ratio:.2f} × sqrt(N))"
        
        return result
    
    def format_residues(self) -> str:
        """Format top residues as readable string."""
        if not self.top_residues_mod30:
            return "n/a"
        parts = [f"{r}:{w:.2f}" for r, w in self.top_residues_mod30[:3]]
        return ", ".join(parts)


@dataclass
class VerificationResult:
    """
    Result of optional post-inference verification.
    """
    factors_found: bool
    p: Optional[int] = None
    q: Optional[int] = None
    
    # Cost metrics
    window_width: Optional[int] = None  # Number of candidates in search space
    checks_attempted: int = 0  # Actual divisibility checks performed
    time_ms: float = 0.0
    
    # Status
    verifier_skipped: bool = False
    skip_reason: Optional[str] = None
    failure_reason: Optional[str] = None  # Why verification failed (e.g., "only trivial factorization")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerificationResult':
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class BaselineComparison:
    """
    Optional baseline comparison metrics.
    """
    method_name: str  # e.g., "Fermat near-square", "Trial division B=10000"
    checks_attempted: int
    time_ms: float
    factors_found: bool
    p: Optional[int] = None
    q: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaselineComparison':
        """Deserialize from dictionary."""
        return cls(**data)


def generate_interpretation(summary: InferenceSummary) -> List[str]:
    """
    Generate cautious interpretation lines based on inference summary.
    
    Returns:
        List of 1-3 interpretation lines (cautious language).
    """
    SMALL_N_THRESHOLD = 8  # bit_length threshold for small-N behavior
    CONFIDENCE_THRESHOLD = 0.05  # Minimum confidence for making claims
    lines = []
    
    # Entropy interpretation (check for None to avoid comparison errors)
    if summary.entropy_pct_reduction is not None:
        if summary.entropy_pct_reduction > 50:
            lines.append("Inference significantly narrowed the search space (>50% entropy reduction).")
        elif summary.entropy_pct_reduction > 20:
            lines.append("Inference moderately narrowed the search space (20-50% entropy reduction).")
        else:
            lines.append("Inference made minimal progress (< 20% entropy reduction).")
    else:
        lines.append("Entropy reduction data not available.")
    
    # Small-N hygiene: explain expected behavior for very small N
    if summary.bit_length is not None and summary.bit_length <= SMALL_N_THRESHOLD:
        if summary.entropy_pct_reduction is not None and summary.entropy_pct_reduction < 5:
            lines.append("N is very small; limited structural inference is expected at this scale.")
    
    # Near-square / skew interpretation (only if confidence sufficient)
    if summary.near_square_score is not None and (summary.confidence is None or summary.confidence >= CONFIDENCE_THRESHOLD):
        ns_label = summary.get_near_square_label()
        if ns_label == "HIGH":
            lines.append("Under current constraints, target may have factors close in magnitude (near-square).")
        elif ns_label == "LOW":
            lines.append("Under current constraints, target may be skewed (factors differ significantly in size).")
    
    # Confidence interpretation (if available)
    if summary.confidence is not None:
        if summary.confidence > 0.8:
            lines.append(f"System expressed high confidence ({summary.confidence:.2f}) in belief state.")
        elif summary.confidence < 0.3:
            lines.append(f"System expressed low confidence ({summary.confidence:.2f}).")
    
    return lines


def compute_adaptive_window(target_n: int, near_square_score: float, assume_odd: bool = True) -> Tuple[int, int]:
    """
    Compute adaptive smaller-factor window based on near_square_score.
    
    Args:
        target_n: The semiprime N
        near_square_score: Score in [0,1] indicating how close to square
        assume_odd: Whether to force odd bounds
    
    Returns:
        (low, high) bounds for smaller factor search
    """
    import math
    
    sqrt_n = math.sqrt(target_n)
    
    # Adaptive window based on near_square_score
    if near_square_score >= 0.70:
        # Near-square: factors close to sqrt(N)
        center = 0.95 * sqrt_n
        half_width = 0.20 * sqrt_n
    elif near_square_score <= 0.30:
        # Skewed: smaller factor much less than sqrt(N)
        center = 0.35 * sqrt_n
        half_width = 0.35 * sqrt_n  # Wider to include smaller factors
    else:
        # Medium: balanced window
        center = 0.60 * sqrt_n
        half_width = 0.30 * sqrt_n
    
    low = max(2, center - half_width)
    high = max(low + 1, center + half_width)
    
    # Clamp high to sqrt(N) since smaller factor <= sqrt(N)
    high = min(high, sqrt_n)
    
    # Convert to integers
    low_int = int(low)
    high_int = int(high)
    
    # Force odd bounds if assume_odd
    if assume_odd:
        if low_int % 2 == 0:
            low_int += 1
        if high_int % 2 == 0:
            high_int -= 1
    
    return low_int, max(low_int, high_int)


    return lines[:3]  # Max 3 lines


def build_inference_summary_text(summary: InferenceSummary, 
                                 verification: Optional[VerificationResult] = None,
                                 baseline: Optional[BaselineComparison] = None,
                                 config: Optional[Dict[str, Any]] = None) -> str:
    """
    Build human-readable summary text from structured data.
    This is the SINGLE SOURCE OF TRUTH for formatting summary information.
    
    Args:
        summary: Inference summary object
        verification: Optional verification result
        baseline: Optional baseline comparison
        config: Optional config dict for additional context
    
    Returns:
        Formatted summary text string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ARITHMETIC FACTOR INFERENCE RUN SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    
    # Target info
    if summary.target_n is not None:
        lines.append(f"Target N: {summary.target_n}")
        if summary.bit_length is not None:
            lines.append(f"Bit length: {summary.bit_length}")
        lines.append("")
    
    # Inference limits (if available)
    if summary.max_steps is not None:
        limit_parts = [f"Max steps: {summary.max_steps}"]
        if summary.min_steps is not None:
            limit_parts.append(f"Min steps: {summary.min_steps}")
        if summary.early_stop_enabled is not None:
            early_stop_str = "enabled" if summary.early_stop_enabled else "disabled"
            limit_parts.append(f"Early-stop: {early_stop_str}")
        lines.append(", ".join(limit_parts))
        lines.append("")
    
    # Termination
    lines.append(f"Termination reason: {summary.termination_reason}")
    lines.append(f"Steps taken: {summary.steps_taken}")
    lines.append("")
    
    # Entropy
    lines.append("ENTROPY METRICS")
    lines.append("-" * 70)
    lines.append(f"Start: {summary.entropy_start:.4f}")
    lines.append(f"End: {summary.entropy_end:.4f}")
    lines.append(f"Delta (end-start): {summary.entropy_delta:+.4f}")
    lines.append(f"Reduction: {summary.entropy_pct_reduction:.2f}%")
    lines.append("")
    
    # Belief state
    lines.append("BELIEF STATE")
    lines.append("-" * 70)
    lines.append(f"Confidence: {summary.confidence if summary.confidence is not None else 'n/a'}")
    lines.append(f"Near-square score: {summary.near_square_score:.3f} ({summary.get_near_square_label()})")
    lines.append(f"Estimated smaller factor magnitude:")
    lines.append(f"  {summary.format_size_window(summary.target_n)}")
    lines.append(f"Top residues (mod 30): {summary.format_residues()}")
    lines.append("")
    
    # Interpretation
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    for line in summary.interpretation_lines:
        lines.append(f"• {line}")
    lines.append("")
    
    # Verification (if available)
    if verification is not None:
        lines.append("VERIFICATION RESULT")
        lines.append("-" * 70)
        if verification.verifier_skipped:
            lines.append(f"Verifier skipped: {verification.skip_reason}")
        else:
            lines.append(f"Factors found: {'YES' if verification.factors_found else 'NO'}")
            if verification.factors_found and verification.p is not None:
                lines.append(f"p = {verification.p}")
                lines.append(f"q = {verification.q}")
            elif not verification.factors_found and verification.failure_reason:
                lines.append(f"Reason: {verification.failure_reason}")
            lines.append(f"Window width: {verification.window_width}")
            lines.append(f"Checks attempted: {verification.checks_attempted}")
            lines.append(f"Time: {verification.time_ms:.2f} ms")
        lines.append("")
    
    # Baseline (if available)
    if baseline is not None:
        lines.append("BASELINE COMPARISON")
        lines.append("-" * 70)
        lines.append(f"Method: {baseline.method_name}")
        lines.append(f"Factors found: {'YES' if baseline.factors_found else 'NO'}")
        if baseline.factors_found:
            lines.append(f"p = {baseline.p}")
            lines.append(f"q = {baseline.q}")
        lines.append(f"Checks attempted: {baseline.checks_attempted}")
        lines.append(f"Time: {baseline.time_ms:.2f} ms")
        lines.append("")
    
    return "\n".join(lines)


def build_inference_run_card_text(summary: InferenceSummary,
                                  verification: Optional[VerificationResult] = None,
                                  baseline: Optional[BaselineComparison] = None,
                                  config: Optional[Dict[str, Any]] = None) -> str:
    """
    Build compact run card text from structured data.
    This is the SINGLE SOURCE OF TRUTH for formatting run card information.
    Uses same formatting rules as build_inference_summary_text but more compact.
    
    Args:
        summary: Inference summary object
        verification: Optional verification result
        baseline: Optional baseline comparison  
        config: Optional config dict for additional context
    
    Returns:
        Formatted run card text string
    """
    lines = []
    lines.append("INFERENCE RUN CARD")
    lines.append("=" * 70)
    lines.append("")
    
    # Target info
    if summary.target_n is not None:
        lines.append(f"Target N: {summary.target_n}")
        if summary.bit_length is not None:
            lines.append(f"Bit length: {summary.bit_length}")
        if config:
            assumptions = config.get('assumptions', {})
            lines.append(f"Assumptions: odd={assumptions.get('is_odd', True)}, semiprime={assumptions.get('is_semiprime', True)}")
        lines.append("")
    
    # Config limits
    if summary.max_steps is not None:
        limit_parts = [f"Max steps: {summary.max_steps}"]
        if summary.min_steps is not None:
            limit_parts.append(f"Min steps: {summary.min_steps}")
        if summary.early_stop_enabled is not None:
            early_stop_str = "enabled" if summary.early_stop_enabled else "disabled"
            limit_parts.append(f"Early-stop: {early_stop_str}")
        lines.append(", ".join(limit_parts))
        lines.append("")
    
    # Belief state summary
    lines.append("BELIEF STATE SUMMARY")
    lines.append("-" * 70)
    lines.append(f"Termination reason: {summary.termination_reason}")
    lines.append(f"Steps taken: {summary.steps_taken}")
    lines.append(f"Entropy: {summary.format_entropy_change()}")
    lines.append(f"Confidence: {summary.confidence:.3f}" if summary.confidence is not None else "Confidence: n/a")
    lines.append(f"Near-square score: {summary.near_square_score:.3f} ({summary.get_near_square_label()})")
    lines.append(f"Estimated smaller factor magnitude:")
    lines.append(f"  {summary.format_size_window(summary.target_n)}")
    lines.append(f"Top residues (mod 30): {summary.format_residues()}")
    lines.append("")
    
    # Interpretation
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    for line in summary.interpretation_lines:
        lines.append(f"• {line}")
    lines.append("")
    
    # Verification (if available)
    if verification is not None:
        lines.append("VERIFICATION RESULT")
        lines.append("-" * 70)
        if verification.verifier_skipped:
            lines.append(f"Verifier skipped: {verification.skip_reason}")
        else:
            lines.append(f"Factors found: {'YES' if verification.factors_found else 'NO'}")
            if verification.factors_found and verification.p is not None:
                lines.append(f"p = {verification.p}")
                lines.append(f"q = {verification.q}")
            elif not verification.factors_found and verification.failure_reason:
                lines.append(f"Reason: {verification.failure_reason}")
            lines.append(f"Window width: {verification.window_width}, Checks: {verification.checks_attempted}, Time: {verification.time_ms:.2f} ms")
        lines.append("")
    
    # Baseline (if available)
    if baseline is not None:
        lines.append("BASELINE COMPARISON")
        lines.append("-" * 70)
        lines.append(f"{baseline.method_name}: {'YES' if baseline.factors_found else 'NO'} ({baseline.checks_attempted} checks, {baseline.time_ms:.2f} ms)")
        if baseline.factors_found:
            lines.append(f"Factors: {baseline.p} × {baseline.q}")
        lines.append("")
    
    return "\n".join(lines)


def write_summary_txt(summary: InferenceSummary, filepath: str, 
                      verification: Optional[VerificationResult] = None,
                      baseline: Optional[BaselineComparison] = None,
                      config: Optional[Dict[str, Any]] = None) -> None:
    """
    Write human-readable summary.txt file using single source of truth function.
    
    Args:
        summary: Inference summary to write
        filepath: Path to summary.txt file
        verification: Optional verification result
        baseline: Optional baseline comparison
        config: Optional config dict
    """
    text = build_inference_summary_text(summary, verification, baseline, config)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)


def write_run_card_txt(summary: InferenceSummary, filepath: str,
                       verification: Optional[VerificationResult] = None,
                       baseline: Optional[BaselineComparison] = None,
                       config: Optional[Dict[str, Any]] = None) -> None:
    """
    Write compact run_card.txt file using single source of truth function.
    
    Args:
        summary: Inference summary to write
        filepath: Path to run_card.txt file
        verification: Optional verification result
        baseline: Optional baseline comparison
        config: Optional config dict
    """
    text = build_inference_run_card_text(summary, verification, baseline, config)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
                f.write(f"Window width: {verification.window_width if verification.window_width else 'n/a'}\n")
                f.write(f"Checks attempted: {verification.checks_attempted}\n")
                f.write(f"Time: {verification.time_ms:.2f} ms\n\n")
        
        # Baseline (if available)
        if baseline is not None:
            f.write("BASELINE COMPARISON\n")
            f.write("-" * 70 + "\n")
            f.write(f"Method: {baseline.method_name}\n")
            f.write(f"Factors found: {'YES' if baseline.factors_found else 'NO'}\n")
            if baseline.factors_found and baseline.p is not None:
                f.write(f"p = {baseline.p}\n")
                f.write(f"q = {baseline.q}\n")
            f.write(f"Checks attempted: {baseline.checks_attempted}\n")
            f.write(f"Time: {baseline.time_ms:.2f} ms\n\n")
        
        f.write("=" * 70 + "\n")
