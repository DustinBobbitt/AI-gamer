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
        if self.near_square_score < 0.3:
            return "LOW"
        elif self.near_square_score < 0.7:
            return "MED"
        else:
            return "HIGH"
    
    def format_entropy_change(self) -> str:
        """Format entropy change as readable string."""
        return f"{self.entropy_start:.3f} → {self.entropy_end:.3f} (Δ{self.entropy_delta:+.3f}, {self.entropy_pct_reduction:.1f}%)"
    
    def format_size_window(self) -> str:
        """Format size window as readable string."""
        if self.size_window_low is None or self.size_window_high is None:
            return "n/a"
        return f"[{self.size_window_low:.2e}, {self.size_window_high:.2e}]"
    
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
    lines = []
    
    # Entropy interpretation
    if summary.entropy_pct_reduction > 50:
        lines.append("Inference significantly narrowed the search space (>50% entropy reduction).")
    elif summary.entropy_pct_reduction > 20:
        lines.append("Inference moderately narrowed the search space (20-50% entropy reduction).")
    else:
        lines.append("Inference made minimal progress (< 20% entropy reduction).")
    
    # Near-square interpretation
    ns_label = summary.get_near_square_label()
    if ns_label == "HIGH":
        lines.append("Target appears near-square (factors likely similar size).")
    elif ns_label == "LOW":
        lines.append("Target appears skewed (factors likely different sizes).")
    
    # Confidence interpretation (if available)
    if summary.confidence is not None:
        if summary.confidence > 0.8:
            lines.append(f"System expressed high confidence ({summary.confidence:.2f}) in belief state.")
        elif summary.confidence < 0.3:
            lines.append(f"System expressed low confidence ({summary.confidence:.2f}), further constraints may help.")
    
    return lines[:3]  # Max 3 lines


def write_summary_txt(summary: InferenceSummary, filepath: str, 
                      verification: Optional[VerificationResult] = None,
                      baseline: Optional[BaselineComparison] = None) -> None:
    """
    Write human-readable summary.txt file.
    
    Args:
        summary: Inference summary to write
        filepath: Path to summary.txt file
        verification: Optional verification result
        baseline: Optional baseline comparison
    """
    with open(filepath, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ARITHMETIC FACTOR INFERENCE RUN SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        # Target info
        if summary.target_n is not None:
            f.write(f"Target N: {summary.target_n}\n")
            if summary.bit_length is not None:
                f.write(f"Bit length: {summary.bit_length}\n")
            f.write("\n")
        
        # Termination
        f.write(f"Termination reason: {summary.termination_reason}\n")
        f.write(f"Steps taken: {summary.steps_taken}\n\n")
        
        # Entropy
        f.write("ENTROPY METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Start: {summary.entropy_start:.4f}\n")
        f.write(f"End: {summary.entropy_end:.4f}\n")
        f.write(f"Delta: {summary.entropy_delta:+.4f}\n")
        f.write(f"Reduction: {summary.entropy_pct_reduction:.2f}%\n\n")
        
        # Belief state
        f.write("BELIEF STATE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Confidence: {summary.confidence if summary.confidence is not None else 'n/a'}\n")
        f.write(f"Near-square score: {summary.near_square_score:.3f} ({summary.get_near_square_label()})\n")
        f.write(f"Size window: {summary.format_size_window()}\n")
        f.write(f"Top residues (mod 30): {summary.format_residues()}\n\n")
        
        # Interpretation
        f.write("INTERPRETATION\n")
        f.write("-" * 70 + "\n")
        for line in summary.interpretation_lines:
            f.write(f"• {line}\n")
        f.write("\n")
        
        # Verification (if available)
        if verification is not None:
            f.write("VERIFICATION RESULT\n")
            f.write("-" * 70 + "\n")
            if verification.verifier_skipped:
                f.write(f"Verifier skipped: {verification.skip_reason}\n\n")
            else:
                f.write(f"Factors found: {'YES' if verification.factors_found else 'NO'}\n")
                if verification.factors_found and verification.p is not None:
                    f.write(f"p = {verification.p}\n")
                    f.write(f"q = {verification.q}\n")
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
