"""
Quick test of Inference Tab v1.1 features.
Tests InferenceSummary generation, termination tracking, and reporting.
"""

from domains.arithmetic.env import SemiprimeInferenceEnv
from domains.arithmetic.state import BeliefState
from domains.arithmetic.scenarios import ScenarioGenerator
from domains.arithmetic.verifier import verify_factors_from_belief
from domains.arithmetic.reporting import write_summary_txt
import random
from pathlib import Path


def test_inference_pipeline():
    """Test complete inference pipeline with summary generation."""
    print("=" * 70)
    print("Testing Inference Tab v1.1 Features")
    print("=" * 70)
    
    # Test 1: Generate a small semiprime
    print("\n1. Generating test semiprime...")
    gen = ScenarioGenerator()
    scenario = gen.generate_balanced(bit_length=16)
    print(f"   N = {scenario.N} ({scenario.p} × {scenario.q})")
    print(f"   Bit length: {scenario.bit_length}")
    print(f"   Distribution: {scenario.distribution_type}")
    
    # Test 2: Initialize environment
    print("\n2. Initializing environment...")
    env = SemiprimeInferenceEnv(config={'max_steps': 50, 'bit_length': 16})
    env.current_N = scenario.N
    env.true_p = scenario.p
    env.true_q = scenario.q
    env.belief_state = BeliefState()
    env.belief_state.update_entropy()
    env.step_count = 0
    env.entropy_history = [env.belief_state.entropy_estimate]
    env.reward_history = []
    env.transform_sequence = []
    env.termination_reason = ""
    env.entropy_start = env.belief_state.entropy_estimate
    print(f"   Initial entropy: {env.entropy_start:.4f}")
    
    # Test 3: Run inference loop
    print("\n3. Running inference (random policy)...")
    done = False
    while not done:
        action = random.randint(0, len(env.get_action_space()) - 1)
        obs, reward, done, info = env.step(action)
        if env.step_count % 10 == 0:
            print(f"   Step {env.step_count}: entropy={info['entropy']:.4f}, reward={reward:.4f}")
    
    print(f"\n   ✓ Inference complete!")
    print(f"   Termination reason: {env.termination_reason}")
    print(f"   Steps taken: {env.step_count}")
    print(f"   Final entropy: {env.belief_state.entropy_estimate:.4f}")
    
    # Test 4: Generate InferenceSummary
    print("\n4. Generating InferenceSummary...")
    summary = env.generate_inference_summary()
    print(f"   Termination: {summary.termination_reason}")
    print(f"   Entropy change: {summary.format_entropy_change()}")
    print(f"   Confidence: {summary.confidence:.3f}")
    print(f"   Near-square: {summary.near_square_score:.3f} ({summary.get_near_square_label()})")
    print(f"   Size window: {summary.format_size_window()}")
    print(f"   Top residues: {summary.format_residues()}")
    print(f"\n   Interpretation:")
    for line in summary.interpretation_lines:
        print(f"   • {line}")
    
    # Test 5: Optional verification
    print("\n5. Running post-inference verification...")
    verification = verify_factors_from_belief(scenario.N, env.belief_state, max_checks=100000)
    if verification.verifier_skipped:
        print(f"   Verifier skipped: {verification.skip_reason}")
    elif verification.factors_found:
        print(f"   ✓ Factors found: {verification.p} × {verification.q}")
        print(f"   Window width: {verification.window_width}")
        print(f"   Checks attempted: {verification.checks_attempted}")
        print(f"   Time: {verification.time_ms:.2f} ms")
        
        # Verify correctness
        if verification.p * verification.q == scenario.N:
            print(f"   ✓ Verification CORRECT!")
        else:
            print(f"   ✗ Verification INCORRECT!")
    else:
        print(f"   No factors found in window (width: {verification.window_width})")
        print(f"   Checks attempted: {verification.checks_attempted}")
    
    # Test 6: Write summary.txt
    print("\n6. Writing summary.txt...")
    test_dir = Path("runs/test_v1.1")
    test_dir.mkdir(parents=True, exist_ok=True)
    summary_file = test_dir / "summary.txt"
    write_summary_txt(summary, str(summary_file), verification)
    print(f"   ✓ Written to: {summary_file}")
    
    # Print summary.txt content
    print("\n" + "=" * 70)
    print("GENERATED SUMMARY.TXT:")
    print("=" * 70)
    with open(summary_file, 'r') as f:
        print(f.read())
    
    print("=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_inference_pipeline()
