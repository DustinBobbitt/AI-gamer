"""
Semiprime Inference Environment - Main RL environment for factor inference.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from domains.base import DomainTask, ScenarioConfig, DomainMetrics, DomainDifficulty
from domains.arithmetic.state import BeliefState
from domains.arithmetic.transforms import Transform, TransformLibrary, TransformResult
from domains.arithmetic.scenarios import SemiprimeScenario, ScenarioGenerator, DistributionType
from domains.arithmetic.reporting import InferenceSummary, generate_interpretation


class SemiprimeInferenceEnv(DomainTask):
    """
    Environment for learning to infer semiprime factors through belief state updates.
    
    Key principles:
    - No trial division loops
    - No candidate enumeration
    - No direct N % k probing as primary mechanism
    - Constraint-based inference only
    - Reward based on entropy reduction
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.domain_name = "ArithmeticFactorInference"
        
        # Configuration
        self.max_steps = config.get('max_steps', 50) if config else 50
        self.bit_length = config.get('bit_length', 32) if config else 32
        self.distribution_type = config.get('distribution_type', 'mixed') if config else 'mixed'
        
        # Current episode state
        self.current_N: Optional[int] = None
        self.true_p: Optional[int] = None
        self.true_q: Optional[int] = None
        self.belief_state: Optional[BeliefState] = None
        self.step_count: int = 0
        
        # Transform library
        self.transforms = TransformLibrary.get_all_transforms()
        
        # Scenario generator
        self.scenario_gen = ScenarioGenerator()
        
        # Episode history
        self.entropy_history: List[float] = []
        self.reward_history: List[float] = []
        self.transform_sequence: List[str] = []
        
        # Termination tracking
        self.termination_reason: str = ""
        self.convergence_threshold: float = 0.001  # Entropy change threshold
        self.convergence_window: int = 5  # Steps to check for convergence
        self.entropy_start: float = 0.0
    
    def make_scenario(self, rng: Any, difficulty: DomainDifficulty) -> ScenarioConfig:
        """Generate a new scenario configuration."""
        # Map difficulty to bit length
        bit_length_map = {
            DomainDifficulty.TRIVIAL: 16,
            DomainDifficulty.EASY: 20,
            DomainDifficulty.MEDIUM: 32,
            DomainDifficulty.HARD: 48,
            DomainDifficulty.EXPERT: 64
        }
        bit_length = bit_length_map.get(difficulty, 32)
        
        return ScenarioConfig(
            domain_name=self.domain_name,
            difficulty=difficulty,
            scenario_id=f"semiprime_{bit_length}bit",
            parameters={
                'bit_length': bit_length,
                'distribution_type': self.distribution_type
            }
        )
    
    def reset(self, scenario_config: ScenarioConfig = None) -> Any:
        """
        Reset to initial state for a new scenario.
        
        Returns:
            observation: BeliefState vector for the brain
        """
        # Generate scenario
        if scenario_config:
            scenario = self.scenario_gen.generate_from_config(scenario_config)
        else:
            scenario = self.scenario_gen.generate_mixed(self.bit_length)
        
        # Store ground truth (hidden from inference)
        self.current_N = scenario.N
        self.true_p = scenario.p
        self.true_q = scenario.q
        self.current_scenario = scenario_config
        
        # Initialize belief state
        self.belief_state = BeliefState()
        self.belief_state.update_entropy()
        
        # Reset episode tracking
        self.step_count = 0
        self.entropy_history = [self.belief_state.entropy_estimate]
        self.reward_history = []
        self.transform_sequence = []
        self.termination_reason = ""
        self.entropy_start = self.belief_state.entropy_estimate
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        Execute one inference step by applying a transform.
        
        Args:
            action: Transform ID to apply
        
        Returns:
            observation: New belief state
            reward: Entropy reduction (information gain)
            done: Whether max steps reached
            info: Additional information
        """
        if self.belief_state is None:
            raise RuntimeError("Must call reset() before step()")
        
        # Get transform and apply it
        transform = TransformLibrary.get_transform_by_id(action)
        prev_entropy = self.belief_state.entropy_estimate
        
        result = transform.apply(self.belief_state, self.current_N)
        self.belief_state = result.new_state
        
        # Compute reward as entropy reduction (information gain)
        new_entropy = self.belief_state.entropy_estimate
        entropy_reduction = prev_entropy - new_entropy
        reward = max(0.0, entropy_reduction)  # Reward for reducing uncertainty
        
        # Bonus for confidence increases
        if self.belief_state.confidence > 0.8:
            reward += 0.5
        
        # Track progress
        self.step_count += 1
        self.entropy_history.append(new_entropy)
        self.reward_history.append(reward)
        self.transform_sequence.append(transform.get_description())
        
        # Check termination conditions
        done, self.termination_reason = self._check_termination()
        
        # Compute evaluation metrics (using hidden ground truth)
        info = self._compute_info()
        info.update(result.info)
        info['transform_applied'] = transform.get_description()
        info['entropy_reduction'] = entropy_reduction
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation for the brain.
        
        Returns belief state vector plus lightweight N features.
        """
        belief_vec = self.belief_state.to_vector()
        
        # Add lightweight N features (no cheating)
        n_features = np.array([
            float(self.current_N.bit_length()) / 64.0,  # Normalized bit length
            float(self.current_N % 2),  # Parity
            float(self.current_N % 3) / 3.0,
            float(self.current_N % 5) / 5.0,
            np.log(self.current_N) / 50.0  # Normalized log
        ])
        
        observation = np.concatenate([belief_vec, n_features])
        return observation
    
    def _check_termination(self) -> Tuple[bool, str]:
        """Check termination conditions and return (done, reason)."""
        # Invalid input check
        if self.current_N is None or self.current_N <= 1:
            return True, "invalid_input"
        
        # Max steps reached
        if self.step_count >= self.max_steps:
            return True, "max_steps"
        
        # High confidence threshold
        if self.belief_state.confidence > 0.95:
            return True, "confidence_reached"
        
        # Entropy convergence check (last N steps have minimal change)
        if len(self.entropy_history) >= self.convergence_window:
            recent_entropies = self.entropy_history[-self.convergence_window:]
            entropy_range = max(recent_entropies) - min(recent_entropies)
            if entropy_range < self.convergence_threshold:
                return True, "entropy_converged"
        
        # Policy stalled check (no change in entropy for M consecutive steps)
        if len(self.entropy_history) >= 10:
            recent_entropies = self.entropy_history[-10:]
            if all(abs(e - recent_entropies[0]) < 1e-6 for e in recent_entropies):
                return True, "policy_stalled"
        
        return False, ""
    
    def _compute_info(self) -> Dict[str, Any]:
        """Compute evaluation metrics using ground truth (for reporting only)."""
        # Compute true size ratio
        true_ratio = min(self.true_p, self.true_q) / max(self.true_p, self.true_q)
        
        # Compute proximity to truth
        estimated_ratio = self.belief_state.size_ratio_estimate
        ratio_error = abs(estimated_ratio - true_ratio)
        
        return {
            'N': self.current_N,
            'true_p': self.true_p,
            'true_q': self.true_q,
            'true_ratio': true_ratio,
            'estimated_ratio': estimated_ratio,
            'ratio_error': ratio_error,
            'entropy': self.belief_state.entropy_estimate,
            'confidence': self.belief_state.confidence,
            'step': self.step_count,
            'termination_reason': self.termination_reason
        }
    
    def get_action_space(self) -> List[int]:
        """Return list of available transform IDs."""
        return list(range(len(self.transforms)))
    
    def list_actions(self) -> List[str]:
        """Return human-readable transform descriptions."""
        return [t.get_description() for t in self.transforms]
    
    def evaluate_policy(self, policy: Any, scenarios: List[ScenarioConfig]) -> DomainMetrics:
        """
        Evaluate policy on multiple scenarios.
        
        Args:
            policy: Callable that maps observation -> action
            scenarios: List of scenario configurations to test
        
        Returns:
            DomainMetrics with evaluation results
        """
        total_reward = 0.0
        best_reward = float('-inf')
        worst_reward = float('inf')
        success_count = 0
        
        for scenario in scenarios:
            obs = self.reset(scenario)
            done = False
            episode_reward = 0.0
            
            while not done:
                # Policy selects action
                if callable(policy):
                    action = policy(obs)
                else:
                    # Assume policy has predict_action method (Brain)
                    action = policy.predict_action(obs)
                
                obs, reward, done, info = self.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
            best_reward = max(best_reward, episode_reward)
            worst_reward = min(worst_reward, episode_reward)
            
            # Success if confidence > threshold
            if info.get('confidence', 0) > 0.8:
                success_count += 1
        
        return DomainMetrics(
            domain_name=self.domain_name,
            scenarios_completed=len(scenarios),
            total_episodes=len(scenarios),
            avg_reward=total_reward / len(scenarios) if scenarios else 0.0,
            best_reward=best_reward,
            worst_reward=worst_reward,
            success_rate=success_count / len(scenarios) if scenarios else 0.0,
            domain_specific={
                'avg_final_entropy': np.mean([h[-1] for h in [self.entropy_history]]),
                'avg_final_confidence': np.mean([info.get('confidence', 0)])
            }
        )
    
    def render(self) -> str:
        """Return string representation of current state."""
        if self.belief_state is None:
            return "ArithmeticFactorInference - Not initialized"
        
        return f"""ArithmeticFactorInference
N = {self.current_N} ({self.current_N.bit_length()} bits)
Step: {self.step_count}/{self.max_steps}
Entropy: {self.belief_state.entropy_estimate:.3f}
Confidence: {self.belief_state.confidence:.3f}
Size Ratio Est: {self.belief_state.size_ratio_estimate:.3f}
Near Square: {self.belief_state.near_square_score:.3f}
"""
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get complete episode summary for reporting."""
        return {
            'N': self.current_N,
            'bit_length': self.current_N.bit_length() if self.current_N else 0,
            'true_factors': [self.true_p, self.true_q],
            'steps': self.step_count,
            'entropy_history': self.entropy_history,
            'reward_history': self.reward_history,
            'transform_sequence': self.transform_sequence,
            'final_belief': self.belief_state.to_dict() if self.belief_state else {},
            'total_reward': sum(self.reward_history),
            'termination_reason': self.termination_reason
        }
    
    def generate_inference_summary(self) -> InferenceSummary:
        """Generate structured InferenceSummary for reporting."""
        if self.belief_state is None:
            raise RuntimeError("Cannot generate summary before episode completes")
        
        # Calculate entropy metrics
        entropy_end = self.belief_state.entropy_estimate
        entropy_delta = self.entropy_start - entropy_end
        entropy_pct = (entropy_delta / self.entropy_start * 100) if self.entropy_start > 0 else 0.0
        
        # Extract top residues
        residue_weights = [(i, w) for i, w in enumerate(self.belief_state.residue_weights_mod30)]
        residue_weights.sort(key=lambda x: x[1], reverse=True)
        valid_residues = [1, 7, 11, 13, 17, 19, 23, 29]
        top_residues = [(valid_residues[i], w) for i, w in residue_weights[:3]]
        
        # Estimate size window (heuristic based on size_ratio and N)
        sqrt_n = np.sqrt(self.current_N)
        if self.belief_state.size_ratio_estimate > 0.8:  # Near-square
            window_low = sqrt_n * 0.9
            window_high = sqrt_n * 1.1
        else:  # Skewed
            window_low = sqrt_n * self.belief_state.size_ratio_estimate * 0.8
            window_high = sqrt_n * self.belief_state.size_ratio_estimate * 1.2
        
        # Create summary
        summary = InferenceSummary(
            termination_reason=self.termination_reason,
            steps_taken=self.step_count,
            entropy_start=self.entropy_start,
            entropy_end=entropy_end,
            entropy_delta=entropy_delta,
            entropy_pct_reduction=entropy_pct,
            confidence=self.belief_state.confidence,
            near_square_score=self.belief_state.near_square_score,
            size_window_low=window_low,
            size_window_high=window_high,
            top_residues_mod30=top_residues,
            interpretation_lines=[],  # Will be filled next
            target_n=self.current_N,
            bit_length=self.current_N.bit_length() if self.current_N else 0
        )
        
        # Generate interpretation
        summary.interpretation_lines = generate_interpretation(summary)
        
        return summary
