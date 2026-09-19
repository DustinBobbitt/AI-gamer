"""Tests for state-aware arithmetic transform selection."""
import unittest

from domains.arithmetic.env import SemiprimeInferenceEnv
from domains.arithmetic.policy import AdaptiveInferencePolicy
from domains.arithmetic.state import BeliefState
from domains.arithmetic.verifier import verify_factors_from_belief


def initialized_env(target_n: int, **config) -> SemiprimeInferenceEnv:
    env = SemiprimeInferenceEnv({"max_steps": 200, **config})
    env.current_N = target_n
    env.belief_state = BeliefState()
    env.belief_state.update_entropy()
    env.entropy_history = [env.belief_state.entropy_estimate]
    env.entropy_start = env.belief_state.entropy_estimate
    return env


class AdaptiveInferencePolicyTests(unittest.TestCase):
    def test_feature_transforms_run_once_then_policy_reaches_fixed_point(self) -> None:
        env = initialized_env(77, disable_early_stop=True)
        policy = AdaptiveInferencePolicy(min_progress=1e-6)
        actions = []

        while len(actions) < env.max_steps:
            action = policy.select_action(env.belief_state, env.current_N)
            if action is None:
                env.finish()
                break
            actions.append(action)
            env.step(action)

        self.assertEqual(actions[:3], [0, 1, 2])
        self.assertLess(len(actions), 20)
        self.assertEqual(env.termination_reason, "policy_complete")
        self.assertEqual(actions.count(0), 1)
        self.assertEqual(actions.count(1), 1)
        self.assertEqual(actions.count(2), 1)

    def test_configured_convergence_threshold_is_used(self) -> None:
        env = initialized_env(77, convergence_threshold=0.25)
        self.assertEqual(env.convergence_threshold, 0.25)

    def test_verifier_searches_the_reported_adaptive_window(self) -> None:
        env = initialized_env(77, disable_early_stop=True)
        policy = AdaptiveInferencePolicy(min_progress=1e-4)
        while (action := policy.select_action(env.belief_state, env.current_N)) is not None:
            env.step(action)

        result = verify_factors_from_belief(77, env.belief_state)

        self.assertTrue(result.factors_found)
        self.assertEqual((result.p, result.q), (7, 11))


if __name__ == "__main__":
    unittest.main()
