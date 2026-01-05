"""
NumberGuessing as a DomainTask.
"""
from __future__ import annotations

from typing import Dict, Any, List, Tuple
import random

from domains.base import DomainTask, ScenarioConfig, DomainMetrics, DomainDifficulty


class NumberGuessingDomain(DomainTask):
    """Number guessing game as a learning domain."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.domain_name = "NumberGuessing"
        self.target = None
        self.min_val = 1
        self.max_val = 100
        self.guesses = 0
        self.max_guesses = 10
        
    def make_scenario(self, rng: Any, difficulty: DomainDifficulty) -> ScenarioConfig:
        range_map = {
            DomainDifficulty.EASY: 50,
            DomainDifficulty.MEDIUM: 100,
            DomainDifficulty.HARD: 1000
        }
        max_val = range_map.get(difficulty, 100)
        
        return ScenarioConfig(
            domain_name=self.domain_name,
            difficulty=difficulty,
            scenario_id=f"numguess_{max_val}",
            parameters={'min_val': 1, 'max_val': max_val}
        )
    
    def reset(self, scenario_config: ScenarioConfig = None) -> Any:
        if scenario_config:
            self.max_val = scenario_config.parameters.get('max_val', 100)
        self.target = random.randint(self.min_val, self.max_val)
        self.guesses = 0
        self.current_scenario = scenario_config
        return self._get_observation()
    
    def _get_observation(self) -> List[float]:
        return [float(self.guesses), float(self.max_val), 0.0]
    
    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Make a guess."""
        self.guesses += 1
        
        if action == self.target:
            reward = 1.0 - (self.guesses / self.max_guesses)
            return self._get_observation(), reward, True, {'success': True}
        
        if self.guesses >= self.max_guesses:
            return self._get_observation(), -1.0, True, {'failure': True}
        
        feedback = 1 if action < self.target else -1
        obs = [float(self.guesses), float(self.max_val), float(feedback)]
        return obs, 0.0, False, {'feedback': feedback}
    
    def get_action_space(self) -> List[int]:
        return list(range(self.min_val, self.max_val + 1))
    
    def list_actions(self) -> List[str]:
        return [f"Guess {i}" for i in range(self.min_val, min(self.max_val + 1, 20))]
    
    def evaluate_policy(self, policy: Any, scenarios: List[ScenarioConfig]) -> DomainMetrics:
        total_reward = 0.0
        successes = 0
        
        for scenario in scenarios:
            obs = self.reset(scenario)
            done = False
            episode_reward = 0.0
            
            while not done:
                action = random.randint(self.min_val, self.max_val)
                obs, reward, done, info = self.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
            if info.get('success'):
                successes += 1
        
        return DomainMetrics(
            domain_name=self.domain_name,
            scenarios_completed=len(scenarios),
            total_episodes=len(scenarios),
            avg_reward=total_reward / len(scenarios) if scenarios else 0.0,
            best_reward=1.0,
            worst_reward=-1.0,
            success_rate=successes / len(scenarios) if scenarios else 0.0
        )
