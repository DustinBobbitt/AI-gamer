"""
TicTacToe as a DomainTask (refactored from game_generator.py).
"""
from __future__ import annotations

from typing import Dict, Any, List, Tuple
import random

from domains.base import DomainTask, ScenarioConfig, DomainMetrics, DomainDifficulty


class TicTacToeDomain(DomainTask):
    """Classic 3x3 Tic-Tac-Toe as a learning domain."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.domain_name = "TicTacToe"
        self.board = None
        self.current_player = 1
        
    def make_scenario(self, rng: Any, difficulty: DomainDifficulty) -> ScenarioConfig:
        return ScenarioConfig(
            domain_name=self.domain_name,
            difficulty=difficulty,
            scenario_id=f"tictactoe_{difficulty.name.lower()}",
            parameters={'board_size': 3}
        )
    
    def reset(self, scenario_config: ScenarioConfig = None) -> Any:
        self.board = [[0 for _ in range(3)] for _ in range(3)]
        self.current_player = 1
        self.current_scenario = scenario_config
        return self._get_observation()
    
    def _get_observation(self) -> List[int]:
        """Flatten board to 1D observation."""
        return [cell for row in self.board for cell in row]
    
    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Place move and check for win/draw."""
        if not (0 <= action < 9):
            return self._get_observation(), -1.0, False, {'invalid_action': True}
        
        row, col = action // 3, action % 3
        
        # Invalid move
        if self.board[row][col] != 0:
            return self._get_observation(), -1.0, False, {'invalid_action': True}
        
        # Place move
        self.board[row][col] = self.current_player
        
        # Check win
        winner = self._check_winner()
        if winner:
            reward = 1.0 if winner == 1 else -1.0
            return self._get_observation(), reward, True, {'winner': winner}
        
        # Check draw
        if all(self.board[r][c] != 0 for r in range(3) for c in range(3)):
            return self._get_observation(), 0.0, True, {'draw': True}
        
        # Switch player
        self.current_player = 3 - self.current_player  # Toggle 1<->2
        
        return self._get_observation(), 0.0, False, {}
    
    def _check_winner(self) -> int:
        """Check if there's a winner. Returns player number or 0."""
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != 0:
                return row[0]
        
        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != 0:
                return self.board[0][col]
        
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            return self.board[0][2]
        
        return 0
    
    def get_action_space(self) -> List[int]:
        """Return list of empty positions."""
        return [r * 3 + c for r in range(3) for c in range(3) if self.board[r][c] == 0]
    
    def list_actions(self) -> List[str]:
        """Return human-readable action descriptions."""
        return [f"Place at ({a//3}, {a%3})" for a in range(9)]
    
    def evaluate_policy(self, policy: Any, scenarios: List[ScenarioConfig]) -> DomainMetrics:
        """Evaluate policy on multiple TicTacToe games."""
        total_reward = 0.0
        wins = 0
        
        for scenario in scenarios:
            obs = self.reset(scenario)
            done = False
            episode_reward = 0.0
            
            while not done:
                legal = self.get_action_space()
                if not legal:
                    break
                
                action = random.choice(legal)  # Simple policy for now
                obs, reward, done, info = self.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
            if episode_reward > 0:
                wins += 1
        
        return DomainMetrics(
            domain_name=self.domain_name,
            scenarios_completed=len(scenarios),
            total_episodes=len(scenarios),
            avg_reward=total_reward / len(scenarios) if scenarios else 0.0,
            best_reward=1.0,
            worst_reward=-1.0,
            success_rate=wins / len(scenarios) if scenarios else 0.0
        )
