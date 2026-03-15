"""
Game Generator - DSL and framework for creating diverse rule-based games.

This module generates many different games with varying complexity to train
the brain on diverse reasoning tasks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple, Dict
import random


class GameDifficulty(Enum):
    """Difficulty levels for generated games."""
    TRIVIAL = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5


@dataclass
class GameState:
    """Represents the current state of a game."""
    observation: Any
    legal_actions: List[int]
    done: bool = False
    winner: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Game(ABC):
    """Abstract base class for all games."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        self.difficulty = GameDifficulty.MEDIUM
        self.state = None
    
    @abstractmethod
    def reset(self) -> GameState:
        """Reset the game to initial state."""
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """
        Execute one game step.
        
        Returns:
            state: New game state
            reward: Reward for this step
            done: Whether game is over
            info: Additional information
        """
        pass
    
    @abstractmethod
    def render(self) -> str:
        """Return string representation of current state."""
        pass
    
    def get_legal_actions(self) -> List[int]:
        """Return list of legal actions in current state."""
        return self.state.legal_actions if self.state else []


class TicTacToe(Game):
    """Classic 3x3 Tic-Tac-Toe game."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.board = None
        self.current_player = 1
        self.difficulty = GameDifficulty.EASY
        self.reset()
    
    def reset(self) -> GameState:
        self.board = [[0 for _ in range(3)] for _ in range(3)]
        self.current_player = 1
        self.state = GameState(
            observation=self._get_observation(),
            legal_actions=list(range(9)),
            done=False
        )
        return self.state
    
    def _get_observation(self) -> List[int]:
        """Flatten board to 1D observation."""
        return [cell for row in self.board for cell in row]
    
    def step(self, action: int) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        row, col = action // 3, action % 3
        
        if self.board[row][col] != 0:
            # Illegal move
            return self.state, -1.0, True, {'illegal_move': True}
        
        self.board[row][col] = self.current_player
        
        # Check win
        winner = self._check_winner()
        done = winner is not None or all(cell != 0 for row in self.board for cell in row)
        
        reward = 0.0
        if winner == self.current_player:
            reward = 1.0
        elif winner is not None:
            reward = -1.0
        elif done:
            reward = 0.0  # Draw
        
        # Switch player
        self.current_player = 3 - self.current_player
        
        # Update legal actions
        legal_actions = [i for i in range(9) if self.board[i//3][i%3] == 0]
        
        self.state = GameState(
            observation=self._get_observation(),
            legal_actions=legal_actions,
            done=done,
            winner=winner
        )
        
        return self.state, reward, done, {}
    
    def _check_winner(self) -> Optional[int]:
        """Check if there's a winner. Returns player number or None."""
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
        
        return None
    
    def render(self) -> str:
        symbols = {0: '.', 1: 'X', 2: 'O'}
        lines = []
        for row in self.board:
            lines.append(' '.join(symbols[cell] for cell in row))
        return '\n'.join(lines)


class NumberGuessing(Game):
    """Simple number guessing game."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.max_number = config.get('max_number', 100) if config else 100
        self.max_attempts = 10
        self.target = None
        self.attempts = 0
        super().__init__(config)
        self.difficulty = GameDifficulty.TRIVIAL
        self.reset()
    
    def reset(self) -> GameState:
        self.target = random.randint(1, self.max_number)
        self.attempts = 0
        self.state = GameState(
            observation=[self.max_number, self.attempts, 0],  # [max, attempts, last_feedback]
            legal_actions=list(range(self.max_number + 1)),
            done=False
        )
        return self.state
    
    def step(self, action: int) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        self.attempts += 1
        
        if action == self.target:
            reward = 1.0
            done = True
            feedback = 0
        elif action < self.target:
            feedback = -1
            reward = -0.01
            done = self.attempts >= self.max_attempts
        else:
            feedback = 1
            reward = -0.01
            done = self.attempts >= self.max_attempts
        
        self.state = GameState(
            observation=[self.max_number, self.attempts, feedback],
            legal_actions=list(range(self.max_number + 1)),
            done=done
        )
        
        return self.state, reward, done, {'target': self.target}
    
    def render(self) -> str:
        return f"Guess a number between 1 and {self.max_number}. Attempts: {self.attempts}/{self.max_attempts}"


class GameGenerator:
    """Generates diverse games for training."""
    
    def __init__(self, allowed_families: Optional[List[str]] = None):
        self.game_classes = [
            TicTacToe,
            NumberGuessing,
        ]
        self._game_class_by_name = {
            game_class.__name__: game_class
            for game_class in self.game_classes
        }
        self.allowed_families = allowed_families or list(self._game_class_by_name.keys())
    
    def _candidate_classes(self, family: Optional[str] = None) -> List[type[Game]]:
        if family and family != "Mixed":
            game_class = self._game_class_by_name.get(family)
            if game_class is None:
                raise ValueError(f"Unknown game family: {family}")
            return [game_class]
        
        candidates = [
            self._game_class_by_name[name]
            for name in self.allowed_families
            if name in self._game_class_by_name
        ]
        if not candidates:
            raise ValueError("No game families are enabled for generation")
        return candidates
    
    def generate_game(self, difficulty: GameDifficulty = None, family: Optional[str] = None) -> Game:
        """Generate a random game, optionally filtered by difficulty."""
        candidate_classes = self._candidate_classes(family)
        game_class = random.choice(candidate_classes)
        game = game_class()
        
        if difficulty is not None:
            # Re-sample if difficulty doesn't match
            attempts = 0
            while game.difficulty != difficulty and attempts < 10:
                game_class = random.choice(candidate_classes)
                game = game_class()
                attempts += 1
        
        return game
    
    def generate_curriculum(self, num_games: int = 100) -> List[Game]:
        """
        Generate a curriculum of games with increasing difficulty.
        
        Returns list of games ordered by difficulty.
        """
        curriculum = []
        difficulties = list(GameDifficulty)
        games_per_difficulty = num_games // len(difficulties)
        
        for difficulty in difficulties:
            for _ in range(games_per_difficulty):
                game = self.generate_game(difficulty)
                curriculum.append(game)
        
        return curriculum
