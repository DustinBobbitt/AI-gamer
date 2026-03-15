"""
Meta-Learning Loop - The main training loop that orchestrates acquisition,
consolidation, and testing phases.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import random
from datetime import datetime

import numpy as np

from brain import Brain
from game_generator import Game, GameGenerator, GameDifficulty
from skill_memory import SkillMemory, Skill
from consolidator import Consolidator, ConsolidationResult

MAX_SEED = 2**32 - 1


@dataclass
class TrainingConfig:
    """Configuration for meta-learning loop."""

    # Overall run
    total_phases: int = 5
    game_family: str = "Mixed"
    consolidation_strategy: Optional[str] = None
    
    # Acquisition phase
    initial_games: int = 50
    episodes_per_game: int = 100
    
    # Consolidation phase
    consolidation_frequency: int = 10  # Every N games
    target_compression: float = 0.3
    
    # Testing
    regression_suite_size: int = 20
    worst_case_threshold: float = 0.1
    
    # Learning
    brain_hidden_size: int = 256
    learning_rate: float = 0.001
    seed: Optional[int] = None
    
    # Output
    output_dir: Path = Path("meta_learning_runs")


@dataclass
class PhaseMetrics:
    """Metrics for a training phase."""
    
    phase_name: str
    games_completed: int
    total_episodes: int
    avg_reward: float
    best_reward: float
    skills_learned: int
    consolidations: int
    phase_duration: float  # seconds


class MetaLearningLoop:
    """
    The main orchestrator for the self-improving game-learning system.
    
    Phases:
    1. Acquisition: Learn many games without pruning
    2. Importance Estimation: Compute Fisher information
    3. Consolidation: Propose and test compression
    4. Regression Testing: Verify no performance loss
    5. Correction: Restore if needed, learn from results
    """
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.config.output_dir = Path(self.config.output_dir)
        if self.config.seed is not None:
            if self.config.seed < 0 or self.config.seed > MAX_SEED:
                raise ValueError(f"Seed must be between 0 and {MAX_SEED}")
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
        
        # Core components
        self.brain = Brain(hidden_size=self.config.brain_hidden_size)
        self.game_generator = GameGenerator()
        self.skill_memory = SkillMemory(storage_path=self.config.output_dir / "skills")
        self.consolidator = Consolidator()
        
        # State
        self.current_episode = 0
        self.games_seen: List[str] = []
        self.regression_suite: List[Game] = []
        self.phase_history: List[PhaseMetrics] = []
        
        # Cold storage for parameter restoration
        self.parameter_checkpoints: List[Brain] = []
    
    def run(self, total_phases: Optional[int] = None, progress_callback=None) -> Dict[str, Any]:
        """
        Run the complete meta-learning loop.
        
        Args:
            total_phases: Number of acquisition-consolidation cycles
            progress_callback: Optional callback for progress updates
        
        Returns:
            Dictionary with training summary
        """
        run_start = datetime.now()
        active_total_phases = total_phases or self.config.total_phases
        
        for phase_num in range(active_total_phases):
            if progress_callback:
                progress_callback(phase_num + 1, active_total_phases, f"Starting phase {phase_num + 1}")
            
            # Phase 1: Acquisition
            acquisition_metrics = self._acquisition_phase(
                phase_num, 
                active_total_phases,
                progress_callback
            )
            self.phase_history.append(acquisition_metrics)
            
            # Phase 2: Importance Estimation
            if progress_callback:
                progress_callback(phase_num + 1, active_total_phases, "Estimating importance")
            self._importance_estimation_phase()
            
            # Phase 3: Consolidation (every N phases)
            if phase_num > 0 and phase_num % self.config.consolidation_frequency == 0:
                if progress_callback:
                    progress_callback(phase_num + 1, active_total_phases, "Consolidating knowledge")
                consolidation_result = self._consolidation_phase(progress_callback)
                
                # Phase 4: Regression Testing
                if progress_callback:
                    progress_callback(phase_num + 1, active_total_phases, "Regression testing")
                self._regression_testing_phase(consolidation_result)
                
                # Phase 5: Correction & Learning
                if progress_callback:
                    progress_callback(phase_num + 1, active_total_phases, "Learning from consolidation")
                self._correction_phase(consolidation_result)
        
        run_duration = (datetime.now() - run_start).total_seconds()
        
        return self._generate_summary(run_duration)
    
    def _acquisition_phase(
        self,
        phase_num: int,
        total_phases: int,
        progress_callback=None,
    ) -> PhaseMetrics:
        """
        Acquisition Phase: Train on new games without pruning.
        """
        phase_start = datetime.now()
        
        base_games, remainder = divmod(self.config.initial_games, total_phases)
        games_this_phase = base_games + (1 if phase_num < remainder else 0)
        total_reward = 0.0
        best_reward = float('-inf')
        skills_learned = 0
        total_episodes = 0
        
        for game_idx in range(games_this_phase):
            # Generate new game
            game = self.game_generator.generate_game(family=self.config.game_family)
            self.games_seen.append(game.name)
            
            # Train on this game
            for episode in range(self.config.episodes_per_game):
                state = game.reset()
                episode_reward = 0.0
                done = False
                
                while not done:
                    # Brain selects action
                    action = self.brain.predict_action(state)
                    
                    # Execute action
                    next_state, reward, done, info = game.step(action)
                    episode_reward += reward
                    
                    # Update brain
                    experience = {
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state,
                        'done': done
                    }
                    self.brain.update(experience)
                    
                    state = next_state
                
                total_reward += episode_reward
                best_reward = max(best_reward, episode_reward)
                total_episodes += 1
                self.current_episode += 1
                
                # Extract skills periodically
                if episode % 20 == 0:
                    skill = self._extract_skill(game, episode_reward)
                    if skill:
                        self.skill_memory.add_skill(skill)
                        skills_learned += 1
            
            # Add successful games to regression suite
            if len(self.regression_suite) < self.config.regression_suite_size:
                self.regression_suite.append(game)
            
            if progress_callback:
                progress_callback(
                    game_idx + 1,
                    games_this_phase, 
                    f"Game {game_idx + 1}/{games_this_phase}: {game.name}"
                )
        
        phase_duration = (datetime.now() - phase_start).total_seconds()
        avg_reward = total_reward / total_episodes if total_episodes > 0 else 0.0
        
        return PhaseMetrics(
            phase_name=f"Acquisition_{phase_num}",
            games_completed=games_this_phase,
            total_episodes=total_episodes,
            avg_reward=avg_reward,
            best_reward=best_reward,
            skills_learned=skills_learned,
            consolidations=0,
            phase_duration=phase_duration
        )
    
    def _importance_estimation_phase(self) -> None:
        """Compute importance matrix using Fisher information."""
        self.brain.estimate_importance()
        self.skill_memory.compute_importance_scores()
    
    def _consolidation_phase(self, progress_callback=None) -> ConsolidationResult:
        """
        Propose and apply consolidation.
        """
        # Save checkpoint before consolidation
        checkpoint = self.brain.clone()
        self.parameter_checkpoints.append(checkpoint)
        
        # Generate consolidation proposal
        proposal = self.consolidator.propose_consolidation(
            self.brain,
            target_compression=self.config.target_compression,
            strategy=self.config.consolidation_strategy,
        )
        
        # Apply consolidation
        consolidated_brain, behavioral_change = self.consolidator.apply_consolidation(
            self.brain,
            proposal
        )
        
        # Evaluate on regression suite
        result = self.consolidator.evaluate_consolidation(
            self.brain,
            consolidated_brain,
            self.regression_suite,
            proposal
        )
        
        # Accept if successful
        if result.accepted:
            self.brain = consolidated_brain
            if progress_callback:
                progress_callback(0, 1, f"Consolidation accepted: {proposal.operation_type}")
        else:
            if progress_callback:
                progress_callback(0, 1, f"Consolidation rejected: {proposal.operation_type}")
        
        return result
    
    def _regression_testing_phase(self, consolidation_result: ConsolidationResult) -> None:
        """
        Test consolidated brain on regression suite.
        """
        # Already done in consolidation evaluation
        # This phase could do more extensive testing
        pass
    
    def _correction_phase(self, consolidation_result: ConsolidationResult) -> None:
        """
        Handle restoration if needed and learn from consolidation result.
        """
        if consolidation_result.restoration_needed and self.parameter_checkpoints:
            # Restore from checkpoint
            self.brain = self.parameter_checkpoints[-1].clone()
            consolidation_result.restoration_needed = True
        
        # Learn from consolidation result
        self.consolidator.learn_from_consolidation(consolidation_result)
        
        # Consolidate skills
        skill_consolidation = self.skill_memory.consolidate_skills()
    
    def _extract_skill(self, game: Game, reward: float) -> Optional[Skill]:
        """Extract a skill from successful game experience."""
        if reward < 0.5:  # Only extract from successful episodes
            return None
        
        skill_id = f"skill_{self.current_episode}_{game.name}"
        
        return Skill(
            skill_id=skill_id,
            name=f"Strategy for {game.name}",
            description=f"Learned from episode {self.current_episode}",
            skill_type='sub_policy',
            parameters={
                'game_type': game.name,
                'reward': reward
            },
            times_used=1,
            success_rate=1.0,
            games_applied=[game.name]
        )
    
    def _generate_summary(self, run_duration: float) -> Dict[str, Any]:
        """Generate training summary."""
        return {
            'run_duration_seconds': run_duration,
            'total_episodes': self.current_episode,
            'total_games_seen': len(self.games_seen),
            'unique_games': len(set(self.games_seen)),
            'brain_version': self.brain.state.version,
            'consolidations_performed': self.brain.state.consolidation_count,
            'phase_history': [
                {
                    'phase': p.phase_name,
                    'games': p.games_completed,
                    'episodes': p.total_episodes,
                    'avg_reward': p.avg_reward,
                    'best_reward': p.best_reward,
                    'skills_learned': p.skills_learned,
                    'duration': p.phase_duration
                }
                for p in self.phase_history
            ],
            'skill_memory_stats': self.skill_memory.get_statistics(),
            'consolidator_stats': self.consolidator.get_statistics(),
            'final_brain_stats': {
                'total_parameters': sum(p.size for p in self.brain.state.parameters.values()),
                'has_importance_matrix': self.brain.state.importance_matrix is not None
            }
        }
    
    def save_state(self, path: Path) -> None:
        """Save complete system state."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save brain
        self.brain.state.save(path / "brain_state.npz")
        
        # Save skill memory
        self.skill_memory.save()
        
        # Save run summary
        summary = self._generate_summary(0)
        with open(path / "run_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
