"""
Skill Memory - Explicit storage of learned skills, sub-policies, and heuristics.

The skill memory allows the system to explicitly store and reuse successful
strategies across different games, enabling transfer learning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import pickle


@dataclass
class Skill:
    """
    Represents a learned skill or sub-policy.
    
    A skill encapsulates a reusable piece of knowledge that can transfer
    across games.
    """
    
    skill_id: str
    name: str
    description: str
    skill_type: str  # 'heuristic', 'sub_policy', 'search_guide', 'meta_strategy'
    
    # Core skill data
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Usage statistics
    times_used: int = 0
    success_rate: float = 0.0
    games_applied: List[str] = field(default_factory=list)
    
    # Importance metrics
    importance_score: float = 0.0
    last_used_episode: int = 0
    
    # Transfer metrics
    transfer_success: Dict[str, float] = field(default_factory=dict)  # game_name -> success_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize skill to dictionary."""
        return {
            'skill_id': self.skill_id,
            'name': self.name,
            'description': self.description,
            'skill_type': self.skill_type,
            'parameters': self.parameters,
            'times_used': self.times_used,
            'success_rate': self.success_rate,
            'games_applied': self.games_applied,
            'importance_score': self.importance_score,
            'last_used_episode': self.last_used_episode,
            'transfer_success': self.transfer_success
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        """Deserialize skill from dictionary."""
        return cls(**data)


class SkillMemory:
    """
    Manages the collection of learned skills.
    
    Responsibilities:
    - Store and retrieve skills
    - Track skill usage and importance
    - Facilitate skill transfer across games
    - Support skill consolidation and pruning
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("skills")
        self.skills: Dict[str, Skill] = {}
        self.skill_graph: Dict[str, List[str]] = {}  # Dependencies between skills
        
        # Load existing skills if available
        if self.storage_path.exists():
            self.load()
    
    def add_skill(self, skill: Skill) -> None:
        """Add a new skill to memory."""
        self.skills[skill.skill_id] = skill
        self._save_skill(skill)
    
    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Retrieve a skill by ID."""
        return self.skills.get(skill_id)
    
    def find_relevant_skills(self, game_name: str, context: Dict[str, Any] = None) -> List[Skill]:
        """
        Find skills that might be relevant for a given game.
        
        Uses transfer success history and skill characteristics.
        """
        relevant = []
        
        for skill in self.skills.values():
            # Check if skill was successful in similar games
            if game_name in skill.transfer_success:
                if skill.transfer_success[game_name] > 0.5:
                    relevant.append(skill)
            # Check if skill has high general importance
            elif skill.importance_score > 0.7:
                relevant.append(skill)
        
        # Sort by importance and success rate
        relevant.sort(key=lambda s: (s.importance_score, s.success_rate), reverse=True)
        return relevant
    
    def update_skill_stats(self, skill_id: str, success: bool, game_name: str, episode: int) -> None:
        """Update statistics after skill usage."""
        skill = self.skills.get(skill_id)
        if skill is None:
            return
        
        skill.times_used += 1
        skill.last_used_episode = episode
        
        # Update success rate with exponential moving average
        alpha = 0.1
        skill.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * skill.success_rate
        
        # Track game-specific transfer
        if game_name not in skill.transfer_success:
            skill.transfer_success[game_name] = 0.0
        skill.transfer_success[game_name] = alpha * (1.0 if success else 0.0) + (1 - alpha) * skill.transfer_success[game_name]
        
        # Add to games applied
        if game_name not in skill.games_applied:
            skill.games_applied.append(game_name)
        
        self._save_skill(skill)
    
    def consolidate_skills(self, importance_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Consolidate skills by merging similar ones and pruning low-importance skills.
        
        Returns:
            Dictionary with consolidation metrics
        """
        initial_count = len(self.skills)
        pruned_skills = []
        
        # Identify low-importance skills
        for skill_id, skill in list(self.skills.items()):
            if skill.importance_score < importance_threshold and skill.times_used < 5:
                pruned_skills.append(skill_id)
                del self.skills[skill_id]
        
        # TODO: Merge similar skills
        # This would involve:
        # - Computing skill similarity metrics
        # - Identifying clusters of similar skills
        # - Creating merged skill representations
        
        return {
            'initial_count': initial_count,
            'final_count': len(self.skills),
            'pruned_count': len(pruned_skills),
            'pruned_skills': pruned_skills,
            'compression_ratio': 1.0 - len(self.skills) / max(initial_count, 1)
        }
    
    def compute_importance_scores(self) -> None:
        """
        Recompute importance scores for all skills based on usage patterns.
        
        Importance considers:
        - Recency of use
        - Success rate
        - Transfer breadth (number of games)
        - General applicability
        """
        for skill in self.skills.values():
            # Recency component (decay over time)
            # Success rate component
            # Transfer breadth component
            transfer_breadth = len(skill.games_applied)
            
            importance = (
                0.3 * skill.success_rate +
                0.3 * min(1.0, transfer_breadth / 5.0) +
                0.4 * min(1.0, skill.times_used / 100.0)
            )
            
            skill.importance_score = importance
    
    def save(self) -> None:
        """Save all skills to disk."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Save skill index
        index_data = {
            'skill_ids': list(self.skills.keys()),
            'skill_graph': self.skill_graph
        }
        with open(self.storage_path / 'index.json', 'w') as f:
            json.dump(index_data, f, indent=2)
        
        # Save individual skills
        for skill in self.skills.values():
            self._save_skill(skill)
    
    def _save_skill(self, skill: Skill) -> None:
        """Save individual skill to disk."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        skill_path = self.storage_path / f"{skill.skill_id}.json"
        with open(skill_path, 'w') as f:
            json.dump(skill.to_dict(), f, indent=2)
    
    def load(self) -> None:
        """Load skills from disk."""
        index_path = self.storage_path / 'index.json'
        if not index_path.exists():
            return
        
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        self.skill_graph = index_data.get('skill_graph', {})
        
        # Load individual skills
        for skill_id in index_data.get('skill_ids', []):
            skill_path = self.storage_path / f"{skill_id}.json"
            if skill_path.exists():
                with open(skill_path, 'r') as f:
                    skill_data = json.load(f)
                self.skills[skill_id] = Skill.from_dict(skill_data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics about skill memory."""
        if not self.skills:
            return {'total_skills': 0}
        
        return {
            'total_skills': len(self.skills),
            'avg_importance': sum(s.importance_score for s in self.skills.values()) / len(self.skills),
            'avg_success_rate': sum(s.success_rate for s in self.skills.values()) / len(self.skills),
            'total_games_covered': len(set(game for s in self.skills.values() for game in s.games_applied)),
            'skill_types': {
                skill_type: sum(1 for s in self.skills.values() if s.skill_type == skill_type)
                for skill_type in set(s.skill_type for s in self.skills.values())
            }
        }
