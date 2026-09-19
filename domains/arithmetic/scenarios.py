"""
Scenario generation for semiprime inference domain.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum

from domains.base import ScenarioConfig, DomainDifficulty
from utils.primes import is_prime, generate_prime


class DistributionType(Enum):
    """Type of semiprime distribution."""
    BALANCED = "balanced"  # p ≈ q (near-square)
    INTERMEDIATE = "intermediate"  # moderate factor ratio
    SKEWED = "skewed"  # p << q
    MIXED = "mixed"  # random mix


@dataclass
class SemiprimeScenario:
    """Configuration for a semiprime factorization scenario."""
    
    N: int  # The semiprime target
    p: int  # True factor (kept hidden during inference)
    q: int  # True factor (kept hidden during inference)
    bit_length: int
    distribution_type: DistributionType
    difficulty: DomainDifficulty


class ScenarioGenerator:
    """Generates semiprime scenarios with various characteristics."""
    
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)
    
    def generate_balanced(self, bit_length: int) -> SemiprimeScenario:
        """
        Generate balanced semiprime where p ≈ q (near-square).
        
        Args:
            bit_length: Target bit length for N
        """
        # Target: p and q both around 2^(bit_length/2)
        half_bits = bit_length // 2
        p_low = 2 ** (half_bits - 1)
        p_high = 2 ** half_bits
        
        # Generate p
        p = self.rng.randint(p_low, p_high)
        while not is_prime(p):
            p = self.rng.randint(p_low, p_high)
        
        # Generate q close to p
        q_low = int(p * 0.9)
        q_high = int(p * 1.1)
        q = self.rng.randint(max(q_low, 3), q_high)
        while not is_prime(q) or q == p:
            q = self.rng.randint(max(q_low, 3), q_high)
        
        N = p * q
        actual_bits = N.bit_length()
        
        difficulty = self._classify_difficulty(bit_length)
        
        return SemiprimeScenario(
            N=N,
            p=min(p, q),
            q=max(p, q),
            bit_length=actual_bits,
            distribution_type=DistributionType.BALANCED,
            difficulty=difficulty
        )
    
    def generate_skewed(self, bit_length: int) -> SemiprimeScenario:
        """
        Generate skewed semiprime where p << q.
        
        Args:
            bit_length: Target bit length for N
        """
        # p is small (8-16 bits), q is large
        p_bits = min(bit_length // 4, 16)
        q_bits = bit_length - p_bits
        
        p_low = 2 ** (p_bits - 1)
        p_high = 2 ** p_bits
        q_low = 2 ** (q_bits - 1)
        q_high = 2 ** q_bits
        
        p = self.rng.randint(p_low, p_high)
        while not is_prime(p):
            p = self.rng.randint(p_low, p_high)
        
        q = self.rng.randint(q_low, q_high)
        while not is_prime(q) or q == p:
            q = self.rng.randint(q_low, q_high)
        
        N = p * q
        actual_bits = N.bit_length()
        
        difficulty = self._classify_difficulty(bit_length)
        
        return SemiprimeScenario(
            N=N,
            p=min(p, q),
            q=max(p, q),
            bit_length=actual_bits,
            distribution_type=DistributionType.SKEWED,
            difficulty=difficulty
        )

    def generate_intermediate(self, bit_length: int) -> SemiprimeScenario:
        """Generate a semiprime with a moderate factor ratio (2 <= q/p <= 8)."""
        half_bits = bit_length // 2
        p_low = 2 ** max(1, half_bits - 1)
        p_high = 2 ** half_bits
        q_low = 2 ** half_bits
        q_high = 2 ** (half_bits + 1)

        while True:
            p = self.rng.randint(p_low, p_high)
            while not is_prime(p):
                p = self.rng.randint(p_low, p_high)
            q = self.rng.randint(q_low, q_high)
            while not is_prime(q) or q == p:
                q = self.rng.randint(q_low, q_high)
            ratio = max(p, q) / min(p, q)
            if 2.0 <= ratio <= 8.0:
                break

        N = p * q
        return SemiprimeScenario(
            N=N,
            p=min(p, q),
            q=max(p, q),
            bit_length=N.bit_length(),
            distribution_type=DistributionType.INTERMEDIATE,
            difficulty=self._classify_difficulty(bit_length),
        )
    
    def generate_mixed(self, bit_length: int) -> SemiprimeScenario:
        """Generate random mix of balanced and skewed."""
        if self.rng.random() < 0.5:
            return self.generate_balanced(bit_length)
        else:
            return self.generate_skewed(bit_length)
    
    def generate_from_config(self, config: ScenarioConfig) -> SemiprimeScenario:
        """Generate from a scenario configuration."""
        params = config.parameters
        bit_length = params.get('bit_length', 32)
        dist_type = params.get('distribution_type', 'mixed')
        
        if dist_type == 'balanced':
            return self.generate_balanced(bit_length)
        elif dist_type == 'intermediate':
            return self.generate_intermediate(bit_length)
        elif dist_type == 'skewed':
            return self.generate_skewed(bit_length)
        else:
            return self.generate_mixed(bit_length)
    
    def _classify_difficulty(self, bit_length: int) -> DomainDifficulty:
        """Classify difficulty based on bit length."""
        if bit_length < 20:
            return DomainDifficulty.EASY
        elif bit_length < 32:
            return DomainDifficulty.MEDIUM
        elif bit_length < 48:
            return DomainDifficulty.HARD
        else:
            return DomainDifficulty.EXPERT
    
    def generate_batch(self, count: int, bit_length: int, distribution: str = 'mixed') -> List[SemiprimeScenario]:
        """Generate batch of scenarios."""
        scenarios = []
        for _ in range(count):
            if distribution == 'balanced':
                scenario = self.generate_balanced(bit_length)
            elif distribution == 'intermediate':
                scenario = self.generate_intermediate(bit_length)
            elif distribution == 'skewed':
                scenario = self.generate_skewed(bit_length)
            else:
                scenario = self.generate_mixed(bit_length)
            scenarios.append(scenario)
        return scenarios
