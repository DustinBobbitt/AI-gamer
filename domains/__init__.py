"""
Domain abstraction layer for multi-domain meta-learning.

Domains provide different problem classes for the brain to learn:
- Toy Games: TicTacToe, NumberGuessing (for sanity checks)
- Arithmetic: Semiprime factor inference (primary domain)
"""
from domains.base import DomainTask, ScenarioConfig, DomainMetrics

__all__ = ['DomainTask', 'ScenarioConfig', 'DomainMetrics']
