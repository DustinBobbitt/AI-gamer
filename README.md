# AI Gamer - Multi-Domain Meta-Learning System

A meta-learning platform that learns transferable reasoning skills across multiple problem domains, with a focus on arithmetic factor inference for semiprimes and optional toy game domains for validation.

## Overview

This system learns constraint-based inference patterns and discovers which internal structures transfer across domains. It implements a self-improving loop that:

1. **Learns how to infer constraints**, not just enumerate solutions
2. **Transfers knowledge** across problem domains while compressing redundant patterns
3. **Measures learning progress** using importance-weighted metrics
4. **Focuses on semiprime factorization** using belief state inference (no trial division)

## Primary Domain: Arithmetic Factor Inference

The system learns to infer factors of semiprimes (N = p×q) through **constraint discovery** rather than exhaustive search:

- **No trial division loops** - learns patterns instead
- **BeliefState representation** - fixed-size token vector (entropy, residue weights, confidence)
- **Transform-based inference** - applies deterministic constraint updates
- **Reward: Information gain** - entropy reduction drives learning
- **Optional verification** - narrow down factor window for final extraction

## Core Architecture

- **Brain** (`brain.py`) - Shared neural policy (θ) trained across domains
- **Domain System** (`domains/`) - Pluggable problem domains:
  - **Arithmetic** (`domains/arithmetic/`) - Semiprime factor inference (primary)
  - **Toy Games** (`domains/toy_games/`) - TicTacToe, NumberGuessing (validation)
- **Skill Memory** (`skill_memory.py`) - Explicit storage of transferable transform sequences
- **Consolidator** (`consolidator.py`) - Meta-system that learns safe knowledge compression
- **Meta-Learning Loop** (`meta_learning_loop.py`) - Main orchestrator across domains

## Learning Loop

1. **Acquisition Phase** - Train on scenarios without pruning (arithmetic or toy domains)
2. **Importance Estimation** - Compute importance metrics (gradient-based, not Hessian Fisher)
3. **Consolidation Phase** - Propose compression via pruning, merging, or quantization
4. **Regression Testing** - Verify performance on held-out scenarios
5. **Correction & Learning** - Restore if needed, learn which consolidation strategies work

## Mathematical Principles

- Brain is a parameter vector **θ**
- Importance represented by diagonal metric **G** (gradient-based, Fisher-inspired)
- Importance-weighted drift measured by **Δθᵀ G Δθ**
- Consolidation is constrained optimization: reduce complexity while bounded drift
- **Arithmetic Domain**: Reward = entropy reduction (Shannon entropy of belief state)

## Installation

```bash
# Clone the repository
git clone https://github.com/DustinBobbitt/AI-gamer.git
cd AI-gamer

# Install dependencies (requires Python 3.10+)
pip install numpy

# Run the GUI
python __main__.py
```

## Usage

### GUI Mode

Launch the graphical interface:

```bash
python __main__.py
```

Configure meta-learning parameters:
- **Training Phases** - Number of acquisition-consolidation cycles
- **Games per Phase** - How many games to learn per phase
- **Episodes per Game** - Training iterations per game
- **Consolidation Frequency** - How often to compress knowledge
- **Target Compression** - Desired parameter reduction ratio

### CLI Mode

Run training from command line:

```python
from meta_learning_loop import MetaLearningLoop, TrainingConfig

config = TrainingConfig(
    initial_games=50,
    episodes_per_game=100,
    consolidation_frequency=10,
    target_compression=0.3
)

loop = MetaLearningLoop(config)
summary = loop.run(total_phases=10)
```

## Key Features

### Transferable Learning
- Skills learned in one game transfer to new games
- Explicit skill memory tracks what works across domains
- Importance metrics identify truly general knowledge

### Learned Forgetting
- System learns which parameters can be safely pruned
- Consolidator improves over time through meta-learning
- Multiple strategies: pruning, merging, compression, distillation

### Safe Consolidation
- Regression suite prevents catastrophic forgetting
- Behavioral distance measured in importance metric
- Automatic restoration from checkpoints if needed

### Evaluation Metrics
- Zero-shot and few-shot performance on new games
- Transfer gain vs training from scratch
- Worst-case regression after consolidation
- Compression ratio vs retained competence
- Skill reuse across unrelated games

## Project Structure

```
├── __init__.py              # Package metadata
├── __main__.py              # Entry point
├── brain.py                 # Core neural model
├── game_generator.py        # Game creation framework
├── skill_memory.py          # Skill storage and retrieval
├── consolidator.py          # Knowledge compression
├── meta_learning_loop.py    # Main training orchestrator
├── gui.py                   # Graphical interface
├── agents/                  # Legacy agent implementations
├── utils/                   # Utility functions
└── README.md               # This file
```

## Goals

Build a general game-learning system that:
- Acquires reasoning skills through diverse games
- Learns how to compress its own "brain" safely
- Improves future learning efficiency by learning what not to keep
- Can be stress-tested on symbolic and mathematical games to probe generality

## License

MIT License - see LICENSE file for details

## Author

Dustin Bobbitt

## Version

2.0.0 - Self-Improving Game-Learning AI with Learned Forgetting
