# Self-Improving Game-Learning AI

A meta-learning system that learns to win diverse games while discovering how to safely compress and retain knowledge through learned forgetting.

## Overview

This system learns transferable reasoning skills across many games and discovers which internal structures are truly general. It implements a self-improving loop that:

1. **Learns how to learn games**, not just solve individual ones
2. **Retains useful logic** across games while discarding redundant knowledge
3. **Measures learning progress** and transfer across increasingly difficult games
4. **Applies meta-skills** to complex symbolic and mathematical tasks

## Core Architecture

- **Brain** (`brain.py`) - Shared neural policy/value model (θ) trained across many games
- **Game Generator** (`game_generator.py`) - DSL framework for creating diverse rule-based games
- **Skill Memory** (`skill_memory.py`) - Explicit storage of skills, heuristics, and sub-policies
- **Consolidator** (`consolidator.py`) - Meta-system that learns how to safely compress knowledge
- **Meta-Learning Loop** (`meta_learning_loop.py`) - Main orchestrator of the learning cycle

## Learning Loop

1. **Acquisition Phase** - Train on games without pruning, keep all learned logic
2. **Importance Estimation** - Compute importance metrics using Fisher information
3. **Consolidation Phase** - Propose compression via pruning, merging, or distillation
4. **Regression Testing** - Verify performance on held-out game suite
5. **Correction & Learning** - Restore if needed, learn from consolidation results

## Mathematical Framing

- Brain is a parameter vector **θ**
- Importance represented by positive semidefinite matrix **G**
- Behavioral change measured by **Δθᵀ G Δθ**
- Consolidation is constrained optimization: reduce complexity while staying close in G-metric
- Forgetting treated as projection in parameter space, not hard deletion

## Installation

```bash
# Clone the repository
git clone https://github.com/DustinBobbitt/Semiprime-Conjecture-Engine.git
cd Semiprime-Conjecture-Engine

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
