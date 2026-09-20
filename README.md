# Semiprime Conjecture Engine

An empirically evaluated arithmetic reasoning workbench with calibrated
factor-geometry beliefs, bounded verification strategies, and retained legacy
toy-game experiments.

## Overview

The primary arithmetic path separates two responsibilities:

1. **Inference** collects only validated factor-blind evidence and reports a
   calibrated balanced/intermediate/skewed posterior.
2. **Verification** uses a transparent, bounded strategy portfolio to attempt
   factor extraction after inference.
3. **Offline training** selects verifier budgets from deterministic training
   corpora and records validation metrics and manifest hashes.
4. **Benchmarks** compare every accepted change with fixed, random, held-out,
   and out-of-distribution baselines.

## Primary Domain: Arithmetic Factor Inference

The current accepted inference evidence is deliberately narrow:

- Exact-perfect-square evidence is used only when the user assumes a semiprime
  and explicitly allows `p=q`.
- Ordinary targets retain an honest calibrated prior when available
  factor-blind evidence cannot identify factor geometry.
- Residue sharpening, smoothness scores, and proximity to an unrelated integer
  square are not treated as calibrated factor evidence.
- Transform entropy is retained only as a legacy scheduling diagnostic.
- Optional verification is separate and reports its strategy trace and work.

## Core Architecture

- **Geometry policy** (`domains/arithmetic/geometry.py`) - Calibrated factor
  geometry posterior and exact-square evidence
- **Verifier policy** (`domains/arithmetic/verifier_policy.py`) - Offline-trained
  per-bit strategy budgets
- **Verifier** (`domains/arithmetic/verifier.py`) - Bounded Fermat, low-factor,
  Pollard-Rho, and legacy fallback strategies
- **Domain System** (`domains/`) - Pluggable problem domains:
  - **Arithmetic** (`domains/arithmetic/`) - Semiprime factor inference (primary)
  - **Toy Games** (`domains/toy_games/`) - TicTacToe, NumberGuessing (validation)
- **Legacy experimental stack** (`brain.py`, `meta_learning_loop.py`,
  `skill_memory.py`, `consolidator.py`) - Preserved toy-game prototypes; these
  placeholder components are not used as evidence of arithmetic learning.

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

### Empirical policy training

The post-inference verifier uses a frozen, schema-versioned budget policy trained
on deterministic balanced and skewed semiprime corpora. Factor labels are used
only by the offline trainer and evaluator; the runtime policy selects a bounded
verification budget from the target bit length.

```bash
# Reproduce the policy checkpoint and its validation metrics
python3 train_verifier_policy.py

# Reproduce calibrated balanced/intermediate/skewed geometry beliefs
python3 train_geometry_policy.py

# Compare learned, fixed-budget, window-only, and inference-policy baselines
python3 benchmark_arithmetic.py --count-per-group 50
```

The checkpoint records training and validation manifest hashes, seeds, and
per-bit success/check metrics in `domains/arithmetic/verifier_policy.json`.
Inference transforms and verification remain separate, and benchmark results
must not attribute verifier gains to transform entropy.

The geometry checkpoint currently reports an indeterminate posterior because
the permitted factor-blind features did not predict factor balance above
chance. This is intentional: its held-out calibration is substantially better
than the retired square-gap confidence, and the GUI does not manufacture
certainty where the experiment found no discriminating evidence.

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
