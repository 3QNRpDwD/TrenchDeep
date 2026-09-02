# Trainer API Optimization

Updated: 2026-09-01

## Implemented

- Added `TrainableModel` and `CheckpointableModel`; paradigm traits now contain only computation contracts.
- Reduced `nn::Model` to an inference contract backed by shared capabilities.
- Added the typed `Trainer` facade, datasets, schedules, stop reasons, step units, and structured results.
- Replaced `StepInfo`/`last_*` with owned `StepOutput` observations and weighted loss aggregation.
- Metric hooks propagate errors; deterministic runtime RNG drives shuffle and RL sampling.
- Added `CheckpointManager`, schema versioning, typed paths, temporary-file commit, and error propagation.
- Reserved optimizer snapshot and loss reduction contracts for follow-up phases.

## Deferred contracts

- `TODO(Phase-6)`: implement optimizer buffers and complete RNG-state persistence.
- `TODO(Phase-7)`: RL checkpoint/resume at episode boundaries; environment state is out of scope.
- `TODO(Phase-B1)`: finalize high-rank, empty-input, masking, and pad-token reduction behavior.
- `TODO(ParallelRuntime)`: add a Send + Sync runtime and hook protocol with parallel training.

## Invariants

- Paradigm-specific model traits and the RL episode loop remain statically typed.
- `fit()` uses the training capability; `fit_checkpointed()` and `resume()` require persistence.
- Resume boundaries are epochs or episodes.
- LatentDiffusion, Encoder, Decoder, and Scheduler stubs remain preserved.
