# Revised P1: initial dependency audit

Latest direction: current Context Diffusion is the shared model implementation;
explicit ctx remains the public entry point, defaulting to P1. Legacy is opt-in
at construction and forwards to original execution without changing its internal
operation/backward flow. Handle/error/lifecycle adaptation remains necessary.
Model-sharing selection is settled; the precise facade extension still needs the
block experiment. Earlier undecided-sharing statements below are historical.

2026-09-08 completion slice: DDPM comparison/replay now use structural names,
shapes and canonical sharing groups. Context enumeration traverses model fields;
legacy build-output accessors traverse original fields and Sequential children,
using the original leaf macro's name/order contract only inside each leaf.
The raw enumeration is validated before constructing name maps; identity-set
coverage checks prevent omitted parameters. Numerical code and Trainer are unchanged.
Earlier statements below about unfinished positional mapping are historical.

Snapshot follow-up: [P1_SNAPSHOT_REFERENCE_AUDIT.md](P1_SNAPSHOT_REFERENCE_AUDIT.md)
records ordinary runtime and CPU-provider copies, borrowed-view feasibility,
alias/reentrancy constraints and a proposed guard boundary. This is inspection,
not a completed zero-copy implementation or benchmark.

This records the first implementation slice of P1_REVISED_PLAN.md, not completion
of selectable execution paths. No model unification/separation decision is made here.

| Component | Existing boundary | Gap found | Minimal change |
|---|---|---|---|
| Epoch training service | Public Tensor/Variable, optimizer, model and loader contracts | Private training scope calls | Public `with_training_scope` closure API; preserve existing guard and cleanup semantics |
| RL Trainer | Shared `finish_step`, existing rollout | Private operation helpers and direct TensorBuffer fields | Receiver operations and public buffer accessors; same rollout algorithm |
| Optimizer | Stable public Parameter IDs and context update API | Direct gradient buffer fields | Public data/shape accessors; same update formulas and state |
| NN layers | Public Layer forward/predict and parameter contracts | No new abstraction needed for this slice | Reuse unchanged |
| DDPM | Existing public Diffusion/Unet | Previously tested different small fixture | Add original 8×8, dim=8, groups=4, batch=2, Adam, 3 epoch loader test |
| Runtime providers | Injectable storage, graph records/order and forward/VJP providers | AutogradEngine delegates graph representation, while backward execution remains in Context | Requires a separate full legacy execution adapter design; do not equate a legacy VJP provider with legacy graph execution |

`with_training_scope` exposes execution semantics, not graph storage or a mutable
guard. It rejects an existing graph/nested training before running the callback,
cleans graph and gradients after success/error, and combines cleanup failures with
the original error. It does not promise parameter rollback. The built-in Trainers
now use the same method that external code can call. An external test exercises it
with independent SlotStore/Tape/ReferenceOps and no default implementations.

The DDPM reference test preserves architecture, Adam settings, input records,
collator, batch size, shuffle setting, epoch count/tolerance and metric assertions.
It uses fixed P1 initialization/model seeds and disables progress rendering for
automated tests. It is NOT a claim of identical legacy random draws or numerical
parity. Original baseline source and existing small comparisons remain unchanged.

Next: capture/replay the original reference timestep/noise and establish named
parameter correspondence, then validate a block-level full execution adapter.
Do not replace Trainer/model classes merely because their current public handles
are implemented by Context. Extend the existing facade only at the proven gap.

## Follow-up boundary inspection (2026-09-07)

`contracts::AutogradEngine` accepts `GradientRecord` and exposes record/get/remove/
nodes/order. It has no execution or gradient-publication method. In
`runtime/backward.rs`, Context seeds its own gradient map, walks engine records,
snapshots inputs/saved tensors and calls each `BackwardOp`, then accumulates VJPs.
Consequently, replacing `AutogradEngine` alone cannot invoke original legacy
Variable backward with its native graph/storage semantics.

`runtime/training.rs` also owns graph-conflict checks, gradient clearing and graph
cleanup. A future full execution adapter must participate in these same lifecycle
operations, as well as forward, native-handle ownership, parameter replacement,
gradient retrieval and capture. Adding only a backward callback would leave native
graphs outside scope cleanup. Keep the existing Trainer and optimizer algorithms;
do not branch there or pass native legacy handles to them.

This is a code audit, not a completed block-adapter experiment. The next bounded
experiment remains TimeEmbedding + one ResidualBlock with native graph backward,
followed by success/error cleanup and nested-session rejection. Model sharing and
the actual facade extension are still undecided; no second runtime was introduced.

The direct-DDPM diagnostic now writes a versioned, self-contained fixture with
initial tensors, original noise/timesteps, expected predictions/losses/gradients,
Adam weights and correction provenance. `replay_reference_diffusion` consumes it
without legacy execution. This advances baseline reproducibility only; positional
parameter mapping and shared-Trainer route switching remain incomplete.
