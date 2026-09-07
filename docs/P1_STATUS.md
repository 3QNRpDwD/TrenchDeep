# P1 implementation and comparison status

Small follow-up fixes: replay validates every step's timestep, tensor lengths,
parameter lengths and finite values before training, and rejects surplus CLI
arguments. The reference integration test passes with malformed-final-step cases
(out-of-range timestep, NaN noise and truncated gradient) plus the existing full
replay/corrupted-prediction checks. A direct CLI invocation confirms surplus
arguments exit with a usage error. `git diff --check` passes. These are targeted
checks; the complete suite counts below belong to the preceding full run.
The handoff now lists only remaining long-term work and corrects stale replay and
post-Conv2D feature-matrix descriptions. Existing worktree changes are preserved.

Follow-up (2026-09-07): the direct reference diagnostic additionally writes
`target/p1/reference-draws/fixture.json`. Replay without the legacy feature using:

```powershell
cargo test --features legacyBenchmark --test legacy_reference_diffusion
cargo run --features enableBackward --example replay_reference_diffusion -- target/p1/reference-draws/fixture.json
```

The version-1 fixture fixes the reference architecture/Adam/input configuration
in the replay implementation and stores initial weights, all three original draws,
expected predictions/losses/gradients/updated weights and correction provenance.
The integration test round-trips the file, replays it and rejects a corrupted
prediction. Version, step count, parameter counts/shapes and correction provenance
are checked. Mapping is still positional; this is not shared-Trainer equivalence.
Generation overwrites the fixture, so copy it elsewhere to retain a particular run.
Earlier individual `step-N.json` files contain draws only and are not CLI inputs.

Follow-up validation: 123 all-feature library/integration tests passed (including
file replay and corrupted-prediction rejection); standalone legacy 309 passed with
the same three ignores; no-default cleanup/providers 12 passed; backward-only and
visualization-only checks passed. The standalone replay command with only
`enableBackward` plus default providers also passed. Legacy hash verification found
no unexpected mismatches. Logs: `target/p1/continued-*.log`.

Latest corrected-baseline verification: 123 all-feature library/integration tests
passed, zero failures/ignores. Standalone legacy tests: 309 passed, zero failures,
three existing ignores. Source verification: one authorized one-line correction,
141 untouched files, zero unexpected mismatches. Older counts below are historical.

Resolved defect discovered during reference-DDPM work: the original Conv2D
input-gradient helper indexes weight as `a[i*k+l]` instead of `a[l*m+i]` in
`matmul_at_b`. A two-channel 1×1 convolution reproduces legacy `[3,7]` versus
finite-difference/Context `[4,6]`. At the user's explicit request, the original
legacy operator now uses the corrected index directly; no correcting adapter was
added. `legacy_conv_gradient` verifies the correction against finite differences.
`legacy_reference_diffusion` now passes predictions, loss, every parameter gradient
and Adam updates over three steps. This is a direct model comparison, not yet the
shared-Trainer route switch. No reference-DDPM timing is published yet.
`BASELINE.json` retains historical hashes; `CORRECTIONS.json` records this one
authorized change and its original/corrected hashes. Older benchmark results used
the uncorrected source and are historical, not results for the corrected baseline.

The original Diffusion model is now exposed through an additive visibility shim.
Its unchanged forward draws are read from the graph and supplied to the existing
Context model. Exact noise/timestep records are saved under
`target/p1/reference-draws/step-N.json` before parity validation. Parameter mapping
is still positional for this diagnostic; structural mapping is not complete.

Revised implementation started: see [P1_DEPENDENCY_AUDIT.md](P1_DEPENDENCY_AUDIT.md).
The existing Trainer now uses public `with_training_scope`; RL operation helpers
and RL/optimizer buffer access were mechanically moved to public APIs. The original
DDPM architecture/Adam/loader configuration has a new P1 execution regression test.
Actual legacy/Context execution-path switching and three-way parity remain pending.
Validation for this slice: 121 all-feature library/integration tests passed with
zero failures/ignores; 12 no-default cleanup/provider tests passed; no-default
backward and visualization builds passed. All 142 original legacy hashes match.

Planning update (2026-09-07): [P1_REVISED_PLAN.md](P1_REVISED_PLAN.md) defines
the revised target: selectable legacy/ExecutionContext paths for the model in
the original `diffusion_train_with_trainer`. Model unification versus separation
is still open for discussion. The implementation results below predate that target
and do not establish that path switching is implemented.
The revised plan additionally requires the same Trainer implementation, and all
upper layers, to depend only on public abstraction APIs. Switching legacy/Context
implementations must not require Trainer changes; this is not yet implemented.
Implementation must first audit and reuse the existing P1 abstractions. Prefer
mechanical public-operation API substitutions where semantics remain intact;
rewrite upper-layer code only where a concrete gap cannot be addressed with a
smaller API/adapter change. This is a planning constraint, not a new completion claim.

P1 is not yet declared complete. The implementation now has replaceable storage,
autograd and operation contracts, common training steps, real U-Net/DDPM, explicit
capture, independent RNG streams and retained comparisons. Full legacy equivalence
has an identified mathematical discrepancy and the comparison coverage below has
deliberate limits. No legacy source is deleted. The sole numerical source change
is the authorized Conv2D correction described above.

## Verified behavior

- Independent slot storage, vector tape and reference operations run the same
  public linear model/optimizer/trainer individually and together. The integration
  test also runs with all default implementations absent from the build.
- Missing storage, autograd, operations and capture return structured dependency
  errors; unsupported operations return capability errors. No-grad inference works
  without autograd. Detach aliasing and context lifetime are tested.
- Shared parameters are registered and clipped once. Initialization, model noise,
  data ordering and RL action streams are separate. Trainer seeding preserves weights.
- Loader setup/forward errors clean the training scope. A primary error and an
  independent cleanup error are both returned. Missing dataset samples return errors.
- Captures contain intermediate gradients before backward cleanup, remain owned
  after context destruction, and do not use a global capture session.
- Training publishes snapshots before batch completion, after graph/gradient
  cleanup. A failing metric hook suppresses both success events and cleans gradients.
- U-Net time embedding, residuals, attention, down/up paths, linear/cosine schedules,
  explicit-noise training/sampling, and checkpoint round-trip are exercised.
- RL uses shared step processing and no-grad policy forward. Interrupt checkpoints
  are saved after completed episode cleanup and retain the existing metadata format.
  Complete resume and optimizer snapshots explicitly remain unsupported in P1.

## Retained comparisons

| Fixture | Verified comparison | Limits |
|---|---|---|
| 16×16 matrix multiplication | Forward values, release timings | One shape; not broad operator coverage |
| Two-stage attention U-Net, `[1,1,4,4]` | Forward, MSE, every parameter gradient, actual SGD updates, release timings | Small CPU fixture, not a large-image performance claim |
| Linear Trainer, eight samples, batches of two | Loader-to-optimizer epoch loss and updated parameters, release timings | Fixed order, no I/O in timed region |
| Bigram | Last-step logits, shift loss, gradients, SGD update | Single sequence fixture; batched shift has separate production tests |
| Pi-model | Output, loss, gradients, SGD update with zero and nonzero noise | Nonzero views are read from the original graph and supplied unchanged to production |
| RL policy | Fixed-action loss, gradients, SGD update | Separate from rollout |
| RL rollout | Three actions, loss, Trainer update | One episode; action RNG seeds explicitly aligned across implementations |
| Legacy MLP/Sigmoid | Discrepancy reproduced and preserved | Performance comparison blocked |

### Legacy Sigmoid discrepancy

`legacy/src/nn/activation/sigmoid.rs` computes `1/(1+exp(x))` in forward. Its
backward uses the positive derivative of `1/(1+exp(-x))`. At `x=1`, baseline
forward is approximately `0.2689414`; production is approximately `0.7310586`.
This is not a tolerance issue. Production retains the mathematically correct
sigmoid, including finite-difference regression coverage. The original baseline
is untouched. `legacy_sigmoid_discrepancy_blocks_mlp_performance_comparison`
asserts the discrepancy rather than ignoring a failed test or widening tolerance.
MLP must not be labeled numerically equivalent or assigned parity-qualified timings.

The new MLP trains on logits with fused softmax cross entropy, while legacy MLP
returns probabilities. Prediction comparisons must first use the same representation;
doing so still exposes the sigmoid discrepancy above.

## Reproduction

From the repository root:

```powershell
python scripts/verify_legacy.py
cargo test --all-features --lib --tests
cargo test --all-features --doc
cargo test --no-default-features --test providers --test metrics --test cleanup
cargo check --no-default-features --features enableBackward
cargo check --no-default-features --features enableVisualization
cargo test --manifest-path legacy/Cargo.toml --lib --features enableBackward
cargo bench --bench p1_compare --features legacyBenchmark
```

The original baseline library remains **309 passing, zero failing, three original
ignored tests**. `legacy/BASELINE.json` verifies all 142 retained source files.
Before the revised implementation, the all-feature library/integration suite had 119 passing tests with no
failures or ignores in the final full run. Documentation tests pass 4/4;
the no-default provider/metrics/cleanup suite passes 13/13. Backward-only and
visualization-only no-default builds pass.
The additive comparison wrapper changes module visibility and macro provenance
only; details are in `legacy/README.md`. No deletion is part of P1 or a follow-up.

Benchmarks compare values before publishing timings. U-Net and Trainer cases
recheck parity throughout repeated updates; failures abort reporting. Tolerance is
absolute **or** relative `1e-3`, with no numerical exception currently permitted.
Initialization is outside timing, with combined fixture setup and separate model construction costs.
One warmup iteration is discarded. Reports contain median, nearest-rank p95,
throughput, all samples, and both implementations' storage/graph handle counts. U-Net timing
includes forward, backward and SGD; Trainer timing includes loader, validation,
backward, optimizer and cleanup. Correctness checks are outside the timed regions.
Counts are live handles, not allocator peak memory.

Results and environment/source metadata are kept under `docs/benchmarks`. These small fixtures show a performance
regression in the new path; they do not establish a general speedup. Owned snapshots
at provider boundaries currently copy inputs; this is an identifiable source of
overhead, not a measured attribution of the entire regression.

## Remaining long-term acceptance work

Post-P1 follow-up (not a P1 acceptance prerequisite): transition ExecutionContext
from its current eager dynamic graph to reusable static execution plans and buffers.
The agreed direction adds a preparation phase before training to analyze forward/
backward lifetimes, references, saved values, buffer reuse and necessary copies.
Batch execution reuses that plan; changed assumptions require another compatible
plan or reanalysis, with runtime decisions limited to genuinely dynamic behavior.
See P1_REVISED_PLAN.md section 10. Ordinary tensor snapshots copy data even without
visualization; their performance impact requires measurement. Visualization-only
overhead is not a required optimization target for this follow-up.

- Validate a native legacy block adapter and decide model sharing scope before
  connecting selectable full execution paths beneath the same public Trainer API.
- Replace positional mapping with names/shapes/shared-parameter correspondence;
  complete common-Trainer reference DDPM E2E and identical-noise sampling parity.
- Extend lifecycle, capture and independent-provider/feature-exclusion verification
  to both execution paths before publishing corrected-baseline benchmarks.

- Resolve the MLP numerical-equivalence criterion without silently changing the
  preserved baseline or corrupting production sigmoid semantics.
- Broaden operation/model/Trainer benchmark coverage beyond the fixtures above.
- Complete the broader failure/observer-order matrix. Existing tests establish the listed
  cases, not every combination of user-supplied providers and callbacks.

These remaining items are visible acceptance gaps; passing the current test suite
does not by itself mean every original P1 acceptance condition is complete.
