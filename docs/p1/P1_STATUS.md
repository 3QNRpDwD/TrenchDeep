# P1 implementation and comparison status

Activation follow-up: user-authorized Tanh/Softmax backward recomputation is now
connected for tracked Legacy execution. Softmax axis gradient shape is fixed.
User's Sigmoid sign correction is preserved with a missing slice borrow repaired.
Provenance is recorded in CORRECTIONS.json; earlier Softmax-blocked notes are historical.

Latest bulk expansion (2026-09-08): see `P1_LEGACY_OPERATIONS.md` for the complete
current support matrix, native numerical differences, and remaining blockers.
Spatial/saved-tensor operations, Pow/approximation, Concat, restricted batched
Matmul, six mean losses, and multi-output inference are connected. Earlier
incremental support lists below are historical. Full Legacy Diffusion training
is not complete: tracked Softmax and original contract mismatches remain.

Shape/matrix route expansion: original ReshapeOp, Transpose and Matmul are now
connected. Reshape preserves element count; Transpose supports identity or one
axis swap for rank >= 2; Matmul supports nonempty rank-1/rank-2 inputs with matching
inner dimensions. Batched matmul and permutations requiring multiple swaps fail
before native execution. Attribute tensors are supplied using the original API:
Reshape's target-shape tensor has a full-sized dummy buffer (not a zero-copy path),
and Transpose receives two axis scalars. Original graph ownership retains these
only as needed. No original forward/backward formula is changed.
Tests compare output shapes/values and both input gradients through transpose →
matmul → reshape → dot, covering all four 1D/2D rank combinations; invalid-shape
and no-grad tests check native graph and tensor reclamation.
Targeted route/operation regressions pass 10/10; Legacy-only route tests pass 4/4.
Source hashes show no unexpected changes. Logs: `target/p1/shape-matrix*.log`.

Small-operation route expansion: native Sub (equal shapes), Neg, Square, Exp,
Sin, Cos, ReLU and SiLU now forward to their original operators with native
backward. Abs, Log and Sqrt are enabled only when graph tracking is off; their
original implementations lack backward, so tracked requests fail before graph
creation. Existing Add/Mul support remains. Shape-changing, reduction, matrix and
larger blocks are still pending. In particular Sum's scalar shape/gradient contract,
Tanh's saved-output expectation and Div's nested operator-storage access require
separate verification; no original formula is modified to force support.

Validation: three category tests compare scalar forward/gradients across five
input values, matrix-shaped no-grad outputs, subtraction derivatives and rejected
unsupported requests. Route + category + reference-DDPM regressions pass 9/9;
Legacy-only route tests pass 4/4. Original hash verification has no unexpected
mismatches. Logs: `target/p1/small-operations*.log`. These fixtures do not establish
all-shape or full-Diffusion route coverage.

Public Legacy route connected for the initial Add/Mul capability slice. A native
TensorStore bridge owns original variables; forward and backward invoke original
operators/graph, not P1 VJPs. Existing public tensor reads, parameter replacement,
gradient API and training scope are used without Trainer changes. The same external
model and built-in UnsupervisedTrainer/SGD run three epochs on P1 and Legacy with
identical final weights. Repeated public scopes preserve returned outputs until
drop and clean graph/gradients on failures. This supersedes earlier route-unavailable
notes below. Unsupported operations, custom ops, detach aliases, explicit backward
seeds, capture and mixed custom-P1-provider/Legacy composition fail explicitly.
The full Context Diffusion route is not yet supported. Raw legacy calls must not
interleave with a session on its owning thread. No static graph work is included.

Connection validation: full all-feature suite passed 133 tests before adding two
additional route-boundary cases; final targeted route suite passed 5/5 and the
Legacy-only/no-default-provider route suite 4/4. No-default cleanup/route/providers
passed 15/15. Preserved-source verification has zero unexpected mismatches.
Logs: `target/p1/connected-*.log`. No claim of full DDPM route parity is made.

Native training prototype extended: parameter replacement validates shape before
calling original Tensor::replace; nested no-grad calls original Function::forward
without graph registration. RAII training_step clears native graph/gradients and
scope-local handles on success, error and unwind. A 16-step scalar update test
verifies values/gradients, no-grad preserves an existing training graph, rejected
replacement preserves the parameter, and the same session is reusable after panic.
All three native-session tests pass with all features (`native-training.log`).
This internal scope returns owned buffers only; public scope handle lifetimes and
route integration remain unfinished. No built-in Trainer routing claim is made.

Native-session boundary experiment added under runtime/legacy_session.rs: native
Variable storage, original Add/Mul forward and original graph backward execute
without P1 VJP. The scalar fan-in case x*x+x produces value 6 and gradient 5 at
x=2. Tests cover exclusive session admission, rejection of pre-existing raw legacy
graphs without clearing them, stale-session handles, and graph/tensor reclamation
on normal drop and panic unwinding. This remains an internal prototype; public
Legacy route still fails explicitly. Raw legacy calls during an active session
are unsupported, and no claim of intercepting those calls is made. Public handle/
storage integration, no-grad, updates and broader operator support remain pending.

Route construction boundary implemented: the existing default/infallible P1
builder remains unchanged; explicit `.route(ExecutionRoute::P1).build()?` preserves
provider composition. Explicit Legacy returns DependencyUnavailable when the legacy
feature is absent and UnsupportedCapability while its native session adapter is
not implemented. No silent fallback or legacy execution claim is made. Tests in
`execution_route.rs` cover both feature configurations. Native forwarding remains
the next implementation step; this does not complete selectable DDPM execution.

Latest plan decision (2026-09-08): use the current Context Diffusion/U-Net as the
shared implementation. Keep explicit ctx construction and passing ctx to models
and Trainers. Default construction remains P1; only Legacy requires an explicit
route selection. The Legacy adapter forwards requests to original operations,
graph and backward without restructuring their internal computation; only handle,
error and session-lifetime boundaries are adapted. The original DDPM remains the
numerical reference. Route selection is planned, not implemented. This supersedes
older undecided model-sharing statements. Static execution remains post-P1.

2026-09-08: reference DDPM parameters now match by structural path, shape and
canonical sharing group rather than whole-model vector order. U-Net exposes
read-only `named_parameters`; the legacy comparison build appends read-only
accessors to an OUT_DIR source copy (preserved source and standalone unit-test
code remain unchanged). Leaf names use the original layer macro's paired
save_state/params contract. Comparison and replay use name-keyed gradient/weight
records. Fixture version 2 replaces version 1; regenerate old fixtures. References
below to positional mapping/version 1 describe the preceding implementation.
The existing checkpoint format is unchanged. Shared-Trainer routing remains open.

Validation for named mapping: all-feature library/integration suite 127 passed;
no-default cleanup/mapping/providers 15 passed; original standalone legacy 309
passed with the same three ignores; backward-only and visualization-only checks
passed. Standalone version-2 replay passed without legacy enabled. After the final
duplicate-name/identity-coverage hardening, targeted DDPM/mapping tests passed 3/3.
Legacy hash verification still reports only the authorized Conv2D correction and
no unexpected mismatches. Logs: `target/p1/named-*.log`.

Latest bounded verification: `tests/cleanup.rs` now runs 64 consecutive public
training scopes with interleaved forward failures and successful backward passes.
The same fan-out/fan-in fixture runs with built-in providers and independent
SlotStore/Tape/ReferenceOps. Each scope clears gradients and graph state; an
explicitly returned output stays live until dropped, after which all GraphStats
fields return to the pre-loop baseline. Targeted cleanup suites pass 5/5 with
all features and 4/4 with no defaults. Logs: `target/p1/batch-lifetime-*.log`.
This verifies live runtime handles, not allocator capacity, RSS, DDPM-wide memory,
or absence of snapshot copies. Production runtime/Trainer code is unchanged.

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
- Reuse the completed DDPM name/shape/sharing correspondence in the future route
  adapter; complete common-Trainer reference DDPM E2E and identical-noise sampling parity.
- Extend lifecycle, capture and independent-provider/feature-exclusion verification
  to both execution paths before publishing corrected-baseline benchmarks.

- Resolve the MLP numerical-equivalence criterion without silently changing the
  preserved baseline or corrupting production sigmoid semantics.
- Broaden operation/model/Trainer benchmark coverage beyond the fixtures above.
- Complete the broader failure/observer-order matrix. Existing tests establish the listed
  cases, not every combination of user-supplied providers and callbacks.

These remaining items are visible acceptance gaps; passing the current test suite
does not by itself mean every original P1 acceptance condition is complete.
