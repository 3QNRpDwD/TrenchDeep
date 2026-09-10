# Legacy route operation forwarding

Source integration update: native operations now compile from root src through
src/legacy.rs in the same crate. The separate package and legacy/ directory are
removed. Provenance lives in src/native_provenance/. This table describes route
capabilities; historical build/test log counts below predate source integration.

Consolidated 2026-09-10. Remaining-work order and acceptance criteria:
[P1_REVISED_PLAN.md](P1_REVISED_PLAN.md). Sigmoid route wiring is complete.
Div/Sum backward fixes and shared Context Diffusion/Trainer training E2E are complete.
Identical-noise product sampling comparison and Abs/Log/Sqrt backward are complete.
Sub nonempty broadcasting and gradient reduction are complete.
Single-input Concat and general Transpose permutations are complete.
Next: remaining Matmul broadcasting contracts.

Current adapter: `src/runtime/legacy_session.rs`. Default route remains P1.
This table supersedes earlier incremental support lists in the handoff/status.
Tanh/Softmax backward were subsequently corrected at the user's request; Sigmoid
forward was corrected by the user. These changes are recorded in CORRECTIONS.json.
Support means native
execution, not universal numerical equivalence with P1.

| Operations | Current Legacy route support |
| --- | --- |
| Add, Mul | Native forward/backward |
| Sub | Native forward/backward with nonempty trailing-axis broadcasting; gradients reduced to each input shape; assignment uses the same forward |
| Neg, Square, Exp, Sin, Cos, ReLU, SiLU, Pow | Native forward/backward |
| ApproxSin, ApproxCos | Native forward/backward; positive finite threshold validated; both originals use fixed-order polynomials and ignore threshold |
| Reshape | Native forward/backward, same element count; full target-shape dummy buffer required by original API |
| Transpose | General validated permutations composed from native axis swaps; rank 0/1 identity uses native Reshape; independent output handle |
| Matmul | Nonempty 1D/2D combinations; matrix batches with equal batch prefixes, or one rank-2 operand; singleton left batch with rank-2 right rejected because original drops batch shape |
| Concat | Native variadic forward/backward, at least one input; single-input output remains an independently owned tensor |
| Conv2d | Native forward/backward; validated NCHW, weight/bias shapes, stride/padding/kernel |
| GroupNorm | Native `apply_with_saved`, preserving x_hat/mean/variance |
| MaxPool2d | Native `apply_with_saved`, preserving mask |
| AvgPool2d, NearestUpsample2d | Native forward/backward; validated spatial attributes |
| MSE, MAE, Huber, BCE, CE, SoftmaxCE | Native mean reduction; Huber delta=1; categorical losses rank 1/2; equal nonempty prediction/target shapes |
| Tanh, Softmax | Native forward/backward; backward recomputes outputs from inputs |
| Abs, Log, Sqrt | Native forward/backward; Abs uses zero subgradient at zero; Log g/x; Sqrt g*0.5/sqrt(x) |
| Div | Native forward/backward for equal shapes; broadcasting rejected |
| Sum | Native forward/backward; scalar gradient expanded to input shape |
| TopK, axis Matmax | Two native inference outputs; tracked use rejected |
| Sigmoid | Native forward/backward; output/gradient/no-grad parity tested for scalar and negative/zero/positive/saturated inputs |
| Global Matmax | Rejected: original returns a zero tensor in place of a scalar argmax index |

Loss targets are copied to separate native leaves so their gradients never flow
into the public target, including when prediction and target share a handle.
Original scalar reductions are reshaped to rank zero through native ReshapeOp.
No-grad retains only primary spatial outputs, discarding unused saved outputs.
TopK/axis Matmax preserve both values and indices through the public handle map.

## Remaining work and original constraints

- Shared P1/native broadcast shape helpers use max(0, 1), which does not preserve
  empty axes. Sub rejects unequal-shape empty inputs before recording/indexing;
  zero-axis broadcasting needs a separate shared contract fix.
- Abs/Log/Sqrt native backward was added after integration (INTEGRATED.json).
  Scalar analytic gradients, weighted multidimensional gradients and no-grad are tested.
  Existing domain behavior is preserved: negative Log returns -Inf on P1 and NaN
  on native. At zero Log/Sqrt have infinite gradients; negative Sqrt is NaN.
- Div backward uses backend kernels without re-entering the operator registry.
- Sum backward expands the scalar upstream gradient to the original input shape.
- Tanh/Softmax input/output contract mismatch is resolved: backward calls the
  concrete operator's forward directly, without re-entering the global registry.
  Softmax uses max-subtracted exp normalization as in SoftmaxCrossEntropyLoss,
  retaining the general VJP rather than the fused cross-entropy derivative.
  Its axis gradient now uses the original axis tensor's shape. Tanh's existing
  exponential forward formula and large-input overflow behavior are unchanged.
- Additional matrix batch/vector broadcasting and singleton batch shapes need
  adapter composition or explicit contract decisions.
- General Transpose uses up to rank-1 native swaps (identity uses one native op).
  Public forward count remains one; native node/copy counts can be higher.
  This does not introduce a zero-copy transpose or static graph execution.
- Non-mean losses and non-default Huber delta are absent from the original
  registered operator interface. High-rank categorical reductions differ.
- Global Matmax needs an index-producing operation or original behavior change.
- TopK/Matmax tracked differentiation is also unavailable in the P1 provider.

Numerical limits retained from the originals: ApproxCos backward uses the
negative degree-15 ApproxSin, while P1 differentiates the degree-14 polynomial;
MAE's derivative at equality and BCE/CE clipping also differ. Representative
parity tests do not establish equality at these boundaries or for all inputs.

Common Context Diffusion/DataLoader/Trainer/Adam training passes three epochs on
Legacy and P1 against native reference draws, predictions, losses, all gradients
and updated weights. Native graph execution and per-epoch cleanup are checked.
Sampling comparison remains pending.
P1 completion still precedes static-graph lifetime/reference/buffer planning.

## Validation

Follow-up Sigmoid validation: legacy_operations 10 passed, execution_route 5
passed; Legacy-only execution_route 4 passed. All 142 source hashes and correction
provenance checked with PowerShell; no mismatches. Original sources unchanged.

`tests/legacy_operations.rs` compares output shapes/values and input gradients
using a weighted dot product. It covers spatial saved tensors, multi-output
inference, batched Matmul, losses with tracked targets, and scope reclamation.
Invalid attributes and unsupported backward paths must leave graph stats intact.
Native unit tests verify corrected Tanh and scalar Sum gradients.
Latest recorded results: all-features lib/integration 145 passed in
`target/p1/activation-all.log`; operation comparisons 9 passed in
`target/p1/activation-fix.log`. Legacy-only route tests passed 4 before the
activation corrections (`target/p1/expanded-legacy-only.log`). Source hash
verification covered 142 files with four authorized corrections and no unexpected
mismatches. These are previous code-session results, not reruns during doc cleanup.
