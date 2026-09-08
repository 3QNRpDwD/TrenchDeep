# Legacy route operation forwarding — 2026-09-08

Current adapter: `src/runtime/legacy_session.rs`. Default route remains P1.
This table supersedes earlier incremental support lists in the handoff/status.
Tanh/Softmax backward were subsequently corrected at the user's request; Sigmoid
forward was corrected by the user. These changes are recorded in CORRECTIONS.json.
Support means native
execution, not universal numerical equivalence with P1.

| Operations | Current Legacy route support |
| --- | --- |
| Add, Mul | Native forward/backward |
| Sub | Forward/backward for equal shapes; broadcasting rejected |
| Neg, Square, Exp, Sin, Cos, ReLU, SiLU, Pow | Native forward/backward |
| ApproxSin, ApproxCos | Native forward/backward; positive finite threshold validated; both originals use fixed-order polynomials and ignore threshold |
| Reshape | Native forward/backward, same element count; full target-shape dummy buffer required by original API |
| Transpose | Rank >= 2, identity or one swapped axis pair |
| Matmul | Nonempty 1D/2D combinations; matrix batches with equal batch prefixes, or one rank-2 operand; singleton left batch with rank-2 right rejected because original drops batch shape |
| Concat | Native variadic forward/backward, at least two inputs |
| Conv2d | Native forward/backward; validated NCHW, weight/bias shapes, stride/padding/kernel |
| GroupNorm | Native `apply_with_saved`, preserving x_hat/mean/variance |
| MaxPool2d | Native `apply_with_saved`, preserving mask |
| AvgPool2d, NearestUpsample2d | Native forward/backward; validated spatial attributes |
| MSE, MAE, Huber, BCE, CE, SoftmaxCE | Native mean reduction; Huber delta=1; categorical losses rank 1/2; equal nonempty prediction/target shapes |
| Tanh, Softmax | Native forward/backward; backward recomputes outputs from inputs |
| Abs, Log, Sqrt, Div, Sum | Inference only; tracked use rejected before graph recording |
| TopK, axis Matmax | Two native inference outputs; tracked use rejected |
| Sigmoid | Original forward corrected by user; adapter support remains to be enabled |
| Global Matmax | Rejected: original returns a zero tensor in place of a scalar argmax index |

Loss targets are copied to separate native leaves so their gradients never flow
into the public target, including when prediction and target share a handle.
Original scalar reductions are reshaped to rank zero through native ReshapeOp.
No-grad retains only primary spatial outputs, discarding unused saved outputs.
TopK/axis Matmax preserve both values and indices through the public handle map.

## Remaining work and original constraints

- Abs/Log/Sqrt have no original backward implementation.
- Div backward re-enters the mutably borrowed operator registry through Neg.
  A direct native diagnostic reproduced BorrowError and subsequent cleanup abort
  (`target/p1/legacy-blockers.log`). The adapter rejects tracked Div; the aborting
  diagnostic is not included in the normal in-process regression suite.
- Sum backward returns its incoming scalar shape without expanding it to the
  input shape; this mismatch has native regression coverage.
- Tanh/Softmax input/output contract mismatch is resolved: backward calls the
  concrete operator's forward directly, without re-entering the global registry.
  Softmax uses max-subtracted exp normalization as in SoftmaxCrossEntropyLoss,
  retaining the general VJP rather than the fused cross-entropy derivative.
  Its axis gradient now uses the original axis tensor's shape. Tanh's existing
  exponential forward formula and large-input overflow behavior are unchanged.
- General transpose permutations, additional matrix batch broadcasting, and
  single-input Concat need adapter composition or explicit contract decisions.
- Non-mean losses and non-default Huber delta are absent from the original
  registered operator interface. High-rank categorical reductions differ.
- Global Matmax needs an index-producing operation or original behavior change.
- TopK/Matmax tracked differentiation is also unavailable in the P1 provider.

Numerical limits retained from the originals: ApproxCos backward uses the
negative degree-15 ApproxSin, while P1 differentiates the degree-14 polynomial;
MAE's derivative at equality and BCE/CE clipping also differ. Representative
parity tests do not establish equality at these boundaries or for all inputs.

Full Context Diffusion training on Legacy still requires an end-to-end check;
the tracked attention Softmax blocker has been resolved.
P1 completion still precedes static-graph lifetime/reference/buffer planning.

## Validation

`tests/legacy_operations.rs` compares output shapes/values and input gradients
using a weighted dot product. It covers spatial saved tensors, multi-output
inference, batched Matmul, losses with tracked targets, and scope reclamation.
Invalid attributes and unsupported backward paths must leave graph stats intact.
Native unit tests verify corrected Tanh and reproduce the remaining Sum mismatch.
Full results: `target/p1/expanded-all.log`; Legacy-only route configuration:
`target/p1/expanded-legacy-only.log`.
Final checks: all-features lib/integration tests 144 passed; Legacy-only route
tests 4 passed; 142 original source hashes have no unexpected mismatches.
