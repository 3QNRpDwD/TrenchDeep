# Preserved comparison engine

`src/` is the byte-for-byte source snapshot of commit `65b4d40` (142 files).
This source is retained for numerical and performance comparisons; deletion belongs
to the user. `BASELINE.json` records its original hashes. Run
`python scripts/verify_legacy.py` from the repository root to verify preservation.

The package name differs from production. `benchmark_lib.rs` includes the unchanged
baseline and adds `comparison.rs`, a visibility bridge for the original models.
The wrapper allows Rust's include-generated macro-path lint because `include!`
changes macro provenance. No calculation, initializer, backward rule, or trainer
loop in the baseline is edited. Unit tests still compile the original test module.
Production loads this package only through the optional `legacyBenchmark` feature.

Known discrepancy: baseline Sigmoid forward computes `1/(1+exp(x))`, whereas its
backward uses a positive sigmoid derivative. MLP performance comparison is blocked;
the baseline is preserved, and production retains the correct sigmoid definition.
See `docs/P1_STATUS.md` for coverage and benchmark limits.
