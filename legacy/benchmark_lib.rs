#![allow(macro_expanded_macro_exports_accessed_by_absolute_paths)]
// include! changes macro expansion provenance, not baseline implementation semantics.
// Preserve source files; the non-test build appends read-only parameter accessors
// to a build-output copy. Standalone baseline unit tests use the source directly.
#[cfg(test)]
include!("src/lib.rs");
#[cfg(not(test))]
include!(concat!(env!("OUT_DIR"), "/reference_src/lib.rs"));
#[cfg(not(test))]
pub mod comparison;
// Original reference models retain their crate::tests::common paths.
// Only this additive visibility shim is compiled outside baseline unit tests.
#[cfg(not(test))]
#[path = "reference_models.rs"]
pub mod tests;
