//! A module containing backends benchmarks.
//!
//! Each submodule here is expected to be activated by a given feature flag (matching the
//! `backend_*` naming), and to contain a benchmark function containing the benchmarking of every
//! entry points exposed by the backend.

#[cfg(feature = "backend_core")]
pub mod core;

#[cfg(feature = "backend_optalysys")]
pub mod optalysys;
