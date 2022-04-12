#![deny(rustdoc::broken_intra_doc_links)]
//! A module containing backends benchmarking facilities.
//!
//! Each `backend_*` submodule here is expected to be activated by a given feature flag
//! (matching the module name), and to contain the instantiation of a generic benchmarking
//! for every implemented operator.

pub mod benchmark;

#[cfg(feature = "backend_core")]
mod core;

// The main entry point. Uses criterion as benchmark harness.
fn main() {
    // We instantiate the benchmarks for different backends depending on the feature flag activated.
    #[cfg(feature = "backend_core")]
    core::bench();

    // We launch the benchmarks.
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
