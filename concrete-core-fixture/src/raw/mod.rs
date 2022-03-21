//! A module manipulating raw messages.
//!
//! For all the fixtures, we need to be able to generate input plaintexts, and analyze output
//! plaintexts. We implement those generation and analysis functions only for the _raw_ `u32` and
//! `u64` types.

pub mod generation;
pub mod statistical_test;
