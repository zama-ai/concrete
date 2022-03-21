#![deny(rustdoc::broken_intra_doc_links)]
//! A library containing generic fixtures for `concrete-core` operators.
//!
//! The central abstraction of this (private) library, is the [`Fixture`](fixture::Fixture) trait
//! which, once implemented for a given engine trait, exposes methods to sample/test/benchmark any
//! implementor of the engine trait in question.

pub mod fixture;
pub mod generation;
pub mod raw;

/// A type representing the number of times we repeat a test for a given set of parameters.
#[derive(Clone, Copy, Debug)]
pub struct Repetitions(pub usize);

/// A type representing the number of samples needed to perform a statistical test.
#[derive(Clone, Copy, Debug)]
pub struct SampleSize(pub usize);
