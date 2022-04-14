//! A module containing random generators objects.
//!
//! See [crate-level](`crate`) explanations.
use crate::seeders::Seed;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// The number of children created when a generator is forked.
#[derive(Debug, Copy, Clone)]
pub struct ChildrenCount(pub usize);

/// The number of bytes each child can generate, when a generator is forked.
#[derive(Debug, Copy, Clone)]
pub struct BytesPerChild(pub usize);

/// A structure representing the number of bytes between two table indices.
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub struct ByteCount(pub u128);

/// An error occuring during a generator fork.
#[derive(Debug)]
pub enum ForkError {
    ForkTooLarge,
    ZeroChildrenCount,
    ZeroBytesPerChild,
}

impl Display for ForkError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ForkError::ForkTooLarge => {
                write!(
                    f,
                    "The children generators would output bytes after the parent bound. "
                )
            }
            ForkError::ZeroChildrenCount => {
                write!(
                    f,
                    "The number of children in the fork must be greater than zero."
                )
            }
            ForkError::ZeroBytesPerChild => {
                write!(
                    f,
                    "The number of bytes per child must be greater than zero."
                )
            }
        }
    }
}
impl Error for ForkError {}

/// A trait for cryptographically secure pseudo-random generators.
///
/// See the [crate-level](#crate) documentation for details.
pub trait RandomGenerator: Iterator<Item = u8> {
    /// The iterator over children generators, returned by `try_fork` in case of success.
    type ChildrenIter: Iterator<Item = Self>;

    /// Creates a new generator from a seed.
    ///
    /// This operation is usually costly to perform, as the round keys need to be generated from the
    /// seed.
    fn new(seed: Seed) -> Self;

    /// Returns the number of bytes that can still be yielded by the generator before reaching its
    /// bound.
    ///
    /// Note:
    /// -----
    ///
    /// A fresh generator can generate 2¹³² bytes. Unfortunately, no rust integer type in is able
    /// to encode such a large number. Consequently [`ByteCount`] uses the largest integer type
    /// available to encode this value: the `u128` type. For this reason, this method does not
    /// effectively return the number of remaining bytes, but instead
    /// `min(2¹²⁸-1, remaining_bytes)`.
    fn remaining_bytes(&self) -> ByteCount;

    /// Returns the next byte of the stream, if the generator did not yet reach its bound.
    fn next_byte(&mut self) -> Option<u8> {
        self.next()
    }

    /// Tries to fork the generator into an iterator of `n_children` new generators, each able to
    /// yield `n_bytes` bytes.
    ///
    /// Note:
    /// -----
    ///
    /// To be successful, the number of remaining bytes for the parent generator must be larger than
    /// `n_children*n_bytes`.
    fn try_fork(
        &mut self,
        n_children: ChildrenCount,
        n_bytes: BytesPerChild,
    ) -> Result<Self::ChildrenIter, ForkError>;
}

/// A trait extending [`RandomGenerator`] to the parallel iterators of `rayon`.
#[cfg(feature = "parallel")]
pub trait ParallelRandomGenerator: RandomGenerator {
    /// The iterator over children generators, returned by `par_try_fork` in case of success.
    type ParChildrenIter: rayon::prelude::IndexedParallelIterator<Item = Self>;

    /// Tries to fork the generator into a parallel iterator of `n_children` new generators, each
    /// able to yield `n_bytes` bytes.
    ///
    /// Note:
    /// -----
    ///
    /// To be successful, the number of remaining bytes for the parent generator must be larger than
    /// `n_children*n_bytes`.
    fn par_try_fork(
        &mut self,
        n_children: ChildrenCount,
        n_bytes: BytesPerChild,
    ) -> Result<Self::ParChildrenIter, ForkError>;
}

mod aes_ctr;

mod implem;
pub use implem::*;

#[cfg(test)]
pub mod generator_generic_test {
    #![allow(unused)] // to please clippy when tests are not activated
    use super::*;
    use rand::Rng;

    const REPEATS: usize = 1_000;

    fn any_seed() -> impl Iterator<Item = Seed> {
        std::iter::repeat_with(|| Seed(rand::thread_rng().gen()))
    }

    /// Checks that the PRNG roughly generates uniform numbers
    pub fn test_uniformity<G: RandomGenerator>() {
        for _ in 0..REPEATS {
            let seed = any_seed().next().unwrap();
            let precision = 10f64.powi(-4);
            let n_samples = 10_000_000_usize;
            let mut generator = G::new(seed);
            let mut counts = [0usize; 256];
            let expected_prob: f64 = 1. / 256.;
            for _ in 0..n_samples {
                counts[generator.next_byte().unwrap() as usize] += 1;
            }
            counts
                .iter()
                .map(|a| (*a as f64) / (n_samples as f64))
                .for_each(|a| assert!((a - expected_prob) < precision))
        }
    }

    /// Checks that given a state and a key, the PRNG is determinist.
    pub fn test_generator_determinism<G: RandomGenerator>() {
        for _ in 0..REPEATS {
            let seed = any_seed().next().unwrap();
            let mut first_generator = G::new(seed);
            let mut second_generator = G::new(seed);
            for _ in 0..1024 {
                assert_eq!(first_generator.next(), second_generator.next());
            }
        }
    }

    /// Checks that forks returns a bounded child, and that the proper number of bytes can be
    /// generated.
    pub fn test_fork_children<G: RandomGenerator>() {
        let mut gen = G::new(any_seed().next().unwrap());
        let mut bounded = gen
            .try_fork(ChildrenCount(1), BytesPerChild(10))
            .unwrap()
            .next()
            .unwrap();
        assert_eq!(bounded.remaining_bytes(), ByteCount(10));
        for _ in 0..10 {
            bounded.next();
        }
    }

    // Checks that a bounded prng returns none when exceeding the allowed number of bytes.
    pub fn test_bounded_none_should_panic<G: RandomGenerator>() {
        let mut gen = G::new(any_seed().next().unwrap());
        let mut bounded = gen
            .try_fork(ChildrenCount(1), BytesPerChild(10))
            .unwrap()
            .next()
            .unwrap();
        for _ in 0..11 {
            assert!(bounded.next().is_some());
        }
    }
}
