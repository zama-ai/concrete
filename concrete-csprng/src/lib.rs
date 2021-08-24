//! Cryptographically secure pseudo random number generator, that uses AES in CTR mode.
//!
//! Welcome to the `concrete-csprng` documentation.
//!
//! # Implementation
//!
//! This crate contains a reasonably fast cryptographically secure pseudo-random number generator.
//! The implementation is based on the AES blockcipher used in counter (CTR) mode, as presented
//! in the ISO/IEC 18033-4 document.
//!
//! When a new generator is created, it can be provided with a seed, which is used as the key of the
//! inner aes blockcipher. The counter is initialized with a constant IV of zero.
//!
//! A generator can be created with a bound, that is a hard limit over which it will stop generating
//! random bytes. By default, this bound is set to its greatest possible value, about 2^128.
//!
//! # Forking
//!
//! This implementation makes it possible to _fork_ a parent generator into a set of children
//! generators which use the same key. Two modes of operation are supported, which give different
//! guarantees on the bytes yielded by the children:
//!
//! + A _sequential fork_ creates a set of sequential generators which can generate a fixed, user
//! selected, amount of randomness. During the fork, the parent generator state is shifted of as
//! many bytes as the children generate, and the children states are set evenly on the state span
//! of the fork. When the children are exhausted in the order yielded by the fork, they are
//! guaranteed to generate the same sequence the parent would have if it had not been forked. Thanks
//! to that, exhausting the leaves of any tree of generators constructed using this fork will give
//! the same sequence of bytes as the parent generator would have. These kind of forks are useful
//! when the number of bytes needed in each children is small and can be known in advance.
//! + An _alternate fork_ creates a set of alternate generators which can generates as much
//! randomness as needed untill they reach their parent's bound. During the fork, the children are
//! initialized by moving the parent's state a step forward, then by multiplying their step by the
//! number of children. This means that generating one byte with each child will yield the same
//! sequence as the unforked parent would have. These are useful when the number of bytes needed in
//! each child cannot be known in advance or might grow.
//!
//! The cost of forking a generator is constant in both modes of operations, and is typically
//! very low (essentially the cost of copying the aes key).

#[cfg(feature = "multithread")]
use rayon::prelude::*;
use std::fmt::{Debug, Display, Formatter, Result};

mod aesni;
mod counter;
mod software;
use crate::counter::{AesKey, BytesPerChild, ChildCount, HardAesCtrGenerator, SoftAesCtrGenerator};
pub use software::set_soft_rdseed_secret;

/// The pseudorandom number generator.
///
/// If the correct instructions sets are available on the machine, an hardware accelerated version
/// of the generator can be used.  
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum RandomGenerator {
    #[doc(hidden)]
    Software(SoftAesCtrGenerator),
    #[doc(hidden)]
    Hardware(HardAesCtrGenerator),
}

impl RandomGenerator {
    /// Builds a new random generator, selecting the hardware implementation if available.
    /// Optionally, a seed can be provided.
    ///
    /// # Note
    ///
    /// If using the `slow` feature, this function will return the non-accelerated variant, even
    /// though the right instructions are available.
    pub fn new(seed: Option<u128>) -> RandomGenerator {
        if cfg!(feature = "slow") {
            return RandomGenerator::new_software(seed);
        }
        RandomGenerator::new_hardware(seed).unwrap_or_else(|| RandomGenerator::new_software(seed))
    }

    /// Builds a new software random generator, optionally seeding it with a given value.
    pub fn new_software(seed: Option<u128>) -> RandomGenerator {
        RandomGenerator::Software(SoftAesCtrGenerator::new(seed.map(AesKey), None, None, None))
    }

    /// Tries to build a new hardware random generator, optionally seeding it with a given value.
    pub fn new_hardware(seed: Option<u128>) -> Option<RandomGenerator> {
        if !is_x86_feature_detected!("aes")
            || !is_x86_feature_detected!("rdseed")
            || !is_x86_feature_detected!("sse2")
        {
            return None;
        }
        Some(RandomGenerator::Hardware(HardAesCtrGenerator::new(
            seed.map(AesKey),
            None,
            None,
            None,
        )))
    }

    /// Yields the next byte from the generator.
    pub fn generate_next(&mut self) -> Option<u8> {
        match self {
            Self::Hardware(ref mut rand) => rand.generate_next(),
            Self::Software(ref mut rand) => rand.generate_next(),
        }
    }

    /// Returns the number of remaining bytes.
    pub fn remaining_bytes(&self) -> u128 {
        match self {
            Self::Hardware(rand) => rand.remaining_bytes(),
            Self::Software(rand) => rand.remaining_bytes(),
        }
    }

    /// Tries to perform a sequential fork of the current generator into `n_child` generators each
    /// able to yield `child_bytes` random bytes.
    ///
    /// If the total number of bytes to be generated exceeds the bound of the current generator,
    /// `None` is returned. Otherwise, we return an iterator over the children generators.
    pub fn try_sequential_fork(
        &mut self,
        n_child: usize,
        child_bytes: usize,
    ) -> Option<impl Iterator<Item = RandomGenerator>> {
        enum GeneratorChildIter<HardIter, SoftIter>
        where
            HardIter: Iterator<Item = HardAesCtrGenerator>,
            SoftIter: Iterator<Item = SoftAesCtrGenerator>,
        {
            Hardware(HardIter),
            Software(SoftIter),
        }

        impl<HardIter, SoftIter> Iterator for GeneratorChildIter<HardIter, SoftIter>
        where
            HardIter: Iterator<Item = HardAesCtrGenerator>,
            SoftIter: Iterator<Item = SoftAesCtrGenerator>,
        {
            type Item = RandomGenerator;

            fn next(&mut self) -> Option<Self::Item> {
                match self {
                    GeneratorChildIter::Hardware(ref mut iter) => {
                        iter.next().map(RandomGenerator::Hardware)
                    }
                    GeneratorChildIter::Software(ref mut iter) => {
                        iter.next().map(RandomGenerator::Software)
                    }
                }
            }
        }
        match self {
            Self::Hardware(ref mut rand) => rand
                .try_sequential_fork(ChildCount(n_child), BytesPerChild(child_bytes))
                .map(GeneratorChildIter::Hardware),
            Self::Software(ref mut rand) => rand
                .try_sequential_fork(ChildCount(n_child), BytesPerChild(child_bytes))
                .map(GeneratorChildIter::Software),
        }
    }

    /// Tries to perform a sequential fork of the current generator into `n_child` generators each
    /// able to yield `child_bytes` random bytes as a parallel iterator.
    ///
    /// If the total number of bytes to be generated exceeds the bound of the current generator,
    /// `None` is returned. Otherwise, we return a parallel iterator over the children generators.
    ///
    /// # Notes
    ///
    /// This method necessitates the "multithread" feature.
    #[cfg(feature = "multithread")]
    pub fn par_try_sequential_fork(
        &mut self,
        n_child: usize,
        child_bytes: usize,
    ) -> Option<impl IndexedParallelIterator<Item = RandomGenerator>> {
        use rayon::iter::plumbing::{Consumer, ProducerCallback, UnindexedConsumer};
        enum GeneratorChildIter<HardIter, SoftIter>
        where
            HardIter: IndexedParallelIterator<Item = HardAesCtrGenerator> + Send + Sync,
            SoftIter: IndexedParallelIterator<Item = SoftAesCtrGenerator> + Send + Sync,
        {
            Hardware(HardIter),
            Software(SoftIter),
        }
        impl<HardIter, SoftIter> ParallelIterator for GeneratorChildIter<HardIter, SoftIter>
        where
            HardIter: IndexedParallelIterator<Item = HardAesCtrGenerator> + Send + Sync,
            SoftIter: IndexedParallelIterator<Item = SoftAesCtrGenerator> + Send + Sync,
        {
            type Item = RandomGenerator;
            fn drive_unindexed<C>(self, consumer: C) -> <C as Consumer<Self::Item>>::Result
            where
                C: UnindexedConsumer<Self::Item>,
            {
                match self {
                    Self::Hardware(iter) => iter
                        .map(RandomGenerator::Hardware)
                        .drive_unindexed(consumer),
                    Self::Software(iter) => iter
                        .map(RandomGenerator::Software)
                        .drive_unindexed(consumer),
                }
            }
        }
        impl<HardIter, SoftIter> IndexedParallelIterator for GeneratorChildIter<HardIter, SoftIter>
        where
            HardIter: IndexedParallelIterator<Item = HardAesCtrGenerator> + Send + Sync,
            SoftIter: IndexedParallelIterator<Item = SoftAesCtrGenerator> + Send + Sync,
        {
            fn len(&self) -> usize {
                match self {
                    Self::Software(iter) => iter.len(),
                    Self::Hardware(iter) => iter.len(),
                }
            }
            fn drive<C: Consumer<Self::Item>>(
                self,
                consumer: C,
            ) -> <C as Consumer<Self::Item>>::Result {
                match self {
                    Self::Software(iter) => iter.map(RandomGenerator::Software).drive(consumer),
                    Self::Hardware(iter) => iter.map(RandomGenerator::Hardware).drive(consumer),
                }
            }
            fn with_producer<CB: ProducerCallback<Self::Item>>(
                self,
                callback: CB,
            ) -> <CB as ProducerCallback<Self::Item>>::Output {
                match self {
                    Self::Software(iter) => {
                        iter.map(RandomGenerator::Software).with_producer(callback)
                    }
                    Self::Hardware(iter) => {
                        iter.map(RandomGenerator::Hardware).with_producer(callback)
                    }
                }
            }
        }

        match self {
            Self::Hardware(ref mut rand) => rand
                .par_try_sequential_fork(ChildCount(n_child), BytesPerChild(child_bytes))
                .map(GeneratorChildIter::Hardware),
            Self::Software(ref mut rand) => rand
                .par_try_sequential_fork(ChildCount(n_child), BytesPerChild(child_bytes))
                .map(GeneratorChildIter::Software),
        }
    }

    /// Tries to perform an alternate fork of the current generator into `n_child` generators each
    /// able to yield as many random bytes as wanted.
    pub fn try_alternate_fork(
        &mut self,
        n_child: usize,
    ) -> Option<impl Iterator<Item = RandomGenerator>> {
        enum GeneratorChildIter<HardIter, SoftIter>
        where
            HardIter: Iterator<Item = HardAesCtrGenerator>,
            SoftIter: Iterator<Item = SoftAesCtrGenerator>,
        {
            Hardware(HardIter),
            Software(SoftIter),
        }

        impl<HardIter, SoftIter> Iterator for GeneratorChildIter<HardIter, SoftIter>
        where
            HardIter: Iterator<Item = HardAesCtrGenerator>,
            SoftIter: Iterator<Item = SoftAesCtrGenerator>,
        {
            type Item = RandomGenerator;

            fn next(&mut self) -> Option<Self::Item> {
                match self {
                    GeneratorChildIter::Hardware(ref mut iter) => {
                        iter.next().map(RandomGenerator::Hardware)
                    }
                    GeneratorChildIter::Software(ref mut iter) => {
                        iter.next().map(RandomGenerator::Software)
                    }
                }
            }
        }
        match self {
            Self::Hardware(ref mut rand) => rand
                .try_alternate_fork(ChildCount(n_child))
                .map(GeneratorChildIter::Hardware),
            Self::Software(ref mut rand) => rand
                .try_alternate_fork(ChildCount(n_child))
                .map(GeneratorChildIter::Software),
        }
    }

    /// Tries to perform an alternate fork of the current generator into `n_child` generators each
    /// able to yield as many random bytes as wanted.
    ///
    /// # Notes
    ///
    /// This method necessitates the "multithread" feature.
    #[cfg(feature = "multithread")]
    pub fn par_try_alternate_fork(
        &mut self,
        n_child: usize,
    ) -> Option<impl IndexedParallelIterator<Item = RandomGenerator>> {
        use rayon::iter::plumbing::{Consumer, ProducerCallback, UnindexedConsumer};
        enum GeneratorChildIter<HardIter, SoftIter>
        where
            HardIter: IndexedParallelIterator<Item = HardAesCtrGenerator> + Send + Sync,
            SoftIter: IndexedParallelIterator<Item = SoftAesCtrGenerator> + Send + Sync,
        {
            Hardware(HardIter),
            Software(SoftIter),
        }
        impl<HardIter, SoftIter> ParallelIterator for GeneratorChildIter<HardIter, SoftIter>
        where
            HardIter: IndexedParallelIterator<Item = HardAesCtrGenerator> + Send + Sync,
            SoftIter: IndexedParallelIterator<Item = SoftAesCtrGenerator> + Send + Sync,
        {
            type Item = RandomGenerator;
            fn drive_unindexed<C>(self, consumer: C) -> <C as Consumer<Self::Item>>::Result
            where
                C: UnindexedConsumer<Self::Item>,
            {
                match self {
                    Self::Hardware(iter) => iter
                        .map(RandomGenerator::Hardware)
                        .drive_unindexed(consumer),
                    Self::Software(iter) => iter
                        .map(RandomGenerator::Software)
                        .drive_unindexed(consumer),
                }
            }
        }
        impl<HardIter, SoftIter> IndexedParallelIterator for GeneratorChildIter<HardIter, SoftIter>
        where
            HardIter: IndexedParallelIterator<Item = HardAesCtrGenerator> + Send + Sync,
            SoftIter: IndexedParallelIterator<Item = SoftAesCtrGenerator> + Send + Sync,
        {
            fn len(&self) -> usize {
                match self {
                    Self::Software(iter) => iter.len(),
                    Self::Hardware(iter) => iter.len(),
                }
            }
            fn drive<C: Consumer<Self::Item>>(
                self,
                consumer: C,
            ) -> <C as Consumer<Self::Item>>::Result {
                match self {
                    Self::Software(iter) => iter.map(RandomGenerator::Software).drive(consumer),
                    Self::Hardware(iter) => iter.map(RandomGenerator::Hardware).drive(consumer),
                }
            }
            fn with_producer<CB: ProducerCallback<Self::Item>>(
                self,
                callback: CB,
            ) -> <CB as ProducerCallback<Self::Item>>::Output {
                match self {
                    Self::Software(iter) => {
                        iter.map(RandomGenerator::Software).with_producer(callback)
                    }
                    Self::Hardware(iter) => {
                        iter.map(RandomGenerator::Hardware).with_producer(callback)
                    }
                }
            }
        }

        match self {
            Self::Hardware(ref mut rand) => rand
                .par_try_alternate_fork(ChildCount(n_child))
                .map(GeneratorChildIter::Hardware),
            Self::Software(ref mut rand) => rand
                .par_try_alternate_fork(ChildCount(n_child))
                .map(GeneratorChildIter::Software),
        }
    }
}

impl Debug for RandomGenerator {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "RandomGenerator")
    }
}

impl Display for RandomGenerator {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "RandomGenerator")
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_uniformity() {
        // Checks that the PRNG generates uniform numbers
        let precision = 10f64.powi(-4);
        let n_samples = 10_000_000_usize;
        let mut generator = RandomGenerator::new(None);
        let mut counts = [0usize; 256];
        let expected_prob: f64 = 1. / 256.;
        for _ in 0..n_samples {
            counts[generator.generate_next().unwrap() as usize] += 1;
        }
        counts
            .iter()
            .map(|a| (*a as f64) / (n_samples as f64))
            .for_each(|a| assert!((a - expected_prob) < precision))
    }

    #[test]
    fn test_generator_determinism() {
        // Checks that given a state and a key, the PRNG is determinist.
        for _ in 0..100 {
            let key = software::dev_random();
            let mut first_generator = RandomGenerator::new(Some(key));
            let mut second_generator = RandomGenerator::new(Some(key));
            for _ in 0..128 {
                assert_eq!(
                    first_generator.generate_next(),
                    second_generator.generate_next()
                );
            }
        }
    }

    #[test]
    fn test_fork() {
        // Checks that forks returns a sequential child, and that the proper number of bytes can be
        // generated.
        let mut gen = RandomGenerator::new(None);
        let mut sequential = gen.try_sequential_fork(1, 10).unwrap().next().unwrap();
        for _ in 0..10 {
            sequential.generate_next();
        }
    }
}
