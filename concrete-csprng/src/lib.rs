//! Cryptographically secure pseudo random number generator, that uses AES in CTR mode.
//!
//! Welcome to the `concrete-csprng` documentation.
//!
//! This crate contains a reasonably fast cryptographically secure pseudo-random number generator.
//! The implementation is based on the AES blockcipher used in counter (CTR) mode, as presented
//! in the ISO/IEC 18033-4 document.

use crate::ctr::{BytesPerChild, ChildCount, HardAesCtrGenerator, SoftAesCtrGenerator, State};
use ctr::AesBatchedGenerator;
pub use software::set_soft_rdseed_secret;
use std::fmt::{Debug, Display, Formatter, Result};

mod aesni;
mod ctr;
mod software;

/// Represents a key used in the AES ciphertext.
#[derive(Clone, Copy)]
pub struct AesKey(u128);

/// The pseudorandom number generator.
///
/// If the correct instructions set are available on the machine, an hardware accelerated version
/// of the generator will be used. If not, a fallback software implementation is used instead.
///
/// # Note
///
/// The software version can also be used on accelerated hardware, by enabling the `slow` feature.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum RandomGenerator {
    Software(SoftAesCtrGenerator),
    Hardware(HardAesCtrGenerator),
}

impl RandomGenerator {
    /// Builds a new random generator, optionally using the input keys and state as initial values.
    pub fn new(key: Option<AesKey>, state: Option<State>) -> RandomGenerator {
        if cfg!(feature = "slow") {
            return RandomGenerator::new_software(key, state);
        }
        RandomGenerator::new_hardware(key, state.clone())
            .unwrap_or_else(|| RandomGenerator::new_software(key.clone(), state.clone()))
    }

    /// Builds a new software random generator, optionally using the input keys and state
    /// as initial values.
    pub fn new_software(key: Option<AesKey>, state: Option<State>) -> RandomGenerator {
        RandomGenerator::Software(SoftAesCtrGenerator::new(key, state, None))
    }

    /// Tries to build a new hardware random generator, optionally using the input keys and state as
    /// initial values.
    pub fn new_hardware(key: Option<AesKey>, state: Option<State>) -> Option<RandomGenerator> {
        if !is_x86_feature_detected!("aes")
            || !is_x86_feature_detected!("rdseed")
            || !is_x86_feature_detected!("sse2")
        {
            return None;
        }
        return Some(RandomGenerator::Hardware(HardAesCtrGenerator::new(
            key, state, None,
        )));
    }

    /// Returns the current state of the generator.
    pub fn get_state(&self) -> &State {
        match self {
            Self::Hardware(ref rand) => rand.get_state(),
            Self::Software(ref rand) => rand.get_state(),
        }
    }

    /// Yields the next byte from the generator.
    pub fn generate_next(&mut self) -> u8 {
        match self {
            Self::Hardware(ref mut rand) => rand.generate_next(),
            Self::Software(ref mut rand) => rand.generate_next(),
        }
    }

    /// Returns whether the generator is bounded.
    pub fn is_bounded(&self) -> bool {
        match self {
            Self::Hardware(rand) => rand.is_bounded(),
            Self::Software(rand) => rand.is_bounded(),
        }
    }

    /// Tries to fork the current generator into `n_child` generators each able to yield
    /// `child_bytes` random bytes.
    ///
    /// If the total number of bytes to be generated exceeds the bound of the current generator,
    /// `None` is returned. Otherwise, we return an iterator over the children generators.
    pub fn try_fork(
        &mut self,
        n_child: ChildCount,
        child_bytes: BytesPerChild,
    ) -> Option<impl Iterator<Item = RandomGenerator>> {
        match self {
            Self::Hardware(ref mut rand) => rand
                .try_fork(n_child, child_bytes)
                .map(|a| GeneratorChildIter::Hardware(a)),
            Self::Software(ref mut rand) => rand
                .try_fork(n_child, child_bytes)
                .map(|a| GeneratorChildIter::Software(a)),
        }
    }
}

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
    use crate::ctr::AesCtr;

    #[test]
    fn test_uniformity() {
        // Checks that the PRNG generates uniform numbers
        let precision = 10f64.powi(-4);
        let n_samples = 10_000_000_usize;
        let mut generator = RandomGenerator::new(None, None);
        let mut counts = [0usize; 256];
        let expected_prob: f64 = 1. / 256.;
        for _ in 0..n_samples {
            counts[generator.generate_next() as usize] += 1;
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
            let state = software::dev_random();
            let mut first_generator = RandomGenerator::new(
                Some(AesKey(key)),
                Some(State::from_aes_counter(AesCtr(state))),
            );
            let mut second_generator = RandomGenerator::new(
                Some(AesKey(key)),
                Some(State::from_aes_counter(AesCtr(state))),
            );
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
        // Checks that forks returns a bounded child, and that the proper number of bytes can be
        // generated.
        let mut gen = RandomGenerator::new(None, None);
        let mut bounded = gen
            .try_fork(ChildCount(1), BytesPerChild(10))
            .unwrap()
            .next()
            .unwrap();
        assert!(bounded.is_bounded());
        assert!(!gen.is_bounded());
        for _ in 0..10 {
            bounded.generate_next();
        }
    }

    #[test]
    #[should_panic]
    fn test_bounded_panic() {
        // Checks that a bounded prng panics when exceeding the allowed number of bytes.
        let mut gen = RandomGenerator::new(None, None);
        let mut bounded = gen
            .try_fork(ChildCount(1), BytesPerChild(10))
            .unwrap()
            .next()
            .unwrap();
        assert!(bounded.is_bounded());
        assert!(!gen.is_bounded());
        for _ in 0..11 {
            bounded.generate_next();
        }
    }
}

#[cfg(all(
    test,
    target_arch = "x86_64",
    target_feature = "aes",
    target_feature = "sse2",
    target_feature = "rdseed"
))]
mod test_aes {
    use super::*;
    use crate::ctr::AesCtr;

    #[test]
    fn test_soft_hard_eq() {
        // Checks that both the software and hardware prng outputs the same values.
        let mut soft = SoftAesCtrGenerator::new(
            Some(AesKey(0)),
            Some(State::from_aes_counter(AesCtr(0))),
            None,
        );
        let mut hard = HardAesCtrGenerator::new(
            Some(AesKey(0)),
            Some(State::from_aes_counter(AesCtr(0))),
            None,
        );
        for _ in 0..1000 {
            assert_eq!(soft.generate_next(), hard.generate_next());
        }
    }
}
