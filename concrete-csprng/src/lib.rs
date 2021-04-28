//! Cryptographically secure pseudo random number generator, that uses AES in CTR mode.
//!
//! Welcome to the `concrete-csprng` documentation.
//!
//! This crate contains a reasonably fast cryptographically secure pseudo-random number generator.
//! The implementation is based on the AES blockcipher used in counter (CTR) mode, as presented
//! in the ISO/IEC 18033-4 document.

use std::fmt::{Debug, Display, Formatter, Result};

mod aesni;
mod software;
pub use software::set_soft_rdseed_secret;

/// The pseudorandom number generator.
///
/// If the correct instructions set are available on the machine, an hardware accelerated version
/// of the generator will be used. If not, a fallback software implementation is used instead.
///
/// # Note
///
/// The software version can also be used on accelerated hardware, by enabling the `slow` feature.
#[allow(clippy::large_enum_variant)]
pub enum RandomGenerator {
    Hardware(aesni::RandomGenerator),
    Software(software::RandomGenerator),
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

impl Default for RandomGenerator {
    fn default() -> Self {
        RandomGenerator::new(None, None)
    }
}

impl RandomGenerator {
    /// Builds a new random generator, optionally using the input keys and state as initial values.
    pub fn new(key: Option<u128>, state: Option<u128>) -> RandomGenerator {
        if !cfg!(feature = "slow") {
            return RandomGenerator::new_software(key, state);
        }
        RandomGenerator::new_hardware(key, state)
            .unwrap_or_else(|| RandomGenerator::new_software(key, state))
    }

    /// Builds a new software random generator, optionally using the input keys and state
    /// as initial values.
    pub fn new_software(key: Option<u128>, state: Option<u128>) -> RandomGenerator {
        RandomGenerator::Software(software::RandomGenerator::new(key, state))
    }

    /// Tries to build a new hardware random generator, optionally using the input keys and state as
    /// initial values.
    pub fn new_hardware(key: Option<u128>, state: Option<u128>) -> Option<RandomGenerator> {
        if !is_x86_feature_detected!("aes")
            || !is_x86_feature_detected!("rdseed")
            || !is_x86_feature_detected!("sse2")
        {
            return None;
        }
        return Some(RandomGenerator::Hardware(aesni::RandomGenerator::new(
            key, state,
        )));
    }

    /// Yields the next byte from the generator.
    pub fn generate_next(&mut self) -> u8 {
        match self {
            RandomGenerator::Hardware(rand) => rand.generate_next(),
            RandomGenerator::Software(rand) => rand.generate_next(),
        }
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
        for _ in 0..100 {
            let key = software::dev_random();
            let state = software::dev_random();
            let mut first_generator = RandomGenerator::new(Some(key), Some(state));
            let mut second_generator = RandomGenerator::new(Some(key), Some(state));
            for _ in 0..128 {
                assert_eq!(
                    first_generator.generate_next(),
                    second_generator.generate_next()
                );
            }
        }
    }
}
