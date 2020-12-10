//! Cryptographically secure pseudo random number generator, that uses AES in CTR mode.
//!
//! Welcome to the `concrete-csprng` documentation.
//!
//! This crate contains a reasonably fast cryptographically secure pseudo-random number generator.
//! The implementation is based on the AES blockcipher used in counter (CTR) mode, as presented
//! in the ISO/IEC 18033-4 document.
//!
//! Currently the implementation uses a mix of `aes` and `sse2` intrinsics, that should be
//! available on most modern laptop and desktop computers.

#[cfg(all(target_arch = "x86_64"))]
pub use aesni::RandomGenerator;

#[cfg(all(target_arch = "x86_64",))]
mod aesni;
