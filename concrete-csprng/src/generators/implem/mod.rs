#[cfg(feature = "generator_x86_64_aesni")]
mod aesni;
#[cfg(feature = "generator_x86_64_aesni")]
pub use aesni::*;

#[cfg(feature = "generator_soft")]
mod soft;
#[cfg(feature = "generator_soft")]
pub use soft::*;
