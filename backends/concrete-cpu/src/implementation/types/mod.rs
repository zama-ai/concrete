#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecompParams {
    pub level: usize,
    pub base_log: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlweParams {
    pub dimension: usize,
    pub polynomial_size: usize,
}

impl GlweParams {
    pub fn lwe_dimension(self) -> usize {
        self.dimension * self.polynomial_size
    }
}

pub fn int_log2(a: usize) -> usize {
    debug_assert!(a.is_power_of_two());
    (a as f64).log2().ceil() as usize
}

mod ciphertext;
pub use ciphertext::*;

mod glwe_ciphertext;
pub use glwe_ciphertext::*;

mod lwe_secret_key;
pub use lwe_secret_key::*;

mod glwe_secret_key;
pub use glwe_secret_key::*;

mod ggsw_ciphertext;
pub use ggsw_ciphertext::*;

mod bootstrap_key;
pub use bootstrap_key::*;

mod keyswitch_key;
pub use keyswitch_key::*;

mod packing_keyswitch_key;
pub use packing_keyswitch_key::*;

pub mod packing_keyswitch_key_list;

mod csprng;
pub use csprng::*;

pub mod ciphertext_list;
pub mod glev_ciphertext;
pub mod polynomial;
pub mod polynomial_list;
