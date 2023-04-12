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
pub use ciphertext::LweCiphertext;

mod glwe_ciphertext;
pub use glwe_ciphertext::GlweCiphertext;

mod lwe_secret_key;
pub use lwe_secret_key::LweSecretKey;

mod glwe_secret_key;
pub use glwe_secret_key::GlweSecretKey;

mod ggsw_ciphertext;
pub use ggsw_ciphertext::GgswCiphertext;

mod bootstrap_key;
pub use bootstrap_key::BootstrapKey;

mod keyswitch_key;
pub use keyswitch_key::LweKeyswitchKey;

mod packing_keyswitch_key;
pub use packing_keyswitch_key::PackingKeyswitchKey;

mod packing_keyswitch_key_list;
pub use packing_keyswitch_key_list::PackingKeyswitchKeyList;

mod csprng;
pub use csprng::*;

mod ciphertext_list;
pub use ciphertext_list::LweCiphertextList;

mod glev_ciphertext;
pub use glev_ciphertext::GlevCiphertext;

mod lev_ciphertext;
pub use lev_ciphertext::LevCiphertext;

mod polynomial;
pub use polynomial::Polynomial;

mod polynomial_list;
pub use polynomial_list::PolynomialList;
