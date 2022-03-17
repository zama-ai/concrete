//! A module containing generic benchmarking functions.
//!
//! Every submodule here is expected to contain a generic `bench` function which can be instantiated
//! with different engine types to benchmark an engine trait.

pub mod cleartext_creation;
pub mod cleartext_vector_creation;
pub mod glwe_ciphertext_decryption;
pub mod glwe_ciphertext_discarding_decryption;
pub mod glwe_ciphertext_discarding_encryption;
pub mod glwe_ciphertext_encryption;
pub mod glwe_ciphertext_vector_decryption;
pub mod glwe_ciphertext_vector_discarding_decryption;
pub mod glwe_ciphertext_vector_discarding_encryption;
pub mod glwe_ciphertext_vector_encryption;
pub mod glwe_ciphertext_vector_zero_encryption;
pub mod glwe_ciphertext_zero_encryption;
pub mod glwe_secret_key_creation;
pub mod lwe_bootstrap_key_conversion;
pub mod lwe_bootstrap_key_creation;
pub mod lwe_ciphertext_cleartext_discarding_multiplication;
pub mod lwe_ciphertext_cleartext_fusing_multiplication;
pub mod lwe_ciphertext_decryption;
pub mod lwe_ciphertext_discarding_addition;
pub mod lwe_ciphertext_discarding_bootstrap;
pub mod lwe_ciphertext_discarding_decryption;
pub mod lwe_ciphertext_discarding_encryption;
pub mod lwe_ciphertext_discarding_extraction;
pub mod lwe_ciphertext_discarding_keyswitch;
pub mod lwe_ciphertext_discarding_opposite;
pub mod lwe_ciphertext_encryption;
pub mod lwe_ciphertext_fusing_addition;
pub mod lwe_ciphertext_fusing_opposite;
pub mod lwe_ciphertext_plaintext_discarding_addition;
pub mod lwe_ciphertext_plaintext_fusing_addition;
pub mod lwe_ciphertext_vector_decryption;
pub mod lwe_ciphertext_vector_discarding_affine_transformation;
pub mod lwe_ciphertext_vector_discarding_decryption;
pub mod lwe_ciphertext_vector_discarding_encryption;
pub mod lwe_ciphertext_vector_encryption;
pub mod lwe_ciphertext_vector_zero_encryption;
pub mod lwe_ciphertext_zero_encryption;
pub mod lwe_keyswitch_key_creation;
pub mod lwe_secret_key_creation;
pub mod plaintext_creation;
pub mod plaintext_vector_creation;
