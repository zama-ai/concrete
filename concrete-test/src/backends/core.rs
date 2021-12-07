use crate::generics::lwe_ciphertext_encryption;
use crate::prototyping::{Precision32, Precision64};
use concrete_core::prelude::*;

#[test]
fn test_binary_lwe_ciphertext_encryption_u32() {
    lwe_ciphertext_encryption::test_binary::<
        Precision32,
        CoreEngine,
        LweSecretKey32,
        Plaintext32,
        LweCiphertext32,
    >();
}

#[test]
fn test_binary_lwe_ciphertext_encryption_u64() {
    lwe_ciphertext_encryption::test_binary::<
        Precision64,
        CoreEngine,
        LweSecretKey64,
        Plaintext64,
        LweCiphertext64,
    >();
}
