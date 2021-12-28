//! A module containing the [engines](crate::specification::engines) exposed by the core backend.

use crate::backends::core::private::crypto::secret::generators::{
    EncryptionRandomGenerator as ImplEncryptionRandomGenerator,
    SecretRandomGenerator as ImplSecretRandomGenerator,
};
use crate::specification::engines::sealed::AbstractEngineSeal;
use crate::specification::engines::AbstractEngine;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// The error which can occur in the execution of FHE operations, due to the core implementation.
///
/// # Note:
///
/// There is currently no such case, as the core implementation is not expected to undergo some
/// major issues unrelated to FHE.
#[derive(Debug)]
pub enum CoreError {
    Borrow,
}

impl Display for CoreError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CoreError::Borrow => {
                write!(f, "The borrowing rules were broken during execution.")
            }
        }
    }
}

impl Error for CoreError {}

use crate::backends::core::private::crypto::bootstrap::FourierBskBuffers;
use crate::prelude::{FourierLweBootstrapKey32, FourierLweBootstrapKey64, LweBootstrapKeyEntity};
use concrete_commons::parameters::{GlweSize, PolynomialSize};
use std::collections::BTreeMap;

#[derive(Ord, PartialOrd, Eq, PartialEq)]
pub(crate) struct FourierBufferKey(pub PolynomialSize, pub GlweSize);

/// The main engine exposed by the core backend.
pub struct CoreEngine {
    secret_generator: ImplSecretRandomGenerator,
    encryption_generator: ImplEncryptionRandomGenerator,
    fourier_bsk_buffers_u32: BTreeMap<FourierBufferKey, FourierBskBuffers<u32>>,
    fourier_bsk_buffers_u64: BTreeMap<FourierBufferKey, FourierBskBuffers<u64>>,
}

impl CoreEngine {
    pub(crate) fn get_fourier_bootstrap_u32_buffer(
        &mut self,
        fourier_bsk: &FourierLweBootstrapKey32,
    ) -> &mut FourierBskBuffers<u32> {
        let poly_size = fourier_bsk.polynomial_size();
        let glwe_size = fourier_bsk.glwe_dimension().to_glwe_size();
        let buffer_key = FourierBufferKey(poly_size, glwe_size);
        self.fourier_bsk_buffers_u32
            .entry(buffer_key)
            .or_insert_with(|| FourierBskBuffers::for_key(fourier_bsk))
    }

    pub(crate) fn get_fourier_bootstrap_u64_buffer(
        &mut self,
        fourier_bsk: &FourierLweBootstrapKey64,
    ) -> &mut FourierBskBuffers<u64> {
        let poly_size = fourier_bsk.polynomial_size();
        let glwe_size = fourier_bsk.glwe_dimension().to_glwe_size();
        let buffer_key = FourierBufferKey(poly_size, glwe_size);
        self.fourier_bsk_buffers_u64
            .entry(buffer_key)
            .or_insert_with(|| FourierBskBuffers::for_key(fourier_bsk))
    }
}

impl AbstractEngineSeal for CoreEngine {}

impl AbstractEngine for CoreEngine {
    type EngineError = CoreError;

    fn new() -> Result<Self, Self::EngineError> {
        Ok(CoreEngine {
            secret_generator: ImplSecretRandomGenerator::new(None),
            encryption_generator: ImplEncryptionRandomGenerator::new(None),
            fourier_bsk_buffers_u32: Default::default(),
            fourier_bsk_buffers_u64: Default::default(),
        })
    }
}

mod cleartext_creation;
mod cleartext_discarding_retrieval;
mod cleartext_retrieval;
mod cleartext_vector_creation;
mod cleartext_vector_discarding_retrieval;
mod cleartext_vector_retrieval;
mod destruction;
mod glwe_ciphertext_decryption;
mod glwe_ciphertext_discarding_decryption;
mod glwe_ciphertext_discarding_encryption;
mod glwe_ciphertext_encryption;
mod glwe_ciphertext_vector_decryption;
mod glwe_ciphertext_vector_discarding_decryption;
mod glwe_ciphertext_vector_discarding_encryption;
mod glwe_ciphertext_vector_encryption;
mod glwe_ciphertext_vector_zero_encryption;
mod glwe_ciphertext_zero_encryption;
mod glwe_secret_key_creation;
mod glwe_secret_key_to_lwe_secret_key_transmutation;
mod lwe_bootstrap_key_conversion;
mod lwe_bootstrap_key_creation;
mod lwe_ciphertext_cleartext_discarding_multiplication;
mod lwe_ciphertext_cleartext_fusing_multiplication;
mod lwe_ciphertext_decryption;
mod lwe_ciphertext_discarding_addition;
mod lwe_ciphertext_discarding_bootstrap;
mod lwe_ciphertext_discarding_decryption;
mod lwe_ciphertext_discarding_encryption;
mod lwe_ciphertext_discarding_extraction;
mod lwe_ciphertext_discarding_keyswitch;
mod lwe_ciphertext_discarding_negation;
mod lwe_ciphertext_encryption;
mod lwe_ciphertext_fusing_addition;
mod lwe_ciphertext_fusing_negation;
mod lwe_ciphertext_plaintext_discarding_addition;
mod lwe_ciphertext_plaintext_fusing_addition;
mod lwe_ciphertext_vector_decryption;
mod lwe_ciphertext_vector_discarding_affine_transformation;
mod lwe_ciphertext_vector_discarding_decryption;
mod lwe_ciphertext_vector_discarding_encryption;
mod lwe_ciphertext_vector_encryption;
mod lwe_ciphertext_vector_zero_encryption;
mod lwe_ciphertext_zero_encryption;
mod lwe_keyswitch_key_creation;
mod lwe_secret_key_creation;
mod plaintext_creation;
mod plaintext_discarding_retrieval;
mod plaintext_retrieval;
mod plaintext_vector_creation;
mod plaintext_vector_discarding_retrieval;
mod plaintext_vector_retrieval;
