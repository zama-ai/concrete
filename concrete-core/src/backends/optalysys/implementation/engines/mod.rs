use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::backends::core::private::crypto::secret::generators::{
    EncryptionRandomGenerator as ImplEncryptionRandomGenerator,
    SecretRandomGenerator as ImplSecretRandomGenerator,
};
use crate::backends::optalysys::entities::{
    OptalysysFourierLweBootstrapKey32, OptalysysFourierLweBootstrapKey64,
};

use crate::backends::optalysys::private::crypto::bootstrap::fourier::buffers::FourierBuffers;
use crate::prelude::sealed::AbstractEngineSeal;
use crate::prelude::LweBootstrapKeyEntity;
use crate::prelude::{AbstractEngine, FourierBufferKey};

#[derive(Debug)]
pub enum OptalysysError {
    DeviceNotFound,
}

impl Display for OptalysysError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            OptalysysError::DeviceNotFound => {
                write!(f, "No Optalysys chip detected on the machine.")
            }
        }
    }
}

impl Error for OptalysysError {}

/// The main engine exposed by the Optalysys backend.
pub struct OptalysysEngine {
    fourier_buffers_u32: BTreeMap<FourierBufferKey, FourierBuffers<u32>>,
    fourier_buffers_u64: BTreeMap<FourierBufferKey, FourierBuffers<u64>>,
}

impl OptalysysEngine {
    pub(crate) fn get_fourier_bootstrap_u32_buffer(
        &mut self,
        fourier_bsk: &OptalysysFourierLweBootstrapKey32,
    ) -> &mut FourierBuffers<u32> {
        let poly_size = fourier_bsk.polynomial_size();
        let glwe_size = fourier_bsk.glwe_dimension().to_glwe_size();
        let buffer_key = FourierBufferKey(poly_size, glwe_size);
        self.fourier_buffers_u32
            .entry(buffer_key)
            .or_insert_with(|| FourierBuffers::for_key(fourier_bsk))
    }

    pub(crate) fn get_fourier_bootstrap_u64_buffer(
        &mut self,
        fourier_bsk: &OptalysysFourierLweBootstrapKey64,
    ) -> &mut FourierBuffers<u64> {
        let poly_size = fourier_bsk.polynomial_size();
        let glwe_size = fourier_bsk.glwe_dimension().to_glwe_size();
        let buffer_key = FourierBufferKey(poly_size, glwe_size);
        self.fourier_buffers_u64
            .entry(buffer_key)
            .or_insert_with(|| FourierBuffers::for_key(fourier_bsk))
    }
}

impl AbstractEngineSeal for OptalysysEngine {}

impl AbstractEngine for OptalysysEngine {
    type EngineError = OptalysysError;

    fn new() -> Result<Self, Self::EngineError> {
        Ok(OptalysysEngine {
            fourier_buffers_u32: Default::default(),
            fourier_buffers_u64: Default::default(),
        })
    }
}

mod destruction;
mod lwe_bootstrap_key_conversion;
mod lwe_ciphertext_discarding_bootstrap;
