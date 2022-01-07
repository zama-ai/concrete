use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::prelude::sealed::AbstractEngineSeal;
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

use crate::backends::optalysys::private::crypto::bootstrap::fourier::buffers::FourierBskBuffers;
use crate::prelude::{
    LweBootstrapKeyEntity, OptalysysFourierLweBootstrapKey32, OptalysysFourierLweBootstrapKey64,
};
use concrete_commons::parameters::{GlweSize, PolynomialSize};
use std::collections::BTreeMap;

/// The main engine exposed by the Optalysys backend.
pub struct OptalysysEngine {
    fourier_bsk_buffers_u32: BTreeMap<FourierBufferKey, FourierBskBuffers<u32>>,
    fourier_bsk_buffers_u64: BTreeMap<FourierBufferKey, FourierBskBuffers<u64>>,
}

impl OptalysysEngine {
    pub(crate) fn get_fourier_bootstrap_u32_buffer(
        &mut self,
        fourier_bsk: &OptalysysFourierLweBootstrapKey32,
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
        fourier_bsk: &OptalysysFourierLweBootstrapKey64,
    ) -> &mut FourierBskBuffers<u64> {
        let poly_size = fourier_bsk.polynomial_size();
        let glwe_size = fourier_bsk.glwe_dimension().to_glwe_size();
        let buffer_key = FourierBufferKey(poly_size, glwe_size);
        self.fourier_bsk_buffers_u64
            .entry(buffer_key)
            .or_insert_with(|| FourierBskBuffers::for_key(fourier_bsk))
    }
}

impl AbstractEngineSeal for OptalysysEngine {}

impl AbstractEngine for OptalysysEngine {
    type EngineError = OptalysysError;

    fn new() -> Result<Self, Self::EngineError> {
        Ok(OptalysysEngine {
            fourier_bsk_buffers_u32: Default::default(),
            fourier_bsk_buffers_u64: Default::default(),
        })
    }
}

mod lwe_ciphertext_discarding_bootstrap;
