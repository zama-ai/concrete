use crate::crypto::bootstrap::FourierBootstrapKey;
use crate::math::fft::Complex64;
use crate::math::tensor::Tensor;
use crate::math::torus::UnsignedTorus;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
};
use concrete_fftw::array::AlignedVec;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Represents the kind of a bsk.
#[derive(Serialize, Deserialize, PartialEq)]
pub enum BskKind {
    Fourier,
    Standard,
}

/// This structure contains only the data of a BSK. Used to implement equality and serialization
/// when the bootstrap key contains other fields (fft, buffers, etc...).
#[derive(Serialize, Deserialize, PartialEq)]
pub struct SurrogateBsk<Cont, Scalar> {
    pub kind: BskKind,
    pub version: String,
    pub tensor: Tensor<Cont>,
    pub poly_size: PolynomialSize,
    pub glwe_size: GlweSize,
    pub decomp_level: DecompositionLevelCount,
    pub decomp_base_log: DecompositionBaseLog,
    pub ciphertext_scalar: PhantomData<Scalar>,
}

impl<Scalar> SurrogateBsk<AlignedVec<Complex64>, Scalar>
where
    Scalar: UnsignedTorus,
{
    /// Turns this surrogate bsk into a fresh fourier bootstrap key.
    pub fn into_fourier_bsk(self) -> FourierBootstrapKey<AlignedVec<Complex64>, Scalar> {
        FourierBootstrapKey::from_container(
            self.tensor.into_container(),
            self.glwe_size,
            self.poly_size,
            self.decomp_level,
            self.decomp_base_log,
        )
    }
}
