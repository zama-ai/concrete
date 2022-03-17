use concrete_fftw::array::AlignedVec;
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use concrete_commons::parameters::{GlweDimension, PolynomialSize};

use crate::backends::core::private::math::fft::Complex64;
use crate::specification::entities::markers::{BinaryKeyDistribution, GlweCiphertextKind};
use crate::specification::entities::{AbstractEntity, GlweCiphertextEntity};

use super::super::super::private::crypto::glwe::{
    FourierGlweCiphertext as ImplFourierGlweCiphertext, GlweCiphertext as ImplGlweCiphertext,
};

/// A structure representing a GLWE ciphertext with 32 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GlweCiphertext32(pub(crate) ImplGlweCiphertext<Vec<u32>>);

impl AbstractEntity for GlweCiphertext32 {
    type Kind = GlweCiphertextKind;
}

impl GlweCiphertextEntity for GlweCiphertext32 {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }
}

/// A structure representing a GLWE ciphertext with 64 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GlweCiphertext64(pub(crate) ImplGlweCiphertext<Vec<u64>>);

impl AbstractEntity for GlweCiphertext64 {
    type Kind = GlweCiphertextKind;
}

impl GlweCiphertextEntity for GlweCiphertext64 {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }
}

/// A structure representing a Fourier GLWE ciphertext with 32 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]

pub struct FourierGlweCiphertext32(
    pub(crate) ImplFourierGlweCiphertext<AlignedVec<Complex64>, u32>,
);
impl AbstractEntity for FourierGlweCiphertext32 {
    type Kind = GlweCiphertextKind;
}
impl GlweCiphertextEntity for FourierGlweCiphertext32 {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.glwe_size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }
}

/// A structure representing a Fourier GLWE ciphertext with 64 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct FourierGlweCiphertext64(
    pub(crate) ImplFourierGlweCiphertext<AlignedVec<Complex64>, u64>,
);
impl AbstractEntity for FourierGlweCiphertext64 {
    type Kind = GlweCiphertextKind;
}
impl GlweCiphertextEntity for FourierGlweCiphertext64 {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.glwe_size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }
}
