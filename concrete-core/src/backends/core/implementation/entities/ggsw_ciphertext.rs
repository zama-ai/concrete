use crate::backends::core::private::crypto::ggsw::{
    FourierGgswCiphertext as ImplFourierGgswCiphertext,
    StandardGgswCiphertext as ImplStandardGgswCiphertext,
};
use crate::backends::core::private::math::fft::Complex64;
use crate::prelude::{BinaryKeyDistribution, GgswCiphertextKind};
use crate::specification::entities::{AbstractEntity, GgswCiphertextEntity};
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
};
use concrete_fftw::array::AlignedVec;
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

/// A structure representing a GGSW ciphertext with 32 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GgswCiphertext32(pub(crate) ImplStandardGgswCiphertext<Vec<u32>>);
impl AbstractEntity for GgswCiphertext32 {
    type Kind = GgswCiphertextKind;
}
impl GgswCiphertextEntity for GgswCiphertext32 {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.glwe_size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }

    fn decomposition_level_count(&self) -> DecompositionLevelCount {
        self.0.decomposition_level_count()
    }

    fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.0.decomposition_base_log()
    }
}

/// A structure representing a GGSW ciphertext with 64 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GgswCiphertext64(pub(crate) ImplStandardGgswCiphertext<Vec<u64>>);
impl AbstractEntity for GgswCiphertext64 {
    type Kind = GgswCiphertextKind;
}
impl GgswCiphertextEntity for GgswCiphertext64 {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.glwe_size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }

    fn decomposition_level_count(&self) -> DecompositionLevelCount {
        self.0.decomposition_level_count()
    }

    fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.0.decomposition_base_log()
    }
}

/// A structure representing a GGSW ciphertext with 64 bits of precision in the Fourier domain.
/// Note: The name FourierGgswCiphertext64 refers to the bit size of the coefficients in the
/// standard domain. Complex coefficients (eg in the Fourier domain) are always represented on 64
/// bits.
#[derive(Debug, Clone, PartialEq)]
pub struct FourierGgswCiphertext64(
    pub(crate) ImplFourierGgswCiphertext<AlignedVec<Complex64>, u64>,
);
impl AbstractEntity for FourierGgswCiphertext64 {
    type Kind = GgswCiphertextKind;
}
impl GgswCiphertextEntity for FourierGgswCiphertext64 {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.glwe_size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }

    fn decomposition_level_count(&self) -> DecompositionLevelCount {
        self.0.decomposition_level_count()
    }

    fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.0.decomposition_base_log()
    }
}

/// A structure representing a GGSW ciphertext with 32 bits of precision in the Fourier domain.
/// Note: The name FourierGgswCiphertext32 refers to the bit size of the coefficients in the
/// standard domain. Complex coefficients (eg in the Fourier domain) are always represented on 64
/// bits.
#[derive(Debug, Clone, PartialEq)]
pub struct FourierGgswCiphertext32(
    pub(crate) ImplFourierGgswCiphertext<AlignedVec<Complex64>, u32>,
);
impl AbstractEntity for FourierGgswCiphertext32 {
    type Kind = GgswCiphertextKind;
}
impl GgswCiphertextEntity for FourierGgswCiphertext32 {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.glwe_size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }

    fn decomposition_level_count(&self) -> DecompositionLevelCount {
        self.0.decomposition_level_count()
    }

    fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.0.decomposition_base_log()
    }
}
