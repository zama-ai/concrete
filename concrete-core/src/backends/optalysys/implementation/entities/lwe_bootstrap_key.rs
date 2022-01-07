use crate::backends::core::private::math::fft::Complex64;
use crate::backends::optalysys::private::crypto::bootstrap::FourierBootstrapKey as ImplFourierBootstrapKey;
use crate::specification::entities::markers::{BinaryKeyDistribution, LweBootstrapKeyKind};
use crate::specification::entities::{AbstractEntity, LweBootstrapKeyEntity};
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};
use concrete_fftw::array::AlignedVec;

/// A structure representing an LWE bootstrap key with 32 bits of precision, in the fourier domain.
#[derive(Debug, Clone, PartialEq)]
pub struct OptalysysFourierLweBootstrapKey32(
    pub(crate) ImplFourierBootstrapKey<AlignedVec<Complex64>, u32>,
);
impl AbstractEntity for OptalysysFourierLweBootstrapKey32 {
    type Kind = LweBootstrapKeyKind;
}
impl LweBootstrapKeyEntity for OptalysysFourierLweBootstrapKey32 {
    type InputKeyDistribution = BinaryKeyDistribution;
    type OutputKeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.glwe_size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }

    fn input_lwe_dimension(&self) -> LweDimension {
        self.0.key_size()
    }

    fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.0.base_log()
    }

    fn decomposition_level_count(&self) -> DecompositionLevelCount {
        self.0.level_count()
    }
}

/// A structure representing an LWE bootstrap key with 64 bits of precision, in the fourier domain.
#[derive(Debug, Clone, PartialEq)]
pub struct OptalysysFourierLweBootstrapKey64(
    pub(crate) ImplFourierBootstrapKey<AlignedVec<Complex64>, u64>,
);
impl AbstractEntity for OptalysysFourierLweBootstrapKey64 {
    type Kind = LweBootstrapKeyKind;
}
impl LweBootstrapKeyEntity for OptalysysFourierLweBootstrapKey64 {
    type InputKeyDistribution = BinaryKeyDistribution;
    type OutputKeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.glwe_size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }

    fn input_lwe_dimension(&self) -> LweDimension {
        self.0.key_size()
    }

    fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.0.base_log()
    }

    fn decomposition_level_count(&self) -> DecompositionLevelCount {
        self.0.level_count()
    }
}
