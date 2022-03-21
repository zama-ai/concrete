use crate::backends::core::private::crypto::ggsw::GgswCiphertext as ImplGgswCiphertext;
use crate::specification::entities::markers::{BinaryKeyDistribution, GgswCiphertextKind};
use crate::specification::entities::{AbstractEntity, GgswCiphertextEntity};
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
};
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

/// A structure representing a GGSW ciphertext with 32 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GgswCiphertext32(ImplGgswCiphertext<Vec<u32>>);
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
pub struct GgswCiphertext64(ImplGgswCiphertext<Vec<u64>>);
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
