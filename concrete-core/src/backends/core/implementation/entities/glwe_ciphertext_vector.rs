use super::super::super::private::crypto::glwe::GlweList as ImplGlweList;
use crate::specification::entities::markers::{BinaryKeyDistribution, GlweCiphertextVectorKind};
use crate::specification::entities::{AbstractEntity, GlweCiphertextVectorEntity};
use concrete_commons::parameters::{GlweCiphertextCount, GlweDimension, PolynomialSize};
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

/// A structure representing a vector of GLWE ciphertexts with 32 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GlweCiphertextVector32(pub(crate) ImplGlweList<Vec<u32>>);
impl AbstractEntity for GlweCiphertextVector32 {
    type Kind = GlweCiphertextVectorKind;
}
impl GlweCiphertextVectorEntity for GlweCiphertextVector32 {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }

    fn glwe_ciphertext_count(&self) -> GlweCiphertextCount {
        GlweCiphertextCount(self.0.ciphertext_count().0)
    }
}

/// A structure representing a vector of GLWE ciphertexts with 64 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GlweCiphertextVector64(pub(crate) ImplGlweList<Vec<u64>>);
impl AbstractEntity for GlweCiphertextVector64 {
    type Kind = GlweCiphertextVectorKind;
}
impl GlweCiphertextVectorEntity for GlweCiphertextVector64 {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }

    fn glwe_ciphertext_count(&self) -> GlweCiphertextCount {
        GlweCiphertextCount(self.0.ciphertext_count().0)
    }
}
