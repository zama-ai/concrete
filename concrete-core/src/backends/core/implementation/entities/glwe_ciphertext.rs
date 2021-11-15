use super::super::super::private::crypto::glwe::GlweCiphertext as ImplGlweCiphertext;
use crate::specification::entities::markers::{BinaryKeyFlavor, GlweCiphertextKind};
use crate::specification::entities::{AbstractEntity, GlweCiphertextEntity};
use concrete_commons::parameters::{GlweDimension, PolynomialSize};

/// A structure representing a GLWE ciphertext with 32 bits of precision.
#[derive(Debug, Clone, PartialEq)]
pub struct GlweCiphertext32(pub(crate) ImplGlweCiphertext<Vec<u32>>);
impl AbstractEntity for GlweCiphertext32 {
    type Kind = GlweCiphertextKind;
}
impl GlweCiphertextEntity for GlweCiphertext32 {
    type KeyFlavor = BinaryKeyFlavor;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }
}

/// A structure representing a GLWE ciphertext with 64 bits of precision.
#[derive(Debug, Clone, PartialEq)]
pub struct GlweCiphertext64(pub(crate) ImplGlweCiphertext<Vec<u64>>);
impl AbstractEntity for GlweCiphertext64 {
    type Kind = GlweCiphertextKind;
}
impl GlweCiphertextEntity for GlweCiphertext64 {
    type KeyFlavor = BinaryKeyFlavor;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }
}
