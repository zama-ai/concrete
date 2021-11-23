use crate::specification::entities::markers::{GlweCiphertextKind, KeyFlavorMarker};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::{GlweDimension, PolynomialSize};

/// A trait implemented by types embodying a GLWE ciphertext.
///
/// A GLWE ciphertext is associated with a
/// [`KeyFlavor`](`GlweCiphertextEntity::KeyFlavor`) type, which conveys the flavor of secret
/// key it was encrypted with.
///
/// # Formal Definition
pub trait GlweCiphertextEntity: AbstractEntity<Kind = GlweCiphertextKind> {
    /// The flavor of key the ciphertext was encrypted with.
    type KeyFlavor: KeyFlavorMarker;

    /// Returns the GLWE dimension of the ciphertext.
    fn glwe_dimension(&self) -> GlweDimension;

    /// Returns the polynomial size of the ciphertext.
    fn polynomial_size(&self) -> PolynomialSize;
}
