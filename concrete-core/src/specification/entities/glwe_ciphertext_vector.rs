use crate::specification::entities::markers::{GlweCiphertextVectorKind, KeyFlavorMarker};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::{GlweCiphertextCount, GlweDimension, PolynomialSize};

/// A trait implemented by types embodying a GLWE ciphertext vector.
///
/// A GLWE ciphertext vector is associated with a
/// [`KeyFlavor`](`GlweCiphertextVectorEntity::KeyFlavor`) type, which conveys the flavor of secret
/// key it was encrypted with.
///
/// # Formal Definition
pub trait GlweCiphertextVectorEntity: AbstractEntity<Kind = GlweCiphertextVectorKind> {
    /// The flavor of key the ciphertext was encrypted with.
    type KeyFlavor: KeyFlavorMarker;

    /// Returns the GLWE dimension of the ciphertexts.
    fn glwe_dimension(&self) -> GlweDimension;

    /// Returns the polynomial size of the ciphertexts.
    fn polynomial_size(&self) -> PolynomialSize;

    /// Returns the number of ciphertexts in the vector.
    fn glwe_ciphertext_count(&self) -> GlweCiphertextCount;
}
