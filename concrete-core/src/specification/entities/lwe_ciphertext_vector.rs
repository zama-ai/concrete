use crate::specification::entities::markers::{KeyFlavorMarker, LweCiphertextVectorKind};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::{LweCiphertextCount, LweDimension};

/// A trait implemented by types embodying an LWE ciphertext vector.
///
/// An LWE ciphertext vector is associated with a
/// [`KeyFlavor`](`LweCiphertextVectorEntity::KeyFlavor`) type, which conveys the flavor of secret
/// key it was encrypted with.
pub trait LweCiphertextVectorEntity: AbstractEntity<Kind = LweCiphertextVectorKind> {
    /// The flavor of key the ciphertext was encrypted with.
    type KeyFlavor: KeyFlavorMarker;

    /// Returns the LWE dimension of the ciphertexts.
    fn lwe_dimension(&self) -> LweDimension;

    /// Returns the number of ciphertexts contained in the vector.
    fn lwe_ciphertext_count(&self) -> LweCiphertextCount;
}
