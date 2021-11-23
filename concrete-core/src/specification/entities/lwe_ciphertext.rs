use crate::specification::entities::markers::{KeyFlavorMarker, LweCiphertextKind};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::LweDimension;

/// A trait implemented by types embodying an LWE ciphertext.
///
/// An LWE ciphertext is associated with a
/// [`KeyFlavor`](`LweCiphertextEntity::KeyFlavor`) type, which conveys the flavor of secret
/// key it was encrypted with.
///
/// # Formal Definition
pub trait LweCiphertextEntity: AbstractEntity<Kind = LweCiphertextKind> {
    /// The flavor of key the ciphertext was encrypted with.
    type KeyFlavor: KeyFlavorMarker;

    /// Returns the LWE dimension of the ciphertext.
    fn lwe_dimension(&self) -> LweDimension;
}
