use crate::prelude::{KeyDistributionMarker, LweCiphertextKind};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::LweDimension;

/// A trait implemented by types embodying an LWE ciphertext.
///
/// An LWE ciphertext is associated with a
/// [`KeyDistribution`](`LweCiphertextEntity::KeyDistribution`) type, which conveys the distribution
/// of the secret key it was encrypted with.
///
/// # Formal Definition
pub trait LweCiphertextEntity: AbstractEntity<Kind = LweCiphertextKind> {
    /// The distribution of the key the ciphertext was encrypted with.
    type KeyDistribution: KeyDistributionMarker;

    /// Returns the LWE dimension of the ciphertext.
    fn lwe_dimension(&self) -> LweDimension;
}
