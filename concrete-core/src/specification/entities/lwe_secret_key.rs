use crate::prelude::{KeyDistributionMarker, LweSecretKeyKind};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::LweDimension;

/// A trait implemented by types embodying an LWE secret key.
///
/// An LWE secret key is associated with a
/// [`KeyDistribution`](`LweSecretKeyEntity::KeyDistribution`) type, which conveys its distribution.
///
/// # Formal Definition
pub trait LweSecretKeyEntity: AbstractEntity<Kind = LweSecretKeyKind> {
    /// The distribution of this key.
    type KeyDistribution: KeyDistributionMarker;

    /// Returns the LWE dimension of the key.
    fn lwe_dimension(&self) -> LweDimension;
}
