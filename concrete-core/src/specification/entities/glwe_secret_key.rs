use crate::prelude::{GlweSecretKeyKind, KeyDistributionMarker};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::{GlweDimension, PolynomialSize};

/// A trait implemented by types embodying a GLWE secret key.
///
/// A GLWE secret key is associated with a
/// [`KeyDistribution`](`GlweSecretKeyEntity::KeyDistribution`) type, which conveys its
/// distribution.
///
/// # Formal Definition
pub trait GlweSecretKeyEntity: AbstractEntity<Kind = GlweSecretKeyKind> {
    /// The distribution of this key.
    type KeyDistribution: KeyDistributionMarker;

    /// Returns the GLWE dimension of the key.
    fn glwe_dimension(&self) -> GlweDimension;

    /// Returns the polynomial size of the key.
    fn polynomial_size(&self) -> PolynomialSize;
}
