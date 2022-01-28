use crate::specification::entities::markers::{GlweRelinearizationKeyKind, KeyDistributionMarker};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
};

/// A trait implemented by types embodying a GLWE relinearization key.
///
/// A GLWE relinearization key, which is the tensor product of two GLWE secret keys
///
/// # Formal Definition
pub trait GlweRelinearizationKeyEntity: AbstractEntity<Kind = GlweRelinearizationKeyKind> {
    /// The distribution of the underlying GLWE secret keys
    type KeyDistribution: KeyDistributionMarker;

    /// Returns the GLWE dimension of the key.
    fn glwe_dimension(&self) -> GlweDimension;

    /// Returns the polynomial size of the key.
    fn polynomial_size(&self) -> PolynomialSize;

    /// Returns the number of decomposition levels of the key.
    fn decomposition_level_count(&self) -> DecompositionLevelCount;

    /// Returns the logarithm of the base used in the key.
    fn decomposition_base_log(&self) -> DecompositionBaseLog;
}
