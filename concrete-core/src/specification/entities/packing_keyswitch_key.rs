use crate::specification::entities::markers::{KeyDistributionMarker, PackingKeyswitchKeyKind};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};

/// A trait implemented by types embodying a packing keyswitch key.
///
/// A packing keyswitch key is associated with two [`KeyDistributionMarker`] types:
///
/// + The [`InputKeyDistribution`](`PackingKeyswitchKeyEntity::InputKeyDistribution`) type conveys
/// the distribution of the input secret key.
/// + The [`OutputKeyDistribution`](`PackingKeyswitchKeyEntity::OutputKeyDistribution`) type conveys
/// the distribution of the output secret key.
///
/// # Formal Definition
pub trait PackingKeyswitchKeyEntity: AbstractEntity<Kind = PackingKeyswitchKeyKind> {
    /// The distribution of the key the input ciphertext is encrypted with.
    type InputKeyDistribution: KeyDistributionMarker;

    /// The distribution of the key the output ciphertext is encrypted with.
    type OutputKeyDistribution: KeyDistributionMarker;

    /// Returns the input LWE dimension of the key.
    fn input_lwe_dimension(&self) -> LweDimension;

    /// Returns the output GLWE dimension of the key.
    fn output_glwe_dimension(&self) -> GlweDimension;

    /// Returns the output polynomial degree of the key.
    fn output_polynomial_size(&self) -> PolynomialSize;

    /// Returns the number of decomposition levels of the key.
    fn decomposition_level_count(&self) -> DecompositionLevelCount;

    /// Returns the logarithm of the base used in the key.
    fn decomposition_base_log(&self) -> DecompositionBaseLog;
}
