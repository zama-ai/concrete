use crate::specification::entities::markers::{KeyFlavorMarker, LweKeyswitchKeyKind};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweDimension};

/// A trait implemented by types embodying an LWE keyswitch key.
///
/// An LWE keyswitch key is associated with two [`KeyFlavorMarker`] types:
///
/// + The [`InputKeyFlavor`](`LweKeyswitchKeyEntity::InputKeyFlavor`) type conveys the flavor of the
/// input secret key. + The [`OutputKeyFlavor`](`LweKeyswitchKeyEntity::OutputKeyFlavor`) type
/// conveys the flavor of the output secret key.
pub trait LweKeyswitchKeyEntity: AbstractEntity<Kind = LweKeyswitchKeyKind> {
    /// The flavor of key the input ciphertext is encrypted with.
    type InputKeyFlavor: KeyFlavorMarker;

    /// The flavor of the key the output ciphertext is encrypted with.
    type OutputKeyFlavor: KeyFlavorMarker;

    /// Returns the input LWE dimension of the key.
    fn input_lwe_dimension(&self) -> LweDimension;

    /// Returns the output lew dimension of the key.
    fn output_lwe_dimension(&self) -> LweDimension;

    /// Returns the number of decomposition levels of the key.
    fn decomposition_level_count(&self) -> DecompositionLevelCount;

    /// Returns the logarithm of the base used in the key.
    fn decomposition_base_log(&self) -> DecompositionBaseLog;
}
