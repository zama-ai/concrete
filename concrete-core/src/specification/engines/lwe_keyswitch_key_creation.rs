use super::engine_error;
use crate::specification::engines::AbstractEngine;

use crate::specification::entities::{LweKeyswitchKeyEntity, LweSecretKeyEntity};
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};

engine_error! {
    LweKeyswitchKeyCreationError for LweKeyswitchKeyCreationEngine @
    NullDecompositionBaseLog => "The key decomposition base log must be greater than zero.",
    NullDecompositionLevelCount => "The key decomposition level count must be greater than zero.",
    DecompositionTooLarge => "The decomposition precision (base log * level count) must not exceed \
                              the precision of the ciphertext."
}

impl<EngineError: std::error::Error> LweKeyswitchKeyCreationError<EngineError> {
    /// Validates the inputs
    pub fn perform_generic_checks(
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        integer_precision: usize,
    ) -> Result<(), Self> {
        if decomposition_base_log.0 == 0 {
            return Err(Self::NullDecompositionBaseLog);
        }

        if decomposition_level_count.0 == 0 {
            return Err(Self::NullDecompositionLevelCount);
        }

        if decomposition_level_count.0 * decomposition_base_log.0 > integer_precision {
            return Err(Self::DecompositionTooLarge);
        }

        Ok(())
    }
}

/// A trait for engines creating LWE keyswitch keys.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation creates an LWE keyswitch key allowing to
/// switch from the `input_key` LWE secret key to the `output_key` LWE secret key.
///
/// # Formal Definition
pub trait LweKeyswitchKeyCreationEngine<InputSecretKey, OutputSecretKey, KeyswitchKey>:
    AbstractEngine
where
    InputSecretKey: LweSecretKeyEntity,
    OutputSecretKey: LweSecretKeyEntity,
    KeyswitchKey: LweKeyswitchKeyEntity<
        InputKeyDistribution = InputSecretKey::KeyDistribution,
        OutputKeyDistribution = OutputSecretKey::KeyDistribution,
    >,
{
    /// Creates an LWE keyswitch key.
    fn create_lwe_keyswitch_key(
        &mut self,
        input_key: &InputSecretKey,
        output_key: &OutputSecretKey,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Result<KeyswitchKey, LweKeyswitchKeyCreationError<Self::EngineError>>;

    /// Unsafely creates an LWE keyswitch key.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweKeyswitchKeyCreationError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn create_lwe_keyswitch_key_unchecked(
        &mut self,
        input_key: &InputSecretKey,
        output_key: &OutputSecretKey,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> KeyswitchKey;
}
