use super::engine_error;
use crate::prelude::{GlweSecretKeyEntity, PackingKeyswitchKeyEntity};
use crate::specification::engines::AbstractEngine;

use crate::specification::entities::LweSecretKeyEntity;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};

engine_error! {
    PackingKeyswitchKeyCreationError for PackingKeyswitchKeyCreationEngine @
    NullDecompositionBaseLog => "The key decomposition base log must be greater than zero.",
    NullDecompositionLevelCount => "The key decomposition level count must be greater than zero.",
    DecompositionTooLarge => "The decomposition precision (base log * level count) must not exceed \
                              the precision of the ciphertext."
}

impl<EngineError: std::error::Error> PackingKeyswitchKeyCreationError<EngineError> {
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

/// A trait for engines creating LWE packing keyswitch keys.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation creates an LWE packing keyswitch key
/// allowing to switch from the `input_key` LWE secret key to the `output_key` GLWE secret key.
///
/// # Formal Definition
pub trait PackingKeyswitchKeyCreationEngine<InputSecretKey, OutputSecretKey, PackingKeyswitchKey>:
    AbstractEngine
where
    InputSecretKey: LweSecretKeyEntity,
    OutputSecretKey: GlweSecretKeyEntity,
    PackingKeyswitchKey: PackingKeyswitchKeyEntity<
        InputKeyDistribution = InputSecretKey::KeyDistribution,
        OutputKeyDistribution = OutputSecretKey::KeyDistribution,
    >,
{
    /// Creates a packing keyswitch key.
    fn create_packing_keyswitch_key(
        &mut self,
        input_key: &InputSecretKey,
        output_key: &OutputSecretKey,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Result<PackingKeyswitchKey, PackingKeyswitchKeyCreationError<Self::EngineError>>;

    /// Unsafely creates a packing keyswitch key.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`PackingKeyswitchKeyCreationError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn create_packing_keyswitch_key_unchecked(
        &mut self,
        input_key: &InputSecretKey,
        output_key: &OutputSecretKey,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> PackingKeyswitchKey;
}
