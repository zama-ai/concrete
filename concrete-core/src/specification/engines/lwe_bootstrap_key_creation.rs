use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{
    GlweSecretKeyEntity, LweBootstrapKeyEntity, LweSecretKeyEntity,
};
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};

engine_error! {
    LweBootstrapKeyCreationError for LweBootstrapKeyCreationEngine @
    NullDecompositionBaseLog => "The key decomposition base log must be greater than zero.",
    NullDecompositionLevelCount => "The key decomposition level count must be greater than zero.",
    DecompositionTooLarge => "The decomposition precision (base log * level count) must not exceed \
                              the precision of the ciphertext."
}

/// A trait for engines creating LWE bootstrap keys.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation creates an LWE bootstrap key from the
/// `input_key` LWE secret key, and the `output_key` GLWE secret key.
///
/// # Formal Definition
pub trait LweBootstrapKeyCreationEngine<LweSecretKey, GlweSecretKey, BootstrapKey>:
    AbstractEngine
where
    BootstrapKey: LweBootstrapKeyEntity,
    LweSecretKey: LweSecretKeyEntity<KeyFlavor = BootstrapKey::InputKeyFlavor>,
    GlweSecretKey: GlweSecretKeyEntity<KeyFlavor = BootstrapKey::OutputKeyFlavor>,
{
    /// Creates an LWE bootstrap key.
    fn create_lwe_bootstrap_key(
        &mut self,
        input_key: &LweSecretKey,
        output_key: &GlweSecretKey,
        decomposition_base_log: DecompositionBaseLog,
        decomposition_level_count: DecompositionLevelCount,
        noise: Variance,
    ) -> Result<BootstrapKey, LweBootstrapKeyCreationError<Self::EngineError>>;

    /// Unsafely creates an LWE bootstrap key.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweBootstrapKeyCreationError`]. For safety concerns _specific_ to an engine, refer to
    /// the implementer safety section.
    unsafe fn create_lwe_bootstrap_key_unchecked(
        &mut self,
        input_key: &LweSecretKey,
        output_key: &GlweSecretKey,
        decomposition_base_log: DecompositionBaseLog,
        decomposition_level_count: DecompositionLevelCount,
        noise: Variance,
    ) -> BootstrapKey;
}
