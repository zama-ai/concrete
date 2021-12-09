use super::engine_error;
use crate::specification::engines::AbstractEngine;

use crate::specification::entities::{LweCiphertextVectorEntity, LweKeyswitchKeyEntity};

engine_error! {
    LweCiphertextVectorDiscardingKeyswitchError for LweCiphertextVectorDiscardingKeyswitchEngine @
    InputLweDimensionMismatch => "The input ciphertext vector and keyswitch key input LWE \
                                  dimension must be the same.",
    OutputLweDimensionMismatch => "The output ciphertext vector and keyswitch key output LWE \
                                   dimension must be the same.",
    CiphertextCountMismatch => "The input and output ciphertexts have different ciphertext counts."
}

/// A trait for engines keyswitching (discarding) LWE ciphertext vectors.
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` LWE ciphertext vector
/// with the element-wise keyswitch of the `input` LWE ciphertext vector, under the `ksk` lwe
/// keyswitch key.
///
/// # Formal Definition
pub trait LweCiphertextVectorDiscardingKeyswitchEngine<
    KeyswitchKey,
    InputCiphertextVector,
    OutputCiphertextVector,
>: AbstractEngine where
    KeyswitchKey: LweKeyswitchKeyEntity,
    InputCiphertextVector:
        LweCiphertextVectorEntity<KeyDistribution = KeyswitchKey::InputKeyDistribution>,
    OutputCiphertextVector:
        LweCiphertextVectorEntity<KeyDistribution = KeyswitchKey::OutputKeyDistribution>,
{
    /// Keyswitch an LWE ciphertext vector.
    fn discard_keyswitch_lwe_ciphertext_vector(
        &mut self,
        output: &mut OutputCiphertextVector,
        input: &InputCiphertextVector,
        ksk: &KeyswitchKey,
    ) -> Result<(), LweCiphertextVectorDiscardingKeyswitchError<Self::EngineError>>;

    /// Unsafely keyswitch an LWE ciphertext vector.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextVectorDiscardingKeyswitchError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn discard_keyswitch_lwe_ciphertext_vector_unchecked(
        &mut self,
        output: &mut OutputCiphertextVector,
        input: &InputCiphertextVector,
        ksk: &KeyswitchKey,
    );
}
