use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    CleartextVector32, CleartextVector64, LweCiphertext32, LweCiphertext64, LweCiphertextVector32,
    LweCiphertextVector64, Plaintext32, Plaintext64,
};
use crate::specification::engines::{
    LweCiphertextVectorDiscardingAffineTransformationEngine,
    LweCiphertextVectorDiscardingAffineTransformationError,
};
use crate::specification::entities::{
    CleartextVectorEntity, LweCiphertextEntity, LweCiphertextVectorEntity,
};

impl
    LweCiphertextVectorDiscardingAffineTransformationEngine<
        LweCiphertextVector32,
        CleartextVector32,
        Plaintext32,
        LweCiphertext32,
    > for CoreEngine
{
    fn discard_affine_transform_lwe_ciphertext_vector(
        &mut self,
        output: &mut LweCiphertext32,
        inputs: &LweCiphertextVector32,
        weights: &CleartextVector32,
        bias: &Plaintext32,
    ) -> Result<(), LweCiphertextVectorDiscardingAffineTransformationError<Self::EngineError>> {
        if output.lwe_dimension() != inputs.lwe_dimension() {
            return Err(
                LweCiphertextVectorDiscardingAffineTransformationError::LweDimensionMismatch,
            );
        }
        if inputs.lwe_ciphertext_count().0 != weights.cleartext_count().0 {
            return Err(
                LweCiphertextVectorDiscardingAffineTransformationError::CleartextCountMismatch,
            );
        }
        unsafe {
            self.discard_affine_transform_lwe_ciphertext_vector_unchecked(
                output, inputs, weights, bias,
            )
        };
        Ok(())
    }

    unsafe fn discard_affine_transform_lwe_ciphertext_vector_unchecked(
        &mut self,
        output: &mut LweCiphertext32,
        inputs: &LweCiphertextVector32,
        weights: &CleartextVector32,
        bias: &Plaintext32,
    ) {
        output
            .0
            .fill_with_multisum_with_bias(&inputs.0, &weights.0, &bias.0);
    }
}

impl
    LweCiphertextVectorDiscardingAffineTransformationEngine<
        LweCiphertextVector64,
        CleartextVector64,
        Plaintext64,
        LweCiphertext64,
    > for CoreEngine
{
    fn discard_affine_transform_lwe_ciphertext_vector(
        &mut self,
        output: &mut LweCiphertext64,
        inputs: &LweCiphertextVector64,
        weights: &CleartextVector64,
        bias: &Plaintext64,
    ) -> Result<(), LweCiphertextVectorDiscardingAffineTransformationError<Self::EngineError>> {
        if output.lwe_dimension() != inputs.lwe_dimension() {
            return Err(
                LweCiphertextVectorDiscardingAffineTransformationError::LweDimensionMismatch,
            );
        }
        if inputs.lwe_ciphertext_count().0 != weights.cleartext_count().0 {
            return Err(
                LweCiphertextVectorDiscardingAffineTransformationError::CleartextCountMismatch,
            );
        }
        unsafe {
            self.discard_affine_transform_lwe_ciphertext_vector_unchecked(
                output, inputs, weights, bias,
            )
        };
        Ok(())
    }

    unsafe fn discard_affine_transform_lwe_ciphertext_vector_unchecked(
        &mut self,
        output: &mut LweCiphertext64,
        inputs: &LweCiphertextVector64,
        weights: &CleartextVector64,
        bias: &Plaintext64,
    ) {
        output
            .0
            .fill_with_multisum_with_bias(&inputs.0, &weights.0, &bias.0);
    }
}
