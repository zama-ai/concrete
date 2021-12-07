//! A module containing entry points to manipulate prototypical entities.
use crate::prototyping::prototypes::*;
use crate::prototyping::{IntegerPrecision, Precision32, Precision64};
use crate::Maker;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};
use concrete_core::prelude::*;

/// A trait for types which allows to manipulate
pub trait Prototyper<Precision: IntegerPrecision> {
    type CleartextPrototype: CleartextPrototype<Precision = Precision>;
    type CleartextVectorPrototype: CleartextVectorPrototype<Precision = Precision>;
    type PlaintextPrototype: PlaintextPrototype<Precision = Precision>;
    type PlaintextVectorPrototype: PlaintextVectorPrototype<Precision = Precision>;
    type BinaryLweSecretKeyPrototype: BinaryLweSecretKeyPrototype<Precision = Precision>;
    type BinaryLweCiphertextPrototype: BinaryLweCiphertextPrototype<Precision = Precision>;
    type BinaryLweCiphertextVectorPrototype: BinaryLweCiphertextVectorPrototype<
        Precision = Precision,
    >;
    type BinaryGlweSecretKeyPrototype: BinaryGlweSecretKeyPrototype<Precision = Precision>;
    type BinaryGlweCiphertextPrototype: BinaryGlweCiphertextPrototype<Precision = Precision>;
    type BinaryGlweCiphertextVectorPrototype: BinaryGlweCiphertextVectorPrototype<
        Precision = Precision,
    >;
    type BinaryBinaryLweKeyswitchKeyPrototype: BinaryBinaryLweKeyswitchKeyPrototype<
        Precision = Precision,
    >;
    type BinaryBinaryLweBootstrapKeyPrototype: BinaryBinaryLweBootstrapKeyPrototype<
        Precision = Precision,
    >;

    fn new_binary_lwe_secret_key(
        &mut self,
        lwe_dimension: LweDimension,
    ) -> Self::BinaryLweSecretKeyPrototype;

    fn new_binary_glwe_secret_key(
        &mut self,
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
    ) -> Self::BinaryGlweSecretKeyPrototype;

    fn new_binary_binary_lwe_keyswitch_key(
        &mut self,
        input_key: &Self::BinaryLweSecretKeyPrototype,
        output_key: &Self::BinaryLweSecretKeyPrototype,
        decomposition_level: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Self::BinaryBinaryLweKeyswitchKeyPrototype;

    fn new_binary_binary_lwe_bootstrap_key(
        &mut self,
        input_key: &Self::BinaryLweSecretKeyPrototype,
        output_key: &Self::BinaryGlweSecretKeyPrototype,
        decomposition_level: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Self::BinaryBinaryLweBootstrapKeyPrototype;

    fn transform_raw_to_plaintext(&mut self, raw: &Precision::Raw) -> Self::PlaintextPrototype;

    fn transform_raw_vec_to_plaintext_vector(
        &mut self,
        raw: &[Precision::Raw],
    ) -> Self::PlaintextVectorPrototype;

    fn transform_plaintext_to_raw(
        &mut self,
        plaintext: &Self::PlaintextPrototype,
    ) -> Precision::Raw;

    fn transform_plaintext_vector_to_raw_vec(
        &mut self,
        plaintext: &Self::PlaintextVectorPrototype,
    ) -> Vec<Precision::Raw>;

    fn transform_raw_to_cleartext(&mut self, raw: &Precision::Raw) -> Self::CleartextPrototype;

    fn transform_raw_vec_to_cleartext_vector(
        &mut self,
        raw: &[Precision::Raw],
    ) -> Self::CleartextVectorPrototype;

    fn transform_cleartext_to_raw(
        &mut self,
        cleartext: &Self::CleartextPrototype,
    ) -> Precision::Raw;

    fn transform_cleartext_vector_to_raw_vec(
        &mut self,
        cleartext: &Self::CleartextVectorPrototype,
    ) -> Vec<Precision::Raw>;

    fn encrypt_plaintext_to_binary_lwe_ciphertext(
        &mut self,
        secret_key: &Self::BinaryLweSecretKeyPrototype,
        plaintext: &Self::PlaintextPrototype,
        noise: Variance,
    ) -> Self::BinaryLweCiphertextPrototype;

    fn encrypt_plaintext_vector_to_binary_lwe_ciphertext_vector(
        &mut self,
        secret_key: &Self::BinaryLweSecretKeyPrototype,
        plaintext_vector: &Self::PlaintextVectorPrototype,
        noise: Variance,
    ) -> Self::BinaryLweCiphertextVectorPrototype;

    fn encrypt_plaintext_vector_to_binary_glwe_ciphertext(
        &mut self,
        secret_key: &Self::BinaryGlweSecretKeyPrototype,
        plaintext_vector: &Self::PlaintextVectorPrototype,
        noise: Variance,
    ) -> Self::BinaryGlweCiphertextPrototype;

    fn encrypt_plaintext_vector_to_binary_glwe_ciphertext_vector(
        &mut self,
        secret_key: &Self::BinaryGlweSecretKeyPrototype,
        plaintext_vector: &Self::PlaintextVectorPrototype,
        noise: Variance,
    ) -> Self::BinaryGlweCiphertextVectorPrototype;

    fn decrypt_binary_lwe_ciphertext_to_plaintext(
        &mut self,
        secret_key: &Self::BinaryLweSecretKeyPrototype,
        ciphertext: &Self::BinaryLweCiphertextPrototype,
    ) -> Self::PlaintextPrototype;

    fn decrypt_binary_lwe_ciphertext_vector_to_plaintext_vector(
        &mut self,
        secret_key: &Self::BinaryLweSecretKeyPrototype,
        ciphertext_vector: &Self::BinaryLweCiphertextVectorPrototype,
    ) -> Self::PlaintextVectorPrototype;

    fn decrypt_binary_glwe_ciphertext_to_plaintext_vector(
        &mut self,
        secret_key: &Self::BinaryGlweSecretKeyPrototype,
        ciphertext: &Self::BinaryGlweCiphertextPrototype,
    ) -> Self::PlaintextVectorPrototype;

    fn decrypt_binary_glwe_ciphertext_vector_to_plaintext_vector(
        &mut self,
        secret_key: &Self::BinaryGlweSecretKeyPrototype,
        ciphertext: &Self::BinaryGlweCiphertextVectorPrototype,
    ) -> Self::PlaintextVectorPrototype;
}

impl Prototyper<Precision32> for Maker {
    type CleartextPrototype = ProtoCleartext32;
    type CleartextVectorPrototype = ProtoCleartextVector32;
    type PlaintextPrototype = ProtoPlaintext32;
    type PlaintextVectorPrototype = ProtoPlaintextVector32;
    type BinaryLweSecretKeyPrototype = ProtoBinaryLweSecretKey32;
    type BinaryLweCiphertextPrototype = ProtoBinaryLweCiphertext32;
    type BinaryLweCiphertextVectorPrototype = ProtoBinaryLweCiphertextVector32;
    type BinaryGlweSecretKeyPrototype = ProtoBinaryGlweSecretKey32;
    type BinaryGlweCiphertextPrototype = ProtoBinaryGlweCiphertext32;
    type BinaryGlweCiphertextVectorPrototype = ProtoBinaryGlweCiphertextVector32;
    type BinaryBinaryLweKeyswitchKeyPrototype = ProtoBinaryBinaryLweKeyswitchKey32;
    type BinaryBinaryLweBootstrapKeyPrototype = ProtoBinaryBinaryLweBootstrapKey32;

    fn new_binary_lwe_secret_key(
        &mut self,
        lwe_dimension: LweDimension,
    ) -> Self::BinaryLweSecretKeyPrototype {
        ProtoBinaryLweSecretKey32(
            self.core_engine
                .create_lwe_secret_key(lwe_dimension)
                .unwrap(),
        )
    }

    fn new_binary_glwe_secret_key(
        &mut self,
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
    ) -> Self::BinaryGlweSecretKeyPrototype {
        ProtoBinaryGlweSecretKey32(
            self.core_engine
                .create_glwe_secret_key(glwe_dimension, polynomial_size)
                .unwrap(),
        )
    }

    fn new_binary_binary_lwe_keyswitch_key(
        &mut self,
        input_key: &Self::BinaryLweSecretKeyPrototype,
        output_key: &Self::BinaryLweSecretKeyPrototype,
        decomposition_level: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Self::BinaryBinaryLweKeyswitchKeyPrototype {
        ProtoBinaryBinaryLweKeyswitchKey32(
            self.core_engine
                .create_lwe_keyswitch_key(
                    &input_key.0,
                    &output_key.0,
                    decomposition_level,
                    decomposition_base_log,
                    noise,
                )
                .unwrap(),
        )
    }

    fn new_binary_binary_lwe_bootstrap_key(
        &mut self,
        input_key: &Self::BinaryLweSecretKeyPrototype,
        output_key: &Self::BinaryGlweSecretKeyPrototype,
        decomposition_level: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Self::BinaryBinaryLweBootstrapKeyPrototype {
        ProtoBinaryBinaryLweBootstrapKey32(
            self.core_engine
                .create_lwe_bootstrap_key(
                    &input_key.0,
                    &output_key.0,
                    decomposition_base_log,
                    decomposition_level,
                    noise,
                )
                .unwrap(),
        )
    }

    fn transform_raw_to_plaintext(&mut self, input: &u32) -> Self::PlaintextPrototype {
        ProtoPlaintext32(self.core_engine.create_plaintext(input).unwrap())
    }

    fn transform_raw_vec_to_plaintext_vector(
        &mut self,
        raw: &[u32],
    ) -> Self::PlaintextVectorPrototype {
        ProtoPlaintextVector32(self.core_engine.create_plaintext_vector(raw).unwrap())
    }

    fn transform_plaintext_to_raw(&mut self, plaintext: &Self::PlaintextPrototype) -> u32 {
        self.core_engine.retrieve_plaintext(&plaintext.0).unwrap()
    }

    fn transform_plaintext_vector_to_raw_vec(
        &mut self,
        plaintext: &Self::PlaintextVectorPrototype,
    ) -> Vec<u32> {
        self.core_engine
            .retrieve_plaintext_vector(&plaintext.0)
            .unwrap()
    }

    fn transform_raw_to_cleartext(&mut self, raw: &u32) -> Self::CleartextPrototype {
        ProtoCleartext32(self.core_engine.create_cleartext(raw).unwrap())
    }

    fn transform_raw_vec_to_cleartext_vector(
        &mut self,
        raw: &[u32],
    ) -> Self::CleartextVectorPrototype {
        ProtoCleartextVector32(self.core_engine.create_cleartext_vector(raw).unwrap())
    }

    fn transform_cleartext_to_raw(&mut self, cleartext: &Self::CleartextPrototype) -> u32 {
        self.core_engine.retrieve_cleartext(&cleartext.0).unwrap()
    }

    fn transform_cleartext_vector_to_raw_vec(
        &mut self,
        cleartext: &Self::CleartextVectorPrototype,
    ) -> Vec<u32> {
        self.core_engine
            .retrieve_cleartext_vector(&cleartext.0)
            .unwrap()
    }

    fn encrypt_plaintext_to_binary_lwe_ciphertext(
        &mut self,
        secret_key: &Self::BinaryLweSecretKeyPrototype,
        plaintext: &Self::PlaintextPrototype,
        noise: Variance,
    ) -> Self::BinaryLweCiphertextPrototype {
        ProtoBinaryLweCiphertext32(
            self.core_engine
                .encrypt_lwe_ciphertext(&secret_key.0, &plaintext.0, noise)
                .unwrap(),
        )
    }

    fn encrypt_plaintext_vector_to_binary_lwe_ciphertext_vector(
        &mut self,
        secret_key: &Self::BinaryLweSecretKeyPrototype,
        plaintext_vector: &Self::PlaintextVectorPrototype,
        noise: Variance,
    ) -> Self::BinaryLweCiphertextVectorPrototype {
        ProtoBinaryLweCiphertextVector32(
            self.core_engine
                .encrypt_lwe_ciphertext_vector(&secret_key.0, &plaintext_vector.0, noise)
                .unwrap(),
        )
    }

    fn encrypt_plaintext_vector_to_binary_glwe_ciphertext(
        &mut self,
        secret_key: &Self::BinaryGlweSecretKeyPrototype,
        plaintext_vector: &Self::PlaintextVectorPrototype,
        noise: Variance,
    ) -> Self::BinaryGlweCiphertextPrototype {
        ProtoBinaryGlweCiphertext32(
            self.core_engine
                .encrypt_glwe_ciphertext(&secret_key.0, &plaintext_vector.0, noise)
                .unwrap(),
        )
    }

    fn encrypt_plaintext_vector_to_binary_glwe_ciphertext_vector(
        &mut self,
        secret_key: &Self::BinaryGlweSecretKeyPrototype,
        plaintext_vector: &Self::PlaintextVectorPrototype,
        noise: Variance,
    ) -> Self::BinaryGlweCiphertextVectorPrototype {
        ProtoBinaryGlweCiphertextVector32(
            self.core_engine
                .encrypt_glwe_ciphertext_vector(&secret_key.0, &plaintext_vector.0, noise)
                .unwrap(),
        )
    }

    fn decrypt_binary_lwe_ciphertext_to_plaintext(
        &mut self,
        secret_key: &Self::BinaryLweSecretKeyPrototype,
        ciphertext: &Self::BinaryLweCiphertextPrototype,
    ) -> Self::PlaintextPrototype {
        ProtoPlaintext32(
            self.core_engine
                .decrypt_lwe_ciphertext(&secret_key.0, &ciphertext.0)
                .unwrap(),
        )
    }

    fn decrypt_binary_lwe_ciphertext_vector_to_plaintext_vector(
        &mut self,
        secret_key: &Self::BinaryLweSecretKeyPrototype,
        ciphertext_vector: &Self::BinaryLweCiphertextVectorPrototype,
    ) -> Self::PlaintextVectorPrototype {
        ProtoPlaintextVector32(
            self.core_engine
                .decrypt_lwe_ciphertext_vector(&secret_key.0, &ciphertext_vector.0)
                .unwrap(),
        )
    }

    fn decrypt_binary_glwe_ciphertext_to_plaintext_vector(
        &mut self,
        secret_key: &Self::BinaryGlweSecretKeyPrototype,
        ciphertext: &Self::BinaryGlweCiphertextPrototype,
    ) -> Self::PlaintextVectorPrototype {
        ProtoPlaintextVector32(
            self.core_engine
                .decrypt_glwe_ciphertext(&secret_key.0, &ciphertext.0)
                .unwrap(),
        )
    }

    fn decrypt_binary_glwe_ciphertext_vector_to_plaintext_vector(
        &mut self,
        secret_key: &Self::BinaryGlweSecretKeyPrototype,
        ciphertext: &Self::BinaryGlweCiphertextVectorPrototype,
    ) -> Self::PlaintextVectorPrototype {
        ProtoPlaintextVector32(
            self.core_engine
                .decrypt_glwe_ciphertext_vector(&secret_key.0, &ciphertext.0)
                .unwrap(),
        )
    }
}

impl Prototyper<Precision64> for Maker {
    type CleartextPrototype = ProtoCleartext64;
    type CleartextVectorPrototype = ProtoCleartextVector64;
    type PlaintextPrototype = ProtoPlaintext64;
    type PlaintextVectorPrototype = ProtoPlaintextVector64;
    type BinaryLweSecretKeyPrototype = ProtoBinaryLweSecretKey64;
    type BinaryLweCiphertextPrototype = ProtoBinaryLweCiphertext64;
    type BinaryLweCiphertextVectorPrototype = ProtoBinaryLweCiphertextVector64;
    type BinaryGlweSecretKeyPrototype = ProtoBinaryGlweSecretKey64;
    type BinaryGlweCiphertextPrototype = ProtoBinaryGlweCiphertext64;
    type BinaryGlweCiphertextVectorPrototype = ProtoBinaryGlweCiphertextVector64;
    type BinaryBinaryLweKeyswitchKeyPrototype = ProtoBinaryBinaryLweKeyswitchKey64;
    type BinaryBinaryLweBootstrapKeyPrototype = ProtoBinaryBinaryLweBootstrapKey64;

    fn new_binary_lwe_secret_key(
        &mut self,
        lwe_dimension: LweDimension,
    ) -> Self::BinaryLweSecretKeyPrototype {
        ProtoBinaryLweSecretKey64(
            self.core_engine
                .create_lwe_secret_key(lwe_dimension)
                .unwrap(),
        )
    }

    fn new_binary_glwe_secret_key(
        &mut self,
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
    ) -> Self::BinaryGlweSecretKeyPrototype {
        ProtoBinaryGlweSecretKey64(
            self.core_engine
                .create_glwe_secret_key(glwe_dimension, polynomial_size)
                .unwrap(),
        )
    }

    fn new_binary_binary_lwe_keyswitch_key(
        &mut self,
        input_key: &Self::BinaryLweSecretKeyPrototype,
        output_key: &Self::BinaryLweSecretKeyPrototype,
        decomposition_level: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Self::BinaryBinaryLweKeyswitchKeyPrototype {
        ProtoBinaryBinaryLweKeyswitchKey64(
            self.core_engine
                .create_lwe_keyswitch_key(
                    &input_key.0,
                    &output_key.0,
                    decomposition_level,
                    decomposition_base_log,
                    noise,
                )
                .unwrap(),
        )
    }

    fn new_binary_binary_lwe_bootstrap_key(
        &mut self,
        input_key: &Self::BinaryLweSecretKeyPrototype,
        output_key: &Self::BinaryGlweSecretKeyPrototype,
        decomposition_level: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Self::BinaryBinaryLweBootstrapKeyPrototype {
        ProtoBinaryBinaryLweBootstrapKey64(
            self.core_engine
                .create_lwe_bootstrap_key(
                    &input_key.0,
                    &output_key.0,
                    decomposition_base_log,
                    decomposition_level,
                    noise,
                )
                .unwrap(),
        )
    }

    fn transform_raw_to_plaintext(&mut self, input: &u64) -> Self::PlaintextPrototype {
        ProtoPlaintext64(self.core_engine.create_plaintext(input).unwrap())
    }

    fn transform_raw_vec_to_plaintext_vector(
        &mut self,
        raw: &[u64],
    ) -> Self::PlaintextVectorPrototype {
        ProtoPlaintextVector64(self.core_engine.create_plaintext_vector(raw).unwrap())
    }

    fn transform_plaintext_to_raw(&mut self, plaintext: &Self::PlaintextPrototype) -> u64 {
        self.core_engine.retrieve_plaintext(&plaintext.0).unwrap()
    }

    fn transform_plaintext_vector_to_raw_vec(
        &mut self,
        plaintext: &Self::PlaintextVectorPrototype,
    ) -> Vec<u64> {
        self.core_engine
            .retrieve_plaintext_vector(&plaintext.0)
            .unwrap()
    }

    fn transform_raw_to_cleartext(&mut self, raw: &u64) -> Self::CleartextPrototype {
        ProtoCleartext64(self.core_engine.create_cleartext(raw).unwrap())
    }

    fn transform_raw_vec_to_cleartext_vector(
        &mut self,
        raw: &[u64],
    ) -> Self::CleartextVectorPrototype {
        ProtoCleartextVector64(self.core_engine.create_cleartext_vector(raw).unwrap())
    }

    fn transform_cleartext_to_raw(&mut self, cleartext: &Self::CleartextPrototype) -> u64 {
        self.core_engine.retrieve_cleartext(&cleartext.0).unwrap()
    }

    fn transform_cleartext_vector_to_raw_vec(
        &mut self,
        cleartext: &Self::CleartextVectorPrototype,
    ) -> Vec<u64> {
        self.core_engine
            .retrieve_cleartext_vector(&cleartext.0)
            .unwrap()
    }

    fn encrypt_plaintext_to_binary_lwe_ciphertext(
        &mut self,
        secret_key: &Self::BinaryLweSecretKeyPrototype,
        plaintext: &Self::PlaintextPrototype,
        noise: Variance,
    ) -> Self::BinaryLweCiphertextPrototype {
        ProtoBinaryLweCiphertext64(
            self.core_engine
                .encrypt_lwe_ciphertext(&secret_key.0, &plaintext.0, noise)
                .unwrap(),
        )
    }

    fn encrypt_plaintext_vector_to_binary_lwe_ciphertext_vector(
        &mut self,
        secret_key: &Self::BinaryLweSecretKeyPrototype,
        plaintext_vector: &Self::PlaintextVectorPrototype,
        noise: Variance,
    ) -> Self::BinaryLweCiphertextVectorPrototype {
        ProtoBinaryLweCiphertextVector64(
            self.core_engine
                .encrypt_lwe_ciphertext_vector(&secret_key.0, &plaintext_vector.0, noise)
                .unwrap(),
        )
    }

    fn encrypt_plaintext_vector_to_binary_glwe_ciphertext(
        &mut self,
        secret_key: &Self::BinaryGlweSecretKeyPrototype,
        plaintext_vector: &Self::PlaintextVectorPrototype,
        noise: Variance,
    ) -> Self::BinaryGlweCiphertextPrototype {
        ProtoBinaryGlweCiphertext64(
            self.core_engine
                .encrypt_glwe_ciphertext(&secret_key.0, &plaintext_vector.0, noise)
                .unwrap(),
        )
    }

    fn encrypt_plaintext_vector_to_binary_glwe_ciphertext_vector(
        &mut self,
        secret_key: &Self::BinaryGlweSecretKeyPrototype,
        plaintext_vector: &Self::PlaintextVectorPrototype,
        noise: Variance,
    ) -> Self::BinaryGlweCiphertextVectorPrototype {
        ProtoBinaryGlweCiphertextVector64(
            self.core_engine
                .encrypt_glwe_ciphertext_vector(&secret_key.0, &plaintext_vector.0, noise)
                .unwrap(),
        )
    }

    fn decrypt_binary_lwe_ciphertext_to_plaintext(
        &mut self,
        secret_key: &Self::BinaryLweSecretKeyPrototype,
        ciphertext: &Self::BinaryLweCiphertextPrototype,
    ) -> Self::PlaintextPrototype {
        ProtoPlaintext64(
            self.core_engine
                .decrypt_lwe_ciphertext(&secret_key.0, &ciphertext.0)
                .unwrap(),
        )
    }

    fn decrypt_binary_lwe_ciphertext_vector_to_plaintext_vector(
        &mut self,
        secret_key: &Self::BinaryLweSecretKeyPrototype,
        ciphertext_vector: &Self::BinaryLweCiphertextVectorPrototype,
    ) -> Self::PlaintextVectorPrototype {
        ProtoPlaintextVector64(
            self.core_engine
                .decrypt_lwe_ciphertext_vector(&secret_key.0, &ciphertext_vector.0)
                .unwrap(),
        )
    }

    fn decrypt_binary_glwe_ciphertext_to_plaintext_vector(
        &mut self,
        secret_key: &Self::BinaryGlweSecretKeyPrototype,
        ciphertext: &Self::BinaryGlweCiphertextPrototype,
    ) -> Self::PlaintextVectorPrototype {
        ProtoPlaintextVector64(
            self.core_engine
                .decrypt_glwe_ciphertext(&secret_key.0, &ciphertext.0)
                .unwrap(),
        )
    }

    fn decrypt_binary_glwe_ciphertext_vector_to_plaintext_vector(
        &mut self,
        secret_key: &Self::BinaryGlweSecretKeyPrototype,
        ciphertext: &Self::BinaryGlweCiphertextVectorPrototype,
    ) -> Self::PlaintextVectorPrototype {
        ProtoPlaintextVector64(
            self.core_engine
                .decrypt_glwe_ciphertext_vector(&secret_key.0, &ciphertext.0)
                .unwrap(),
        )
    }
}
