use crate::generation::prototypes::{
    GlweCiphertextVectorPrototype, ProtoBinaryGlweCiphertextVector32,
    ProtoBinaryGlweCiphertextVector64, ProtoPlaintextVector32, ProtoPlaintextVector64,
};
use crate::generation::prototyping::glwe_secret_key::PrototypesGlweSecretKey;
use crate::generation::prototyping::plaintext_vector::PrototypesPlaintextVector;
use crate::generation::{IntegerPrecision, Maker, Precision32, Precision64};
use concrete_commons::dispersion::Variance;
use concrete_core::prelude::markers::{BinaryKeyDistribution, KeyDistributionMarker};
use concrete_core::prelude::{
    GlweCiphertextVectorDecryptionEngine, GlweCiphertextVectorEncryptionEngine,
};

/// A trait allowing to manipulate GLWE ciphertext vector prototypes.
pub trait PrototypesGlweCiphertextVector<
    Precision: IntegerPrecision,
    KeyDistribution: KeyDistributionMarker,
>:
    PrototypesPlaintextVector<Precision> + PrototypesGlweSecretKey<Precision, KeyDistribution>
{
    type GlweCiphertextVectorProto: GlweCiphertextVectorPrototype<
        Precision = Precision,
        KeyDistribution = KeyDistribution,
    >;
    fn encrypt_plaintext_vector_to_glwe_ciphertext_vector(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        plaintext_vector: &Self::PlaintextVectorProto,
        noise: Variance,
    ) -> Self::GlweCiphertextVectorProto;

    fn decrypt_glwe_ciphertext_vector_to_plaintext_vector(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        ciphertext: &Self::GlweCiphertextVectorProto,
    ) -> Self::PlaintextVectorProto;
}

impl PrototypesGlweCiphertextVector<Precision32, BinaryKeyDistribution> for Maker {
    type GlweCiphertextVectorProto = ProtoBinaryGlweCiphertextVector32;

    fn encrypt_plaintext_vector_to_glwe_ciphertext_vector(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        plaintext_vector: &Self::PlaintextVectorProto,
        noise: Variance,
    ) -> Self::GlweCiphertextVectorProto {
        ProtoBinaryGlweCiphertextVector32(
            self.core_engine
                .encrypt_glwe_ciphertext_vector(&secret_key.0, &plaintext_vector.0, noise)
                .unwrap(),
        )
    }

    fn decrypt_glwe_ciphertext_vector_to_plaintext_vector(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        ciphertext: &Self::GlweCiphertextVectorProto,
    ) -> Self::PlaintextVectorProto {
        ProtoPlaintextVector32(
            self.core_engine
                .decrypt_glwe_ciphertext_vector(&secret_key.0, &ciphertext.0)
                .unwrap(),
        )
    }
}

impl PrototypesGlweCiphertextVector<Precision64, BinaryKeyDistribution> for Maker {
    type GlweCiphertextVectorProto = ProtoBinaryGlweCiphertextVector64;

    fn encrypt_plaintext_vector_to_glwe_ciphertext_vector(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        plaintext_vector: &Self::PlaintextVectorProto,
        noise: Variance,
    ) -> Self::GlweCiphertextVectorProto {
        ProtoBinaryGlweCiphertextVector64(
            self.core_engine
                .encrypt_glwe_ciphertext_vector(&secret_key.0, &plaintext_vector.0, noise)
                .unwrap(),
        )
    }

    fn decrypt_glwe_ciphertext_vector_to_plaintext_vector(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        ciphertext: &Self::GlweCiphertextVectorProto,
    ) -> Self::PlaintextVectorProto {
        ProtoPlaintextVector64(
            self.core_engine
                .decrypt_glwe_ciphertext_vector(&secret_key.0, &ciphertext.0)
                .unwrap(),
        )
    }
}
