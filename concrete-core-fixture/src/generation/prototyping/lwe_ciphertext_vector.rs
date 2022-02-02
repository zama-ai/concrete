use crate::generation::prototypes::{
    LweCiphertextVectorPrototype, ProtoBinaryLweCiphertextVector32,
    ProtoBinaryLweCiphertextVector64, ProtoPlaintextVector32, ProtoPlaintextVector64,
};
use crate::generation::prototyping::lwe_secret_key::PrototypesLweSecretKey;
use crate::generation::prototyping::plaintext_vector::PrototypesPlaintextVector;
use crate::generation::{IntegerPrecision, Maker, Precision32, Precision64};
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{LweCiphertextCount, LweDimension};
use concrete_core::prelude::markers::{BinaryKeyDistribution, KeyDistributionMarker};
use concrete_core::prelude::{
    LweCiphertextVectorDecryptionEngine, LweCiphertextVectorEncryptionEngine,
    LweCiphertextVectorTrivialEncryptionEngine, PlaintextVectorCreationEngine,
};

/// A trait allowing to manipulate lwe ciphertext vector prototypes.
pub trait PrototypesLweCiphertextVector<
    Precision: IntegerPrecision,
    KeyDistribution: KeyDistributionMarker,
>:
    PrototypesPlaintextVector<Precision> + PrototypesLweSecretKey<Precision, KeyDistribution>
{
    type LweCiphertextVectorProto: LweCiphertextVectorPrototype<
        Precision = Precision,
        KeyDistribution = KeyDistribution,
    >;
    fn trivial_encrypt_zeros_to_lwe_ciphertext_vector(
        &mut self,
        lwe_dimension: LweDimension,
        count: LweCiphertextCount,
    ) -> Self::LweCiphertextVectorProto;
    fn trivial_encrypt_plaintext_vector_to_lwe_ciphertext(
        &mut self,
        lwe_dimension: LweDimension,
        plaintext_vector: &Self::PlaintextVectorProto,
    ) -> Self::LweCiphertextVectorProto;
    fn encrypt_plaintext_vector_to_lwe_ciphertext_vector(
        &mut self,
        secret_key: &Self::LweSecretKeyProto,
        plaintext_vector: &Self::PlaintextVectorProto,
        noise: Variance,
    ) -> Self::LweCiphertextVectorProto;

    fn decrypt_lwe_ciphertext_vector_to_plaintext_vector(
        &mut self,
        secret_key: &Self::LweSecretKeyProto,
        ciphertext_vector: &Self::LweCiphertextVectorProto,
    ) -> Self::PlaintextVectorProto;
}

impl PrototypesLweCiphertextVector<Precision32, BinaryKeyDistribution> for Maker {
    type LweCiphertextVectorProto = ProtoBinaryLweCiphertextVector32;

    fn trivial_encrypt_zeros_to_lwe_ciphertext_vector(
        &mut self,
        lwe_dimension: LweDimension,
        count: LweCiphertextCount,
    ) -> Self::LweCiphertextVectorProto {
        let plaintext_vector = self
            .core_engine
            .create_plaintext_vector(&vec![0u32; count.0])
            .unwrap();
        ProtoBinaryLweCiphertextVector32(
            self.core_engine
                .trivially_encrypt_lwe_ciphertext_vector(
                    lwe_dimension.to_lwe_size(),
                    &plaintext_vector,
                )
                .unwrap(),
        )
    }

    fn trivial_encrypt_plaintext_vector_to_lwe_ciphertext(
        &mut self,
        lwe_dimension: LweDimension,
        plaintext_vector: &Self::PlaintextVectorProto,
    ) -> Self::LweCiphertextVectorProto {
        ProtoBinaryLweCiphertextVector32(
            self.core_engine
                .trivially_encrypt_lwe_ciphertext_vector(
                    lwe_dimension.to_lwe_size(),
                    &plaintext_vector.0,
                )
                .unwrap(),
        )
    }

    fn encrypt_plaintext_vector_to_lwe_ciphertext_vector(
        &mut self,
        secret_key: &Self::LweSecretKeyProto,
        plaintext_vector: &Self::PlaintextVectorProto,
        noise: Variance,
    ) -> Self::LweCiphertextVectorProto {
        ProtoBinaryLweCiphertextVector32(
            self.core_engine
                .encrypt_lwe_ciphertext_vector(&secret_key.0, &plaintext_vector.0, noise)
                .unwrap(),
        )
    }

    fn decrypt_lwe_ciphertext_vector_to_plaintext_vector(
        &mut self,
        secret_key: &Self::LweSecretKeyProto,
        ciphertext_vector: &Self::LweCiphertextVectorProto,
    ) -> Self::PlaintextVectorProto {
        ProtoPlaintextVector32(
            self.core_engine
                .decrypt_lwe_ciphertext_vector(&secret_key.0, &ciphertext_vector.0)
                .unwrap(),
        )
    }
}

impl PrototypesLweCiphertextVector<Precision64, BinaryKeyDistribution> for Maker {
    type LweCiphertextVectorProto = ProtoBinaryLweCiphertextVector64;

    fn trivial_encrypt_zeros_to_lwe_ciphertext_vector(
        &mut self,
        lwe_dimension: LweDimension,
        count: LweCiphertextCount,
    ) -> Self::LweCiphertextVectorProto {
        let plaintext_vector = self
            .core_engine
            .create_plaintext_vector(&vec![0u64; count.0])
            .unwrap();
        ProtoBinaryLweCiphertextVector64(
            self.core_engine
                .trivially_encrypt_lwe_ciphertext_vector(
                    lwe_dimension.to_lwe_size(),
                    &plaintext_vector,
                )
                .unwrap(),
        )
    }

    fn trivial_encrypt_plaintext_vector_to_lwe_ciphertext(
        &mut self,
        lwe_dimension: LweDimension,
        plaintext_vector: &Self::PlaintextVectorProto,
    ) -> Self::LweCiphertextVectorProto {
        ProtoBinaryLweCiphertextVector64(
            self.core_engine
                .trivially_encrypt_lwe_ciphertext_vector(
                    lwe_dimension.to_lwe_size(),
                    &plaintext_vector.0,
                )
                .unwrap(),
        )
    }

    fn encrypt_plaintext_vector_to_lwe_ciphertext_vector(
        &mut self,
        secret_key: &Self::LweSecretKeyProto,
        plaintext_vector: &Self::PlaintextVectorProto,
        noise: Variance,
    ) -> Self::LweCiphertextVectorProto {
        ProtoBinaryLweCiphertextVector64(
            self.core_engine
                .encrypt_lwe_ciphertext_vector(&secret_key.0, &plaintext_vector.0, noise)
                .unwrap(),
        )
    }

    fn decrypt_lwe_ciphertext_vector_to_plaintext_vector(
        &mut self,
        secret_key: &Self::LweSecretKeyProto,
        ciphertext_vector: &Self::LweCiphertextVectorProto,
    ) -> Self::PlaintextVectorProto {
        ProtoPlaintextVector64(
            self.core_engine
                .decrypt_lwe_ciphertext_vector(&secret_key.0, &ciphertext_vector.0)
                .unwrap(),
        )
    }
}
