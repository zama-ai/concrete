use crate::generation::prototypes::{
    GlweCiphertextPrototype, ProtoBinaryGlweCiphertext32, ProtoBinaryGlweCiphertext64,
    ProtoPlaintextVector32, ProtoPlaintextVector64,
};
use crate::generation::prototyping::glwe_secret_key::PrototypesGlweSecretKey;
use crate::generation::prototyping::plaintext_vector::PrototypesPlaintextVector;
use crate::generation::{IntegerPrecision, Maker, Precision32, Precision64};
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{GlweDimension, PolynomialSize};
use concrete_core::prelude::markers::{BinaryKeyDistribution, KeyDistributionMarker};
use concrete_core::prelude::{
    GlweCiphertextConsumingRetrievalEngine, GlweCiphertextCreationEngine,
    GlweCiphertextDecryptionEngine, GlweCiphertextEncryptionEngine, GlweCiphertextEntity,
    GlweCiphertextTrivialDecryptionEngine, GlweCiphertextTrivialEncryptionEngine,
    PlaintextVectorCreationEngine,
};

/// A trait allowing to manipulate GLWE ciphertext prototypes.
pub trait PrototypesGlweCiphertext<
    Precision: IntegerPrecision,
    KeyDistribution: KeyDistributionMarker,
>:
    PrototypesPlaintextVector<Precision> + PrototypesGlweSecretKey<Precision, KeyDistribution>
{
    type GlweCiphertextProto: GlweCiphertextPrototype<
        Precision = Precision,
        KeyDistribution = KeyDistribution,
    >;
    fn trivially_encrypt_zeros_to_glwe_ciphertext(
        &mut self,
        glwe_dimension: GlweDimension,
        poly_size: PolynomialSize,
    ) -> Self::GlweCiphertextProto;
    fn trivially_encrypt_plaintext_vector_to_glwe_ciphertext(
        &mut self,
        glwe_dimension: GlweDimension,
        plaintext_vector: &Self::PlaintextVectorProto,
    ) -> Self::GlweCiphertextProto;
    fn trivially_decrypt_glwe_ciphertext(
        &mut self,
        ciphertext: &Self::GlweCiphertextProto,
    ) -> Self::PlaintextVectorProto;
    fn encrypt_plaintext_vector_to_glwe_ciphertext(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        plaintext_vector: &Self::PlaintextVectorProto,
        noise: Variance,
    ) -> Self::GlweCiphertextProto;
    fn decrypt_glwe_ciphertext_to_plaintext_vector(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        ciphertext: &Self::GlweCiphertextProto,
    ) -> Self::PlaintextVectorProto;
    fn transform_raw_vec_to_ciphertext(
        &mut self,
        raw: &[Precision::Raw],
        polynomial_size: PolynomialSize,
    ) -> Self::GlweCiphertextProto;
    fn transform_ciphertext_to_raw_vec(
        &mut self,
        ciphertext: &Self::GlweCiphertextProto,
    ) -> (Vec<Precision::Raw>, PolynomialSize);
}

impl PrototypesGlweCiphertext<Precision32, BinaryKeyDistribution> for Maker {
    type GlweCiphertextProto = ProtoBinaryGlweCiphertext32;

    fn trivially_encrypt_zeros_to_glwe_ciphertext(
        &mut self,
        glwe_dimension: GlweDimension,
        poly_size: PolynomialSize,
    ) -> Self::GlweCiphertextProto {
        let plaintext_vector = self
            .core_engine
            .create_plaintext_vector(&vec![0u32; poly_size.0])
            .unwrap();
        ProtoBinaryGlweCiphertext32(
            self.core_engine
                .trivially_encrypt_glwe_ciphertext(glwe_dimension.to_glwe_size(), &plaintext_vector)
                .unwrap(),
        )
    }

    fn trivially_encrypt_plaintext_vector_to_glwe_ciphertext(
        &mut self,
        glwe_dimension: GlweDimension,
        plaintext_vector: &Self::PlaintextVectorProto,
    ) -> Self::GlweCiphertextProto {
        ProtoBinaryGlweCiphertext32(
            self.core_engine
                .trivially_encrypt_glwe_ciphertext(
                    glwe_dimension.to_glwe_size(),
                    &plaintext_vector.0,
                )
                .unwrap(),
        )
    }

    fn trivially_decrypt_glwe_ciphertext(
        &mut self,
        ciphertext: &Self::GlweCiphertextProto,
    ) -> Self::PlaintextVectorProto {
        ProtoPlaintextVector32(
            self.core_engine
                .trivially_decrypt_glwe_ciphertext(&ciphertext.0)
                .unwrap(),
        )
    }

    fn encrypt_plaintext_vector_to_glwe_ciphertext(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        plaintext_vector: &Self::PlaintextVectorProto,
        noise: Variance,
    ) -> Self::GlweCiphertextProto {
        ProtoBinaryGlweCiphertext32(
            self.core_engine
                .encrypt_glwe_ciphertext(&secret_key.0, &plaintext_vector.0, noise)
                .unwrap(),
        )
    }

    fn decrypt_glwe_ciphertext_to_plaintext_vector(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        ciphertext: &Self::GlweCiphertextProto,
    ) -> Self::PlaintextVectorProto {
        ProtoPlaintextVector32(
            self.core_engine
                .decrypt_glwe_ciphertext(&secret_key.0, &ciphertext.0)
                .unwrap(),
        )
    }

    fn transform_raw_vec_to_ciphertext(
        &mut self,
        raw: &[u32],
        polynomial_size: PolynomialSize,
    ) -> Self::GlweCiphertextProto {
        let glwe_ciphertext = self
            .core_engine
            .create_glwe_ciphertext(raw.to_owned(), polynomial_size)
            .unwrap();
        ProtoBinaryGlweCiphertext32(glwe_ciphertext)
    }

    fn transform_ciphertext_to_raw_vec(
        &mut self,
        ciphertext: &Self::GlweCiphertextProto,
    ) -> (Vec<u32>, PolynomialSize) {
        let ciphertext = ciphertext.0.to_owned();
        let polynomial_size = ciphertext.polynomial_size();
        (
            self.core_engine
                .consume_retrieve_glwe_ciphertext(ciphertext)
                .unwrap(),
            polynomial_size,
        )
    }
}

impl PrototypesGlweCiphertext<Precision64, BinaryKeyDistribution> for Maker {
    type GlweCiphertextProto = ProtoBinaryGlweCiphertext64;

    fn trivially_encrypt_zeros_to_glwe_ciphertext(
        &mut self,
        glwe_dimension: GlweDimension,
        poly_size: PolynomialSize,
    ) -> Self::GlweCiphertextProto {
        let plaintext_vector = self
            .core_engine
            .create_plaintext_vector(&vec![0u64; poly_size.0])
            .unwrap();
        ProtoBinaryGlweCiphertext64(
            self.core_engine
                .trivially_encrypt_glwe_ciphertext(glwe_dimension.to_glwe_size(), &plaintext_vector)
                .unwrap(),
        )
    }

    fn trivially_encrypt_plaintext_vector_to_glwe_ciphertext(
        &mut self,
        glwe_dimension: GlweDimension,
        plaintext_vector: &Self::PlaintextVectorProto,
    ) -> Self::GlweCiphertextProto {
        ProtoBinaryGlweCiphertext64(
            self.core_engine
                .trivially_encrypt_glwe_ciphertext(
                    glwe_dimension.to_glwe_size(),
                    &plaintext_vector.0,
                )
                .unwrap(),
        )
    }

    fn trivially_decrypt_glwe_ciphertext(
        &mut self,
        ciphertext: &Self::GlweCiphertextProto,
    ) -> Self::PlaintextVectorProto {
        ProtoPlaintextVector64(
            self.core_engine
                .trivially_decrypt_glwe_ciphertext(&ciphertext.0)
                .unwrap(),
        )
    }

    fn encrypt_plaintext_vector_to_glwe_ciphertext(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        plaintext_vector: &Self::PlaintextVectorProto,
        noise: Variance,
    ) -> Self::GlweCiphertextProto {
        ProtoBinaryGlweCiphertext64(
            self.core_engine
                .encrypt_glwe_ciphertext(&secret_key.0, &plaintext_vector.0, noise)
                .unwrap(),
        )
    }

    fn decrypt_glwe_ciphertext_to_plaintext_vector(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        ciphertext: &Self::GlweCiphertextProto,
    ) -> Self::PlaintextVectorProto {
        ProtoPlaintextVector64(
            self.core_engine
                .decrypt_glwe_ciphertext(&secret_key.0, &ciphertext.0)
                .unwrap(),
        )
    }

    fn transform_raw_vec_to_ciphertext(
        &mut self,
        raw: &[u64],
        polynomial_size: PolynomialSize,
    ) -> Self::GlweCiphertextProto {
        let glwe_ciphertext = self
            .core_engine
            .create_glwe_ciphertext(raw.to_owned(), polynomial_size)
            .unwrap();
        ProtoBinaryGlweCiphertext64(glwe_ciphertext)
    }

    fn transform_ciphertext_to_raw_vec(
        &mut self,
        ciphertext: &Self::GlweCiphertextProto,
    ) -> (Vec<u64>, PolynomialSize) {
        let ciphertext = ciphertext.0.to_owned();
        let polynomial_size = ciphertext.polynomial_size();
        (
            self.core_engine
                .consume_retrieve_glwe_ciphertext(ciphertext)
                .unwrap(),
            polynomial_size,
        )
    }
}
