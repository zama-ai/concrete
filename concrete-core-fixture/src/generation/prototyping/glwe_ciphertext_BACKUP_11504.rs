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
    GlweCiphertextDecryptionEngine, GlweCiphertextEncryptionEngine,
<<<<<<< HEAD
    GlweCiphertextTrivialEncryptionEngine, PlaintextVectorCreationEngine,
=======
    GlweCiphertextTrivialDecryptionEngine, GlweCiphertextTrivialEncryptionEngine,
    PlaintextVectorCreationEngine,
>>>>>>> f4445e4535ecbd23d22476f0e244528721f4f792
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
<<<<<<< HEAD
    fn trivial_encrypt_zeros_to_glwe_ciphertext(
=======
    fn trivially_encrypt_zeros_to_glwe_ciphertext(
>>>>>>> f4445e4535ecbd23d22476f0e244528721f4f792
        &mut self,
        glwe_dimension: GlweDimension,
        poly_size: PolynomialSize,
    ) -> Self::GlweCiphertextProto;
<<<<<<< HEAD
    fn trivial_encrypt_plaintext_vector_to_glwe_ciphertext(
=======
    fn trivially_encrypt_plaintext_vector_to_glwe_ciphertext(
>>>>>>> f4445e4535ecbd23d22476f0e244528721f4f792
        &mut self,
        glwe_dimension: GlweDimension,
        plaintext_vector: &Self::PlaintextVectorProto,
    ) -> Self::GlweCiphertextProto;
<<<<<<< HEAD
=======
    fn trivially_decrypt_glwe_ciphertext(
        &mut self,
        ciphertext: &Self::GlweCiphertextProto,
    ) -> Self::PlaintextVectorProto;
>>>>>>> f4445e4535ecbd23d22476f0e244528721f4f792
    fn encrypt_plaintext_vector_to_glwe_ciphertext(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        plaintext_vector: &Self::PlaintextVectorProto,
        noise: Variance,
    ) -> Self::GlweCiphertextProto;
<<<<<<< HEAD

=======
>>>>>>> f4445e4535ecbd23d22476f0e244528721f4f792
    fn decrypt_glwe_ciphertext_to_plaintext_vector(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        ciphertext: &Self::GlweCiphertextProto,
    ) -> Self::PlaintextVectorProto;
}

impl PrototypesGlweCiphertext<Precision32, BinaryKeyDistribution> for Maker {
    type GlweCiphertextProto = ProtoBinaryGlweCiphertext32;

<<<<<<< HEAD
    fn trivial_encrypt_zeros_to_glwe_ciphertext(
=======
    fn trivially_encrypt_zeros_to_glwe_ciphertext(
>>>>>>> f4445e4535ecbd23d22476f0e244528721f4f792
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

<<<<<<< HEAD
    fn trivial_encrypt_plaintext_vector_to_glwe_ciphertext(
=======
    fn trivially_encrypt_plaintext_vector_to_glwe_ciphertext(
>>>>>>> f4445e4535ecbd23d22476f0e244528721f4f792
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

<<<<<<< HEAD
=======
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

>>>>>>> f4445e4535ecbd23d22476f0e244528721f4f792
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
}

impl PrototypesGlweCiphertext<Precision64, BinaryKeyDistribution> for Maker {
    type GlweCiphertextProto = ProtoBinaryGlweCiphertext64;

<<<<<<< HEAD
    fn trivial_encrypt_zeros_to_glwe_ciphertext(
=======
    fn trivially_encrypt_zeros_to_glwe_ciphertext(
>>>>>>> f4445e4535ecbd23d22476f0e244528721f4f792
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

<<<<<<< HEAD
    fn trivial_encrypt_plaintext_vector_to_glwe_ciphertext(
=======
    fn trivially_encrypt_plaintext_vector_to_glwe_ciphertext(
>>>>>>> f4445e4535ecbd23d22476f0e244528721f4f792
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

<<<<<<< HEAD
=======
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

>>>>>>> f4445e4535ecbd23d22476f0e244528721f4f792
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
}
