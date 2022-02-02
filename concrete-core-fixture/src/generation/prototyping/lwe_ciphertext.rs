use crate::generation::prototypes::{
    LweCiphertextPrototype, ProtoBinaryLweCiphertext32, ProtoBinaryLweCiphertext64,
    ProtoPlaintext32, ProtoPlaintext64,
};
use crate::generation::prototyping::lwe_secret_key::PrototypesLweSecretKey;
use crate::generation::prototyping::plaintext::PrototypesPlaintext;
use crate::generation::{IntegerPrecision, Maker, Precision32, Precision64};
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::LweDimension;
use concrete_core::prelude::markers::{BinaryKeyDistribution, KeyDistributionMarker};
use concrete_core::prelude::{
    LweCiphertextDecryptionEngine, LweCiphertextEncryptionEngine,
    LweCiphertextTrivialEncryptionEngine, PlaintextCreationEngine,
};

/// A trait allowing to manipulate LWE ciphertext prototypes.
pub trait PrototypesLweCiphertext<
    Precision: IntegerPrecision,
    KeyDistribution: KeyDistributionMarker,
>: PrototypesPlaintext<Precision> + PrototypesLweSecretKey<Precision, KeyDistribution>
{
    type LweCiphertextProto: LweCiphertextPrototype<
        Precision = Precision,
        KeyDistribution = KeyDistribution,
    >;
    fn trivial_encrypt_zero_to_lwe_ciphertext(
        &mut self,
        lwe_dimension: LweDimension,
    ) -> Self::LweCiphertextProto;
    fn trivial_encrypt_plaintext_to_lwe_ciphertext(
        &mut self,
        lwe_dimension: LweDimension,
        plaintext: &Self::PlaintextProto,
    ) -> Self::LweCiphertextProto;
    fn encrypt_plaintext_to_lwe_ciphertext(
        &mut self,
        secret_key: &Self::LweSecretKeyProto,
        plaintext: &Self::PlaintextProto,
        noise: Variance,
    ) -> Self::LweCiphertextProto;
    fn decrypt_lwe_ciphertext_to_plaintext(
        &mut self,
        secret_key: &Self::LweSecretKeyProto,
        ciphertext: &Self::LweCiphertextProto,
    ) -> Self::PlaintextProto;
}

impl PrototypesLweCiphertext<Precision32, BinaryKeyDistribution> for Maker {
    type LweCiphertextProto = ProtoBinaryLweCiphertext32;

    fn trivial_encrypt_zero_to_lwe_ciphertext(
        &mut self,
        lwe_dimension: LweDimension,
    ) -> Self::LweCiphertextProto {
        let plaintext = self.core_engine.create_plaintext(&0u32).unwrap();
        ProtoBinaryLweCiphertext32(
            self.core_engine
                .trivially_encrypt_lwe_ciphertext(lwe_dimension.to_lwe_size(), &plaintext)
                .unwrap(),
        )
    }

    fn trivial_encrypt_plaintext_to_lwe_ciphertext(
        &mut self,
        lwe_dimension: LweDimension,
        plaintext: &Self::PlaintextProto,
    ) -> Self::LweCiphertextProto {
        ProtoBinaryLweCiphertext32(
            self.core_engine
                .trivially_encrypt_lwe_ciphertext(lwe_dimension.to_lwe_size(), &plaintext.0)
                .unwrap(),
        )
    }

    fn encrypt_plaintext_to_lwe_ciphertext(
        &mut self,
        secret_key: &Self::LweSecretKeyProto,
        plaintext: &Self::PlaintextProto,
        noise: Variance,
    ) -> Self::LweCiphertextProto {
        ProtoBinaryLweCiphertext32(
            self.core_engine
                .encrypt_lwe_ciphertext(&secret_key.0, &plaintext.0, noise)
                .unwrap(),
        )
    }

    fn decrypt_lwe_ciphertext_to_plaintext(
        &mut self,
        secret_key: &Self::LweSecretKeyProto,
        ciphertext: &Self::LweCiphertextProto,
    ) -> Self::PlaintextProto {
        ProtoPlaintext32(
            self.core_engine
                .decrypt_lwe_ciphertext(&secret_key.0, &ciphertext.0)
                .unwrap(),
        )
    }
}

impl PrototypesLweCiphertext<Precision64, BinaryKeyDistribution> for Maker {
    type LweCiphertextProto = ProtoBinaryLweCiphertext64;

    fn trivial_encrypt_zero_to_lwe_ciphertext(
        &mut self,
        lwe_dimension: LweDimension,
    ) -> Self::LweCiphertextProto {
        let plaintext = self.core_engine.create_plaintext(&0u64).unwrap();
        ProtoBinaryLweCiphertext64(
            self.core_engine
                .trivially_encrypt_lwe_ciphertext(lwe_dimension.to_lwe_size(), &plaintext)
                .unwrap(),
        )
    }

    fn trivial_encrypt_plaintext_to_lwe_ciphertext(
        &mut self,
        lwe_dimension: LweDimension,
        plaintext: &Self::PlaintextProto,
    ) -> Self::LweCiphertextProto {
        ProtoBinaryLweCiphertext64(
            self.core_engine
                .trivially_encrypt_lwe_ciphertext(lwe_dimension.to_lwe_size(), &plaintext.0)
                .unwrap(),
        )
    }

    fn encrypt_plaintext_to_lwe_ciphertext(
        &mut self,
        secret_key: &Self::LweSecretKeyProto,
        plaintext: &Self::PlaintextProto,
        noise: Variance,
    ) -> Self::LweCiphertextProto {
        ProtoBinaryLweCiphertext64(
            self.core_engine
                .encrypt_lwe_ciphertext(&secret_key.0, &plaintext.0, noise)
                .unwrap(),
        )
    }

    fn decrypt_lwe_ciphertext_to_plaintext(
        &mut self,
        secret_key: &Self::LweSecretKeyProto,
        ciphertext: &Self::LweCiphertextProto,
    ) -> Self::PlaintextProto {
        ProtoPlaintext64(
            self.core_engine
                .decrypt_lwe_ciphertext(&secret_key.0, &ciphertext.0)
                .unwrap(),
        )
    }
}
