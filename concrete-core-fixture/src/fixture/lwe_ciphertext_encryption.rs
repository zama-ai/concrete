use crate::fixture::Fixture;
use crate::generation::prototyping::{
    PrototypesLweCiphertext, PrototypesLweSecretKey, PrototypesPlaintext,
};
use crate::generation::synthesizing::{
    SynthesizesLweCiphertext, SynthesizesLweSecretKey, SynthesizesPlaintext,
};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::LweDimension;
use concrete_core::prelude::{
    LweCiphertextEncryptionEngine, LweCiphertextEntity, LweSecretKeyEntity, PlaintextEntity,
};

/// A fixture for the types implementing the `LweCiphertextEncryptionEngine` trait.
pub struct LweCiphertextEncryptionFixture;

#[derive(Debug)]
pub struct LweCiphertextEncryptionParameters {
    pub noise: Variance,
    pub lwe_dimension: LweDimension,
}

impl<Precision, Engine, Plaintext, SecretKey, Ciphertext>
    Fixture<Precision, Engine, (Plaintext, SecretKey, Ciphertext)>
    for LweCiphertextEncryptionFixture
where
    Precision: IntegerPrecision,
    Engine: LweCiphertextEncryptionEngine<SecretKey, Plaintext, Ciphertext>,
    Plaintext: PlaintextEntity,
    SecretKey: LweSecretKeyEntity,
    Ciphertext: LweCiphertextEntity<KeyDistribution = SecretKey::KeyDistribution>,
    Maker: SynthesizesPlaintext<Precision, Plaintext>
        + SynthesizesLweSecretKey<Precision, SecretKey>
        + SynthesizesLweCiphertext<Precision, Ciphertext>,
{
    type Parameters = LweCiphertextEncryptionParameters;
    type RepetitionPrototypes = (<Maker as PrototypesLweSecretKey<Precision, Ciphertext::KeyDistribution>>::LweSecretKeyProto, );
    type SamplePrototypes = (
        <Maker as PrototypesPlaintext<Precision>>::PlaintextProto,
        Precision::Raw,
    );
    type PreExecutionContext = (Plaintext, SecretKey);
    type PostExecutionContext = (Plaintext, SecretKey, Ciphertext);
    type Criteria = (Variance,);
    type Outcome = (Precision::Raw, Precision::Raw);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                LweCiphertextEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(100),
                },
                LweCiphertextEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(300),
                },
                LweCiphertextEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(600),
                },
                LweCiphertextEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(1000),
                },
                LweCiphertextEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(3000),
                },
                LweCiphertextEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(6000),
                },
            ]
            .into_iter(),
        )
    }

    fn generate_random_repetition_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
    ) -> Self::RepetitionPrototypes {
        let proto_secret_key = maker.new_lwe_secret_key(parameters.lwe_dimension);
        (proto_secret_key,)
    }

    fn generate_random_sample_prototypes(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        let raw_plaintext = Precision::Raw::uniform();
        let proto_plaintext = maker.transform_raw_to_plaintext(&raw_plaintext);
        (proto_plaintext, raw_plaintext)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_secret_key,) = repetition_proto;
        let (proto_plaintext, _) = sample_proto;
        let synth_plaintext = maker.synthesize_plaintext(proto_plaintext);
        let synth_secret_key = maker.synthesize_lwe_secret_key(proto_secret_key);
        (synth_plaintext, synth_secret_key)
    }

    fn execute_engine(
        parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (plaintext, secret_key) = context;
        let ciphertext = unsafe {
            engine.encrypt_lwe_ciphertext_unchecked(&secret_key, &plaintext, parameters.noise)
        };
        (plaintext, secret_key, ciphertext)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (plaintext, secret_key, ciphertext) = context;
        let (proto_secret_key,) = repetition_proto;
        let (_, raw_plaintext) = sample_proto;
        let proto_output_ciphertext = maker.unsynthesize_lwe_ciphertext(&ciphertext);
        maker.destroy_lwe_ciphertext(ciphertext);
        maker.destroy_plaintext(plaintext);
        maker.destroy_lwe_secret_key(secret_key);
        let proto_plaintext =
            maker.decrypt_lwe_ciphertext_to_plaintext(proto_secret_key, &proto_output_ciphertext);
        (
            *raw_plaintext,
            maker.transform_plaintext_to_raw(&proto_plaintext),
        )
    }

    fn compute_criteria(
        parameters: &Self::Parameters,
        _maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::Criteria {
        (parameters.noise,)
    }

    fn verify(criteria: &Self::Criteria, outputs: &[Self::Outcome]) -> bool {
        let (means, actual): (Vec<_>, Vec<_>) = outputs.iter().cloned().unzip();
        assert_noise_distribution(&actual, means.as_slice(), criteria.0)
    }
}
