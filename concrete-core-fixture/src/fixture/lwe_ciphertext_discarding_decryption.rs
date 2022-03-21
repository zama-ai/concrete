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
    LweCiphertextDiscardingDecryptionEngine, LweCiphertextEntity, LweSecretKeyEntity,
    PlaintextEntity,
};

/// A fixture for the types implementing the `LweCiphertextDiscardingDecryptionEngine` trait.
pub struct LweCiphertextDiscardingDecryptionFixture;

#[derive(Debug)]
pub struct LweCiphertextDiscardingDecryptionParameters {
    pub noise: Variance,
    pub lwe_dimension: LweDimension,
}

#[allow(clippy::type_complexity)]
impl<Precision, Engine, Plaintext, SecretKey, Ciphertext>
    Fixture<Precision, Engine, (Ciphertext, SecretKey, Plaintext)>
    for LweCiphertextDiscardingDecryptionFixture
where
    Precision: IntegerPrecision,
    Engine: LweCiphertextDiscardingDecryptionEngine<SecretKey, Ciphertext, Plaintext>,
    Ciphertext: LweCiphertextEntity<KeyDistribution = SecretKey::KeyDistribution>,
    SecretKey: LweSecretKeyEntity,
    Plaintext: PlaintextEntity,
    Maker: SynthesizesPlaintext<Precision, Plaintext>
        + SynthesizesLweSecretKey<Precision, SecretKey>
        + SynthesizesLweCiphertext<Precision, Ciphertext>,
{
    type Parameters = LweCiphertextDiscardingDecryptionParameters;
    type RepetitionPrototypes = (
        <Maker as PrototypesLweSecretKey<Precision, Ciphertext::KeyDistribution>>::LweSecretKeyProto,
    );
    type SamplePrototypes = (
        <Maker as PrototypesLweCiphertext<Precision, Ciphertext::KeyDistribution>>::LweCiphertextProto,
        <Maker as PrototypesPlaintext<Precision>>::PlaintextProto,
    );
    type PreExecutionContext = (Ciphertext, SecretKey, Plaintext);
    type PostExecutionContext = (Ciphertext, SecretKey, Plaintext);
    type Criteria = (Variance,);
    type Outcome = (Precision::Raw, Precision::Raw);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                LweCiphertextDiscardingDecryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(100),
                },
                LweCiphertextDiscardingDecryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(300),
                },
                LweCiphertextDiscardingDecryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(600),
                },
                LweCiphertextDiscardingDecryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(1000),
                },
                LweCiphertextDiscardingDecryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(3000),
                },
                LweCiphertextDiscardingDecryptionParameters {
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
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        let (proto_key,) = repetition_proto;
        let proto_plaintext = maker.transform_raw_to_plaintext(&Precision::Raw::uniform());
        let proto_ciph = maker.encrypt_plaintext_to_lwe_ciphertext(
            proto_key,
            &proto_plaintext,
            parameters.noise,
        );
        (proto_ciph, proto_plaintext)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_secret_key,) = repetition_proto;
        let (proto_ciph, proto_plaintext) = sample_proto;
        let synth_plaintext = maker.synthesize_plaintext(proto_plaintext);
        let synth_secret_key = maker.synthesize_lwe_secret_key(proto_secret_key);
        let synth_ciph = maker.synthesize_lwe_ciphertext(proto_ciph);
        (synth_ciph, synth_secret_key, synth_plaintext)
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (ciphertext, secret_key, mut plaintext) = context;
        unsafe {
            engine.discard_decrypt_lwe_ciphertext_unchecked(
                &secret_key,
                &mut plaintext,
                &ciphertext,
            )
        };
        (ciphertext, secret_key, plaintext)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (ciphertext, secret_key, plaintext) = context;
        let (_, proto_plaintext) = sample_proto;
        let proto_output_plaintext = maker.unsynthesize_plaintext(&plaintext);
        maker.destroy_lwe_ciphertext(ciphertext);
        maker.destroy_plaintext(plaintext);
        maker.destroy_lwe_secret_key(secret_key);
        (
            maker.transform_plaintext_to_raw(proto_plaintext),
            maker.transform_plaintext_to_raw(&proto_output_plaintext),
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
