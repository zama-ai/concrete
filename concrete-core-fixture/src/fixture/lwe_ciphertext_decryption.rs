use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::LweDimension;
use concrete_core::prelude::{
    LweCiphertextDecryptionEngine, LweCiphertextEntity, LweSecretKeyEntity, PlaintextEntity,
};

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

/// A fixture for the types implementing the `LweCiphertextDecryptionEngine` trait.
pub struct LweCiphertextDecryptionFixture;

#[derive(Debug)]
pub struct LweCiphertextDecryptionParameters {
    pub noise: Variance,
    pub lwe_dimension: LweDimension,
}

impl<Precision, Engine, Plaintext, SecretKey, Ciphertext>
    Fixture<Precision, Engine, (Plaintext, SecretKey, Ciphertext)>
    for LweCiphertextDecryptionFixture
where
    Precision: IntegerPrecision,
    Engine: LweCiphertextDecryptionEngine<SecretKey, Ciphertext, Plaintext>,
    Plaintext: PlaintextEntity,
    SecretKey: LweSecretKeyEntity,
    Ciphertext: LweCiphertextEntity<KeyDistribution = SecretKey::KeyDistribution>,
    Maker: SynthesizesPlaintext<Precision, Plaintext>
        + SynthesizesLweSecretKey<Precision, SecretKey>
        + SynthesizesLweCiphertext<Precision, Ciphertext>,
{
    type Parameters = LweCiphertextDecryptionParameters;
    type RepetitionPrototypes = (<Maker as PrototypesLweSecretKey<Precision, Ciphertext::KeyDistribution>>::LweSecretKeyProto, );
    type SamplePrototypes = (
        <Maker as PrototypesPlaintext<Precision>>::PlaintextProto,
        <Maker as PrototypesLweCiphertext<Precision, Ciphertext::KeyDistribution>>::LweCiphertextProto,
    );
    type PreExecutionContext = (SecretKey, Ciphertext);
    type PostExecutionContext = (SecretKey, Ciphertext, Plaintext);
    type Criteria = (Variance,);
    type Outcome = (Precision::Raw, Precision::Raw);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                LweCiphertextDecryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(100),
                },
                LweCiphertextDecryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(300),
                },
                LweCiphertextDecryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(600),
                },
                LweCiphertextDecryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(1000),
                },
                LweCiphertextDecryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(3000),
                },
                LweCiphertextDecryptionParameters {
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
        let (proto_secret_key,) = repetition_proto;
        let raw_plaintext = Precision::Raw::uniform();
        let proto_plaintext = maker.transform_raw_to_plaintext(&raw_plaintext);
        let proto_ciphertext = maker.encrypt_plaintext_to_lwe_ciphertext(
            proto_secret_key,
            &proto_plaintext,
            parameters.noise,
        );
        (proto_plaintext, proto_ciphertext)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_secret_key,) = repetition_proto;
        let (_, proto_ciphertext) = sample_proto;
        let synth_secret_key = maker.synthesize_lwe_secret_key(proto_secret_key);
        let synth_ciphertext = maker.synthesize_lwe_ciphertext(proto_ciphertext);
        (synth_secret_key, synth_ciphertext)
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (secret_key, ciphertext) = context;
        let plaintext =
            unsafe { engine.decrypt_lwe_ciphertext_unchecked(&secret_key, &ciphertext) };
        (secret_key, ciphertext, plaintext)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (secret_key, ciphertext, plaintext) = context;
        let (proto_plaintext, _) = sample_proto;
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
