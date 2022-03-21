use concrete_commons::dispersion::{DispersionParameter, LogStandardDev, Variance};
use concrete_commons::numeric::UnsignedInteger;
use concrete_commons::parameters::LweDimension;
use concrete_core::prelude::{
    CleartextEntity, LweCiphertextCleartextFusingMultiplicationEngine, LweCiphertextEntity,
};

use crate::fixture::Fixture;
use crate::generation::prototyping::{
    PrototypesCleartext, PrototypesLweCiphertext, PrototypesLweSecretKey, PrototypesPlaintext,
};
use crate::generation::synthesizing::{SynthesizesCleartext, SynthesizesLweCiphertext};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;

/// A fixture for the types implementing the `LweCiphertextCleartextFusingMultiplicationEngine`
/// trait.
pub struct LweCiphertextCleartextFusingMultiplicationFixture;

#[derive(Debug)]
pub struct LweCiphertextCleartextFusingMultiplicationParameters {
    pub noise: Variance,
    pub lwe_dimension: LweDimension,
}

#[allow(clippy::type_complexity)]
impl<Precision, Engine, Ciphertext, Cleartext> Fixture<Precision, Engine, (Ciphertext, Cleartext)>
    for LweCiphertextCleartextFusingMultiplicationFixture
where
    Precision: IntegerPrecision,
    Engine: LweCiphertextCleartextFusingMultiplicationEngine<Ciphertext, Cleartext>,
    Ciphertext: LweCiphertextEntity,
    Cleartext: CleartextEntity,
    Maker: SynthesizesCleartext<Precision, Cleartext>
        + SynthesizesLweCiphertext<Precision, Ciphertext>,
{
    type Parameters = LweCiphertextCleartextFusingMultiplicationParameters;
    type RepetitionPrototypes = (
        <Maker as PrototypesLweSecretKey<Precision, Ciphertext::KeyDistribution>>::LweSecretKeyProto,
        <Maker as PrototypesCleartext<Precision>>::CleartextProto,
    );
    type SamplePrototypes = (
        <Maker as PrototypesPlaintext<Precision>>::PlaintextProto,
        <Maker as PrototypesLweCiphertext<Precision, Ciphertext::KeyDistribution>>::LweCiphertextProto,
    );
    type PreExecutionContext = (Ciphertext, Cleartext);
    type PostExecutionContext = (Ciphertext, Cleartext);
    type Criteria = (Variance,);
    type Outcome = (Precision::Raw, Precision::Raw);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![LweCiphertextCleartextFusingMultiplicationParameters {
                noise: Variance(LogStandardDev::from_log_standard_dev(-15.).get_variance()),
                lwe_dimension: LweDimension(600),
            }]
            .into_iter(),
        )
    }

    fn generate_random_repetition_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
    ) -> Self::RepetitionPrototypes {
        let proto_secret_key = maker.new_lwe_secret_key(parameters.lwe_dimension);
        let raw_cleartext = Precision::Raw::uniform_zero_centered(1024);
        let proto_cleartext = maker.transform_raw_to_cleartext(&raw_cleartext);
        (proto_secret_key, proto_cleartext)
    }

    fn generate_random_sample_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        let (proto_secret_key, _) = repetition_proto;
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
        let (_, proto_cleartext) = repetition_proto;
        let (_, proto_ciphertext) = sample_proto;
        let synth_ciphertext = maker.synthesize_lwe_ciphertext(proto_ciphertext);
        let synth_cleartext = maker.synthesize_cleartext(proto_cleartext);
        (synth_ciphertext, synth_cleartext)
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (mut ciphertext, cleartext) = context;
        unsafe { engine.fuse_mul_lwe_ciphertext_cleartext_unchecked(&mut ciphertext, &cleartext) };
        (ciphertext, cleartext)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (ciphertext, cleartext) = context;
        let (proto_plaintext, ..) = sample_proto;
        let (proto_secret_key, proto_cleartext) = repetition_proto;
        let raw_plaintext = maker.transform_plaintext_to_raw(proto_plaintext);
        let raw_cleartext = maker.transform_cleartext_to_raw(proto_cleartext);
        let expected_mean = raw_plaintext.wrapping_mul(raw_cleartext);
        let proto_output_ciphertext = maker.unsynthesize_lwe_ciphertext(&ciphertext);
        let proto_output_plaintext =
            maker.decrypt_lwe_ciphertext_to_plaintext(proto_secret_key, &proto_output_ciphertext);
        maker.destroy_lwe_ciphertext(ciphertext);
        maker.destroy_cleartext(cleartext);
        (
            expected_mean,
            maker.transform_plaintext_to_raw(&proto_output_plaintext),
        )
    }

    fn compute_criteria(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::Criteria {
        let (_, proto_cleartext) = repetition_proto;
        let raw_cleartext = maker.transform_cleartext_to_raw(proto_cleartext);
        let predicted_variance: Variance =
            concrete_npe::estimate_integer_plaintext_multiplication_noise::<Precision::Raw, _>(
                parameters.noise,
                raw_cleartext,
            );
        (predicted_variance,)
    }

    fn verify(criteria: &Self::Criteria, outputs: &[Self::Outcome]) -> bool {
        let (means, actual): (Vec<_>, Vec<_>) = outputs.iter().cloned().unzip();
        assert_noise_distribution(&actual, means.as_slice(), criteria.0)
    }
}
