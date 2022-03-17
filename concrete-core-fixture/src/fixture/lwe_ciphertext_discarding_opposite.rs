use crate::fixture::Fixture;
use crate::generation::prototyping::{
    PrototypesLweCiphertext, PrototypesLweSecretKey, PrototypesPlaintext,
};
use crate::generation::synthesizing::SynthesizesLweCiphertext;
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::{DispersionParameter, LogStandardDev, Variance};
use concrete_commons::numeric::UnsignedInteger;
use concrete_commons::parameters::LweDimension;
use concrete_core::prelude::{LweCiphertextDiscardingOppositeEngine, LweCiphertextEntity};

/// A fixture for the types implementing the `LweCiphertextDiscardingOppositeEngine`
/// trait.
pub struct LweCiphertextDiscardingOppositeFixture;

#[derive(Debug)]
pub struct LweCiphertextDiscardingOppositeParameters {
    pub noise: Variance,
    pub lwe_dimension: LweDimension,
}

#[allow(clippy::type_complexity)]
impl<Precision, Engine, InputCiphertext, OutputCiphertext>
    Fixture<Precision, Engine, (InputCiphertext, OutputCiphertext)>
    for LweCiphertextDiscardingOppositeFixture
where
    Precision: IntegerPrecision,
    Engine: LweCiphertextDiscardingOppositeEngine<InputCiphertext, OutputCiphertext>,
    InputCiphertext: LweCiphertextEntity,
    OutputCiphertext: LweCiphertextEntity<KeyDistribution = InputCiphertext::KeyDistribution>,
    Maker: SynthesizesLweCiphertext<Precision, InputCiphertext>
        + SynthesizesLweCiphertext<Precision, OutputCiphertext>,
{
    type Parameters = LweCiphertextDiscardingOppositeParameters;
    type RepetitionPrototypes = (
        <Maker as PrototypesLweSecretKey<Precision, InputCiphertext::KeyDistribution>>::LweSecretKeyProto,
    );
    type SamplePrototypes = (
        <Maker as PrototypesPlaintext<Precision>>::PlaintextProto,
        <Maker as PrototypesLweCiphertext<Precision, InputCiphertext::KeyDistribution>>::LweCiphertextProto,
        <Maker as PrototypesLweCiphertext<Precision, InputCiphertext::KeyDistribution>>::LweCiphertextProto,
    );
    type PreExecutionContext = (InputCiphertext, OutputCiphertext);
    type PostExecutionContext = (InputCiphertext, OutputCiphertext);
    type Criteria = (Variance,);
    type Outcome = (Precision::Raw, Precision::Raw);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![LweCiphertextDiscardingOppositeParameters {
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
        (maker.new_lwe_secret_key(parameters.lwe_dimension),)
    }

    fn generate_random_sample_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        let (proto_secret_key,) = repetition_proto;
        let raw_plaintext = Precision::Raw::uniform();
        let proto_plaintext = maker.transform_raw_to_plaintext(&raw_plaintext);
        let proto_input_ciphertext = maker.encrypt_plaintext_to_lwe_ciphertext(
            proto_secret_key,
            &proto_plaintext,
            parameters.noise,
        );
        let proto_output_ciphertext =
            maker.trivially_encrypt_zero_to_lwe_ciphertext(parameters.lwe_dimension);
        (
            proto_plaintext,
            proto_input_ciphertext,
            proto_output_ciphertext,
        )
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (_, proto_input_ciphertext, proto_output_ciphertext) = sample_proto;
        let synth_input_ciphertext = maker.synthesize_lwe_ciphertext(proto_input_ciphertext);
        let synth_output_ciphertext = maker.synthesize_lwe_ciphertext(proto_output_ciphertext);
        (synth_input_ciphertext, synth_output_ciphertext)
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (input_ciphertext, mut output_ciphertext) = context;
        unsafe {
            engine.discard_opp_lwe_ciphertext_unchecked(&mut output_ciphertext, &input_ciphertext)
        };
        (input_ciphertext, output_ciphertext)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (input_ciphertext, output_ciphertext) = context;
        let (proto_plaintext, ..) = sample_proto;
        let (proto_secret_key,) = repetition_proto;
        let raw_plaintext = maker.transform_plaintext_to_raw(proto_plaintext);
        let expected_mean = raw_plaintext.wrapping_neg();
        let proto_output_ciphertext = maker.unsynthesize_lwe_ciphertext(&output_ciphertext);
        let proto_output_plaintext =
            maker.decrypt_lwe_ciphertext_to_plaintext(proto_secret_key, &proto_output_ciphertext);
        maker.destroy_lwe_ciphertext(input_ciphertext);
        maker.destroy_lwe_ciphertext(output_ciphertext);
        (
            expected_mean,
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
