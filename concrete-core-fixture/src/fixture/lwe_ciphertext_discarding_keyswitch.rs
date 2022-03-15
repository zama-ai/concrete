use crate::fixture::Fixture;
use crate::generation::prototyping::{
    PrototypesLweCiphertext, PrototypesLweKeyswitchKey, PrototypesLweSecretKey, PrototypesPlaintext,
};
use crate::generation::synthesizing::{SynthesizesLweCiphertext, SynthesizesLweKeyswitchKey};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::{DispersionParameter, LogStandardDev, Variance};
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweDimension};
use concrete_core::prelude::{
    LweCiphertextDiscardingKeyswitchEngine, LweCiphertextEntity, LweKeyswitchKeyEntity,
};
use concrete_npe::estimate_keyswitch_noise_lwe_to_glwe_with_constant_terms;

/// A fixture for the types implementing the `LweCiphertextDiscardingKeyswitchEngine` trait.
pub struct LweCiphertextDiscardingKeyswitchFixture;

#[derive(Debug)]
pub struct LweCiphertextDiscardingKeyswitchParameters {
    pub n_bit_msg: usize,
    pub input_noise: Variance,
    pub ksk_noise: Variance,
    pub input_lwe_dimension: LweDimension,
    pub output_lwe_dimension: LweDimension,
    pub decomp_level_count: DecompositionLevelCount,
    pub decomp_base_log: DecompositionBaseLog,
}

impl<Precision, Engine, KeyswitchKey, InputCiphertext, OutputCiphertext>
    Fixture<Precision, Engine, (KeyswitchKey, InputCiphertext, OutputCiphertext)>
    for LweCiphertextDiscardingKeyswitchFixture
where
    Precision: IntegerPrecision,
    Engine: LweCiphertextDiscardingKeyswitchEngine<KeyswitchKey, InputCiphertext, OutputCiphertext>,
    InputCiphertext: LweCiphertextEntity,
    OutputCiphertext: LweCiphertextEntity,
    KeyswitchKey: LweKeyswitchKeyEntity<
        InputKeyDistribution = InputCiphertext::KeyDistribution,
        OutputKeyDistribution = OutputCiphertext::KeyDistribution,
    >,
    Maker: SynthesizesLweKeyswitchKey<Precision, KeyswitchKey>
        + SynthesizesLweCiphertext<Precision, InputCiphertext>
        + SynthesizesLweCiphertext<Precision, OutputCiphertext>,
{
    type Parameters = LweCiphertextDiscardingKeyswitchParameters;
    type RepetitionPrototypes = (
        <Maker as PrototypesLweSecretKey<Precision, InputCiphertext::KeyDistribution>>::LweSecretKeyProto,
        <Maker as PrototypesLweSecretKey<Precision, OutputCiphertext::KeyDistribution>>::LweSecretKeyProto,
        <Maker as PrototypesLweKeyswitchKey<Precision, InputCiphertext::KeyDistribution, OutputCiphertext::KeyDistribution>>::LweKeyswitchKeyProto,
    );
    type SamplePrototypes = (
        <Maker as PrototypesPlaintext<Precision>>::PlaintextProto,
        <Maker as PrototypesLweCiphertext<Precision, InputCiphertext::KeyDistribution>>::LweCiphertextProto,
        <Maker as PrototypesLweCiphertext<Precision, OutputCiphertext::KeyDistribution>>::LweCiphertextProto,
    );
    type PreExecutionContext = (OutputCiphertext, InputCiphertext, KeyswitchKey);
    type PostExecutionContext = (OutputCiphertext, InputCiphertext, KeyswitchKey);
    type Criteria = (Variance,);
    type Outcome = (Precision::Raw, Precision::Raw);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![LweCiphertextDiscardingKeyswitchParameters {
                n_bit_msg: 8,
                input_noise: Variance(LogStandardDev::from_log_standard_dev(-10.).get_variance()),
                ksk_noise: Variance(LogStandardDev::from_log_standard_dev(-25.).get_variance()),
                input_lwe_dimension: LweDimension(600),
                output_lwe_dimension: LweDimension(1024),
                decomp_level_count: DecompositionLevelCount(8),
                decomp_base_log: DecompositionBaseLog(3),
            }]
            .into_iter(),
        )
    }

    fn generate_random_repetition_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
    ) -> Self::RepetitionPrototypes {
        let proto_output_secret_key = <Maker as PrototypesLweSecretKey<
            Precision,
            OutputCiphertext::KeyDistribution,
        >>::new_lwe_secret_key(
            maker, parameters.output_lwe_dimension
        );
        let proto_input_secret_key = <Maker as PrototypesLweSecretKey<
            Precision,
            InputCiphertext::KeyDistribution,
        >>::new_lwe_secret_key(
            maker, parameters.input_lwe_dimension
        );
        let proto_keyswitch_key = maker.new_lwe_keyswitch_key(
            &proto_input_secret_key,
            &proto_output_secret_key,
            parameters.decomp_level_count,
            parameters.decomp_base_log,
            parameters.ksk_noise,
        );
        (
            proto_input_secret_key,
            proto_output_secret_key,
            proto_keyswitch_key,
        )
    }

    fn generate_random_sample_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        let (proto_input_secret_key, ..) = repetition_proto;
        let raw_plaintext = Precision::Raw::uniform_n_msb(parameters.n_bit_msg);
        let proto_plaintext = maker.transform_raw_to_plaintext(&raw_plaintext);
        let proto_input_ciphertext = <Maker as PrototypesLweCiphertext<
            Precision,
            InputCiphertext::KeyDistribution,
        >>::encrypt_plaintext_to_lwe_ciphertext(
            maker,
            proto_input_secret_key,
            &proto_plaintext,
            parameters.input_noise,
        );
        let proto_output_ciphertext = <Maker as PrototypesLweCiphertext<
            Precision,
            OutputCiphertext::KeyDistribution,
        >>::trivially_encrypt_zero_to_lwe_ciphertext(
            maker, parameters.output_lwe_dimension
        );
        (
            proto_plaintext,
            proto_input_ciphertext,
            proto_output_ciphertext,
        )
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (_, _, proto_keyswitch_key) = repetition_proto;
        let (_, proto_input_ciphertext, proto_output_ciphertext) = sample_proto;
        let synth_keywsitch_key = maker.synthesize_lwe_keyswitch_key(proto_keyswitch_key);
        let synth_input_ciphertext = maker.synthesize_lwe_ciphertext(proto_input_ciphertext);
        let synth_output_ciphertext = maker.synthesize_lwe_ciphertext(proto_output_ciphertext);
        (
            synth_output_ciphertext,
            synth_input_ciphertext,
            synth_keywsitch_key,
        )
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (mut output_ciphertext, input_ciphertext, keyswitch_key) = context;
        unsafe {
            engine.discard_keyswitch_lwe_ciphertext_unchecked(
                &mut output_ciphertext,
                &input_ciphertext,
                &keyswitch_key,
            )
        };
        (output_ciphertext, input_ciphertext, keyswitch_key)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (output_ciphertext, input_ciphertext, keyswitch_key) = context;
        let (_, proto_output_secret_key, _) = repetition_proto;
        let (proto_plaintext, ..) = sample_proto;
        let proto_output_ciphertext = maker.unsynthesize_lwe_ciphertext(&output_ciphertext);
        let proto_output_plaintext = <Maker as PrototypesLweCiphertext<
            Precision,
            OutputCiphertext::KeyDistribution,
        >>::decrypt_lwe_ciphertext_to_plaintext(
            maker,
            proto_output_secret_key,
            &proto_output_ciphertext,
        );
        maker.destroy_lwe_ciphertext(input_ciphertext);
        maker.destroy_lwe_ciphertext(output_ciphertext);
        maker.destroy_lwe_keyswitch_key(keyswitch_key);
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
        let predicted_variance: Variance = estimate_keyswitch_noise_lwe_to_glwe_with_constant_terms::<
            Precision::Raw,
            _,
            _,
            OutputCiphertext::KeyDistribution,
        >(
            parameters.input_lwe_dimension,
            parameters.input_noise,
            parameters.ksk_noise,
            parameters.decomp_base_log,
            parameters.decomp_level_count,
        );
        (predicted_variance,)
    }

    fn verify(criteria: &Self::Criteria, outputs: &[Self::Outcome]) -> bool {
        let (means, actual): (Vec<_>, Vec<_>) = outputs.iter().cloned().unzip();
        assert_noise_distribution(&actual, means.as_slice(), criteria.0)
    }
}
