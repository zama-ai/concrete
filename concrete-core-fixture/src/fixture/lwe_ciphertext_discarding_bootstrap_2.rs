use crate::fixture::Fixture;
use crate::generation::prototyping::{
    PrototypesGlweCiphertext, PrototypesGlweSecretKey, PrototypesLweBootstrapKey,
    PrototypesLweCiphertext, PrototypesLweSecretKey, PrototypesPlaintext,
    PrototypesPlaintextVector,
};
use crate::generation::synthesizing::{
    SynthesizesGlweCiphertext, SynthesizesLweBootstrapKey, SynthesizesLweCiphertext,
};
use crate::generation::{IntegerPrecision, Maker};
use concrete_commons::dispersion::{DispersionParameter, LogStandardDev, Variance};
use concrete_commons::numeric::{CastInto, Numeric};
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};
use concrete_core::prelude::{
    GlweCiphertextEntity, LweBootstrapKeyEntity, LweCiphertextDiscardingBootstrapEngine,
    LweCiphertextEntity,
};

/// A fixture for the types implementing the `LweCiphertextDiscardingBootstrapEngine` trait.
pub struct LweCiphertextDiscardingBootstrapFixture2;

#[derive(Debug)]
pub struct LweCiphertextDiscardingBootstrapParameters2 {
    pub noise: Variance,
    pub lwe_dimension: LweDimension,
    pub glwe_dimension: GlweDimension,
    pub poly_size: PolynomialSize,
    pub decomp_level_count: DecompositionLevelCount,
    pub decomp_base_log: DecompositionBaseLog,
}

#[allow(clippy::type_complexity)]
impl<Precision, Engine, BootstrapKey, Accumulator, InputCiphertext, OutputCiphertext>
    Fixture<Precision, Engine, (BootstrapKey, Accumulator, InputCiphertext, OutputCiphertext)>
    for LweCiphertextDiscardingBootstrapFixture2
where
    Precision: IntegerPrecision,
    Engine: LweCiphertextDiscardingBootstrapEngine<
        BootstrapKey,
        Accumulator,
        InputCiphertext,
        OutputCiphertext,
    >,
    InputCiphertext: LweCiphertextEntity,
    OutputCiphertext: LweCiphertextEntity,
    Accumulator: GlweCiphertextEntity<KeyDistribution = OutputCiphertext::KeyDistribution>,
    BootstrapKey: LweBootstrapKeyEntity<
        InputKeyDistribution = InputCiphertext::KeyDistribution,
        OutputKeyDistribution = OutputCiphertext::KeyDistribution,
    >,
    Maker: SynthesizesLweBootstrapKey<Precision, BootstrapKey>
        + SynthesizesGlweCiphertext<Precision, Accumulator>
        + SynthesizesLweCiphertext<Precision, InputCiphertext>
        + SynthesizesLweCiphertext<Precision, OutputCiphertext>,
{
    type Parameters = LweCiphertextDiscardingBootstrapParameters2;
    type RepetitionPrototypes = (
        <Maker as PrototypesGlweCiphertext<Precision, OutputCiphertext::KeyDistribution>>::GlweCiphertextProto,
        <Maker as PrototypesLweSecretKey<Precision, InputCiphertext::KeyDistribution>>::LweSecretKeyProto,
        <Maker as PrototypesGlweSecretKey<Precision, OutputCiphertext::KeyDistribution>>::GlweSecretKeyProto,
        <Maker as PrototypesLweBootstrapKey<Precision, InputCiphertext::KeyDistribution, OutputCiphertext::KeyDistribution>>::LweBootstrapKeyProto,
    );
    type SamplePrototypes = (
        <Maker as PrototypesPlaintext<Precision>>::PlaintextProto,
        <Maker as PrototypesLweCiphertext<Precision, InputCiphertext::KeyDistribution>>::LweCiphertextProto,
        <Maker as PrototypesLweCiphertext<Precision, OutputCiphertext::KeyDistribution>>::LweCiphertextProto,
    );
    type PreExecutionContext = (BootstrapKey, Accumulator, OutputCiphertext, InputCiphertext);
    type PostExecutionContext = (BootstrapKey, Accumulator, OutputCiphertext, InputCiphertext);
    type Criteria = (i64,);
    type Outcome = (Precision::Raw, Precision::Raw);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![LweCiphertextDiscardingBootstrapParameters2 {
                noise: Variance(LogStandardDev::from_log_standard_dev(-29.).get_variance()),
                lwe_dimension: LweDimension(630),
                glwe_dimension: GlweDimension(1),
                poly_size: PolynomialSize(1024),
                decomp_level_count: DecompositionLevelCount(3),
                decomp_base_log: DecompositionBaseLog(7),
            }]
            .into_iter(),
        )
    }

    fn generate_random_repetition_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
    ) -> Self::RepetitionPrototypes {
        let log_degree = f64::log2(parameters.poly_size.0 as f64) as i32;
        let raw_plaintext_vector: Vec<Precision::Raw> = (0..parameters.poly_size.0)
            .map(|i| {
                (i as f64 * 2_f64.powi(Precision::Raw::BITS as i32 - log_degree - 1)).cast_into()
            })
            .collect();
        let proto_plaintext_vector =
            maker.transform_raw_vec_to_plaintext_vector(raw_plaintext_vector.as_slice());
        let proto_accumulator = maker.trivially_encrypt_plaintext_vector_to_glwe_ciphertext(
            parameters.glwe_dimension,
            &proto_plaintext_vector,
        );
        let proto_lwe_secret_key = <Maker as PrototypesLweSecretKey<
            Precision,
            InputCiphertext::KeyDistribution,
        >>::new_lwe_secret_key(maker, parameters.lwe_dimension);
        let proto_glwe_secret_key = <Maker as PrototypesGlweSecretKey<
            Precision,
            OutputCiphertext::KeyDistribution,
        >>::new_glwe_secret_key(
            maker, parameters.glwe_dimension, parameters.poly_size
        );
        let proto_bootstrap_key = maker.new_lwe_bootstrap_key(
            &proto_lwe_secret_key,
            &proto_glwe_secret_key,
            parameters.decomp_level_count,
            parameters.decomp_base_log,
            parameters.noise,
        );
        (
            proto_accumulator,
            proto_lwe_secret_key,
            proto_glwe_secret_key,
            proto_bootstrap_key,
        )
    }

    fn generate_random_sample_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        let (_, proto_lwe_secret_key, ..) = repetition_proto;
        let log_degree = f64::log2(parameters.poly_size.0 as f64) as i32;
        let raw_plaintext = ((parameters.poly_size.0 as f64
            - (10. * f64::sqrt((parameters.lwe_dimension.0 as f64) / 16.0)))
            * 2_f64.powi(Precision::Raw::BITS as i32 - log_degree - 1))
        .cast_into();
        let proto_plaintext = maker.transform_raw_to_plaintext(&raw_plaintext);
        let proto_input_ciphertext = <Maker as PrototypesLweCiphertext<
            Precision,
            InputCiphertext::KeyDistribution,
        >>::encrypt_plaintext_to_lwe_ciphertext(
            maker,
            proto_lwe_secret_key,
            &proto_plaintext,
            parameters.noise,
        );
        let proto_output_ciphertext = <Maker as PrototypesLweCiphertext<
            Precision,
            OutputCiphertext::KeyDistribution,
        >>::trivially_encrypt_zero_to_lwe_ciphertext(
            maker,
            LweDimension(parameters.glwe_dimension.0 * parameters.poly_size.0),
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
        let (proto_accumulator, _, _, proto_bootstrap_key) = repetition_proto;
        let (_, proto_input_ciphertext, proto_output_ciphertext) = sample_proto;
        let synth_bootstrap_key = maker.synthesize_lwe_bootstrap_key(proto_bootstrap_key);
        let synth_accumulator = maker.synthesize_glwe_ciphertext(proto_accumulator);
        let synth_input_ciphertext = maker.synthesize_lwe_ciphertext(proto_input_ciphertext);
        let synth_output_ciphertext = maker.synthesize_lwe_ciphertext(proto_output_ciphertext);
        (
            synth_bootstrap_key,
            synth_accumulator,
            synth_output_ciphertext,
            synth_input_ciphertext,
        )
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (bootstrap_key, accumulator, mut output_ciphertext, input_ciphertext) = context;
        unsafe {
            engine.discard_bootstrap_lwe_ciphertext_unchecked(
                &mut output_ciphertext,
                &input_ciphertext,
                &accumulator,
                &bootstrap_key,
            )
        };
        (
            bootstrap_key,
            accumulator,
            output_ciphertext,
            input_ciphertext,
        )
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (bootstrap_key, accumulator, output_ciphertext, input_ciphertext) = context;
        let (_, _, proto_glwe_secret_key, _) = repetition_proto;
        let (proto_plaintext, ..) = sample_proto;
        let proto_output_ciphertext = maker.unsynthesize_lwe_ciphertext(&output_ciphertext);
        let proto_output_lwe_secret_key =
            maker.transmute_glwe_secret_key_to_lwe_secret_key(proto_glwe_secret_key);
        let proto_output_plaintext = <Maker as PrototypesLweCiphertext<
            Precision,
            OutputCiphertext::KeyDistribution,
        >>::decrypt_lwe_ciphertext_to_plaintext(
            maker,
            &proto_output_lwe_secret_key,
            &proto_output_ciphertext,
        );
        maker.destroy_lwe_ciphertext(input_ciphertext);
        maker.destroy_lwe_ciphertext(output_ciphertext);
        maker.destroy_lwe_bootstrap_key(bootstrap_key);
        maker.destroy_glwe_ciphertext(accumulator);
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
        let log_degree = f64::log2(parameters.poly_size.0 as f64) as i32;
        let delta_max: i64 = ((5. * f64::sqrt((parameters.lwe_dimension.0 as f64) / 16.0))
            * 2_f64.powi(Precision::Raw::BITS as i32 - log_degree - 1))
            as i64;
        (delta_max,)
    }

    fn verify(criteria: &Self::Criteria, outputs: &[Self::Outcome]) -> bool {
        let (delta_max,) = criteria;
        for (expected, obtained) in outputs.iter() {
            if (<Precision::Raw as CastInto<i64>>::cast_into(*expected)
                - <Precision::Raw as CastInto<i64>>::cast_into(*obtained))
            .abs()
                > *delta_max
            {
                return false;
            }
        }
        true
    }
}
