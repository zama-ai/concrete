use crate::fixture::{fix_estimate_keyswitch_noise_lwe_to_glwe_with_constant_terms, Fixture};
use crate::generation::prototyping::{
    PrototypesGlweCiphertext, PrototypesGlweSecretKey, PrototypesLweCiphertextVector,
    PrototypesLweSecretKey, PrototypesPackingKeyswitchKey, PrototypesPlaintextVector,
};
use crate::generation::synthesizing::{
    SynthesizesGlweCiphertext, SynthesizesLweCiphertextVector, SynthesizesPackingKeyswitchKey,
};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};
use concrete_core::prelude::{
    DispersionParameter, GlweCiphertextEntity, LogStandardDev, LweCiphertextCount,
    LweCiphertextVectorEntity, LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchEngine,
    PackingKeyswitchKeyEntity,
};

/// A fixture for the types implementing the
/// `LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchEngine` trait.
pub struct LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchFixture;

#[derive(Debug)]
pub struct LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchParameters {
    pub input_lwe_noise: Variance,
    pub pksk_noise: Variance,
    pub input_lwe_dimension: LweDimension,
    pub input_lwe_count: LweCiphertextCount,
    pub output_glwe_dimension: GlweDimension,
    pub output_polynomial_size: PolynomialSize,
    pub decomposition_level: DecompositionLevelCount,
    pub decomposition_base_log: DecompositionBaseLog,
}

impl<Precision, Engine, InputCiphertextVector, PackingKeyswitchKey, OutputCiphertext>
    Fixture<Precision, Engine, (InputCiphertextVector, PackingKeyswitchKey, OutputCiphertext)>
    for LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchFixture
where
    Precision: IntegerPrecision,
    Engine: LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchEngine<
        PackingKeyswitchKey,
        InputCiphertextVector,
        OutputCiphertext,
    >,
    InputCiphertextVector: LweCiphertextVectorEntity,
    PackingKeyswitchKey: PackingKeyswitchKeyEntity<
        InputKeyDistribution = InputCiphertextVector::KeyDistribution,
        OutputKeyDistribution = OutputCiphertext::KeyDistribution,
    >,
    OutputCiphertext: GlweCiphertextEntity,
    Maker: SynthesizesLweCiphertextVector<Precision, InputCiphertextVector>
        + SynthesizesGlweCiphertext<Precision, OutputCiphertext>
        + SynthesizesPackingKeyswitchKey<Precision, PackingKeyswitchKey>,
{
    type Parameters = LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchParameters;
    type RepetitionPrototypes =
        (
            <Maker as PrototypesPackingKeyswitchKey<
                Precision,
                InputCiphertextVector::KeyDistribution,
                OutputCiphertext::KeyDistribution,
            >>::PackingKeyswitchKeyProto,
            <Maker as PrototypesLweSecretKey<
                Precision,
                InputCiphertextVector::KeyDistribution,
            >>::LweSecretKeyProto,
            <Maker as PrototypesGlweSecretKey<
                Precision,
                OutputCiphertext::KeyDistribution,
            >>::GlweSecretKeyProto,
        );
    type SamplePrototypes = (
        <Maker as PrototypesPlaintextVector<Precision>>::PlaintextVectorProto,
        <Maker as PrototypesLweCiphertextVector<
            Precision,
            InputCiphertextVector::KeyDistribution,
        >>::LweCiphertextVectorProto,
        <Maker as PrototypesGlweCiphertext<
            Precision,
            OutputCiphertext::KeyDistribution,
        >>::GlweCiphertextProto,
    );
    type PreExecutionContext = (OutputCiphertext, InputCiphertextVector, PackingKeyswitchKey);
    type PostExecutionContext = (OutputCiphertext, InputCiphertextVector, PackingKeyswitchKey);
    type Outcome = (Vec<Precision::Raw>, Vec<Precision::Raw>);
    type Criteria = (Variance,);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchParameters {
                    input_lwe_noise: Variance(
                        LogStandardDev::from_log_standard_dev(-10.).get_variance(),
                    ),
                    pksk_noise: Variance(
                        LogStandardDev::from_log_standard_dev(-25.).get_variance(),
                    ),
                    input_lwe_dimension: LweDimension(200),
                    input_lwe_count: LweCiphertextCount(10),
                    output_glwe_dimension: GlweDimension(1),
                    output_polynomial_size: PolynomialSize(256),
                    decomposition_level: DecompositionLevelCount(3),
                    decomposition_base_log: DecompositionBaseLog(7),
                },
                LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchParameters {
                    input_lwe_noise: Variance(
                        LogStandardDev::from_log_standard_dev(-10.).get_variance(),
                    ),
                    pksk_noise: Variance(
                        LogStandardDev::from_log_standard_dev(-25.).get_variance(),
                    ),
                    input_lwe_dimension: LweDimension(200),
                    input_lwe_count: LweCiphertextCount(10),
                    output_glwe_dimension: GlweDimension(2),
                    output_polynomial_size: PolynomialSize(256),
                    decomposition_level: DecompositionLevelCount(3),
                    decomposition_base_log: DecompositionBaseLog(7),
                },
            ]
            .into_iter(),
        )
    }

    fn generate_random_repetition_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
    ) -> Self::RepetitionPrototypes {
        let proto_secret_key_input = <Maker as PrototypesLweSecretKey<
            Precision,
            InputCiphertextVector::KeyDistribution,
        >>::new_lwe_secret_key(
            maker, parameters.input_lwe_dimension
        );
        let proto_secret_key_output = <Maker as PrototypesGlweSecretKey<
            Precision,
            OutputCiphertext::KeyDistribution,
        >>::new_glwe_secret_key(
            maker,
            parameters.output_glwe_dimension,
            parameters.output_polynomial_size,
        );
        let proto_pksk = maker.new_packing_keyswitch_key(
            &proto_secret_key_input,
            &proto_secret_key_output,
            parameters.decomposition_level,
            parameters.decomposition_base_log,
            parameters.pksk_noise,
        );
        (proto_pksk, proto_secret_key_input, proto_secret_key_output)
    }

    fn generate_random_sample_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        let (_, proto_input_secret_key, _) = repetition_proto;
        let raw_plaintext_vector = Precision::Raw::uniform_vec(parameters.input_lwe_count.0);
        let proto_plaintext_vector =
            maker.transform_raw_vec_to_plaintext_vector(raw_plaintext_vector.as_slice());
        let proto_input_ciphertext_vector = <Maker as PrototypesLweCiphertextVector<
            Precision,
            InputCiphertextVector::KeyDistribution,
        >>::encrypt_plaintext_vector_to_lwe_ciphertext_vector(
            maker,
            proto_input_secret_key,
            &proto_plaintext_vector,
            parameters.input_lwe_noise,
        );
        let proto_output_ciphertext = <Maker as PrototypesGlweCiphertext<
            Precision,
            OutputCiphertext::KeyDistribution,
        >>::trivially_encrypt_zeros_to_glwe_ciphertext(
            maker,
            parameters.output_glwe_dimension,
            parameters.output_polynomial_size,
        );
        (
            proto_plaintext_vector,
            proto_input_ciphertext_vector,
            proto_output_ciphertext,
        )
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_packing_keyswitch_key, ..) = repetition_proto;
        let (_, proto_input_ciphertext_vector, proto_output_ciphertext) = sample_proto;
        let synth_packing_keywsitch_key =
            maker.synthesize_packing_keyswitch_key(proto_packing_keyswitch_key);
        let synth_input_ciphertext_vector =
            maker.synthesize_lwe_ciphertext_vector(proto_input_ciphertext_vector);
        let synth_output_ciphertext = maker.synthesize_glwe_ciphertext(proto_output_ciphertext);
        (
            synth_output_ciphertext,
            synth_input_ciphertext_vector,
            synth_packing_keywsitch_key,
        )
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (mut output_ciphertext, input_ciphertext_vector, pksk) = context;
        unsafe {
            engine.discard_packing_keyswitch_lwe_ciphertext_vector_unchecked(
                &mut output_ciphertext,
                &input_ciphertext_vector,
                &pksk,
            );
        };
        (output_ciphertext, input_ciphertext_vector, pksk)
    }

    fn process_context(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (output_ciphertext, input_ciphertext, keyswitch_key) = context;
        let (_, _, proto_output_secret_key) = repetition_proto;
        let (proto_plaintext, ..) = sample_proto;
        let proto_output_ciphertext = maker.unsynthesize_glwe_ciphertext(&output_ciphertext);
        let proto_output_plaintext = <Maker as PrototypesGlweCiphertext<
            Precision,
            OutputCiphertext::KeyDistribution,
        >>::decrypt_glwe_ciphertext_to_plaintext_vector(
            maker,
            proto_output_secret_key,
            &proto_output_ciphertext,
        );
        maker.destroy_lwe_ciphertext_vector(input_ciphertext);
        maker.destroy_glwe_ciphertext(output_ciphertext);
        maker.destroy_packing_keyswitch_key(keyswitch_key);
        (
            maker.transform_plaintext_vector_to_raw_vec(proto_plaintext),
            maker
                .transform_plaintext_vector_to_raw_vec(&proto_output_plaintext)
                .iter()
                .take(parameters.input_lwe_count.0)
                .cloned()
                .collect::<Vec<Precision::Raw>>(),
        )
    }

    fn compute_criteria(
        parameters: &Self::Parameters,
        _maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::Criteria {
        let predicted_variance: Variance =
            fix_estimate_keyswitch_noise_lwe_to_glwe_with_constant_terms::<
                Precision::Raw,
                _,
                _,
                OutputCiphertext::KeyDistribution,
            >(
                parameters.input_lwe_dimension,
                parameters.input_lwe_noise,
                parameters.pksk_noise,
                parameters.decomposition_base_log,
                parameters.decomposition_level,
            );
        (Variance(
            predicted_variance.0 * parameters.input_lwe_count.0 as f64,
        ),)
    }

    fn verify(criteria: &Self::Criteria, outputs: &[Self::Outcome]) -> bool {
        let (means, actual): (Vec<_>, Vec<_>) = outputs.iter().cloned().unzip();
        let means = means
            .iter()
            .flat_map(|r| r.iter())
            .copied()
            .collect::<Vec<_>>();
        let actual = actual
            .iter()
            .flat_map(|r| r.iter())
            .copied()
            .collect::<Vec<_>>();
        assert_noise_distribution(&actual, means.as_slice(), criteria.0)
    }
}
