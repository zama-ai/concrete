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
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_delta_std_dev;
use concrete_commons::dispersion::{DispersionParameter, LogStandardDev, Variance};
use concrete_commons::key_kinds::{BinaryKeyKind, GaussianKeyKind, TernaryKeyKind};
use concrete_commons::numeric::{Numeric, UnsignedInteger};
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};
use concrete_core::prelude::markers::{
    BinaryKeyDistribution, GaussianKeyDistribution, KeyDistributionMarker, TernaryKeyDistribution,
};
use concrete_core::prelude::{
    GlweCiphertextEntity, LweBootstrapKeyEntity, LweCiphertextDiscardingBootstrapEngine,
    LweCiphertextEntity,
};
use std::any::TypeId;

/// A fixture for the types implementing the `LweCiphertextDiscardingBootstrapEngine` trait.
pub struct LweCiphertextDiscardingBootstrapFixture;

#[derive(Debug)]
pub struct LweCiphertextDiscardingBootstrapParameters {
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
    for LweCiphertextDiscardingBootstrapFixture
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
    type Parameters = LweCiphertextDiscardingBootstrapParameters;
    type RepetitionPrototypes = (
        <Maker as PrototypesPlaintextVector<Precision>>::PlaintextVectorProto,
        <Maker as PrototypesLweSecretKey<Precision, InputCiphertext::KeyDistribution>>::LweSecretKeyProto,
        <Maker as PrototypesGlweSecretKey<Precision, OutputCiphertext::KeyDistribution>>::GlweSecretKeyProto,
        <Maker as PrototypesLweBootstrapKey<Precision, InputCiphertext::KeyDistribution, OutputCiphertext::KeyDistribution>>::LweBootstrapKeyProto,
    );
    type SamplePrototypes = (
        <Maker as PrototypesPlaintext<Precision>>::PlaintextProto,
        <Maker as PrototypesGlweCiphertext<Precision, OutputCiphertext::KeyDistribution>>::GlweCiphertextProto,
        <Maker as PrototypesLweCiphertext<Precision, InputCiphertext::KeyDistribution>>::LweCiphertextProto,
        <Maker as PrototypesLweCiphertext<Precision, OutputCiphertext::KeyDistribution>>::LweCiphertextProto,
    );
    type PreExecutionContext = (BootstrapKey, Accumulator, OutputCiphertext, InputCiphertext);
    type PostExecutionContext = (BootstrapKey, Accumulator, OutputCiphertext, InputCiphertext);
    type Criteria = (Variance,);
    type Outcome = (Precision::Raw, Precision::Raw);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                LweCiphertextDiscardingBootstrapParameters {
                    noise: Variance(LogStandardDev::from_log_standard_dev(-29.).get_variance()),
                    lwe_dimension: LweDimension(630),
                    glwe_dimension: GlweDimension(1),
                    poly_size: PolynomialSize(512),
                    decomp_level_count: DecompositionLevelCount(3),
                    decomp_base_log: DecompositionBaseLog(7),
                },
                LweCiphertextDiscardingBootstrapParameters {
                    noise: Variance(LogStandardDev::from_log_standard_dev(-29.).get_variance()),
                    lwe_dimension: LweDimension(630),
                    glwe_dimension: GlweDimension(1),
                    poly_size: PolynomialSize(1024),
                    decomp_level_count: DecompositionLevelCount(3),
                    decomp_base_log: DecompositionBaseLog(7),
                },
                LweCiphertextDiscardingBootstrapParameters {
                    noise: Variance(LogStandardDev::from_log_standard_dev(-29.).get_variance()),
                    lwe_dimension: LweDimension(630),
                    glwe_dimension: GlweDimension(1),
                    poly_size: PolynomialSize(2048),
                    decomp_level_count: DecompositionLevelCount(3),
                    decomp_base_log: DecompositionBaseLog(7),
                },
            ]
            .into_iter(),
        )
    }

    fn generate_random_repetition_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
    ) -> Self::RepetitionPrototypes {
        let raw_plaintext_vector =
            vec![Precision::Raw::ONE << (Precision::Raw::BITS - 3); parameters.poly_size.0];
        let proto_plaintext_vector =
            maker.transform_raw_vec_to_plaintext_vector(raw_plaintext_vector.as_slice());
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
            proto_plaintext_vector,
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
        let (proto_plaintext_vector, proto_lwe_secret_key, ..) = repetition_proto;
        let raw_plaintext = Precision::Raw::uniform_n_msb_padding(1, 3);
        let proto_plaintext = maker.transform_raw_to_plaintext(&raw_plaintext);
        let proto_accumulator = maker.trivial_encrypt_plaintext_vector_to_glwe_ciphertext(
            parameters.glwe_dimension,
            proto_plaintext_vector,
        );
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
        >>::trivial_encrypt_zero_to_lwe_ciphertext(
            maker,
            LweDimension(parameters.glwe_dimension.0 * parameters.poly_size.0),
        );
        (
            proto_plaintext,
            proto_accumulator,
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
        let (_, _, _, proto_bootstrap_key) = repetition_proto;
        let (_, proto_accumulator, proto_input_ciphertext, proto_output_ciphertext) = sample_proto;
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
            maker.convert_glwe_secret_key_to_lwe_secret_key(proto_glwe_secret_key);
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
        let predicted_variance: Variance =
            fix_estimate_pbs_noise::<Precision::Raw, Variance, OutputCiphertext::KeyDistribution>(
                parameters.lwe_dimension,
                parameters.poly_size,
                parameters.glwe_dimension,
                parameters.decomp_base_log,
                parameters.decomp_level_count,
                parameters.noise,
            );
        (predicted_variance,)
    }

    fn verify(criteria: &Self::Criteria, outputs: &[Self::Outcome]) -> bool {
        let (means, actual): (Vec<_>, Vec<_>) = outputs.iter().cloned().unzip();
        assert_delta_std_dev(&actual, means.as_slice(), criteria.0)
    }
}

// The current NPE does not use the key distribution markers of concrete-core. This function makes
// the mapping. This function should be removed as soon as the npe uses the types of concrete-core.
fn fix_estimate_pbs_noise<T, D, K>(
    lwe_mask_size: LweDimension,
    poly_size: PolynomialSize,
    rlwe_mask_size: GlweDimension,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
    dispersion_bsk: D,
) -> Variance
where
    T: UnsignedInteger,
    D: DispersionParameter,
    K: KeyDistributionMarker,
{
    let k_type_id = TypeId::of::<K>();
    if k_type_id == TypeId::of::<BinaryKeyDistribution>() {
        concrete_npe::estimate_pbs_noise::<T, D, BinaryKeyKind>(
            lwe_mask_size,
            poly_size,
            rlwe_mask_size,
            base_log,
            level,
            dispersion_bsk,
        )
    } else if k_type_id == TypeId::of::<TernaryKeyDistribution>() {
        concrete_npe::estimate_pbs_noise::<T, D, TernaryKeyKind>(
            lwe_mask_size,
            poly_size,
            rlwe_mask_size,
            base_log,
            level,
            dispersion_bsk,
        )
    } else if k_type_id == TypeId::of::<GaussianKeyDistribution>() {
        concrete_npe::estimate_pbs_noise::<T, D, GaussianKeyKind>(
            lwe_mask_size,
            poly_size,
            rlwe_mask_size,
            base_log,
            level,
            dispersion_bsk,
        )
    } else {
        panic!("Unknown key distribution encountered.")
    }
}
