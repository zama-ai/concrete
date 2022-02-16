use crate::fixture::Fixture;
use crate::generation::prototyping::{
    PrototypesGgswCiphertext, PrototypesGlweCiphertext, PrototypesGlweSecretKey,
    PrototypesPlaintext, PrototypesPlaintextVector,
};
use crate::generation::synthesizing::{SynthesizesGgswCiphertext, SynthesizesGlweCiphertext};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::{DispersionParameter, LogStandardDev, Variance};
use concrete_commons::key_kinds::{BinaryKeyKind, GaussianKeyKind, TernaryKeyKind};
use concrete_commons::numeric::UnsignedInteger;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
};
use concrete_core::prelude::markers::{
    BinaryKeyDistribution, GaussianKeyDistribution, KeyDistributionMarker, TernaryKeyDistribution,
};
use concrete_core::prelude::{
    GgswCiphertextEntity, GlweCiphertextEntity, GlweCiphertextGgswCiphertextExternalProductEngine,
};
use std::any::TypeId;

/// A fixture for the types implementing the `GlweCiphertextGgswCiphertextExternalProduct`
/// trait.
pub struct GlweCiphertextGgswCiphertextExternalProductFixture;

#[derive(Debug)]
pub struct GlweCiphertextGgswCiphertextExternalProductParameters {
    pub ggsw_noise: Variance,
    pub glwe_noise: Variance,
    pub glwe_dimension: GlweDimension,
    pub polynomial_size: PolynomialSize,
    pub decomposition_base_log: DecompositionBaseLog,
    pub decomposition_level_count: DecompositionLevelCount,
}

impl<Precision, Engine, GlweInput, GgswInput, GlweOutput>
    Fixture<Precision, Engine, (GlweInput, GgswInput, GlweOutput)>
    for GlweCiphertextGgswCiphertextExternalProductFixture
where
    Precision: IntegerPrecision,
    Engine: GlweCiphertextGgswCiphertextExternalProductEngine<GlweInput, GgswInput, GlweOutput>,
    GlweInput: GlweCiphertextEntity,
    GgswInput: GgswCiphertextEntity<KeyDistribution = GlweInput::KeyDistribution>,
    GlweOutput: GlweCiphertextEntity<KeyDistribution = GlweInput::KeyDistribution>,
    Maker: SynthesizesGlweCiphertext<Precision, GlweInput>
        + SynthesizesGlweCiphertext<Precision, GlweOutput>
        + SynthesizesGgswCiphertext<Precision, GgswInput>,
{
    type Parameters = GlweCiphertextGgswCiphertextExternalProductParameters;
    type Criteria = (Variance,);
    type Outcome = (Vec<Precision::Raw>, Vec<Precision::Raw>);
    type RepetitionPrototypes = (
        <Maker as PrototypesPlaintext<Precision>>::PlaintextProto,
        <Maker as PrototypesGlweSecretKey<Precision, GlweInput::KeyDistribution>>::GlweSecretKeyProto,
        <Maker as PrototypesGgswCiphertext<Precision, GlweInput::KeyDistribution>>::GgswCiphertextProto,
    );
    type SamplePrototypes = (
        <Maker as PrototypesPlaintextVector<Precision>>::PlaintextVectorProto,
        <Maker as PrototypesGlweCiphertext<Precision, GlweInput::KeyDistribution>>::GlweCiphertextProto,
    );
    type PreExecutionContext = (GlweInput, GgswInput);
    type PostExecutionContext = (GlweInput, GgswInput, GlweOutput);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                GlweCiphertextGgswCiphertextExternalProductParameters {
                    ggsw_noise: Variance(LogStandardDev(-25.).get_variance()),
                    glwe_noise: Variance(LogStandardDev(-20.).get_variance()),
                    glwe_dimension: GlweDimension(2),
                    polynomial_size: PolynomialSize(512),
                    decomposition_base_log: DecompositionBaseLog(6),
                    decomposition_level_count: DecompositionLevelCount(4),
                },
                GlweCiphertextGgswCiphertextExternalProductParameters {
                    ggsw_noise: Variance(LogStandardDev(-25.).get_variance()),
                    glwe_noise: Variance(LogStandardDev(-20.).get_variance()),
                    glwe_dimension: GlweDimension(2),
                    polynomial_size: PolynomialSize(1024),
                    decomposition_base_log: DecompositionBaseLog(6),
                    decomposition_level_count: DecompositionLevelCount(4),
                },
                GlweCiphertextGgswCiphertextExternalProductParameters {
                    ggsw_noise: Variance(LogStandardDev(-25.).get_variance()),
                    glwe_noise: Variance(LogStandardDev(-20.).get_variance()),
                    glwe_dimension: GlweDimension(2),
                    polynomial_size: PolynomialSize(2048),
                    decomposition_base_log: DecompositionBaseLog(6),
                    decomposition_level_count: DecompositionLevelCount(4),
                },
            ]
            .into_iter(),
        )
    }

    fn generate_random_repetition_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
    ) -> Self::RepetitionPrototypes {
        let proto_secret_key =
            maker.new_glwe_secret_key(parameters.glwe_dimension, parameters.polynomial_size);
        let raw_plaintext = Precision::Raw::pick(&[
            Precision::Raw::zero(),
            Precision::Raw::one(),
            Precision::Raw::power_of_two(1),
        ]);
        let proto_plaintext = maker.transform_raw_to_plaintext(&raw_plaintext);
        let proto_ggsw = maker.encrypt_plaintext_to_ggsw_ciphertext(
            &proto_secret_key,
            &proto_plaintext,
            parameters.ggsw_noise,
            parameters.decomposition_level_count,
            parameters.decomposition_base_log,
        );
        (proto_plaintext, proto_secret_key, proto_ggsw)
    }

    fn generate_random_sample_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        let (_, proto_secret_key, _) = repetition_proto;
        let raw_plaintext_vector = Precision::Raw::uniform_vec(parameters.polynomial_size.0);
        let proto_plaintext_vector =
            maker.transform_raw_vec_to_plaintext_vector(&raw_plaintext_vector);
        let proto_glwe_ciphertext = maker.encrypt_plaintext_vector_to_glwe_ciphertext(
            proto_secret_key,
            &proto_plaintext_vector,
            parameters.glwe_noise,
        );
        (proto_plaintext_vector, proto_glwe_ciphertext)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (_, proto_glwe_ciphertext) = sample_proto;
        let (_, _, proto_ggsw_ciphertext) = repetition_proto;
        let synth_glwe_ciphertext = maker.synthesize_glwe_ciphertext(proto_glwe_ciphertext);
        let synth_ggsw_ciphertext = maker.synthesize_ggsw_ciphertext(proto_ggsw_ciphertext);
        (synth_glwe_ciphertext, synth_ggsw_ciphertext)
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (glwe_ciphertext, ggsw_ciphertext) = context;
        let output_glwe_ciphertext = unsafe {
            engine.compute_external_product_glwe_ciphertext_ggsw_ciphertext_unchecked(
                &glwe_ciphertext,
                &ggsw_ciphertext,
            )
        };
        (glwe_ciphertext, ggsw_ciphertext, output_glwe_ciphertext)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (glwe_ciphertext, ggsw_ciphertext, output_glwe_ciphertext) = context;
        let (proto_plaintext, proto_secret_key, _) = repetition_proto;
        let (proto_input_plaintext_vector, ..) = sample_proto;
        let proto_output_glwe_ciphertext =
            maker.unsynthesize_glwe_ciphertext(&output_glwe_ciphertext);
        maker.destroy_glwe_ciphertext(glwe_ciphertext);
        maker.destroy_ggsw_ciphertext(ggsw_ciphertext);
        maker.destroy_glwe_ciphertext(output_glwe_ciphertext);
        let proto_output_plaintext_vector = maker.decrypt_glwe_ciphertext_to_plaintext_vector(
            proto_secret_key,
            &proto_output_glwe_ciphertext,
        );
        let raw_input_plaintext = maker.transform_plaintext_to_raw(proto_plaintext);
        let raw_input_plaintext_vector = maker
            .transform_plaintext_vector_to_raw_vec(proto_input_plaintext_vector)
            .into_iter()
            .map(|v| v * raw_input_plaintext)
            .collect();
        (
            raw_input_plaintext_vector,
            maker.transform_plaintext_vector_to_raw_vec(&proto_output_plaintext_vector),
        )
    }

    fn compute_criteria(
        parameters: &Self::Parameters,
        _maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::Criteria {
        let output_variance = fix_estimate_external_product_noise_with_binary_ggsw::<
            Precision::Raw,
            Variance,
            Variance,
            GlweInput::KeyDistribution,
        >(
            parameters.polynomial_size,
            parameters.glwe_dimension,
            parameters.glwe_noise,
            parameters.ggsw_noise,
            parameters.decomposition_base_log,
            parameters.decomposition_level_count,
        );
        (output_variance,)
    }

    fn verify(criteria: &Self::Criteria, outputs: &[Self::Outcome]) -> bool {
        let (means, actual): (Vec<_>, Vec<_>) = outputs.iter().cloned().unzip();
        let means = means.into_iter().flatten().collect::<Vec<_>>();
        let actual = actual.into_iter().flatten().collect::<Vec<_>>();
        assert_noise_distribution(&actual, means.as_slice(), criteria.0)
    }
}

// FIXME:
// The current NPE does not use the key distribution markers of concrete-core. This function makes
// the mapping. This function should be removed as soon as the npe uses the types of concrete-core.
fn fix_estimate_external_product_noise_with_binary_ggsw<T, D1, D2, K>(
    poly_size: PolynomialSize,
    rlwe_mask_size: GlweDimension,
    var_glwe: D1,
    var_ggsw: D2,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance
where
    T: UnsignedInteger,
    D1: DispersionParameter,
    D2: DispersionParameter,
    K: KeyDistributionMarker,
{
    let k_type_id = TypeId::of::<K>();
    if k_type_id == TypeId::of::<BinaryKeyDistribution>() {
        concrete_npe::estimate_external_product_noise_with_binary_ggsw::<T, D1, D2, BinaryKeyKind>(
            poly_size,
            rlwe_mask_size,
            var_glwe,
            var_ggsw,
            base_log,
            level,
        )
    } else if k_type_id == TypeId::of::<TernaryKeyDistribution>() {
        concrete_npe::estimate_external_product_noise_with_binary_ggsw::<T, D1, D2, TernaryKeyKind>(
            poly_size,
            rlwe_mask_size,
            var_glwe,
            var_ggsw,
            base_log,
            level,
        )
    } else if k_type_id == TypeId::of::<GaussianKeyDistribution>() {
        concrete_npe::estimate_external_product_noise_with_binary_ggsw::<T, D1, D2, GaussianKeyKind>(
            poly_size,
            rlwe_mask_size,
            var_glwe,
            var_ggsw,
            base_log,
            level,
        )
    } else {
        panic!("Unknown key distribution encountered.")
    }
}
