use crate::fixture::Fixture;
use crate::generation::prototyping::{
    PrototypesGlweCiphertext, PrototypesGlweSecretKey, PrototypesLweCiphertext,
    PrototypesPlaintext, PrototypesPlaintextVector,
};
use crate::generation::synthesizing::{SynthesizesGlweCiphertext, SynthesizesLweCiphertext};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{GlweDimension, LweDimension, MonomialIndex, PolynomialSize};
use concrete_core::prelude::{
    GlweCiphertextEntity, LweCiphertextDiscardingExtractionEngine, LweCiphertextEntity,
};

/// A fixture for the types implementing the `LweCiphertextDiscardingExtractionEngine` trait.
pub struct LweCiphertextDiscardingExtractionFixture;

#[derive(Debug)]
pub struct LweCiphertextDiscardingExtractionParameters {
    pub noise: Variance,
    pub glwe_dimension: GlweDimension,
    pub poly_size: PolynomialSize,
    pub nth: MonomialIndex,
}

#[allow(clippy::type_complexity)]
impl<Precision, Engine, GlweCiphertext, LweCiphertext>
    Fixture<Precision, Engine, (GlweCiphertext, LweCiphertext)>
    for LweCiphertextDiscardingExtractionFixture
where
    Precision: IntegerPrecision,
    Engine: LweCiphertextDiscardingExtractionEngine<GlweCiphertext, LweCiphertext>,
    GlweCiphertext: GlweCiphertextEntity,
    LweCiphertext: LweCiphertextEntity<KeyDistribution = GlweCiphertext::KeyDistribution>,
    Maker: SynthesizesLweCiphertext<Precision, LweCiphertext>
        + SynthesizesGlweCiphertext<Precision, GlweCiphertext>,
{
    type Parameters = LweCiphertextDiscardingExtractionParameters;
    type RepetitionPrototypes = (
        <Maker as PrototypesGlweSecretKey<Precision, GlweCiphertext::KeyDistribution>>::GlweSecretKeyProto,
    );
    type SamplePrototypes = (
        <Maker as PrototypesPlaintextVector<Precision>>::PlaintextVectorProto,
        <Maker as PrototypesGlweCiphertext<Precision, GlweCiphertext::KeyDistribution>>::GlweCiphertextProto,
        <Maker as PrototypesLweCiphertext<Precision, GlweCiphertext::KeyDistribution>>::LweCiphertextProto,
    );
    type PreExecutionContext = (GlweCiphertext, LweCiphertext);
    type PostExecutionContext = (GlweCiphertext, LweCiphertext);
    type Criteria = (Variance,);
    type Outcome = (Precision::Raw, Precision::Raw);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![LweCiphertextDiscardingExtractionParameters {
                noise: Variance(0.00000001),
                glwe_dimension: GlweDimension(200),
                poly_size: PolynomialSize(256),
                nth: MonomialIndex(0),
            }]
            .into_iter(),
        )
    }

    fn generate_random_repetition_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
    ) -> Self::RepetitionPrototypes {
        let proto_secret_key =
            maker.new_glwe_secret_key(parameters.glwe_dimension, parameters.poly_size);
        (proto_secret_key,)
    }

    fn generate_random_sample_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        let (proto_secret_key,) = repetition_proto;
        let raw_plaintext_vector = Precision::Raw::uniform_vec(parameters.poly_size.0);
        let proto_plaintext_vector =
            maker.transform_raw_vec_to_plaintext_vector(&raw_plaintext_vector);
        let proto_glwe_ciphertext = maker.encrypt_plaintext_vector_to_glwe_ciphertext(
            proto_secret_key,
            &proto_plaintext_vector,
            parameters.noise,
        );
        let proto_lwe_ciphertext = maker.trivially_encrypt_zero_to_lwe_ciphertext(LweDimension(
            parameters.glwe_dimension.0 * parameters.poly_size.0,
        ));
        (
            proto_plaintext_vector,
            proto_glwe_ciphertext,
            proto_lwe_ciphertext,
        )
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (_, proto_glwe_ciphertext, proto_lwe_ciphertext) = sample_proto;
        let synth_glwe_ciphertext = maker.synthesize_glwe_ciphertext(proto_glwe_ciphertext);
        let synth_lwe_cipehrtext = maker.synthesize_lwe_ciphertext(proto_lwe_ciphertext);
        (synth_glwe_ciphertext, synth_lwe_cipehrtext)
    }

    fn execute_engine(
        parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (glwe_ciphertext, mut lwe_ciphertext) = context;
        unsafe {
            engine.discard_extract_lwe_ciphertext_unchecked(
                &mut lwe_ciphertext,
                &glwe_ciphertext,
                parameters.nth,
            )
        };
        (glwe_ciphertext, lwe_ciphertext)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (glwe_ciphertext, lwe_ciphertext) = context;
        let (proto_glwe_secret_key,) = repetition_proto;
        let (proto_plaintext_vector, ..) = sample_proto;
        let proto_output_ciphertext = maker.unsynthesize_lwe_ciphertext(&lwe_ciphertext);
        let proto_lwe_secret_key =
            maker.transmute_glwe_secret_key_to_lwe_secret_key(proto_glwe_secret_key);
        let proto_output_plaintext = maker
            .decrypt_lwe_ciphertext_to_plaintext(&proto_lwe_secret_key, &proto_output_ciphertext);
        maker.destroy_lwe_ciphertext(lwe_ciphertext);
        maker.destroy_glwe_ciphertext(glwe_ciphertext);
        (
            maker.transform_plaintext_vector_to_raw_vec(proto_plaintext_vector)[_parameters.nth.0],
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
