use crate::fixture::Fixture;
use crate::generation::prototyping::{
    PrototypesGlweCiphertext, PrototypesGlweSecretKey, PrototypesPlaintextVector,
};
use crate::generation::synthesizing::{SynthesizesGlweCiphertext, SynthesizesGlweSecretKey};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{GlweDimension, PolynomialSize};
use concrete_core::prelude::{
    GlweCiphertextEntity, GlweCiphertextZeroEncryptionEngine, GlweSecretKeyEntity,
};

/// A fixture for the types implementing the `GlweCiphertextZeroEncryptionEngine` trait.
pub struct GlweCiphertextZeroEncryptionFixture;

#[derive(Debug)]
pub struct GlweCiphertextZeroEncryptionParameters {
    pub noise: Variance,
    pub glwe_dimension: GlweDimension,
    pub polynomial_size: PolynomialSize,
}

impl<Precision, Engine, SecretKey, Ciphertext> Fixture<Precision, Engine, (SecretKey, Ciphertext)>
    for GlweCiphertextZeroEncryptionFixture
where
    Precision: IntegerPrecision,
    Engine: GlweCiphertextZeroEncryptionEngine<SecretKey, Ciphertext>,
    SecretKey: GlweSecretKeyEntity,
    Ciphertext: GlweCiphertextEntity<KeyDistribution = SecretKey::KeyDistribution>,
    Maker: SynthesizesGlweSecretKey<Precision, SecretKey>
        + SynthesizesGlweCiphertext<Precision, Ciphertext>,
{
    type Parameters = GlweCiphertextZeroEncryptionParameters;
    type RepetitionPrototypes = (
        <Maker as PrototypesGlweSecretKey<Precision, SecretKey::KeyDistribution>>::GlweSecretKeyProto,
    );
    type SamplePrototypes = ();
    type PreExecutionContext = (SecretKey,);
    type PostExecutionContext = (SecretKey, Ciphertext);
    type Criteria = (Variance,);
    type Outcome = (Vec<Precision::Raw>, Vec<Precision::Raw>);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                GlweCiphertextZeroEncryptionParameters {
                    noise: Variance(0.00000001),
                    glwe_dimension: GlweDimension(200),
                    polynomial_size: PolynomialSize(256),
                },
                GlweCiphertextZeroEncryptionParameters {
                    noise: Variance(0.00000001),
                    glwe_dimension: GlweDimension(1),
                    polynomial_size: PolynomialSize(2),
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
        (proto_secret_key,)
    }

    fn generate_random_sample_prototypes(
        _parameters: &Self::Parameters,
        _maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        _sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_secret_key,) = repetition_proto;
        (maker.synthesize_glwe_secret_key(proto_secret_key),)
    }

    fn execute_engine(
        parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (secret_key,) = context;
        let ciphertext =
            unsafe { engine.zero_encrypt_glwe_ciphertext_unchecked(&secret_key, parameters.noise) };
        (secret_key, ciphertext)
    }

    fn process_context(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        _sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (proto_secret_key,) = repetition_proto;
        let (secret_key, ciphertext) = context;
        let proto_output_ciphertext = maker.unsynthesize_glwe_ciphertext(&ciphertext);
        let proto_output_plaintext_vector = maker.decrypt_glwe_ciphertext_to_plaintext_vector(
            proto_secret_key,
            &proto_output_ciphertext,
        );
        maker.destroy_glwe_secret_key(secret_key);
        maker.destroy_glwe_ciphertext(ciphertext);
        (
            Precision::Raw::zero_vec(parameters.polynomial_size.0),
            maker.transform_plaintext_vector_to_raw_vec(&proto_output_plaintext_vector),
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
        let means: Vec<Precision::Raw> = means.into_iter().flatten().collect();
        let actual: Vec<Precision::Raw> = actual.into_iter().flatten().collect();
        assert_noise_distribution(&actual, means.as_slice(), criteria.0)
    }
}
