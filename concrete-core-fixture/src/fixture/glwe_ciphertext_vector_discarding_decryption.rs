use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{GlweDimension, PolynomialSize};
use concrete_core::prelude::{
    GlweCiphertextCount, GlweCiphertextVectorDiscardingDecryptionEngine,
    GlweCiphertextVectorEntity, GlweSecretKeyEntity, PlaintextVectorEntity,
};

use crate::fixture::Fixture;
use crate::generation::prototyping::{
    PrototypesGlweCiphertextVector, PrototypesGlweSecretKey, PrototypesPlaintextVector,
};
use crate::generation::synthesizing::{
    SynthesizesGlweCiphertextVector, SynthesizesGlweSecretKey, SynthesizesPlaintextVector,
};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;

/// A fixture for the types implementing the `GlweCiphertextVectorDiscardingDecryptionEngine` trait.
pub struct GlweCiphertextVectorDiscardingDecryptionFixture;

#[derive(Debug)]
pub struct GlweCiphertextVectorDiscardingDecryptionParameters {
    pub noise: Variance,
    pub glwe_dimension: GlweDimension,
    pub polynomial_size: PolynomialSize,
    pub count: GlweCiphertextCount,
}

impl<Precision, Engine, PlaintextVector, SecretKey, CiphertextVector>
    Fixture<Precision, Engine, (PlaintextVector, SecretKey, CiphertextVector)>
    for GlweCiphertextVectorDiscardingDecryptionFixture
where
    Precision: IntegerPrecision,
    Engine: GlweCiphertextVectorDiscardingDecryptionEngine<
        SecretKey,
        CiphertextVector,
        PlaintextVector,
    >,
    PlaintextVector: PlaintextVectorEntity,
    SecretKey: GlweSecretKeyEntity,
    CiphertextVector: GlweCiphertextVectorEntity<KeyDistribution = SecretKey::KeyDistribution>,
    Maker: SynthesizesPlaintextVector<Precision, PlaintextVector>
        + SynthesizesGlweSecretKey<Precision, SecretKey>
        + SynthesizesGlweCiphertextVector<Precision, CiphertextVector>,
{
    type Parameters = GlweCiphertextVectorDiscardingDecryptionParameters;
    type RepetitionPrototypes = (
        <Maker as PrototypesGlweSecretKey<Precision, SecretKey::KeyDistribution>>::GlweSecretKeyProto,
    );
    type SamplePrototypes = (
        <Maker as PrototypesPlaintextVector<Precision>>::PlaintextVectorProto,
        <Maker as PrototypesGlweCiphertextVector<Precision, SecretKey::KeyDistribution>>::GlweCiphertextVectorProto,
        <Maker as PrototypesPlaintextVector<Precision>>::PlaintextVectorProto
    );
    type PreExecutionContext = (SecretKey, CiphertextVector, PlaintextVector);
    type PostExecutionContext = (SecretKey, CiphertextVector, PlaintextVector);
    type Criteria = (Variance,);
    type Outcome = (Vec<Precision::Raw>, Vec<Precision::Raw>);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                GlweCiphertextVectorDiscardingDecryptionParameters {
                    noise: Variance(0.00000001),
                    glwe_dimension: GlweDimension(200),
                    polynomial_size: PolynomialSize(256),
                    count: GlweCiphertextCount(10),
                },
                GlweCiphertextVectorDiscardingDecryptionParameters {
                    noise: Variance(0.00000001),
                    glwe_dimension: GlweDimension(1),
                    polynomial_size: PolynomialSize(256),
                    count: GlweCiphertextCount(10),
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
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        let (proto_secret_key,) = repetition_proto;
        let raw_plaintext_vector =
            Precision::Raw::uniform_vec(parameters.polynomial_size.0 * parameters.count.0);
        let proto_plaintext_vector =
            maker.transform_raw_vec_to_plaintext_vector(raw_plaintext_vector.as_slice());
        let proto_ciphertext_vector = maker.encrypt_plaintext_vector_to_glwe_ciphertext_vector(
            proto_secret_key,
            &proto_plaintext_vector,
            parameters.noise,
        );
        let raw_output_plaintext_vector =
            Precision::Raw::uniform_vec(parameters.polynomial_size.0 * parameters.count.0);
        let proto_output_plaintext_vector =
            maker.transform_raw_vec_to_plaintext_vector(&raw_output_plaintext_vector);
        (
            proto_plaintext_vector,
            proto_ciphertext_vector,
            proto_output_plaintext_vector,
        )
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_secret_key,) = repetition_proto;
        let (_, proto_ciphertext_vector, proto_output_plaintext_vector) = sample_proto;
        let secret_key = maker.synthesize_glwe_secret_key(proto_secret_key);
        let ciphertext_vector = maker.synthesize_glwe_ciphertext_vector(proto_ciphertext_vector);
        let plaintext_vector = maker.synthesize_plaintext_vector(proto_output_plaintext_vector);
        (secret_key, ciphertext_vector, plaintext_vector)
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (secret_key, ciphertext_vector, mut plaintext_vector) = context;
        unsafe {
            engine.discard_decrypt_glwe_ciphertext_vector_unchecked(
                &secret_key,
                &mut plaintext_vector,
                &ciphertext_vector,
            )
        };
        (secret_key, ciphertext_vector, plaintext_vector)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (proto_plaintext_vector, ..) = sample_proto;
        let (secret_key, ciphertext_vector, plaintext_vector) = context;
        let proto_output_plaintext_vector = maker.unsynthesize_plaintext_vector(&plaintext_vector);
        maker.destroy_glwe_ciphertext_vector(ciphertext_vector);
        maker.destroy_glwe_secret_key(secret_key);
        maker.destroy_plaintext_vector(plaintext_vector);
        (
            maker.transform_plaintext_vector_to_raw_vec(proto_plaintext_vector),
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
        assert_noise_distribution(actual.as_slice(), means.as_slice(), criteria.0)
    }
}
