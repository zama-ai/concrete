use crate::fixture::Fixture;
use crate::generation::prototyping::{PrototypesGlweCiphertextVector, PrototypesPlaintextVector};
use crate::generation::synthesizing::{
    SynthesizesGlweCiphertextVector, SynthesizesPlaintextVector,
};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{GlweCiphertextCount, GlweDimension, PolynomialSize};
use concrete_core::prelude::{
    GlweCiphertextVectorEntity, GlweCiphertextVectorTrivialEncryptionEngine, PlaintextVectorEntity,
};

/// A fixture for the types implementing the `GlweCiphertextVectorTrivialEncryptionEngine` trait.
pub struct GlweCiphertextVectorTrivialEncryptionFixture;

#[derive(Debug)]
pub struct GlweCiphertextVectorTrivialEncryptionParameters {
    pub glwe_dimension: GlweDimension,
    pub polynomial_size: PolynomialSize,
    pub count: GlweCiphertextCount,
}

impl<Precision, Engine, PlaintextVector, CiphertextVector>
    Fixture<Precision, Engine, (PlaintextVector, CiphertextVector)>
    for GlweCiphertextVectorTrivialEncryptionFixture
where
    Precision: IntegerPrecision,
    Engine: GlweCiphertextVectorTrivialEncryptionEngine<PlaintextVector, CiphertextVector>,
    PlaintextVector: PlaintextVectorEntity,
    CiphertextVector: GlweCiphertextVectorEntity,
    Maker: SynthesizesPlaintextVector<Precision, PlaintextVector>
        + SynthesizesGlweCiphertextVector<Precision, CiphertextVector>,
{
    type Parameters = GlweCiphertextVectorTrivialEncryptionParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes =
        (<Maker as PrototypesPlaintextVector<Precision>>::PlaintextVectorProto,);
    type PreExecutionContext = (PlaintextVector,);
    type PostExecutionContext = (PlaintextVector, CiphertextVector);
    type Criteria = (Variance,);
    type Outcome = (Vec<Precision::Raw>, Vec<Precision::Raw>);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                GlweCiphertextVectorTrivialEncryptionParameters {
                    glwe_dimension: GlweDimension(200),
                    polynomial_size: PolynomialSize(256),
                    count: GlweCiphertextCount(100),
                },
                GlweCiphertextVectorTrivialEncryptionParameters {
                    glwe_dimension: GlweDimension(1),
                    polynomial_size: PolynomialSize(2),
                    count: GlweCiphertextCount(1),
                },
            ]
            .into_iter(),
        )
    }

    fn generate_random_repetition_prototypes(
        _parameters: &Self::Parameters,
        _maker: &mut Maker,
    ) -> Self::RepetitionPrototypes {
    }

    fn generate_random_sample_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        let raw_plaintext_vector =
            Precision::Raw::uniform_vec(parameters.polynomial_size.0 * parameters.count.0);
        let proto_plaintext_vector =
            maker.transform_raw_vec_to_plaintext_vector(raw_plaintext_vector.as_slice());
        (proto_plaintext_vector,)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_plaintext_vector,) = sample_proto;
        (maker.synthesize_plaintext_vector(proto_plaintext_vector),)
    }

    fn execute_engine(
        parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (plaintext_vector,) = context;
        let ciphertext_vector = unsafe {
            engine.trivially_encrypt_glwe_ciphertext_vector_unchecked(
                parameters.glwe_dimension.to_glwe_size(),
                parameters.count,
                &plaintext_vector,
            )
        };
        (plaintext_vector, ciphertext_vector)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (proto_plaintext_vector,) = sample_proto;
        let (plaintext_vector, ciphertext_vector) = context;
        let proto_output_ciphertext_vector =
            maker.unsynthesize_glwe_ciphertext_vector(&ciphertext_vector);
        let proto_output_plaintext_vector = maker
            .trivially_decrypt_glwe_ciphertext_vector_to_plaintext_vector(
                &proto_output_ciphertext_vector,
            );
        maker.destroy_plaintext_vector(plaintext_vector);
        maker.destroy_glwe_ciphertext_vector(ciphertext_vector);
        (
            maker.transform_plaintext_vector_to_raw_vec(proto_plaintext_vector),
            maker.transform_plaintext_vector_to_raw_vec(&proto_output_plaintext_vector),
        )
    }

    fn compute_criteria(
        _parameters: &Self::Parameters,
        _maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::Criteria {
        (Variance(0.),)
    }

    fn verify(criteria: &Self::Criteria, outputs: &[Self::Outcome]) -> bool {
        let (means, actual): (Vec<_>, Vec<_>) = outputs.iter().cloned().unzip();
        let means: Vec<Precision::Raw> = means.into_iter().flatten().collect();
        let actual: Vec<Precision::Raw> = actual.into_iter().flatten().collect();
        assert_noise_distribution(&actual, means.as_slice(), criteria.0)
    }
}
