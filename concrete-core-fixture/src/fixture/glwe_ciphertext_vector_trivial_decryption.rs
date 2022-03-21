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
    GlweCiphertextVectorEntity, GlweCiphertextVectorTrivialDecryptionEngine, PlaintextVectorEntity,
};

/// A fixture for the types implementing the `GlweCiphertextVectorTrivialDecryptionEngine` trait.
pub struct GlweCiphertextVectorTrivialDecryptionFixture;

#[derive(Debug)]
pub struct GlweCiphertextVectorTrivialDecryptionParameters {
    pub glwe_dimension: GlweDimension,
    pub polynomial_size: PolynomialSize,
    pub count: GlweCiphertextCount,
}

impl<Precision, Engine, PlaintextVector, CiphertextVector>
    Fixture<Precision, Engine, (PlaintextVector, CiphertextVector)>
    for GlweCiphertextVectorTrivialDecryptionFixture
where
    Precision: IntegerPrecision,
    Engine: GlweCiphertextVectorTrivialDecryptionEngine<CiphertextVector, PlaintextVector>,
    PlaintextVector: PlaintextVectorEntity,
    CiphertextVector: GlweCiphertextVectorEntity,
    Maker: SynthesizesPlaintextVector<Precision, PlaintextVector>
        + SynthesizesGlweCiphertextVector<Precision, CiphertextVector>,
{
    type Parameters = GlweCiphertextVectorTrivialDecryptionParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes =
        (
            <Maker as PrototypesPlaintextVector<Precision>>::PlaintextVectorProto,
            <Maker as PrototypesGlweCiphertextVector<
                Precision,
                CiphertextVector::KeyDistribution,
            >>::GlweCiphertextVectorProto,
        );
    type PreExecutionContext = (CiphertextVector,);
    type PostExecutionContext = (CiphertextVector, PlaintextVector);
    type Criteria = (Variance,);
    type Outcome = (Vec<Precision::Raw>, Vec<Precision::Raw>);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                GlweCiphertextVectorTrivialDecryptionParameters {
                    glwe_dimension: GlweDimension(200),
                    polynomial_size: PolynomialSize(256),
                    count: GlweCiphertextCount(100),
                },
                GlweCiphertextVectorTrivialDecryptionParameters {
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
        let proto_ciphertext_vector = maker
            .trivially_encrypt_plaintext_vector_to_glwe_ciphertext_vector(
                parameters.glwe_dimension.to_glwe_size(),
                parameters.count,
                &proto_plaintext_vector,
            );
        (proto_plaintext_vector, proto_ciphertext_vector)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (_, proto_ciphertext_vector) = sample_proto;
        let ciphertext_vector = maker.synthesize_glwe_ciphertext_vector(proto_ciphertext_vector);
        (ciphertext_vector,)
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (ciphertext_vector,) = context;
        let plaintext_vector = unsafe {
            engine.trivially_decrypt_glwe_ciphertext_vector_unchecked(&ciphertext_vector)
        };
        (ciphertext_vector, plaintext_vector)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (proto_plaintext_vector, _) = sample_proto;
        let (ciphertext_vector, plaintext_vector) = context;
        let proto_output_plaintext_vector = maker.unsynthesize_plaintext_vector(&plaintext_vector);
        maker.destroy_glwe_ciphertext_vector(ciphertext_vector);
        maker.destroy_plaintext_vector(plaintext_vector);
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
        assert_noise_distribution(actual.as_slice(), means.as_slice(), criteria.0)
    }
}
