use crate::fixture::Fixture;
use crate::generation::prototyping::{PrototypesLweCiphertextVector, PrototypesPlaintextVector};
use crate::generation::synthesizing::{SynthesizesLweCiphertextVector, SynthesizesPlaintextVector};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{LweCiphertextCount, LweDimension};
use concrete_core::prelude::{
    LweCiphertextVectorEntity, LweCiphertextVectorTrivialEncryptionEngine, PlaintextVectorEntity,
};

/// A fixture for the types implementing the `LweCiphertextVectorTrivialEncryptionEngine` trait.
pub struct LweCiphertextVectorTrivialEncryptionFixture;

#[derive(Debug)]
pub struct LweCiphertextVectorTrivialEncryptionParameters {
    pub lwe_dimension: LweDimension,
    pub count: LweCiphertextCount,
}

impl<Precision, Engine, PlaintextVector, CiphertextVector>
    Fixture<Precision, Engine, (PlaintextVector, CiphertextVector)>
    for LweCiphertextVectorTrivialEncryptionFixture
where
    Precision: IntegerPrecision,
    Engine: LweCiphertextVectorTrivialEncryptionEngine<PlaintextVector, CiphertextVector>,
    PlaintextVector: PlaintextVectorEntity,
    CiphertextVector: LweCiphertextVectorEntity,
    Maker: SynthesizesPlaintextVector<Precision, PlaintextVector>
        + SynthesizesLweCiphertextVector<Precision, CiphertextVector>,
{
    type Parameters = LweCiphertextVectorTrivialEncryptionParameters;
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
                LweCiphertextVectorTrivialEncryptionParameters {
                    lwe_dimension: LweDimension(200),
                    count: LweCiphertextCount(100),
                },
                LweCiphertextVectorTrivialEncryptionParameters {
                    lwe_dimension: LweDimension(1),
                    count: LweCiphertextCount(1),
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
        let raw_plaintext_vector = Precision::Raw::uniform_vec(parameters.count.0);
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
            engine.trivially_encrypt_lwe_ciphertext_vector_unchecked(
                parameters.lwe_dimension.to_lwe_size(),
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
            maker.unsynthesize_lwe_ciphertext_vector(&ciphertext_vector);
        let proto_output_plaintext_vector = maker
            .trivially_decrypt_lwe_ciphertext_vector_to_plaintext_vector(
                &proto_output_ciphertext_vector,
            );
        maker.destroy_plaintext_vector(plaintext_vector);
        maker.destroy_lwe_ciphertext_vector(ciphertext_vector);
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
