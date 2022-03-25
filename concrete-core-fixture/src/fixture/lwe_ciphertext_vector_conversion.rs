use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{LweCiphertextCount, LweDimension};
use concrete_core::prelude::{LweCiphertextVectorConversionEngine, LweCiphertextVectorEntity};

use crate::fixture::Fixture;
use crate::generation::prototyping::{
    PrototypesLweCiphertextVector, PrototypesLweSecretKey, PrototypesPlaintextVector,
};
use crate::generation::synthesizing::SynthesizesLweCiphertextVector;
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;

/// A fixture for the types implementing the `LweCiphertextVectorConversionEngine` trait.
pub struct LweCiphertextVectorConversionFixture;

#[derive(Debug)]
pub struct LweCiphertextVectorConversionParameters {
    pub noise: Variance,
    pub lwe_dimension: LweDimension,
    pub lwe_ciphertext_count: LweCiphertextCount,
}

impl<Precision, Engine, CudaCiphertextVector, CiphertextVector>
    Fixture<Precision, Engine, (CudaCiphertextVector, CiphertextVector)>
    for LweCiphertextVectorConversionFixture
where
    Precision: IntegerPrecision,
    Engine: LweCiphertextVectorConversionEngine<CudaCiphertextVector, CiphertextVector>,
    CudaCiphertextVector: LweCiphertextVectorEntity,
    CiphertextVector:
        LweCiphertextVectorEntity<KeyDistribution = CudaCiphertextVector::KeyDistribution>,
    Maker: SynthesizesLweCiphertextVector<Precision, CudaCiphertextVector>,
{
    type Parameters = LweCiphertextVectorConversionParameters;
    type RepetitionPrototypes = (
        <Maker as PrototypesLweSecretKey<Precision, CiphertextVector::KeyDistribution>>::LweSecretKeyProto,
    );
    type SamplePrototypes =
        (
            <Maker as PrototypesLweCiphertextVector<
                Precision,
                CiphertextVector::KeyDistribution,
            >>::LweCiphertextVectorProto,
        );
    type PreExecutionContext = (CudaCiphertextVector,);
    type PostExecutionContext = (CudaCiphertextVector,);
    type Criteria = (Variance,);
    type Outcome = (Vec<Precision::Raw>, Vec<Precision::Raw>);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                LweCiphertextVectorConversionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(100),
                    lwe_ciphertext_count: LweCiphertextCount(1),
                },
                LweCiphertextVectorConversionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(100),
                    lwe_ciphertext_count: LweCiphertextCount(100),
                },
                LweCiphertextVectorConversionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(300),
                    lwe_ciphertext_count: LweCiphertextCount(100),
                },
                LweCiphertextVectorConversionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(600),
                    lwe_ciphertext_count: LweCiphertextCount(100),
                },
                LweCiphertextVectorConversionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(1000),
                    lwe_ciphertext_count: LweCiphertextCount(100),
                },
                LweCiphertextVectorConversionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(3000),
                    lwe_ciphertext_count: LweCiphertextCount(100),
                },
                LweCiphertextVectorConversionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(6000),
                    lwe_ciphertext_count: LweCiphertextCount(100),
                },
            ]
            .into_iter(),
        )
    }

    fn generate_random_repetition_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
    ) -> Self::RepetitionPrototypes {
        let proto_secret_key = maker.new_lwe_secret_key(parameters.lwe_dimension);
        (proto_secret_key,)
    }

    fn generate_random_sample_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        let (key,) = repetition_proto;
        let raw_plaintext_vector = Precision::Raw::uniform_vec(parameters.lwe_ciphertext_count.0);
        let proto_plaintext_vector =
            maker.transform_raw_vec_to_plaintext_vector(raw_plaintext_vector.as_slice());
        let proto_ciphertext_vector = maker.encrypt_plaintext_vector_to_lwe_ciphertext_vector(
            key,
            &proto_plaintext_vector,
            parameters.noise,
        );
        (proto_ciphertext_vector,)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_ciphertext_vector,) = sample_proto;
        (maker.synthesize_lwe_ciphertext_vector(proto_ciphertext_vector),)
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        _engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (ciphertext_vector,) = context;
        (ciphertext_vector,)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (key,) = repetition_proto;
        let (proto_ciphertext_vector,) = sample_proto;
        let (output_ciphertext_vector,) = context;
        let proto_output_ciphertext_vector =
            maker.unsynthesize_lwe_ciphertext_vector(&output_ciphertext_vector);
        let proto_plaintext_vector =
            maker.decrypt_lwe_ciphertext_vector_to_plaintext_vector(key, proto_ciphertext_vector);
        let proto_output_plaintext_vector = <Maker as PrototypesLweCiphertextVector<
            Precision,
            CiphertextVector::KeyDistribution,
        >>::decrypt_lwe_ciphertext_vector_to_plaintext_vector(
            maker,
            key,
            &proto_output_ciphertext_vector,
        );
        maker.destroy_lwe_ciphertext_vector(output_ciphertext_vector);
        (
            maker.transform_plaintext_vector_to_raw_vec(&proto_plaintext_vector),
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
