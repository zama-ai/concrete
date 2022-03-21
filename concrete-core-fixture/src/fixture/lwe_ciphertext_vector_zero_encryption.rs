use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::LweDimension;
use concrete_core::prelude::{
    LweCiphertextCount, LweCiphertextVectorEntity, LweCiphertextVectorZeroEncryptionEngine,
    LweSecretKeyEntity,
};

use crate::fixture::Fixture;
use crate::generation::prototyping::{
    PrototypesLweCiphertextVector, PrototypesLweSecretKey, PrototypesPlaintextVector,
};
use crate::generation::synthesizing::{SynthesizesLweCiphertextVector, SynthesizesLweSecretKey};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;

/// A fixture for the types implementing the `LweCiphertextVectorZeroEncryptionEngine` trait.
pub struct LweCiphertextVectorZeroEncryptionFixture;

#[derive(Debug)]
pub struct LweCiphertextVectorZeroEncryptionParameters {
    pub noise: Variance,
    pub lwe_dimension: LweDimension,
    pub lwe_ciphertext_count: LweCiphertextCount,
}

impl<Precision, Engine, SecretKey, CiphertextVector>
    Fixture<Precision, Engine, (SecretKey, CiphertextVector)>
    for LweCiphertextVectorZeroEncryptionFixture
where
    Precision: IntegerPrecision,
    Engine: LweCiphertextVectorZeroEncryptionEngine<SecretKey, CiphertextVector>,
    SecretKey: LweSecretKeyEntity,
    CiphertextVector: LweCiphertextVectorEntity<KeyDistribution = SecretKey::KeyDistribution>,
    Maker: SynthesizesLweSecretKey<Precision, SecretKey>
        + SynthesizesLweCiphertextVector<Precision, CiphertextVector>,
{
    type Parameters = LweCiphertextVectorZeroEncryptionParameters;
    type RepetitionPrototypes = (<Maker as PrototypesLweSecretKey<Precision, CiphertextVector::KeyDistribution>>::LweSecretKeyProto, );
    type SamplePrototypes = ();
    type PreExecutionContext = (SecretKey,);
    type PostExecutionContext = (SecretKey, CiphertextVector);
    type Criteria = (Variance,);
    type Outcome = (Vec<Precision::Raw>, Vec<Precision::Raw>);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                LweCiphertextVectorZeroEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(100),
                    lwe_ciphertext_count: LweCiphertextCount(1),
                },
                LweCiphertextVectorZeroEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(100),
                    lwe_ciphertext_count: LweCiphertextCount(100),
                },
                LweCiphertextVectorZeroEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(300),
                    lwe_ciphertext_count: LweCiphertextCount(100),
                },
                LweCiphertextVectorZeroEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(600),
                    lwe_ciphertext_count: LweCiphertextCount(100),
                },
                LweCiphertextVectorZeroEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(1000),
                    lwe_ciphertext_count: LweCiphertextCount(100),
                },
                LweCiphertextVectorZeroEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(3000),
                    lwe_ciphertext_count: LweCiphertextCount(100),
                },
                LweCiphertextVectorZeroEncryptionParameters {
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
        (maker.synthesize_lwe_secret_key(proto_secret_key),)
    }

    fn execute_engine(
        parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (secret_key,) = context;
        let ciphertext_vector = unsafe {
            engine.zero_encrypt_lwe_ciphertext_vector_unchecked(
                &secret_key,
                parameters.noise,
                parameters.lwe_ciphertext_count,
            )
        };
        (secret_key, ciphertext_vector)
    }

    fn process_context(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        _sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (secret_key, ciphertext_vector) = context;
        let (proto_secret_key,) = repetition_proto;
        let proto_output_ciphertext_vector =
            maker.unsynthesize_lwe_ciphertext_vector(&ciphertext_vector);
        maker.destroy_lwe_ciphertext_vector(ciphertext_vector);
        maker.destroy_lwe_secret_key(secret_key);
        let proto_plaintext = maker.decrypt_lwe_ciphertext_vector_to_plaintext_vector(
            proto_secret_key,
            &proto_output_ciphertext_vector,
        );
        (
            Precision::Raw::zero_vec(parameters.lwe_ciphertext_count.0),
            maker.transform_plaintext_vector_to_raw_vec(&proto_plaintext),
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
