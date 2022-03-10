use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::LweDimension;
use concrete_core::prelude::{
    LweCiphertextCount, LweCiphertextVectorDiscardingDecryptionEngine, LweCiphertextVectorEntity,
    LweSecretKeyEntity, PlaintextVectorEntity,
};

use crate::fixture::Fixture;
use crate::generation::prototyping::{
    PrototypesLweCiphertextVector, PrototypesLweSecretKey, PrototypesPlaintextVector,
};
use crate::generation::synthesizing::{
    SynthesizesLweCiphertextVector, SynthesizesLweSecretKey, SynthesizesPlaintextVector,
};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;

/// A fixture for the types implementing the `LweCiphertextVectorDiscardingDecryptionEngine` trait.
pub struct LweCiphertextVectorDiscardingDecryptionFixture;

#[derive(Debug)]
pub struct LweCiphertextVectorDiscardingDecryptionParameters {
    pub noise: Variance,
    pub lwe_dimension: LweDimension,
    pub lwe_ciphertext_count: LweCiphertextCount,
}

#[allow(clippy::type_complexity)]
impl<Precision, Engine, PlaintextVector, SecretKey, CiphertextVector>
    Fixture<Precision, Engine, (PlaintextVector, SecretKey, CiphertextVector)>
    for LweCiphertextVectorDiscardingDecryptionFixture
where
    Precision: IntegerPrecision,
    Engine:
        LweCiphertextVectorDiscardingDecryptionEngine<SecretKey, CiphertextVector, PlaintextVector>,
    PlaintextVector: PlaintextVectorEntity,
    SecretKey: LweSecretKeyEntity,
    CiphertextVector: LweCiphertextVectorEntity<KeyDistribution = SecretKey::KeyDistribution>,
    Maker: SynthesizesPlaintextVector<Precision, PlaintextVector>
        + SynthesizesLweSecretKey<Precision, SecretKey>
        + SynthesizesLweCiphertextVector<Precision, CiphertextVector>,
{
    type Parameters = LweCiphertextVectorDiscardingDecryptionParameters;
    type RepetitionPrototypes = (
        <Maker as PrototypesLweSecretKey<Precision, CiphertextVector::KeyDistribution>>::LweSecretKeyProto,
    );
    type SamplePrototypes =
        (
            <Maker as PrototypesPlaintextVector<Precision>>::PlaintextVectorProto,
            <Maker as PrototypesLweCiphertextVector<
                Precision,
                CiphertextVector::KeyDistribution,
            >>::LweCiphertextVectorProto,
            <Maker as PrototypesPlaintextVector<Precision>>::PlaintextVectorProto,
        );
    type PreExecutionContext = (SecretKey, CiphertextVector, PlaintextVector);
    type PostExecutionContext = (SecretKey, CiphertextVector, PlaintextVector);
    type Criteria = (Variance,);
    type Outcome = (Vec<Precision::Raw>, Vec<Precision::Raw>);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                LweCiphertextVectorDiscardingDecryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(100),
                    lwe_ciphertext_count: LweCiphertextCount(1),
                },
                LweCiphertextVectorDiscardingDecryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(100),
                    lwe_ciphertext_count: LweCiphertextCount(100),
                },
                LweCiphertextVectorDiscardingDecryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(300),
                    lwe_ciphertext_count: LweCiphertextCount(100),
                },
                LweCiphertextVectorDiscardingDecryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(600),
                    lwe_ciphertext_count: LweCiphertextCount(100),
                },
                LweCiphertextVectorDiscardingDecryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(1000),
                    lwe_ciphertext_count: LweCiphertextCount(100),
                },
                LweCiphertextVectorDiscardingDecryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(3000),
                    lwe_ciphertext_count: LweCiphertextCount(100),
                },
                LweCiphertextVectorDiscardingDecryptionParameters {
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
        let (proto_secret_key,) = repetition_proto;
        let raw_plaintext_vector = Precision::Raw::uniform_vec(parameters.lwe_ciphertext_count.0);
        let proto_plaintext_vector =
            maker.transform_raw_vec_to_plaintext_vector(&raw_plaintext_vector);
        let proto_ciphertext_vector = maker.encrypt_plaintext_vector_to_lwe_ciphertext_vector(
            proto_secret_key,
            &proto_plaintext_vector,
            parameters.noise,
        );
        let raw_output_plaintext_vector =
            Precision::Raw::uniform_vec(parameters.lwe_ciphertext_count.0);
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
        let synth_secret_key = maker.synthesize_lwe_secret_key(proto_secret_key);
        let synth_ciphertext_vector =
            maker.synthesize_lwe_ciphertext_vector(proto_ciphertext_vector);
        let synth_output_plaintext_vector =
            maker.synthesize_plaintext_vector(proto_output_plaintext_vector);
        (
            synth_secret_key,
            synth_ciphertext_vector,
            synth_output_plaintext_vector,
        )
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (secret_key, ciphertext_vector, mut output_plaintext_vector) = context;
        unsafe {
            engine.discard_decrypt_lwe_ciphertext_vector_unchecked(
                &secret_key,
                &mut output_plaintext_vector,
                &ciphertext_vector,
            )
        };
        (secret_key, ciphertext_vector, output_plaintext_vector)
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
        maker.destroy_lwe_ciphertext_vector(ciphertext_vector);
        maker.destroy_lwe_secret_key(secret_key);
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
        assert_noise_distribution(&actual, means.as_slice(), criteria.0)
    }
}
