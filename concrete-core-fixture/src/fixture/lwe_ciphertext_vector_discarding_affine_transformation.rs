use crate::fixture::Fixture;
use crate::generation::prototyping::{
    PrototypesCleartextVector, PrototypesLweCiphertext, PrototypesLweCiphertextVector,
    PrototypesLweSecretKey, PrototypesPlaintext, PrototypesPlaintextVector,
};
use crate::generation::synthesizing::{
    SynthesizesCleartextVector, SynthesizesLweCiphertext, SynthesizesLweCiphertextVector,
    SynthesizesPlaintext,
};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::{DispersionParameter, LogStandardDev, Variance};
use concrete_commons::numeric::UnsignedInteger;
use concrete_commons::parameters::{LweCiphertextCount, LweDimension};
use concrete_core::prelude::{
    CleartextVectorEntity, LweCiphertextEntity,
    LweCiphertextVectorDiscardingAffineTransformationEngine, LweCiphertextVectorEntity,
    PlaintextEntity,
};

/// A fixture for the types implementing the
/// `LweCiphertextVectorDiscardingAffineTransformationEngine` trait.
pub struct LweCiphertextVectorDiscardingAffineTransformationFixture;

#[derive(Debug)]
pub struct LweCiphertextVectorDiscardingAffineTransformationParameters {
    pub nb_ct: LweCiphertextCount,
    pub noise: Variance,
    pub lwe_dimension: LweDimension,
}

#[allow(clippy::type_complexity)]
impl<Precision, Engine, CiphertextVector, CleartextVector, Plaintext, OutputCiphertext>
    Fixture<
        Precision,
        Engine,
        (
            CiphertextVector,
            CleartextVector,
            Plaintext,
            OutputCiphertext,
        ),
    > for LweCiphertextVectorDiscardingAffineTransformationFixture
where
    Precision: IntegerPrecision,
    Engine: LweCiphertextVectorDiscardingAffineTransformationEngine<
        CiphertextVector,
        CleartextVector,
        Plaintext,
        OutputCiphertext,
    >,
    CiphertextVector: LweCiphertextVectorEntity,
    CleartextVector: CleartextVectorEntity,
    Plaintext: PlaintextEntity,
    OutputCiphertext: LweCiphertextEntity<KeyDistribution = CiphertextVector::KeyDistribution>,
    Maker: SynthesizesLweCiphertextVector<Precision, CiphertextVector>
        + SynthesizesCleartextVector<Precision, CleartextVector>
        + SynthesizesPlaintext<Precision, Plaintext>
        + SynthesizesLweCiphertext<Precision, OutputCiphertext>,
{
    type Parameters = LweCiphertextVectorDiscardingAffineTransformationParameters;
    type RepetitionPrototypes = (
        <Maker as PrototypesLweSecretKey<Precision, CiphertextVector::KeyDistribution>>::LweSecretKeyProto,
        <Maker as PrototypesCleartextVector<Precision>>::CleartextVectorProto,
        <Maker as PrototypesPlaintext<Precision>>::PlaintextProto
    );
    type SamplePrototypes = (
        <Maker as PrototypesLweCiphertext<Precision, CiphertextVector::KeyDistribution>>::LweCiphertextProto,
        <Maker as PrototypesPlaintextVector<Precision>>::PlaintextVectorProto,
        <Maker as PrototypesLweCiphertextVector<Precision, CiphertextVector::KeyDistribution>>::LweCiphertextVectorProto,
    );
    type PreExecutionContext = (
        OutputCiphertext,
        CiphertextVector,
        CleartextVector,
        Plaintext,
    );
    type PostExecutionContext = (
        OutputCiphertext,
        CiphertextVector,
        CleartextVector,
        Plaintext,
    );
    type Criteria = (Variance,);
    type Outcome = (Precision::Raw, Precision::Raw);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                LweCiphertextVectorDiscardingAffineTransformationParameters {
                    nb_ct: LweCiphertextCount(100),
                    noise: Variance(LogStandardDev::from_log_standard_dev(-25.).get_variance()),
                    lwe_dimension: LweDimension(1000),
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
        let raw_cleartext_vector =
            Precision::Raw::uniform_zero_centered_vec(512, parameters.nb_ct.0);
        let raw_plaintext = Precision::Raw::uniform_between(0..1024usize);
        let proto_cleartext_vector =
            maker.transform_raw_vec_to_cleartext_vector(raw_cleartext_vector.as_slice());
        let proto_plaintext = maker.transform_raw_to_plaintext(&raw_plaintext);
        (proto_secret_key, proto_cleartext_vector, proto_plaintext)
    }

    fn generate_random_sample_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        let (proto_secret_key, ..) = repetition_proto;
        let raw_plaintext_vector = Precision::Raw::uniform_vec(parameters.nb_ct.0);
        let proto_plaintext_vector =
            maker.transform_raw_vec_to_plaintext_vector(raw_plaintext_vector.as_slice());
        let proto_ciphertext_vector = maker.encrypt_plaintext_vector_to_lwe_ciphertext_vector(
            proto_secret_key,
            &proto_plaintext_vector,
            parameters.noise,
        );
        let proto_output_ciphertext =
            maker.trivial_encrypt_zero_to_lwe_ciphertext(parameters.lwe_dimension);
        (
            proto_output_ciphertext,
            proto_plaintext_vector,
            proto_ciphertext_vector,
        )
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (_, proto_cleartext_vector, proto_plaintext) = repetition_proto;
        let (proto_output_ciphertext, _, proto_ciphertext_vector) = sample_proto;
        (
            maker.synthesize_lwe_ciphertext(proto_output_ciphertext),
            maker.synthesize_lwe_ciphertext_vector(proto_ciphertext_vector),
            maker.synthesize_cleartext_vector(proto_cleartext_vector),
            maker.synthesize_plaintext(proto_plaintext),
        )
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (mut output_ciphertext, ciphertext_vector, weights, bias) = context;
        unsafe {
            engine.discard_affine_transform_lwe_ciphertext_vector_unchecked(
                &mut output_ciphertext,
                &ciphertext_vector,
                &weights,
                &bias,
            )
        };
        (output_ciphertext, ciphertext_vector, weights, bias)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (output_ciphertext, ciphertext_vector, weights, bias) = context;
        let (proto_secret_key, prototype_cleartext_vector, prototype_plaintext) = repetition_proto;
        let (_, proto_plaintext_vector, _) = sample_proto;
        let proto_output_ciphertext = maker.unsynthesize_lwe_ciphertext(&output_ciphertext);
        let proto_output_plaintext =
            maker.decrypt_lwe_ciphertext_to_plaintext(proto_secret_key, &proto_output_ciphertext);
        let raw_plaintext_vector =
            maker.transform_plaintext_vector_to_raw_vec(proto_plaintext_vector);
        let raw_cleartext_vector =
            maker.transform_cleartext_vector_to_raw_vec(prototype_cleartext_vector);
        let raw_bias = maker.transform_plaintext_to_raw(prototype_plaintext);
        let predicted_output = raw_plaintext_vector
            .iter()
            .zip(raw_cleartext_vector.iter())
            .fold(raw_bias, |a, (c, w)| a.wrapping_add(c.wrapping_mul(*w)));
        maker.destroy_lwe_ciphertext(output_ciphertext);
        maker.destroy_lwe_ciphertext_vector(ciphertext_vector);
        maker.destroy_cleartext_vector(weights);
        maker.destroy_plaintext(bias);
        (
            predicted_output,
            maker.transform_plaintext_to_raw(&proto_output_plaintext),
        )
    }

    fn compute_criteria(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::Criteria {
        let (_, proto_cleartext_vector, _) = repetition_proto;
        let raw_weight_vector = maker.transform_cleartext_vector_to_raw_vec(proto_cleartext_vector);
        let predicted_variance: Variance =
            concrete_npe::estimate_weighted_sum_noise::<Precision::Raw, _>(
                &vec![parameters.noise; parameters.nb_ct.0],
                &raw_weight_vector,
            );
        (predicted_variance,)
    }

    fn verify(criteria: &Self::Criteria, outputs: &[Self::Outcome]) -> bool {
        let (means, actual): (Vec<_>, Vec<_>) = outputs.iter().cloned().unzip();
        assert_noise_distribution(&actual, means.as_slice(), criteria.0)
    }
}
