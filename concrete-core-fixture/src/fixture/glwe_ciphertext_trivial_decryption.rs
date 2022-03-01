use crate::fixture::Fixture;
use crate::generation::prototyping::{PrototypesGlweCiphertext, PrototypesPlaintextVector};
use crate::generation::synthesizing::{SynthesizesGlweCiphertext, SynthesizesPlaintextVector};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{GlweDimension, PolynomialSize};
use concrete_core::prelude::{
    GlweCiphertextEntity, GlweCiphertextTrivialDecryptionEngine, PlaintextVectorEntity,
};

/// A fixture for the types implementing the `GlweCiphertextTrivialDecryptionEngine` trait.
pub struct GlweCiphertextTrivialDecryptionFixture;

#[derive(Debug)]
pub struct GlweCiphertextTrivialDecryptionParameters {
    pub glwe_dimension: GlweDimension,
    pub polynomial_size: PolynomialSize,
}

impl<Precision, Engine, PlaintextVector, Ciphertext>
    Fixture<Precision, Engine, (PlaintextVector, Ciphertext)>
    for GlweCiphertextTrivialDecryptionFixture
where
    Precision: IntegerPrecision,
    Engine: GlweCiphertextTrivialDecryptionEngine<Ciphertext, PlaintextVector>,
    PlaintextVector: PlaintextVectorEntity,
    Ciphertext: GlweCiphertextEntity,
    Maker: SynthesizesPlaintextVector<Precision, PlaintextVector>
        + SynthesizesGlweCiphertext<Precision, Ciphertext>,
{
    type Parameters = GlweCiphertextTrivialDecryptionParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes = (
        <Maker as PrototypesPlaintextVector<Precision>>::PlaintextVectorProto,
        <Maker as PrototypesGlweCiphertext<Precision, Ciphertext::KeyDistribution>>::GlweCiphertextProto,
    );
    type PreExecutionContext = (Ciphertext,);
    type PostExecutionContext = (Ciphertext, PlaintextVector);
    type Criteria = (Variance,);
    type Outcome = (Vec<Precision::Raw>, Vec<Precision::Raw>);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![GlweCiphertextTrivialDecryptionParameters {
                glwe_dimension: GlweDimension(200),
                polynomial_size: PolynomialSize(200),
            }]
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
        let raw_plaintext_vector = Precision::Raw::uniform_vec(parameters.polynomial_size.0);
        let proto_plaintext_vector =
            maker.transform_raw_vec_to_plaintext_vector(raw_plaintext_vector.as_slice());
        let proto_ciphertext = maker.trivial_encrypt_plaintext_vector_to_glwe_ciphertext(
            parameters.glwe_dimension,
            &proto_plaintext_vector,
        );
        (proto_plaintext_vector, proto_ciphertext)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (_, proto_ciphertext_vector) = sample_proto;
        let ciphertext = maker.synthesize_glwe_ciphertext(proto_ciphertext_vector);
        (ciphertext,)
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (ciphertext,) = context;
        let plaintext_vector =
            unsafe { engine.trivially_decrypt_glwe_ciphertext_unchecked(&ciphertext) };
        (ciphertext, plaintext_vector)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (proto_plaintext_vector, _) = sample_proto;
        let (ciphertext, plaintext_vector) = context;
        let proto_output_plaintext_vector = maker.unsynthesize_plaintext_vector(&plaintext_vector);
        maker.destroy_glwe_ciphertext(ciphertext);
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
