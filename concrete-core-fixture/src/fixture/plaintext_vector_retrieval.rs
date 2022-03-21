use crate::fixture::Fixture;
use crate::generation::prototyping::PrototypesPlaintextVector;
use crate::generation::synthesizing::SynthesizesPlaintextVector;
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::PlaintextCount;

use concrete_core::prelude::{PlaintextVectorEntity, PlaintextVectorRetrievalEngine};

/// A fixture for the types implementing the `PlaintextVectorRetrievalEngine` trait.
pub struct PlaintextVectorRetrievalFixture;

#[derive(Debug)]
pub struct PlaintextVectorRetrievalParameters {
    count: PlaintextCount,
}

impl<Precision, Engine, PlaintextVector> Fixture<Precision, Engine, (PlaintextVector,)>
    for PlaintextVectorRetrievalFixture
where
    Precision: IntegerPrecision,
    Engine: PlaintextVectorRetrievalEngine<PlaintextVector, Precision::Raw>,
    PlaintextVector: PlaintextVectorEntity,
    Maker: SynthesizesPlaintextVector<Precision, PlaintextVector>,
{
    type Parameters = PlaintextVectorRetrievalParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes =
        (<Maker as PrototypesPlaintextVector<Precision>>::PlaintextVectorProto,);
    type PreExecutionContext = (PlaintextVector,);
    type PostExecutionContext = (PlaintextVector, Vec<Precision::Raw>);
    type Criteria = (Variance,);
    type Outcome = (Vec<Precision::Raw>, Vec<Precision::Raw>);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                PlaintextVectorRetrievalParameters {
                    count: PlaintextCount(100),
                },
                PlaintextVectorRetrievalParameters {
                    count: PlaintextCount(1),
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
            maker.transform_raw_vec_to_plaintext_vector(&raw_plaintext_vector);
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
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (plaintext_vector,) = context;
        let raw_output_vector =
            unsafe { engine.retrieve_plaintext_vector_unchecked(&plaintext_vector) };
        (plaintext_vector, raw_output_vector)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        _sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (plaintext_vector, raw_output_vector) = context;
        let proto_output_plaintext = maker.unsynthesize_plaintext_vector(&plaintext_vector);
        maker.destroy_plaintext_vector(plaintext_vector);
        (
            maker.transform_plaintext_vector_to_raw_vec(&proto_output_plaintext),
            raw_output_vector,
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
