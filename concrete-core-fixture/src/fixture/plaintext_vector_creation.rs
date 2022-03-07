use crate::fixture::Fixture;
use crate::generation::prototyping::PrototypesPlaintextVector;
use crate::generation::synthesizing::SynthesizesPlaintextVector;
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::PlaintextCount;

use concrete_core::prelude::{PlaintextVectorCreationEngine, PlaintextVectorEntity};

/// A fixture for the types implementing the `PlaintextVectorCreationEngine` trait.
pub struct PlaintextVectorCreationFixture;

#[derive(Debug)]
pub struct PlaintextVectorCreationParameters {
    count: PlaintextCount,
}

impl<Precision, Engine, PlaintextVector> Fixture<Precision, Engine, (PlaintextVector,)>
    for PlaintextVectorCreationFixture
where
    Precision: IntegerPrecision,
    Engine: PlaintextVectorCreationEngine<Precision::Raw, PlaintextVector>,
    PlaintextVector: PlaintextVectorEntity,
    Maker: SynthesizesPlaintextVector<Precision, PlaintextVector>,
{
    type Parameters = PlaintextVectorCreationParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes = (Vec<Precision::Raw>,);
    type PreExecutionContext = (Vec<Precision::Raw>,);
    type PostExecutionContext = (PlaintextVector,);
    type Criteria = (Variance,);
    type Outcome = (Vec<Precision::Raw>, Vec<Precision::Raw>);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                PlaintextVectorCreationParameters {
                    count: PlaintextCount(1),
                },
                PlaintextVectorCreationParameters {
                    count: PlaintextCount(500),
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
        _maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        (Precision::Raw::uniform_vec(parameters.count.0),)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        _maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        sample_proto.to_owned()
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (raw_plaintext_vector,) = context;
        let plaintext_vector =
            unsafe { engine.create_plaintext_vector_unchecked(&raw_plaintext_vector) };
        (plaintext_vector,)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (plaintext_vector,) = context;
        let proto_output_plaintext = maker.unsynthesize_plaintext_vector(&plaintext_vector);
        maker.destroy_plaintext_vector(plaintext_vector);
        (
            sample_proto.0.to_owned(),
            maker.transform_plaintext_vector_to_raw_vec(&proto_output_plaintext),
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
