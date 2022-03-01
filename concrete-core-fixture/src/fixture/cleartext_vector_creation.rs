use crate::fixture::Fixture;
use crate::generation::prototyping::PrototypesCleartextVector;
use crate::generation::synthesizing::SynthesizesCleartextVector;
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::CleartextCount;

use concrete_core::prelude::{CleartextVectorCreationEngine, CleartextVectorEntity};

/// A fixture for the types implementing the `CleartextVectorCreationEngine` trait.
pub struct CleartextVectorCreationFixture;

#[derive(Debug)]
pub struct CleartextVectorCreationParameters {
    count: CleartextCount,
}

impl<Precision, Engine, CleartextVector> Fixture<Precision, Engine, (CleartextVector,)>
    for CleartextVectorCreationFixture
where
    Precision: IntegerPrecision,
    Engine: CleartextVectorCreationEngine<Precision::Raw, CleartextVector>,
    CleartextVector: CleartextVectorEntity,
    Maker: SynthesizesCleartextVector<Precision, CleartextVector>,
{
    type Parameters = CleartextVectorCreationParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes = (Vec<Precision::Raw>,);
    type PreExecutionContext = (Vec<Precision::Raw>,);
    type PostExecutionContext = (CleartextVector,);
    type Criteria = (Variance,);
    type Outcome = (Vec<Precision::Raw>, Vec<Precision::Raw>);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![CleartextVectorCreationParameters {
                count: CleartextCount(500),
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
        let (raw_cleartext_vector,) = context;
        let cleartext_vector =
            unsafe { engine.create_cleartext_vector_unchecked(&raw_cleartext_vector) };
        (cleartext_vector,)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (cleartext_vector,) = context;
        let proto_output_cleartext = maker.unsynthesize_cleartext_vector(&cleartext_vector);
        maker.destroy_cleartext_vector(cleartext_vector);
        (
            sample_proto.0.to_owned(),
            maker.transform_cleartext_vector_to_raw_vec(&proto_output_cleartext),
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
