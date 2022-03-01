use crate::fixture::Fixture;
use crate::generation::prototyping::PrototypesCleartextVector;
use crate::generation::synthesizing::SynthesizesCleartextVector;
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::CleartextCount;

use concrete_core::prelude::{CleartextVectorDiscardingRetrievalEngine, CleartextVectorEntity};

/// A fixture for the types implementing the `CleartextVectorDiscardingRetrievalEngine` trait.
pub struct CleartextVectorDiscardingRetrievalFixture;

#[derive(Debug)]
pub struct CleartextVectorDiscardingRetrievalParameters {
    count: CleartextCount,
}

impl<Precision, Engine, CleartextVector> Fixture<Precision, Engine, (CleartextVector,)>
    for CleartextVectorDiscardingRetrievalFixture
where
    Precision: IntegerPrecision,
    Engine: CleartextVectorDiscardingRetrievalEngine<CleartextVector, Precision::Raw>,
    CleartextVector: CleartextVectorEntity,
    Maker: SynthesizesCleartextVector<Precision, CleartextVector>,
{
    type Parameters = CleartextVectorDiscardingRetrievalParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes = (
        <Maker as PrototypesCleartextVector<Precision>>::CleartextVectorProto,
        Vec<Precision::Raw>,
    );
    type PreExecutionContext = (CleartextVector, Vec<Precision::Raw>);
    type PostExecutionContext = (CleartextVector, Vec<Precision::Raw>);
    type Criteria = (Variance,);
    type Outcome = (Vec<Precision::Raw>, Vec<Precision::Raw>);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![CleartextVectorDiscardingRetrievalParameters {
                count: CleartextCount(100),
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
        let raw_cleartext_vector = Precision::Raw::uniform_vec(parameters.count.0);
        let proto_cleartext_vector =
            maker.transform_raw_vec_to_cleartext_vector(&raw_cleartext_vector);
        let raw_output_vector = Precision::Raw::zero_vec(parameters.count.0);
        (proto_cleartext_vector, raw_output_vector)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_cleartext_vector, raw_output) = sample_proto;
        (
            maker.synthesize_cleartext_vector(proto_cleartext_vector),
            raw_output.to_owned(),
        )
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (cleartext, mut raw_output) = context;
        unsafe { engine.discard_retrieve_cleartext_vector_unchecked(&mut raw_output, &cleartext) };
        (cleartext, raw_output)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        _sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (cleartext_vector, raw_output_vector) = context;
        let proto_output_cleartext_vector = maker.unsynthesize_cleartext_vector(&cleartext_vector);
        maker.destroy_cleartext_vector(cleartext_vector);
        (
            maker.transform_cleartext_vector_to_raw_vec(&proto_output_cleartext_vector),
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
