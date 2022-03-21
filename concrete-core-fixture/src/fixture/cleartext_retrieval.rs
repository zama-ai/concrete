use crate::fixture::Fixture;
use crate::generation::prototyping::PrototypesCleartext;
use crate::generation::synthesizing::SynthesizesCleartext;
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;

use concrete_core::prelude::{CleartextEntity, CleartextRetrievalEngine};

/// A fixture for the types implementing the `CleartextRetrievalEngine` trait.
pub struct CleartextRetrievalFixture;

#[derive(Debug)]
pub struct CleartextRetrievalParameters;

impl<Precision, Engine, Cleartext> Fixture<Precision, Engine, (Cleartext,)>
    for CleartextRetrievalFixture
where
    Precision: IntegerPrecision,
    Engine: CleartextRetrievalEngine<Cleartext, Precision::Raw>,
    Cleartext: CleartextEntity,
    Maker: SynthesizesCleartext<Precision, Cleartext>,
{
    type Parameters = CleartextRetrievalParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes = (<Maker as PrototypesCleartext<Precision>>::CleartextProto,);
    type PreExecutionContext = (Cleartext,);
    type PostExecutionContext = (Cleartext, Precision::Raw);
    type Criteria = (Variance,);
    type Outcome = (Precision::Raw, Precision::Raw);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(vec![CleartextRetrievalParameters].into_iter())
    }

    fn generate_random_repetition_prototypes(
        _parameters: &Self::Parameters,
        _maker: &mut Maker,
    ) -> Self::RepetitionPrototypes {
    }

    fn generate_random_sample_prototypes(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        let raw_cleartext = Precision::Raw::uniform();
        let proto_cleartext = maker.transform_raw_to_cleartext(&raw_cleartext);
        (proto_cleartext,)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_cleartext,) = sample_proto;
        (maker.synthesize_cleartext(proto_cleartext),)
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (cleartext,) = context;
        let raw_output = unsafe { engine.retrieve_cleartext_unchecked(&cleartext) };
        (cleartext, raw_output)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        _sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (cleartext, raw_output) = context;
        let proto_output_cleartext = maker.unsynthesize_cleartext(&cleartext);
        maker.destroy_cleartext(cleartext);
        (
            maker.transform_cleartext_to_raw(&proto_output_cleartext),
            raw_output,
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
        assert_noise_distribution(actual.as_slice(), means.as_slice(), criteria.0)
    }
}
