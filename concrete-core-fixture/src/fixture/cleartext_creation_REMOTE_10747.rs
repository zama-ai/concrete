use crate::fixture::Fixture;
use crate::generation::prototyping::PrototypesCleartext;
use crate::generation::synthesizing::SynthesizesCleartext;
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;

use concrete_core::prelude::{CleartextCreationEngine, CleartextEntity};

/// A fixture for the types implementing the `CleartextCreationEngine` trait.
pub struct CleartextCreationFixture;

#[derive(Debug)]
pub struct CleartextCreationParameters;

impl<Precision, Engine, Cleartext> Fixture<Precision, Engine, (Cleartext,)>
    for CleartextCreationFixture
where
    Precision: IntegerPrecision,
    Engine: CleartextCreationEngine<Precision::Raw, Cleartext>,
    Cleartext: CleartextEntity,
    Maker: SynthesizesCleartext<Precision, Cleartext>,
{
    type Parameters = CleartextCreationParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes = (Precision::Raw,);
    type PreExecutionContext = (Precision::Raw,);
    type PostExecutionContext = (Cleartext,);
    type Criteria = (Variance,);
    type Outcome = (Precision::Raw, Precision::Raw);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(vec![CleartextCreationParameters].into_iter())
    }

    fn generate_random_repetition_prototypes(
        _parameters: &Self::Parameters,
        _maker: &mut Maker,
    ) -> Self::RepetitionPrototypes {
    }

    fn generate_random_sample_prototypes(
        _parameters: &Self::Parameters,
        _maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        (Precision::Raw::uniform(),)
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
        let (raw_cleartext,) = context;
        let cleartext = unsafe { engine.create_cleartext_unchecked(&raw_cleartext) };
        (cleartext,)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (cleartext,) = context;
        let proto_output_cleartext = maker.unsynthesize_cleartext(&cleartext);
        maker.destroy_cleartext(cleartext);
        (
            sample_proto.0,
            maker.transform_cleartext_to_raw(&proto_output_cleartext),
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
