use crate::fixture::Fixture;
use crate::generation::prototyping::PrototypesPlaintext;
use crate::generation::synthesizing::SynthesizesPlaintext;
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;

use concrete_core::prelude::{PlaintextCreationEngine, PlaintextEntity};

/// A fixture for the types implementing the `PlaintextCreationEngine` trait.
pub struct PlaintextCreationFixture;

#[derive(Debug)]
pub struct PlaintextCreationParameters;

impl<Precision, Engine, Plaintext> Fixture<Precision, Engine, (Plaintext,)>
    for PlaintextCreationFixture
where
    Precision: IntegerPrecision,
    Engine: PlaintextCreationEngine<Precision::Raw, Plaintext>,
    Plaintext: PlaintextEntity,
    Maker: SynthesizesPlaintext<Precision, Plaintext>,
{
    type Parameters = PlaintextCreationParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes = (Precision::Raw,);
    type PreExecutionContext = (Precision::Raw,);
    type PostExecutionContext = (Plaintext,);
    type Criteria = (Variance,);
    type Outcome = (Precision::Raw, Precision::Raw);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(vec![PlaintextCreationParameters].into_iter())
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
        let (raw_plaintext,) = context;
        let plaintext = unsafe { engine.create_plaintext_unchecked(&raw_plaintext) };
        (plaintext,)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (plaintext,) = context;
        let proto_output_plaintext = maker.unsynthesize_plaintext(&plaintext);
        maker.destroy_plaintext(plaintext);
        (
            sample_proto.0,
            maker.transform_plaintext_to_raw(&proto_output_plaintext),
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
