use crate::fixture::Fixture;
use crate::generation::prototyping::PrototypesPlaintext;
use crate::generation::synthesizing::SynthesizesPlaintext;
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;

use concrete_core::prelude::{PlaintextEntity, PlaintextRetrievalEngine};

/// A fixture for the types implementing the `PlaintextRetrievalEngine` trait.
pub struct PlaintextRetrievalFixture;

#[derive(Debug)]
pub struct PlaintextRetrievalParameters;

impl<Precision, Engine, Plaintext> Fixture<Precision, Engine, (Plaintext,)>
    for PlaintextRetrievalFixture
where
    Precision: IntegerPrecision,
    Engine: PlaintextRetrievalEngine<Plaintext, Precision::Raw>,
    Plaintext: PlaintextEntity,
    Maker: SynthesizesPlaintext<Precision, Plaintext>,
{
    type Parameters = PlaintextRetrievalParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes = (<Maker as PrototypesPlaintext<Precision>>::PlaintextProto,);
    type PreExecutionContext = (Plaintext,);
    type PostExecutionContext = (Plaintext, Precision::Raw);
    type Criteria = (Variance,);
    type Outcome = (Precision::Raw, Precision::Raw);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(vec![PlaintextRetrievalParameters].into_iter())
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
        let raw_plaintext = Precision::Raw::uniform();
        let proto_plaintext = maker.transform_raw_to_plaintext(&raw_plaintext);
        (proto_plaintext,)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_plaintext,) = sample_proto;
        (maker.synthesize_plaintext(proto_plaintext),)
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (plaintext,) = context;
        let raw_output = unsafe { engine.retrieve_plaintext_unchecked(&plaintext) };
        (plaintext, raw_output)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        _sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (plaintext, raw_output) = context;
        let proto_output_plaintext = maker.unsynthesize_plaintext(&plaintext);
        maker.destroy_plaintext(plaintext);
        (
            maker.transform_plaintext_to_raw(&proto_output_plaintext),
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
