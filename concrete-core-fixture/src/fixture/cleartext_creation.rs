use crate::fixture::Fixture;
use crate::generation::prototyping::PrototypesCleartext;
use crate::generation::synthesizing::SynthesizesCleartext;
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use crate::SampleSize;
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
    type RawInputs = (Precision::Raw,);
    type RawOutputs = (Precision::Raw,);
    type PreExecutionContext = (Precision::Raw,);
    type SecretKeyPrototypes = ();
    type PostExecutionContext = (Cleartext,);
    type Prediction = (Vec<Precision::Raw>, Variance);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(vec![CleartextCreationParameters].into_iter())
    }

    fn generate_random_raw_inputs(_parameters: &Self::Parameters) -> Self::RawInputs {
        (Precision::Raw::uniform(),)
    }

    fn compute_prediction(
        _parameters: &Self::Parameters,
        raw_inputs: &Self::RawInputs,
        sample_size: SampleSize,
    ) -> Self::Prediction {
        let (raw_cleartext,) = raw_inputs;
        (vec![*raw_cleartext; sample_size.0], Variance(0.))
    }

    fn check_prediction(
        _parameters: &Self::Parameters,
        forecast: &Self::Prediction,
        actual: &[Self::RawOutputs],
    ) -> bool {
        let (means, noise) = forecast;
        let actual = actual.iter().map(|r| r.0).collect::<Vec<_>>();
        assert_noise_distribution(&actual, means.as_slice(), *noise)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        _maker: &mut Maker,
        raw_inputs: &Self::RawInputs,
    ) -> (Self::SecretKeyPrototypes, Self::PreExecutionContext) {
        let (raw_cleartext,) = raw_inputs;
        ((), (*raw_cleartext,))
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
        _secret_keys: Self::SecretKeyPrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::RawOutputs {
        let (cleartext,) = context;
        let proto_cleartext = maker.unsynthesize_cleartext(&cleartext);
        maker.destroy_cleartext(cleartext);
        (maker.transform_cleartext_to_raw(&proto_cleartext),)
    }
}
