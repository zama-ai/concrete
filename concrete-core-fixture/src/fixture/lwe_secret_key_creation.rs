use crate::fixture::Fixture;
use crate::generation::synthesizing::SynthesizesLweSecretKey;
use crate::generation::{IntegerPrecision, Maker};
use concrete_commons::parameters::LweDimension;
use concrete_core::prelude::{LweSecretKeyCreationEngine, LweSecretKeyEntity};

/// A fixture for the types implementing the `LweSecretKeyCreationEngine` trait.
pub struct LweSecretKeyCreationFixture;

#[derive(Debug)]
pub struct LweSecretKeyCreationParameters {
    pub lwe_dimension: LweDimension,
}

impl<Precision, Engine, SecretKey> Fixture<Precision, Engine, (SecretKey,)>
    for LweSecretKeyCreationFixture
where
    Precision: IntegerPrecision,
    Engine: LweSecretKeyCreationEngine<SecretKey>,
    SecretKey: LweSecretKeyEntity,
    Maker: SynthesizesLweSecretKey<Precision, SecretKey>,
{
    type Parameters = LweSecretKeyCreationParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes = ();
    type PreExecutionContext = ();
    type PostExecutionContext = (SecretKey,);
    type Criteria = ();
    type Outcome = ();

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![LweSecretKeyCreationParameters {
                lwe_dimension: LweDimension(630),
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
        _parameters: &Self::Parameters,
        _maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        _maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        _sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
    }

    fn execute_engine(
        parameters: &Self::Parameters,
        engine: &mut Engine,
        _context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let sk = unsafe { engine.create_lwe_secret_key_unchecked(parameters.lwe_dimension) };
        (sk,)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        _sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (sk,) = context;
        maker.destroy_lwe_secret_key(sk);
        unimplemented!()
    }

    fn compute_criteria(
        _parameters: &Self::Parameters,
        _maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::Criteria {
        unimplemented!()
    }

    fn verify(_criteria: &Self::Criteria, _outputs: &[Self::Outcome]) -> bool {
        unimplemented!()
    }
}
