use crate::fixture::Fixture;
use crate::generation::synthesizing::SynthesizesGlweSecretKey;
use crate::generation::{IntegerPrecision, Maker};
use concrete_commons::parameters::{GlweDimension, PolynomialSize};
use concrete_core::prelude::{GlweSecretKeyCreationEngine, GlweSecretKeyEntity};

/// A fixture for the types implementing the `GlweSecretKeyCreationEngine` trait.
pub struct GlweSecretKeyCreationFixture;

#[derive(Debug)]
pub struct GlweSecretKeyCreationParameters {
    pub glwe_dimension: GlweDimension,
    pub polynomial_size: PolynomialSize,
}

impl<Precision, Engine, SecretKey> Fixture<Precision, Engine, (SecretKey,)>
    for GlweSecretKeyCreationFixture
where
    Precision: IntegerPrecision,
    Engine: GlweSecretKeyCreationEngine<SecretKey>,
    SecretKey: GlweSecretKeyEntity,
    Maker: SynthesizesGlweSecretKey<Precision, SecretKey>,
{
    type Parameters = GlweSecretKeyCreationParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes = ();
    type PreExecutionContext = ();
    type PostExecutionContext = (SecretKey,);
    type Criteria = ();
    type Outcome = ();

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![GlweSecretKeyCreationParameters {
                glwe_dimension: GlweDimension(1),
                polynomial_size: PolynomialSize(1024),
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
        let sk = unsafe {
            engine.create_glwe_secret_key_unchecked(
                parameters.glwe_dimension,
                parameters.polynomial_size,
            )
        };
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
        maker.destroy_glwe_secret_key(sk);
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
