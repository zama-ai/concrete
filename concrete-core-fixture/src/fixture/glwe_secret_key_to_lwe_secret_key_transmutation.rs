use crate::fixture::Fixture;
use crate::generation::prototyping::PrototypesGlweSecretKey;
use crate::generation::synthesizing::{SynthesizesGlweSecretKey, SynthesizesLweSecretKey};
use crate::generation::{IntegerPrecision, Maker};
use concrete_commons::parameters::{GlweDimension, PolynomialSize};
use concrete_core::prelude::{
    GlweSecretKeyEntity, GlweToLweSecretKeyTransmutationEngine, LweSecretKeyEntity,
};

/// A fixture for the types implementing the `LweKeyswitchKeyCreationEngine` trait.
pub struct LweKeyswitchKeyCreationFixture;

#[derive(Debug)]
pub struct LweKeyswitchKeyCreationParameters {
    pub glwe_dimension: GlweDimension,
    pub polynomial_size: PolynomialSize,
}

impl<Precision, Engine, InputSecretKey, OutputSecretKey>
    Fixture<Precision, Engine, (InputSecretKey, OutputSecretKey)> for LweKeyswitchKeyCreationFixture
where
    Precision: IntegerPrecision,
    Engine: GlweToLweSecretKeyTransmutationEngine<InputSecretKey, OutputSecretKey>,
    InputSecretKey: GlweSecretKeyEntity,
    OutputSecretKey: LweSecretKeyEntity<KeyDistribution = InputSecretKey::KeyDistribution>,
    Maker: SynthesizesLweSecretKey<Precision, OutputSecretKey>
        + SynthesizesGlweSecretKey<Precision, InputSecretKey>,
{
    type Parameters = LweKeyswitchKeyCreationParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes = (
        <Maker as PrototypesGlweSecretKey<Precision, InputSecretKey::KeyDistribution>>::GlweSecretKeyProto,
    );
    type PreExecutionContext = (InputSecretKey,);
    type PostExecutionContext = (OutputSecretKey,);
    type Criteria = ();
    type Outcome = ();

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![LweKeyswitchKeyCreationParameters {
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
        parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        let proto_secret_key_in =
            maker.new_glwe_secret_key(parameters.glwe_dimension, parameters.polynomial_size);
        (proto_secret_key_in,)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_secret_key_in,) = sample_proto;
        let synth_secret_key_in = maker.synthesize_glwe_secret_key(proto_secret_key_in);
        (synth_secret_key_in,)
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (sk_in,) = context;
        let sk_out = unsafe { engine.transmute_glwe_secret_key_to_lwe_secret_key_unchecked(sk_in) };
        (sk_out,)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        _sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (sk_out,) = context;
        maker.destroy_lwe_secret_key(sk_out);
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
