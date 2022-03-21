use crate::fixture::Fixture;
use crate::generation::prototyping::{
    PrototypesGlweSecretKey, PrototypesLweBootstrapKey, PrototypesLweSecretKey,
};
use crate::generation::synthesizing::SynthesizesLweBootstrapKey;
use crate::generation::{IntegerPrecision, Maker};
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};
use concrete_core::prelude::{LweBootstrapKeyConversionEngine, LweBootstrapKeyEntity};

/// A fixture for the types implementing the `LweBootstrapKeyConversionEngine` trait.
pub struct LweSecretKeyConversionFixture;

#[derive(Debug)]
pub struct LweSecretKeyConversionParameters {
    pub lwe_dimension: LweDimension,
    pub glwe_dimension: GlweDimension,
    pub polynomial_size: PolynomialSize,
    pub level: DecompositionLevelCount,
    pub base_log: DecompositionBaseLog,
    pub noise: Variance,
}

impl<Precision, Engine, InputKey, OutputKey> Fixture<Precision, Engine, (InputKey, OutputKey)>
    for LweSecretKeyConversionFixture
where
    Precision: IntegerPrecision,
    Engine: LweBootstrapKeyConversionEngine<InputKey, OutputKey>,
    InputKey: LweBootstrapKeyEntity,
    OutputKey: LweBootstrapKeyEntity<
        InputKeyDistribution = InputKey::InputKeyDistribution,
        OutputKeyDistribution = InputKey::OutputKeyDistribution,
    >,
    Maker: SynthesizesLweBootstrapKey<Precision, InputKey>
        + SynthesizesLweBootstrapKey<Precision, OutputKey>,
{
    type Parameters = LweSecretKeyConversionParameters;
    type RepetitionPrototypes = (
        <Maker as PrototypesLweBootstrapKey<
            Precision,
            InputKey::InputKeyDistribution,
            InputKey::OutputKeyDistribution,
        >>::LweBootstrapKeyProto,
    );
    type SamplePrototypes = ();
    type PreExecutionContext = (InputKey,);
    type PostExecutionContext = (OutputKey,);
    type Criteria = ();
    type Outcome = ();

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![LweSecretKeyConversionParameters {
                lwe_dimension: LweDimension(630),
                glwe_dimension: GlweDimension(1),
                polynomial_size: PolynomialSize(1024),
                level: DecompositionLevelCount(3),
                base_log: DecompositionBaseLog(7),
                noise: Variance(0.00000001),
            }]
            .into_iter(),
        )
    }

    fn generate_random_repetition_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
    ) -> Self::RepetitionPrototypes {
        let input_key = <Maker as PrototypesLweSecretKey<
            Precision,
            InputKey::InputKeyDistribution,
        >>::new_lwe_secret_key(maker, parameters.lwe_dimension);
        let output_key =
            maker.new_glwe_secret_key(parameters.glwe_dimension, parameters.polynomial_size);
        let proto_bsk_in = maker.new_lwe_bootstrap_key(
            &input_key,
            &output_key,
            parameters.level,
            parameters.base_log,
            parameters.noise,
        );
        (proto_bsk_in,)
    }

    fn generate_random_sample_prototypes(
        _parameters: &Self::Parameters,
        _maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        _sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_bsk_in,) = repetition_proto;
        let synth_bsk = maker.synthesize_lwe_bootstrap_key(proto_bsk_in);
        (synth_bsk,)
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (bsk_in,) = context;
        let bsk_out = unsafe { engine.convert_lwe_bootstrap_key_unchecked(&bsk_in) };
        (bsk_out,)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        _sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (bsk,) = context;
        maker.destroy_lwe_bootstrap_key(bsk);
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
