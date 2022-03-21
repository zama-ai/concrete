use crate::fixture::Fixture;
use crate::generation::prototyping::PrototypesLweSecretKey;
use crate::generation::synthesizing::{SynthesizesLweKeyswitchKey, SynthesizesLweSecretKey};
use crate::generation::{IntegerPrecision, Maker};
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweDimension};
use concrete_core::prelude::{
    LweKeyswitchKeyCreationEngine, LweKeyswitchKeyEntity, LweSecretKeyEntity,
};

/// A fixture for the types implementing the `LweKeyswitchKeyCreationEngine` trait.
pub struct LweKeyswitchKeyCreationFixture;

#[derive(Debug)]
pub struct LweKeyswitchKeyCreationParameters {
    pub noise: Variance,
    pub lwe_dimension_in: LweDimension,
    pub lwe_dimension_out: LweDimension,
    pub level: DecompositionLevelCount,
    pub base_log: DecompositionBaseLog,
}

impl<Precision, Engine, InputSecretKey, OutputSecretKey, LweKeyswitchKey>
    Fixture<Precision, Engine, (InputSecretKey, OutputSecretKey, LweKeyswitchKey)>
    for LweKeyswitchKeyCreationFixture
where
    Precision: IntegerPrecision,
    Engine: LweKeyswitchKeyCreationEngine<InputSecretKey, OutputSecretKey, LweKeyswitchKey>,
    InputSecretKey: LweSecretKeyEntity,
    OutputSecretKey: LweSecretKeyEntity,
    LweKeyswitchKey: LweKeyswitchKeyEntity<
        InputKeyDistribution = InputSecretKey::KeyDistribution,
        OutputKeyDistribution = OutputSecretKey::KeyDistribution,
    >,
    Maker: SynthesizesLweKeyswitchKey<Precision, LweKeyswitchKey>
        + SynthesizesLweSecretKey<Precision, InputSecretKey>
        + SynthesizesLweSecretKey<Precision, OutputSecretKey>,
{
    type Parameters = LweKeyswitchKeyCreationParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes = (
        <Maker as PrototypesLweSecretKey<Precision, InputSecretKey::KeyDistribution>>::LweSecretKeyProto,
        <Maker as PrototypesLweSecretKey<Precision, OutputSecretKey::KeyDistribution>>::LweSecretKeyProto,
    );
    type PreExecutionContext = (InputSecretKey, OutputSecretKey);
    type PostExecutionContext = (LweKeyswitchKey,);
    type Criteria = ();
    type Outcome = ();

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![LweKeyswitchKeyCreationParameters {
                noise: Variance(0.00000001),
                lwe_dimension_in: LweDimension(1024),
                lwe_dimension_out: LweDimension(630),
                level: DecompositionLevelCount(3),
                base_log: DecompositionBaseLog(7),
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
        let proto_secret_key_in = <Maker as PrototypesLweSecretKey<
            Precision,
            InputSecretKey::KeyDistribution,
        >>::new_lwe_secret_key(maker, parameters.lwe_dimension_in);
        let proto_secret_key_out = <Maker as PrototypesLweSecretKey<
            Precision,
            OutputSecretKey::KeyDistribution,
        >>::new_lwe_secret_key(
            maker, parameters.lwe_dimension_out
        );
        (proto_secret_key_in, proto_secret_key_out)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_secret_key_in, proto_secret_key_out) = sample_proto;
        let synth_secret_key_in = maker.synthesize_lwe_secret_key(proto_secret_key_in);
        let synth_secret_key_out = maker.synthesize_lwe_secret_key(proto_secret_key_out);
        (synth_secret_key_in, synth_secret_key_out)
    }

    fn execute_engine(
        parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (sk_in, sk_out) = context;
        let ksk = unsafe {
            engine.create_lwe_keyswitch_key_unchecked(
                &sk_in,
                &sk_out,
                parameters.level,
                parameters.base_log,
                parameters.noise,
            )
        };
        (ksk,)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        _sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (ksk,) = context;
        maker.destroy_lwe_keyswitch_key(ksk);
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
