use crate::fixture::Fixture;
use crate::generation::prototyping::{PrototypesContainer, PrototypesLweCiphertext};
use crate::generation::synthesizing::{SynthesizesContainer, SynthesizesLweCiphertext};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::LweDimension;

use concrete_core::prelude::{LweCiphertextCreationEngine, LweCiphertextEntity};

#[derive(Debug)]
pub struct LweCiphertextCreationParameters {
    pub lwe_dimension: LweDimension,
}

/// A fixture for the types implementing the `LweCiphertextCreationEngine` trait with LWE
/// ciphertexts.
pub struct LweCiphertextCreationFixture;

impl<Precision, Engine, LweCiphertext, Container>
    Fixture<Precision, Engine, (LweCiphertext, Container)> for LweCiphertextCreationFixture
where
    Precision: IntegerPrecision,
    Engine: LweCiphertextCreationEngine<Container, LweCiphertext>,
    LweCiphertext: LweCiphertextEntity,
    Maker: SynthesizesLweCiphertext<Precision, LweCiphertext>
        + SynthesizesContainer<Precision, Container>,
{
    type Parameters = LweCiphertextCreationParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes = (<Maker as PrototypesContainer<Precision>>::ContainerProto,);
    type PreExecutionContext = (Container,);
    type PostExecutionContext = (LweCiphertext,);
    type Criteria = (Variance,);
    type Outcome = (Vec<Precision::Raw>, Vec<Precision::Raw>);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                LweCiphertextCreationParameters {
                    lwe_dimension: LweDimension(1),
                },
                LweCiphertextCreationParameters {
                    lwe_dimension: LweDimension(512),
                },
                LweCiphertextCreationParameters {
                    lwe_dimension: LweDimension(751),
                },
            ]
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
        let num_elements = parameters.lwe_dimension.to_lwe_size().0;
        (maker.transform_raw_vec_to_container(&Precision::Raw::uniform_vec(num_elements)),)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        (maker.synthesize_container(&sample_proto.0),)
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (underlying_container,) = context;
        let lwe_ciphertext =
            unsafe { engine.create_lwe_ciphertext_unchecked(underlying_container) };
        (lwe_ciphertext,)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (lwe_ciphertext,) = context;
        let ciphertext_proto = maker.unsynthesize_lwe_ciphertext(lwe_ciphertext);
        (
            maker.transform_container_to_raw_vec(&sample_proto.0),
            maker.transform_ciphertext_to_raw_vec(&ciphertext_proto),
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
        let means: Vec<Precision::Raw> = means.into_iter().flatten().collect();
        let actual: Vec<Precision::Raw> = actual.into_iter().flatten().collect();
        assert_noise_distribution(actual.as_slice(), means.as_slice(), criteria.0)
    }
}
