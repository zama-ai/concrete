use crate::fixture::Fixture;
use crate::generation::prototyping::{PrototypesContainer, PrototypesGlweCiphertext};
use crate::generation::synthesizing::{SynthesizesContainer, SynthesizesGlweCiphertext};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{GlweDimension, PolynomialSize};

use concrete_core::prelude::{GlweCiphertextCreationEngine, GlweCiphertextEntity};

#[derive(Debug)]
pub struct GlweCiphertextCreationParameters {
    pub glwe_dimension: GlweDimension,
    pub polynomial_size: PolynomialSize,
}

/// A fixture for the types implementing the `GlweCiphertextCreationEngine` trait with GLWE
/// ciphertexts.
pub struct GlweCiphertextCreationFixture;

impl<Precision, Engine, GlweCiphertext, Container>
    Fixture<Precision, Engine, (GlweCiphertext, Container)> for GlweCiphertextCreationFixture
where
    Precision: IntegerPrecision,
    Engine: GlweCiphertextCreationEngine<Container, GlweCiphertext>,
    GlweCiphertext: GlweCiphertextEntity,
    Maker: SynthesizesGlweCiphertext<Precision, GlweCiphertext>
        + SynthesizesContainer<Precision, Container>,
{
    type Parameters = GlweCiphertextCreationParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes = (
        <Maker as PrototypesContainer<Precision>>::ContainerProto,
        PolynomialSize,
    );
    type PreExecutionContext = (Container, PolynomialSize);
    type PostExecutionContext = (GlweCiphertext,);
    type Criteria = (Variance,);
    type Outcome = (Vec<Precision::Raw>, Vec<Precision::Raw>);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                GlweCiphertextCreationParameters {
                    glwe_dimension: GlweDimension(1),
                    polynomial_size: PolynomialSize(512),
                },
                GlweCiphertextCreationParameters {
                    glwe_dimension: GlweDimension(2),
                    polynomial_size: PolynomialSize(1024),
                },
                GlweCiphertextCreationParameters {
                    glwe_dimension: GlweDimension(2),
                    polynomial_size: PolynomialSize(2048),
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
        let Self::Parameters {
            glwe_dimension,
            polynomial_size,
        } = parameters;
        let num_elements = glwe_dimension.to_glwe_size().0 * polynomial_size.0;
        (
            maker.transform_raw_vec_to_container(&Precision::Raw::uniform_vec(num_elements)),
            *polynomial_size,
        )
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        (maker.synthesize_container(&sample_proto.0), sample_proto.1)
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (underlying_container, polynomial_size) = context;
        let glwe_ciphertext = unsafe {
            engine.create_glwe_ciphertext_unchecked(underlying_container, polynomial_size)
        };
        (glwe_ciphertext,)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (glwe_ciphertext,) = context;
        let ciphertext_proto = maker.unsynthesize_glwe_ciphertext(glwe_ciphertext);
        (
            maker.transform_container_to_raw_vec(&sample_proto.0),
            maker.transform_ciphertext_to_raw_vec(&ciphertext_proto).0,
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
