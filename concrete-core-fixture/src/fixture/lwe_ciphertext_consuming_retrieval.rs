use crate::fixture::Fixture;
use crate::generation::prototyping::{PrototypesContainer, PrototypesLweCiphertext};
use crate::generation::synthesizing::{SynthesizesContainer, SynthesizesLweCiphertext};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::LweDimension;

use concrete_core::prelude::{LweCiphertextConsumingRetrievalEngine, LweCiphertextEntity};

#[derive(Debug)]
pub struct LweCiphertextConsumingRetrievalParameters {
    pub lwe_dimension: LweDimension,
}

/// A fixture for the types implementing the `LweCiphertextConsumingRetrievalEngine` trait with LWE
/// ciphertexts.
pub struct LweCiphertextConsumingRetrievalFixture;

impl<Precision, Engine, LweCiphertext, Container>
    Fixture<Precision, Engine, (LweCiphertext, Container)>
    for LweCiphertextConsumingRetrievalFixture
where
    Precision: IntegerPrecision,
    Engine: LweCiphertextConsumingRetrievalEngine<LweCiphertext, Container>,
    LweCiphertext: LweCiphertextEntity,
    Maker: SynthesizesLweCiphertext<Precision, LweCiphertext>
        + SynthesizesContainer<Precision, Container>,
{
    type Parameters = LweCiphertextConsumingRetrievalParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes =
        (<Maker as PrototypesLweCiphertext<Precision, LweCiphertext::KeyDistribution>>::LweCiphertextProto,);
    type PreExecutionContext = (LweCiphertext,);
    type PostExecutionContext = (Container,);
    type Criteria = (Variance,);
    type Outcome = (Vec<Precision::Raw>, Vec<Precision::Raw>);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                LweCiphertextConsumingRetrievalParameters {
                    lwe_dimension: LweDimension(1),
                },
                LweCiphertextConsumingRetrievalParameters {
                    lwe_dimension: LweDimension(512),
                },
                LweCiphertextConsumingRetrievalParameters {
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
        let proto_ciphertext =
            maker.transform_raw_vec_to_ciphertext(&Precision::Raw::uniform_vec(num_elements));
        (proto_ciphertext,)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_ciphertext,) = sample_proto;
        (maker.synthesize_lwe_ciphertext(proto_ciphertext),)
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (ciphertext,) = context;
        let raw_ciphertext =
            unsafe { engine.consume_retrieve_lwe_ciphertext_unchecked(ciphertext) };
        (raw_ciphertext,)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (proto_ciphertext,) = sample_proto;
        let (raw_ciphertext,) = context;
        let proto_container = maker.unsynthesize_container(raw_ciphertext);
        (
            maker.transform_ciphertext_to_raw_vec(proto_ciphertext),
            maker.transform_container_to_raw_vec(&proto_container),
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
