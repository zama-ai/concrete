use crate::fixture::Fixture;
use crate::generation::prototyping::{PrototypesLweCiphertext, PrototypesPlaintext};
use crate::generation::synthesizing::{SynthesizesLweCiphertext, SynthesizesPlaintext};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::LweDimension;
use concrete_core::prelude::{
    LweCiphertextEntity, LweCiphertextTrivialDecryptionEngine, PlaintextEntity,
};

/// A fixture for the types implementing the `LweCiphertextTrivialDecryptionEngine` trait.
pub struct LweCiphertextTrivialDecryptionFixture;

#[derive(Debug)]
pub struct LweCiphertextTrivialDecryptionParameters {
    pub lwe_dimension: LweDimension,
}

impl<Precision, Engine, Plaintext, Ciphertext> Fixture<Precision, Engine, (Plaintext, Ciphertext)>
    for LweCiphertextTrivialDecryptionFixture
where
    Precision: IntegerPrecision,
    Engine: LweCiphertextTrivialDecryptionEngine<Ciphertext, Plaintext>,
    Plaintext: PlaintextEntity,
    Ciphertext: LweCiphertextEntity,
    Maker: SynthesizesPlaintext<Precision, Plaintext>
        + SynthesizesLweCiphertext<Precision, Ciphertext>,
{
    type Parameters = LweCiphertextTrivialDecryptionParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes = (
        <Maker as PrototypesLweCiphertext<
            Precision,
            Ciphertext::KeyDistribution,
        >>::LweCiphertextProto,
        Precision::Raw,
    );
    type PreExecutionContext = (Ciphertext,);
    type PostExecutionContext = (Plaintext, Ciphertext);
    type Criteria = (Variance,);
    type Outcome = (Precision::Raw, Precision::Raw);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                LweCiphertextTrivialDecryptionParameters {
                    lwe_dimension: LweDimension(100),
                },
                LweCiphertextTrivialDecryptionParameters {
                    lwe_dimension: LweDimension(300),
                },
                LweCiphertextTrivialDecryptionParameters {
                    lwe_dimension: LweDimension(600),
                },
                LweCiphertextTrivialDecryptionParameters {
                    lwe_dimension: LweDimension(1000),
                },
                LweCiphertextTrivialDecryptionParameters {
                    lwe_dimension: LweDimension(3000),
                },
                LweCiphertextTrivialDecryptionParameters {
                    lwe_dimension: LweDimension(6000),
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
        let raw_plaintext = Precision::Raw::uniform();
        let proto_plaintext = maker.transform_raw_to_plaintext(&raw_plaintext);
        let proto_ciphertext = maker.trivially_encrypt_plaintext_to_lwe_ciphertext(
            parameters.lwe_dimension,
            &proto_plaintext,
        );
        (proto_ciphertext, raw_plaintext)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_ciphertext, _) = sample_proto;
        let synth_ciphertext = maker.synthesize_lwe_ciphertext(proto_ciphertext);
        (synth_ciphertext,)
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (ciphertext,) = context;
        let plaintext = unsafe { engine.trivially_decrypt_lwe_ciphertext_unchecked(&ciphertext) };
        (plaintext, ciphertext)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (plaintext, ciphertext) = context;
        let (_, raw_plaintext) = sample_proto;
        let proto_plaintext = maker.unsynthesize_plaintext(&plaintext);
        maker.destroy_lwe_ciphertext(ciphertext);
        maker.destroy_plaintext(plaintext);
        (
            *raw_plaintext,
            maker.transform_plaintext_to_raw(&proto_plaintext),
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
        assert_noise_distribution(&actual, means.as_slice(), criteria.0)
    }
}
