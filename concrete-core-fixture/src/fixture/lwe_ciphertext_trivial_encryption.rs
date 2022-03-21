use crate::fixture::Fixture;
use crate::generation::prototyping::{PrototypesLweCiphertext, PrototypesPlaintext};
use crate::generation::synthesizing::{SynthesizesLweCiphertext, SynthesizesPlaintext};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::LweDimension;
use concrete_core::prelude::{
    LweCiphertextEntity, LweCiphertextTrivialEncryptionEngine, PlaintextEntity,
};

/// A fixture for the types implementing the `LweCiphertextTrivialEncryptionEngine` trait.
pub struct LweCiphertextTrivialEncryptionFixture;

#[derive(Debug)]
pub struct LweCiphertextTrivialEncryptionParameters {
    pub lwe_dimension: LweDimension,
}

impl<Precision, Engine, Plaintext, Ciphertext> Fixture<Precision, Engine, (Plaintext, Ciphertext)>
    for LweCiphertextTrivialEncryptionFixture
where
    Precision: IntegerPrecision,
    Engine: LweCiphertextTrivialEncryptionEngine<Plaintext, Ciphertext>,
    Plaintext: PlaintextEntity,
    Ciphertext: LweCiphertextEntity,
    Maker: SynthesizesPlaintext<Precision, Plaintext>
        + SynthesizesLweCiphertext<Precision, Ciphertext>,
{
    type Parameters = LweCiphertextTrivialEncryptionParameters;
    type RepetitionPrototypes = ();
    type SamplePrototypes = (
        <Maker as PrototypesPlaintext<Precision>>::PlaintextProto,
        Precision::Raw,
    );
    type PreExecutionContext = (Plaintext,);
    type PostExecutionContext = (Plaintext, Ciphertext);
    type Criteria = (Variance,);
    type Outcome = (Precision::Raw, Precision::Raw);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                LweCiphertextTrivialEncryptionParameters {
                    lwe_dimension: LweDimension(100),
                },
                LweCiphertextTrivialEncryptionParameters {
                    lwe_dimension: LweDimension(300),
                },
                LweCiphertextTrivialEncryptionParameters {
                    lwe_dimension: LweDimension(600),
                },
                LweCiphertextTrivialEncryptionParameters {
                    lwe_dimension: LweDimension(1000),
                },
                LweCiphertextTrivialEncryptionParameters {
                    lwe_dimension: LweDimension(3000),
                },
                LweCiphertextTrivialEncryptionParameters {
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
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        let raw_plaintext = Precision::Raw::uniform();
        let proto_plaintext = maker.transform_raw_to_plaintext(&raw_plaintext);
        (proto_plaintext, raw_plaintext)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_plaintext, _) = sample_proto;
        let synth_plaintext = maker.synthesize_plaintext(proto_plaintext);
        (synth_plaintext,)
    }

    fn execute_engine(
        parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (plaintext,) = context;
        let ciphertext = unsafe {
            engine.trivially_encrypt_lwe_ciphertext_unchecked(
                parameters.lwe_dimension.to_lwe_size(),
                &plaintext,
            )
        };
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
        let proto_output_ciphertext = maker.unsynthesize_lwe_ciphertext(&ciphertext);
        maker.destroy_lwe_ciphertext(ciphertext);
        maker.destroy_plaintext(plaintext);
        let proto_plaintext =
            maker.trivially_decrypt_lwe_ciphertext_to_plaintext(&proto_output_ciphertext);
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
