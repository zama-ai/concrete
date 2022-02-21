use crate::fixture::Fixture;
use crate::generation::prototyping::{
    PrototypesLweCiphertext, PrototypesLweSecretKey, PrototypesPlaintext,
};
use crate::generation::synthesizing::{
    SynthesizesLweCiphertext, SynthesizesLweSecretKey, SynthesizesPlaintext,
};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use crate::SampleSize;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::LweDimension;
use concrete_core::prelude::{
    LweCiphertextEncryptionEngine, LweCiphertextEntity, LweSecretKeyEntity, PlaintextEntity,
};

/// A fixture for the types implementing the `LweCiphertextEncryptionEngine` trait.
pub struct LweCiphertextEncryptionFixture;

#[derive(Debug)]
pub struct LweCiphertextEncryptionParameters {
    pub noise: Variance,
    pub lwe_dimension: LweDimension,
}

impl<Precision, Engine, Plaintext, SecretKey, Ciphertext>
    Fixture<Precision, Engine, (Plaintext, SecretKey, Ciphertext)>
    for LweCiphertextEncryptionFixture
where
    Precision: IntegerPrecision,
    Engine: LweCiphertextEncryptionEngine<SecretKey, Plaintext, Ciphertext>,
    Plaintext: PlaintextEntity,
    SecretKey: LweSecretKeyEntity,
    Ciphertext: LweCiphertextEntity<KeyDistribution = SecretKey::KeyDistribution>,
    Maker: SynthesizesPlaintext<Precision, Plaintext>
        + SynthesizesLweSecretKey<Precision, SecretKey>
        + SynthesizesLweCiphertext<Precision, Ciphertext>,
{
    type Parameters = LweCiphertextEncryptionParameters;
    type RawInputs = (Precision::Raw,);
    type RawOutputs = (Precision::Raw,);
    type SecretKeyPrototypes = (<Maker as PrototypesLweSecretKey<Precision, Ciphertext::KeyDistribution>>::LweSecretKeyProto, );
    type PreExecutionContext = (Plaintext, SecretKey);
    type PostExecutionContext = (Plaintext, SecretKey, Ciphertext);
    type Prediction = (Vec<Precision::Raw>, Variance);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                LweCiphertextEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(100),
                },
                LweCiphertextEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(300),
                },
                LweCiphertextEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(600),
                },
                LweCiphertextEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(1000),
                },
                LweCiphertextEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(3000),
                },
                LweCiphertextEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(6000),
                },
            ]
            .into_iter(),
        )
    }

    fn generate_random_raw_inputs(_parameters: &Self::Parameters) -> Self::RawInputs {
        (Precision::Raw::uniform(),)
    }

    fn compute_prediction(
        parameters: &Self::Parameters,
        raw_inputs: &Self::RawInputs,
        sample_size: SampleSize,
    ) -> Self::Prediction {
        let (raw_plaintext,) = raw_inputs;
        (vec![*raw_plaintext; sample_size.0], parameters.noise)
    }

    fn check_prediction(
        _parameters: &Self::Parameters,
        forecast: &Self::Prediction,
        actual: &[Self::RawOutputs],
    ) -> bool {
        let (means, noise) = forecast;
        let actual = actual.iter().map(|r| r.0).collect::<Vec<_>>();
        assert_noise_distribution(&actual, means.as_slice(), *noise)
    }

    fn prepare_context(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        raw_inputs: &Self::RawInputs,
    ) -> (Self::SecretKeyPrototypes, Self::PreExecutionContext) {
        let (raw_plaintext,) = raw_inputs;
        let proto_plaintext = maker.transform_raw_to_plaintext(raw_plaintext);
        let proto_secret_key = maker.new_lwe_secret_key(parameters.lwe_dimension);
        let synth_plaintext = maker.synthesize_plaintext(&proto_plaintext);
        let synth_secret_key = maker.synthesize_lwe_secret_key(&proto_secret_key);
        ((proto_secret_key,), (synth_plaintext, synth_secret_key))
    }

    fn execute_engine(
        parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (plaintext, secret_key) = context;
        let ciphertext = unsafe {
            engine.encrypt_lwe_ciphertext_unchecked(&secret_key, &plaintext, parameters.noise)
        };
        (plaintext, secret_key, ciphertext)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        secret_keys: Self::SecretKeyPrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::RawOutputs {
        let (plaintext, secret_key, ciphertext) = context;
        let (proto_secret_key,) = secret_keys;
        let proto_output_ciphertext = maker.unsynthesize_lwe_ciphertext(&ciphertext);
        maker.destroy_lwe_ciphertext(ciphertext);
        maker.destroy_plaintext(plaintext);
        maker.destroy_lwe_secret_key(secret_key);
        let proto_plaintext =
            maker.decrypt_lwe_ciphertext_to_plaintext(&proto_secret_key, &proto_output_ciphertext);
        (maker.transform_plaintext_to_raw(&proto_plaintext),)
    }
}
