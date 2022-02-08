use crate::prototyping::prototyper::Prototyper;
use crate::prototyping::IntegerPrecision;
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use crate::synthesizing::{
    SynthesizableBinaryLweCiphertext, SynthesizableBinaryLweSecretKey, SynthesizablePlaintext,
};
use crate::Maker;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::LweDimension;
use concrete_core::prelude::LweCiphertextEncryptionEngine;
use lazy_static::lazy_static;

pub fn test_binary<Precision, Engine, SecretKey, Plaintext, Ciphertext>()
where
    Precision: IntegerPrecision,
    Maker: Prototyper<Precision>,
    Engine: LweCiphertextEncryptionEngine<SecretKey, Plaintext, Ciphertext>,
    SecretKey: SynthesizableBinaryLweSecretKey<Precision>,
    Plaintext: SynthesizablePlaintext<Precision>,
    Ciphertext: SynthesizableBinaryLweCiphertext<Precision>,
{
    let mut maker = Maker::default();
    let mut engine = Engine::new().unwrap();

    for p in &*PARAMETERS {
        for _ in 0..crate::REPETITIONS.0 {
            let expected = Precision::Raw::uniform_vec(crate::SAMPLE_SIZE.0);
            let mut achieved = Precision::Raw::one_vec(crate::SAMPLE_SIZE.0);

            for i in 0..crate::SAMPLE_SIZE.0 {
                let proto_lwe_secret_key = maker.new_binary_lwe_secret_key(p.lwe_dimension);
                let proto_plaintext = maker.transform_raw_to_plaintext(&expected[i]);
                let lwe_secret_key = SecretKey::from_prototype(&mut maker, &proto_lwe_secret_key);
                let plaintext = Plaintext::from_prototype(&mut maker, &proto_plaintext);
                let ciphertext = engine
                    .encrypt_lwe_ciphertext(&lwe_secret_key, &plaintext, p.noise)
                    .unwrap();
                let proto_ciphertext = Ciphertext::into_prototype(&mut maker, &ciphertext);
                let proto_plaintext = maker.decrypt_binary_lwe_ciphertext_to_plaintext(
                    &proto_lwe_secret_key,
                    &proto_ciphertext,
                );
                let raw_plaintext = maker.transform_plaintext_to_raw(&proto_plaintext);
                achieved[i] = raw_plaintext;
            }

            assert_noise_distribution(achieved.as_slice(), expected.as_slice(), p.noise);
        }
    }
}

pub struct Parameters {
    pub noise: Variance,
    pub lwe_dimension: LweDimension,
}

lazy_static! {
    static ref PARAMETERS: Vec<Parameters> = vec![
        Parameters {
            noise: Variance(0.00000001),
            lwe_dimension: LweDimension(100)
        },
        Parameters {
            noise: Variance(0.00000001),
            lwe_dimension: LweDimension(300)
        },
        Parameters {
            noise: Variance(0.00000001),
            lwe_dimension: LweDimension(600)
        },
        Parameters {
            noise: Variance(0.00000001),
            lwe_dimension: LweDimension(1000)
        },
        Parameters {
            noise: Variance(0.00000001),
            lwe_dimension: LweDimension(3000)
        },
        Parameters {
            noise: Variance(0.00000001),
            lwe_dimension: LweDimension(6000)
        }
    ];
}
