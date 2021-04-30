//! A module using a software fallback implementation of `aes128-ctr` random number generator.
use crate::ctr::{AesCtr, AesBatch};
use crate::{AesBatchedGenerator, AesKey};
use aes_soft::cipher::generic_array::GenericArray;
use aes_soft::cipher::{BlockCipher, NewBlockCipher};
use aes_soft::Aes128;
use std::cell::UnsafeCell;
use std::io::Read;

thread_local! {
    static RDSEED_COUNTER: UnsafeCell<u128> = UnsafeCell::new(
        std::time::UNIX_EPOCH
            .elapsed()
            .expect("Failed to initialized software rdseed counter.")
            .as_nanos()
    );
    static RDSEED_SECRET: UnsafeCell<u128> = UnsafeCell::new(0);
    static RDSEED_SEEDED: UnsafeCell<bool> = UnsafeCell::new(false);
}

/// Sets the secret used to seed the software version of the prng.
///
/// When using the software variant of the CSPRNG, we do not have access to the (trusted)
/// hardware source of randomness to seed the generator. Instead, we use a value from
/// `/dev/random`, which can be easy to temper with. To mitigate this risk, the user can provide
/// a secret value that is included in the seed of the prng. Note that to ensure maximal
/// security, this value should be different each time a new application using concrete is started.
pub fn set_soft_rdseed_secret(secret: u128) {
    RDSEED_SECRET.with(|f| {
        let _secret = unsafe { &mut *{ f.get() } };
        *_secret = secret;
    });
    RDSEED_SEEDED.with(|f| {
        let _seeded = unsafe { &mut *{ f.get() } };
        *_seeded = true;
    })
}

fn rdseed() -> u128 {
    RDSEED_SEEDED.with(|f| {
        let is_seeded = unsafe { &*{ f.get() } };
        if !*is_seeded {
            println!(
                "WARNING: You are currently using the software variant of concrete-csprng \
                which does not have access to a hardware source of randomness. To ensure the \
                security of your application, please arrange to provide a secret by using the \
                `concrete_csprng::set_soft_rdseed_secret` function."
            );
        }
    });
    let mut output: u128 = 0;
    RDSEED_SECRET.with(|f| {
        let secret = unsafe { &*{ f.get() } };
        RDSEED_COUNTER.with(|f| {
            let counter = unsafe { &mut *{ f.get() } };
            output = *secret ^ *counter ^ dev_random();
            *counter = counter.wrapping_add(1);
        })
    });
    output
}

#[derive(Clone)]
pub struct Generator {
    // Aes structure
    aes: Aes128,
}

impl AesBatchedGenerator for Generator {
    fn new(key: Option<AesKey>) -> Generator {
        let key: [u8; 16] = key.map(|AesKey(k)| k).unwrap_or_else(rdseed).to_ne_bytes();
        let key = GenericArray::clone_from_slice(&key[..]);
        let aes = Aes128::new(&key);
        Generator { aes }
    }

    fn generate_batch(&mut self, AesCtr(aes_ctr): AesCtr) -> AesBatch {
        aes_encrypt_many(
            aes_ctr,
            aes_ctr + 1,
            aes_ctr + 2,
            aes_ctr + 3,
            aes_ctr + 4,
            aes_ctr + 5,
            aes_ctr + 6,
            aes_ctr + 7,
            &self.aes,
        )
    }
}

pub fn dev_random() -> u128 {
    let mut random = std::fs::File::open("/dev/random").expect("Failed to open /dev/random .");
    let mut buf = [0u8; 16];
    random
        .read_exact(&mut buf[..])
        .expect("Failed to read from /dev/random .");
    u128::from_ne_bytes(buf)
}

// Uses aes to encrypt many values at once. This allows a substantial speedup (around 30%)
// compared to the naive approach.
#[allow(clippy::too_many_arguments)]
fn aes_encrypt_many(
    message_1: u128,
    message_2: u128,
    message_3: u128,
    message_4: u128,
    message_5: u128,
    message_6: u128,
    message_7: u128,
    message_8: u128,
    cipher: &Aes128,
) -> [u8; 128] {
    let mut b1 = GenericArray::clone_from_slice(&message_1.to_ne_bytes()[..]);
    let mut b2 = GenericArray::clone_from_slice(&message_2.to_ne_bytes()[..]);
    let mut b3 = GenericArray::clone_from_slice(&message_3.to_ne_bytes()[..]);
    let mut b4 = GenericArray::clone_from_slice(&message_4.to_ne_bytes()[..]);
    let mut b5 = GenericArray::clone_from_slice(&message_5.to_ne_bytes()[..]);
    let mut b6 = GenericArray::clone_from_slice(&message_6.to_ne_bytes()[..]);
    let mut b7 = GenericArray::clone_from_slice(&message_7.to_ne_bytes()[..]);
    let mut b8 = GenericArray::clone_from_slice(&message_8.to_ne_bytes()[..]);

    cipher.encrypt_block(&mut b1);
    cipher.encrypt_block(&mut b2);
    cipher.encrypt_block(&mut b3);
    cipher.encrypt_block(&mut b4);
    cipher.encrypt_block(&mut b5);
    cipher.encrypt_block(&mut b6);
    cipher.encrypt_block(&mut b7);
    cipher.encrypt_block(&mut b8);

    let output_array: [[u8; 16]; 8] = [
        b1.into(),
        b2.into(),
        b3.into(),
        b4.into(),
        b5.into(),
        b6.into(),
        b7.into(),
        b8.into(),
    ];

    unsafe { *{ output_array.as_ptr() as *const [u8; 128] } }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::convert::TryInto;

    // Test vector for aes128, from the FIPS publication 197
    const CIPHER_KEY: u128 = u128::from_be(0x000102030405060708090a0b0c0d0e0f);
    const PLAINTEXT: u128 = u128::from_be(0x00112233445566778899aabbccddeeff);
    const CIPHERTEXT: u128 = u128::from_be(0x69c4e0d86a7b0430d8cdb78070b4c55a);

    #[test]
    fn test_encrypt_many_messages() {
        // Checks that encrypting many plaintext at the same time gives the correct output.
        let key: [u8; 16] = CIPHER_KEY.to_ne_bytes();
        let aes = Aes128::new(&GenericArray::from(key));
        let ciphertexts = aes_encrypt_many(
            PLAINTEXT, PLAINTEXT, PLAINTEXT, PLAINTEXT, PLAINTEXT, PLAINTEXT, PLAINTEXT, PLAINTEXT,
            &aes,
        );
        let ciphertexts: [u8; 128] = ciphertexts[..].try_into().unwrap();
        for i in 0..8 {
            assert_eq!(
                u128::from_ne_bytes(ciphertexts[16 * i..16 * (i + 1)].try_into().unwrap()),
                CIPHERTEXT
            );
        }
    }

    #[test]
    fn test_uniformity() {
        // Checks that the PRNG generates uniform numbers
        let precision = 10f64.powi(-4);
        let n_samples = 1_000_000_usize;
        let mut generator = Generator::new(None);
        let mut counts = [0usize; 256];
        let expected_prob: f64 = 1. / 256.;
        for counter in 0..n_samples {
            let batch = generator.generate_batch(AesCtr(counter as u128));
            for i in 0..128 {
                counts[batch[i] as usize] += 1;
            }
        }
        counts
            .iter()
            .map(|a| (*a as f64) / ((n_samples * 128) as f64))
            .for_each(|a| assert!((a - expected_prob) < precision))
    }
}
