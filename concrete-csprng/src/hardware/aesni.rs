//! A module implementing an `aes128-counter` random number generator, using `aesni` instructions.
//!
//! This module implements a cryptographically secure pseudorandom number generator
//! (CS-PRNG), using a fast streamcipher: aes128 in counter-mode (CTR). The implementation
//! is based on the [intel aesni white paper 323641-001 revision 3.0](https://www.intel.com/content/dam/doc/white-paper/advanced-encryption-standard-new-instructions-set-paper.pdf).
use crate::counter::{AesBatchedGenerator, AesCtr, AesKey};
use std::arch::x86_64::{
    __m128i, _mm_aesenc_si128, _mm_aesenclast_si128, _mm_aeskeygenassist_si128, _mm_load_si128,
    _mm_shuffle_epi32, _mm_slli_si128, _mm_store_si128, _mm_xor_si128,
};
use std::mem::transmute;

#[derive(Clone)]
pub struct Generator {
    // The set of round keys used for the aes encryption
    round_keys: [__m128i; 11],
}

impl AesBatchedGenerator for Generator {
    fn new(key: Option<AesKey>) -> Generator {
        Self::try_new(key).expect(
            "One of the `aes`, `rdseed`, or `sse2` instructions set was not fount. It is \
                 currently mandatory to use `concrete-csprng`.",
        )
    }

    fn try_new(key: Option<AesKey>) -> Option<Self> {
        if is_x86_feature_detected!("aes")
            && is_x86_feature_detected!("rdseed")
            && is_x86_feature_detected!("sse2")
        {
            let round_keys =
                generate_round_keys(key.unwrap_or_else(generate_initialization_vector));
            Some(Generator { round_keys })
        } else {
            None
        }
    }

    fn generate_batch(&mut self, AesCtr(aes_ctr): AesCtr) -> [u8; 128] {
        si128arr_to_u8arr(aes_encrypt_many(
            &u128_to_si128(aes_ctr),
            &u128_to_si128(aes_ctr + 1),
            &u128_to_si128(aes_ctr + 2),
            &u128_to_si128(aes_ctr + 3),
            &u128_to_si128(aes_ctr + 4),
            &u128_to_si128(aes_ctr + 5),
            &u128_to_si128(aes_ctr + 6),
            &u128_to_si128(aes_ctr + 7),
            &self.round_keys,
        ))
    }
}

fn generate_initialization_vector() -> AesKey {
    // The initialization vector is a random value from rdseed
    AesKey(si128_to_u128(rdseed_random_m128()))
}

fn generate_round_keys(key: AesKey) -> [__m128i; 11] {
    // The secret key is a random value from rdseed.
    let key = u128_to_si128(key.0);
    let mut keys: [__m128i; 11] = [u128_to_si128(0); 11];
    aes_128_key_expansion(key, &mut keys);
    keys
}

// Generates a random 128 bits value from rdseed
fn rdseed_random_m128() -> __m128i {
    let mut rand1: u64 = 0;
    let mut rand2: u64 = 0;
    unsafe {
        loop {
            if core::arch::x86_64::_rdseed64_step(&mut rand1) == 1 {
                break;
            }
        }
        loop {
            if core::arch::x86_64::_rdseed64_step(&mut rand2) == 1 {
                break;
            }
        }
        std::mem::transmute::<(u64, u64), __m128i>((rand1, rand2))
    }
}

// Uses aes to encrypt many values at once. This allows a substantial speedup (around 30%)
// compared to the naive approach.
#[allow(clippy::too_many_arguments)]
fn aes_encrypt_many(
    message_1: &__m128i,
    message_2: &__m128i,
    message_3: &__m128i,
    message_4: &__m128i,
    message_5: &__m128i,
    message_6: &__m128i,
    message_7: &__m128i,
    message_8: &__m128i,
    keys: &[__m128i; 11],
) -> [__m128i; 8] {
    unsafe {
        let message_1 = _mm_load_si128(message_1 as *const __m128i);
        let message_2 = _mm_load_si128(message_2 as *const __m128i);
        let message_3 = _mm_load_si128(message_3 as *const __m128i);
        let message_4 = _mm_load_si128(message_4 as *const __m128i);
        let message_5 = _mm_load_si128(message_5 as *const __m128i);
        let message_6 = _mm_load_si128(message_6 as *const __m128i);
        let message_7 = _mm_load_si128(message_7 as *const __m128i);
        let message_8 = _mm_load_si128(message_8 as *const __m128i);

        let mut tmp_1 = _mm_xor_si128(message_1, keys[0]);
        let mut tmp_2 = _mm_xor_si128(message_2, keys[0]);
        let mut tmp_3 = _mm_xor_si128(message_3, keys[0]);
        let mut tmp_4 = _mm_xor_si128(message_4, keys[0]);
        let mut tmp_5 = _mm_xor_si128(message_5, keys[0]);
        let mut tmp_6 = _mm_xor_si128(message_6, keys[0]);
        let mut tmp_7 = _mm_xor_si128(message_7, keys[0]);
        let mut tmp_8 = _mm_xor_si128(message_8, keys[0]);

        for key in keys.iter().take(10).skip(1) {
            tmp_1 = _mm_aesenc_si128(tmp_1, *key);
            tmp_2 = _mm_aesenc_si128(tmp_2, *key);
            tmp_3 = _mm_aesenc_si128(tmp_3, *key);
            tmp_4 = _mm_aesenc_si128(tmp_4, *key);
            tmp_5 = _mm_aesenc_si128(tmp_5, *key);
            tmp_6 = _mm_aesenc_si128(tmp_6, *key);
            tmp_7 = _mm_aesenc_si128(tmp_7, *key);
            tmp_8 = _mm_aesenc_si128(tmp_8, *key);
        }

        tmp_1 = _mm_aesenclast_si128(tmp_1, keys[10]);
        tmp_2 = _mm_aesenclast_si128(tmp_2, keys[10]);
        tmp_3 = _mm_aesenclast_si128(tmp_3, keys[10]);
        tmp_4 = _mm_aesenclast_si128(tmp_4, keys[10]);
        tmp_5 = _mm_aesenclast_si128(tmp_5, keys[10]);
        tmp_6 = _mm_aesenclast_si128(tmp_6, keys[10]);
        tmp_7 = _mm_aesenclast_si128(tmp_7, keys[10]);
        tmp_8 = _mm_aesenclast_si128(tmp_8, keys[10]);

        [tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, tmp_7, tmp_8]
    }
}

fn aes_128_assist(temp1: __m128i, temp2: __m128i) -> __m128i {
    let mut temp3: __m128i;
    let mut temp2 = temp2;
    let mut temp1 = temp1;
    unsafe {
        temp2 = _mm_shuffle_epi32(temp2, 0xff);
        temp3 = _mm_slli_si128(temp1, 0x4);
        temp1 = _mm_xor_si128(temp1, temp3);
        temp3 = _mm_slli_si128(temp3, 0x4);
        temp1 = _mm_xor_si128(temp1, temp3);
        temp3 = _mm_slli_si128(temp3, 0x4);
        temp1 = _mm_xor_si128(temp1, temp3);
        temp1 = _mm_xor_si128(temp1, temp2);
    }
    temp1
}

fn aes_128_key_expansion(key: __m128i, keys: &mut [__m128i; 11]) {
    let (mut temp1, mut temp2): (__m128i, __m128i);
    temp1 = key;
    unsafe {
        _mm_store_si128(keys.as_mut_ptr(), temp1);
        temp2 = _mm_aeskeygenassist_si128(temp1, 0x01);
        temp1 = aes_128_assist(temp1, temp2);
        _mm_store_si128(keys.as_mut_ptr().offset(1), temp1);
        temp2 = _mm_aeskeygenassist_si128(temp1, 0x02);
        temp1 = aes_128_assist(temp1, temp2);
        _mm_store_si128(keys.as_mut_ptr().offset(2), temp1);
        temp2 = _mm_aeskeygenassist_si128(temp1, 0x04);
        temp1 = aes_128_assist(temp1, temp2);
        _mm_store_si128(keys.as_mut_ptr().offset(3), temp1);
        temp2 = _mm_aeskeygenassist_si128(temp1, 0x08);
        temp1 = aes_128_assist(temp1, temp2);
        _mm_store_si128(keys.as_mut_ptr().offset(4), temp1);
        temp2 = _mm_aeskeygenassist_si128(temp1, 0x10);
        temp1 = aes_128_assist(temp1, temp2);
        _mm_store_si128(keys.as_mut_ptr().offset(5), temp1);
        temp2 = _mm_aeskeygenassist_si128(temp1, 0x20);
        temp1 = aes_128_assist(temp1, temp2);
        _mm_store_si128(keys.as_mut_ptr().offset(6), temp1);
        temp2 = _mm_aeskeygenassist_si128(temp1, 0x40);
        temp1 = aes_128_assist(temp1, temp2);
        _mm_store_si128(keys.as_mut_ptr().offset(7), temp1);
        temp2 = _mm_aeskeygenassist_si128(temp1, 0x80);
        temp1 = aes_128_assist(temp1, temp2);
        _mm_store_si128(keys.as_mut_ptr().offset(8), temp1);
        temp2 = _mm_aeskeygenassist_si128(temp1, 0x1b);
        temp1 = aes_128_assist(temp1, temp2);
        _mm_store_si128(keys.as_mut_ptr().offset(9), temp1);
        temp2 = _mm_aeskeygenassist_si128(temp1, 0x36);
        temp1 = aes_128_assist(temp1, temp2);
        _mm_store_si128(keys.as_mut_ptr().offset(10), temp1);
    }
}

fn u128_to_si128(input: u128) -> __m128i {
    unsafe { transmute(input) }
}

fn si128_to_u128(input: __m128i) -> u128 {
    unsafe { transmute(input) }
}

fn si128arr_to_u8arr(input: [__m128i; 8]) -> [u8; 128] {
    unsafe { transmute(input) }
}

#[cfg(all(
    test,
    target_arch = "x86_64",
    target_feature = "aes",
    target_feature = "sse2",
    target_feature = "rdseed"
))]
mod test {
    use super::*;
    use crate::hardware::test::{CIPHERTEXT, CIPHER_KEY, KEY_SCHEDULE, PLAINTEXT};

    #[test]
    fn test_generate_key_schedule() {
        // Checks that the round keys are correctly generated from the sample key from FIPS
        let key = u128_to_si128(CIPHER_KEY);
        let mut keys: [__m128i; 11] = [u128_to_si128(0); 11];
        aes_128_key_expansion(key, &mut keys);
        for (expected, actual) in KEY_SCHEDULE.iter().zip(keys.iter()) {
            assert_eq!(*expected, si128_to_u128(*actual));
        }
    }

    #[test]
    fn test_encrypt_many_messages() {
        // Checks that encrypting many plaintext at the same time gives the correct output.
        let message = u128_to_si128(PLAINTEXT);
        let key = u128_to_si128(CIPHER_KEY);
        let mut keys: [__m128i; 11] = [u128_to_si128(0); 11];
        aes_128_key_expansion(key, &mut keys);
        let ciphertexts = aes_encrypt_many(
            &message, &message, &message, &message, &message, &message, &message, &message, &keys,
        );
        for ct in &ciphertexts {
            assert_eq!(CIPHERTEXT, si128_to_u128(*ct));
        }
    }
}
