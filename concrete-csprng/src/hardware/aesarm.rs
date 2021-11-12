use crate::counter::{AesBatchedGenerator, AesCtr, AesKey};

use core::arch::aarch64::{
    uint8x16_t, vaeseq_u8, vaesmcq_u8, vdupq_n_u32, vdupq_n_u8, veorq_u8, vgetq_lane_u32,
    vreinterpretq_u32_u8, vreinterpretq_u8_u32,
};
use std::mem::transmute;

#[derive(Clone)]
pub struct Generator {
    round_keys: [uint8x16_t; Self::NUM_ROUND_KEYS],
}

impl Generator {
    // We supports only 128 bit key, which has 10 AES rounds
    const NUM_ROUNDS: usize = 10;
    const NUM_ROUND_KEYS: usize = Self::NUM_ROUNDS + 1;
    const KEY_SIZE_IN_BYTES: usize = 128 / 8;
    // 32 bit word size
    const WORD_SIZE: usize = 4;
    const NUM_WORDS_IN_KEY: usize = Self::KEY_SIZE_IN_BYTES / Self::WORD_SIZE;

    // Values for the AES AesKeyExtension step
    const RCONS: [u32; 10] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36];

    /// Creates a new Generator from a AesKey
    /// This does the AesKeyExtension
    ///
    /// # SAFETY
    ///
    /// You must make sure the CPU's arch is`aarch64` and has
    /// `neon` and `aes` features.
    unsafe fn with_key(key: AesKey) -> Self {
        debug_assert_eq!(Self::NUM_WORDS_IN_KEY, 4);

        let mut round_keys: [uint8x16_t; Self::NUM_ROUND_KEYS] = std::mem::zeroed();
        round_keys[0] = transmute(key.0);

        let words = std::slice::from_raw_parts_mut(
            round_keys.as_mut_ptr() as *mut u32,
            Self::NUM_ROUND_KEYS * Self::NUM_WORDS_IN_KEY,
        );

        debug_assert_eq!(words.len(), 44);
        // Skip the words of the first key, its already done
        for i in Self::NUM_WORDS_IN_KEY..words.len() {
            if (i % Self::NUM_WORDS_IN_KEY) == 0 {
                words[i] = words[i - Self::NUM_WORDS_IN_KEY]
                    ^ sub_word(words[i - 1]).rotate_right(8)
                    ^ Self::RCONS[(i / Self::NUM_WORDS_IN_KEY) - 1];
            } else {
                words[i] = words[i - Self::NUM_WORDS_IN_KEY] ^ words[i - 1];
            }
            // Note: there is also a special thing to do when
            // i mod SElf::NUM_WORDS_IN_KEY == 4 but it cannot happen on 128 bits keys
        }

        Self { round_keys }
    }

    // Encrypts a 128-bit message
    ///
    /// # SAFETY
    ///
    /// You must make sure the CPU's arch is`aarch64` and has
    /// `neon` and `aes` features.
    unsafe fn encrypt(&self, message: u128) -> u128 {
        // Notes:
        // According the [ARM Manual](https://developer.arm.com/documentation/ddi0487/gb/):
        // `vaeseq_u8` is the following AES operations:
        //      1. AddRoundKey (XOR)
        //      2. ShiftRows
        //      3. SubBytes
        // `vaesmcq_u8` is MixColumns
        let mut data: uint8x16_t = transmute(message);

        for i in 0..Self::NUM_ROUNDS - 1 {
            data = vaesmcq_u8(vaeseq_u8(data, self.round_keys[i]));
        }

        data = vaeseq_u8(data, self.round_keys[Self::NUM_ROUNDS - 1]);
        data = veorq_u8(data, self.round_keys[Self::NUM_ROUND_KEYS - 1]);

        transmute(data)
    }
}

impl AesBatchedGenerator for Generator {
    fn new(key: Option<AesKey>) -> Self {
        Self::try_new(key).expect("CPU does not support 'neon' and/or 'aes' instructions")
    }

    fn try_new(key: Option<AesKey>) -> Option<Self> {
        if is_aarch64_feature_detected!("neon") && is_aarch64_feature_detected!("aes") {
            let key = key.unwrap_or_else(generate_random_key);
            unsafe { Some(Self::with_key(key)) }
        } else {
            None
        }
    }

    fn generate_batch(&mut self, ctr: AesCtr) -> [u8; 128] {
        let mut output = [0u8; 128];
        // We want 128 bytes of output, the ctr gives 128 bit message (16 bytes)
        for (i, out) in output.chunks_exact_mut(16).enumerate() {
            let encrypted = unsafe {
                // Safe because we prevent the user from creating the Generator
                // on non-supported hardware
                self.encrypt(ctr.0 + i as u128)
            };
            out.copy_from_slice(&encrypted.to_ne_bytes());
        }
        output
    }
}

/// Does the AES SubWord operation for the Key Expansion step
///
/// # SAFETY
///
/// You must make sure the CPU's arch is`aarch64` and has
/// `neon` and `aes` features.
unsafe fn sub_word(word: u32) -> u32 {
    let data = vreinterpretq_u8_u32(vdupq_n_u32(word));
    let zero_key = vdupq_n_u8(0u8);
    let temp = vaeseq_u8(data, zero_key);
    // vaeseq_u8 does SubBytes(ShiftRow(XOR(data, key))
    // But because we used a zero aes key,the XOR did not alter data
    // We now have temp = SubBytes(ShiftRow(data))

    // Since in AES ShiftRow operation, the first row is not shifted
    // We can just get that one to have our SubWord(word) result
    vgetq_lane_u32::<0>(vreinterpretq_u32_u8(temp))
}

fn generate_random_key() -> AesKey {
    let mut bytes = [0u8; std::mem::size_of::<u128>()];
    secure_enclave::generate_random_bytes(&mut bytes)
        .expect("Failed to generate random bytes using `SecRandomCopyBytes");
    AesKey(u128::from_le_bytes(bytes))
}

/// There is no `rseed` equivalent in the ARM specification until `ARMv8.5-A`.
/// However it seems that these instructions are not exposed in `core::arch::aarch64`.
///
/// Our primary interest for supporting aarch64 targets is AppleSilicon support
/// which for the M1 macs available, they are based on the `ARMv8.4-A` set.
///
/// So we fall back to using a function from Apple's API which
/// uses the [Secure Enclave] to generate cryptographically secure random bytes.
///
/// [Secure Enclave]: https://support.apple.com/fr-fr/guide/security/sec59b0b31ff/web
mod secure_enclave {
    pub enum __SecRandom {}
    pub type SecRandomRef = *const __SecRandom;
    use libc::{c_int, c_void};

    #[link(name = "Security", kind = "framework")]
    extern "C" {
        pub static kSecRandomDefault: SecRandomRef;

        pub fn SecRandomCopyBytes(rnd: SecRandomRef, count: usize, bytes: *mut c_void) -> c_int;
    }

    pub fn generate_random_bytes(bytes: &mut [u8]) -> std::io::Result<()> {
        // As per Apple's documentation:
        // - https://developer.apple.com/documentation/security/randomization_services?language=objc
        // - https://developer.apple.com/documentation/security/1399291-secrandomcopybytes?language=objc
        //
        // The `SecRandomCopyBytes` "Generate cryptographically secure random numbers"
        unsafe {
            let res = SecRandomCopyBytes(
                kSecRandomDefault,
                bytes.len(),
                bytes.as_mut_ptr() as *mut c_void,
            );
            if res != 0 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        }
    }
}

#[cfg(all(
    test,
    target_arch = "aarch64",
    target_os = "macos",
    target_feature = "neon",
    target_feature = "aes"
))]
mod test {
    use super::*;
    use crate::hardware::test::{CIPHERTEXT, CIPHER_KEY, KEY_SCHEDULE, PLAINTEXT};
    use std::mem::transmute;

    #[test]
    fn test_generate_key_schedule() {
        // Checks that the round keys are correctly generated from the sample key from FIPS
        // let key = vld1q_u8(CIPHER_KEY as *const u8);
        let generator = super::Generator::try_new(Some(AesKey(CIPHER_KEY))).unwrap();
        for (i, (expected, actual)) in KEY_SCHEDULE
            .iter()
            .zip(generator.round_keys.iter())
            .enumerate()
        {
            let actual: u128 = unsafe { transmute(*actual) };
            assert_eq!(
                *expected, actual,
                "Keys {} not equals, expected {} got {}",
                i, expected, actual
            );
        }
    }

    #[test]
    fn test_encrypt_message() {
        let message = PLAINTEXT;
        let key = AesKey(CIPHER_KEY);

        let generator = super::Generator::try_new(Some(key)).unwrap();
        let cipher_text = unsafe { generator.encrypt(message) };
        assert_eq!(cipher_text, CIPHERTEXT);
    }
}
