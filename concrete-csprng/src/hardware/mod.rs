#[cfg(target_arch = "x86_64")]
mod aesni;

#[cfg(target_arch = "x86_64")]
pub type Generator = aesni::Generator;

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod aesarm;

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub type Generator = aesarm::Generator;

#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
compile_error!("aarch64 / arm64 is only supported on macOS");

#[cfg(all(
    test,
    any(
        all(
            target_arch = "x86_64",
            target_feature = "aes",
            target_feature = "sse2",
            target_feature = "rdseed"
        ),
        all(
            target_arch = "aarch64",
            target_os = "macos",
            target_feature = "neon",
            target_feature = "aes"
        )
    )
))]
mod test {
    use super::Generator;
    use crate::counter::{AesBatchedGenerator, AesCtr};

    // Test vector for aes128, from the FIPS publication 197
    pub(super) const CIPHER_KEY: u128 = u128::from_be(0x000102030405060708090a0b0c0d0e0f);
    pub(super) const KEY_SCHEDULE: [u128; 11] = [
        u128::from_be(0x000102030405060708090a0b0c0d0e0f),
        u128::from_be(0xd6aa74fdd2af72fadaa678f1d6ab76fe),
        u128::from_be(0xb692cf0b643dbdf1be9bc5006830b3fe),
        u128::from_be(0xb6ff744ed2c2c9bf6c590cbf0469bf41),
        u128::from_be(0x47f7f7bc95353e03f96c32bcfd058dfd),
        u128::from_be(0x3caaa3e8a99f9deb50f3af57adf622aa),
        u128::from_be(0x5e390f7df7a69296a7553dc10aa31f6b),
        u128::from_be(0x14f9701ae35fe28c440adf4d4ea9c026),
        u128::from_be(0x47438735a41c65b9e016baf4aebf7ad2),
        u128::from_be(0x549932d1f08557681093ed9cbe2c974e),
        u128::from_be(0x13111d7fe3944a17f307a78b4d2b30c5),
    ];
    pub(super) const PLAINTEXT: u128 = u128::from_be(0x00112233445566778899aabbccddeeff);
    pub(super) const CIPHERTEXT: u128 = u128::from_be(0x69c4e0d86a7b0430d8cdb78070b4c55a);

    #[test]
    fn test_uniformity() {
        // Checks that the PRNG generates uniform numbers
        let precision = 10f64.powi(-4);
        let n_samples = 10_000_000_usize;
        let mut generator = Generator::try_new(None).unwrap();
        let mut counts = [0usize; 256];
        let expected_prob: f64 = 1. / 256.;
        for counter in 0..n_samples {
            let generated = generator.generate_batch(AesCtr(counter as u128));
            for i in 0..128 {
                counts[generated[i] as usize] += 1;
            }
        }
        counts
            .iter()
            .map(|a| (*a as f64) / ((n_samples * 128) as f64))
            .for_each(|a| assert!((a - expected_prob) < precision))
    }
}
