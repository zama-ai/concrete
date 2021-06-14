use criterion::{criterion_group, criterion_main, Benchmark, BenchmarkId, Criterion};
use itertools::iproduct;
use rand::Rng;

use concrete_commons::{CastFrom, CastInto, Numeric};

use concrete_core::crypto::bootstrap::BootstrapKey;
use concrete_core::crypto::cross::{bootstrap, cmux, constant_sample_extract, external_product};
use concrete_core::crypto::encoding::{Plaintext, PlaintextList};
use concrete_core::crypto::glwe::{GlweCiphertext, GlweList};
use concrete_core::crypto::lwe::{LweCiphertext, LweKeyswitchKey};
use concrete_core::crypto::secret::{GlweSecretKey, LweSecretKey};
use concrete_core::crypto::{
    CiphertextCount, GlweDimension, LweDimension, LweSize, PlaintextCount, UnsignedTorus,
};
use concrete_core::math::decomposition::{DecompositionBaseLog, DecompositionLevelCount};
use concrete_core::math::dispersion::{DispersionParameter, LogStandardDev, Variance};
use concrete_core::math::fft::{Complex64, Fft, FourierPolynomial};
use concrete_core::math::polynomial::PolynomialSize;
use concrete_core::math::random::{
    fill_with_random_uniform, fill_with_random_uniform_boolean, random_uniform_n_msb,
    RandomGenerable, UniformMsb,
};
use concrete_core::math::tensor::{
    AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, IntoTensor, Tensor,
};

pub fn bench<T: UnsignedTorus + RandomGenerable<UniformMsb>>(c: &mut Criterion) {
    // fix a set of parameters
    let base_log_level = vec![(4, 3), (6, 2), (10, 1)]; // a parameter of the gadget matrix
    let dimensions_before = vec![1024];
    let dimensions_after = vec![512];

    let n_bit_msg = 8; // bit precision of the plaintext
    let std_input = f64::powi(2., -10); // standard deviation of the encrypted messages to KS
    let std_ksk = f64::powi(2., -25); // standard deviation of the ksk

    let params = iproduct!(base_log_level, dimensions_before, dimensions_after);

    let mut group = c.benchmark_group("compilo-keyswitch");
    for p in params {
        // group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!(
                "p={}-bg={}-l={}-nin={}-nout={}",
                T::BITS,
                (p.0).0,
                (p.0).1,
                p.1,
                p.2
            )),
            &p,
            |b, p| {
                let base_log = DecompositionBaseLog((p.0).0);
                let level = DecompositionLevelCount((p.0).1);
                let dimension_before = LweDimension(p.1);
                let dimension_after = LweDimension(p.2);

                // --------> all allocation
                // fill the messages with random Torus element set to zeros but the n_bit_msg MSB uniformly picked
                let mut messages = Plaintext(T::ZERO); // the message to encrypt
                                                       // messages.0 = random_uniform_n_msb(n_bit_msg);

                // create and fill the before and the after keys with random bits
                // let mut sk_before = LweSecretKey::generate(dimension_before);
                // let mut sk_after = LweSecretKey::generate(dimension_after);
                let mut sk_before = LweSecretKey::from_container(vec![T::ZERO; dimension_before.0]);
                let mut sk_after = LweSecretKey::from_container(vec![T::ZERO; dimension_after.0]);

                // create the before ciphertexts and the after ciphertexts
                let mut ciphertexts_before =
                    LweCiphertext::allocate(T::ZERO, dimension_before.to_lwe_size());
                let mut ciphertexts_after =
                    LweCiphertext::allocate(T::ZERO, dimension_after.to_lwe_size());

                // key switching key generation
                let mut ksk = LweKeyswitchKey::allocate(
                    T::ZERO,
                    level,
                    base_log,
                    dimension_before,
                    dimension_after,
                );
                //ksk.fill_with_keyswitch_key(&sk_before, &sk_after, LogStandardDev(-25.));

                //sk_before.encrypt_lwe(&mut ciphertexts_before, &messages, LogStandardDev(-30.));

                b.iter(|| {
                    // key switch before -> after
                    ksk.keyswitch_ciphertext(&mut ciphertexts_after, &ciphertexts_before);
                });
            },
        );
    }
    group.finish();
}

pub fn bench_32(c: &mut Criterion) {
    bench::<u32>(c);
}

pub fn bench_64(c: &mut Criterion) {
    bench::<u64>(c);
}
