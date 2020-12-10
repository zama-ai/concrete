use criterion::{criterion_group, criterion_main, Benchmark, BenchmarkId, Criterion};
use itertools::iproduct;
use rand::Rng;

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
use concrete_core::numeric::{CastFrom, CastInto, Numeric};

pub fn bench<T: UnsignedTorus + CastFrom<u64>>(c: &mut Criterion) {
    let lwe_dimensions = vec![512]; // 512;
    let l_gadgets = vec![1, 3, 10];
    let rlwe_dimensions = vec![1, 2, 3];
    let degrees = vec![1024];
    let n_slots = 1;
    let base_log = 7;
    let std = f64::powi(2., -23);

    let params = iproduct!(lwe_dimensions, l_gadgets, rlwe_dimensions, degrees);

    let mut group = c.benchmark_group("compilo-bootstrap");
    for p in params {
        // group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!(
                "p={}-n={}-l={}-k={}-N={}",
                T::BITS,
                p.0,
                p.1,
                p.2,
                p.3
            )),
            &p,
            |b, p| {
                // --------> all allocation
                let polynomial_size = PolynomialSize(p.3);
                let rlwe_dimension = GlweDimension(p.2);
                let lwe_dimension = LweDimension(p.0);
                let level = DecompositionLevelCount(p.1);
                let base_log = DecompositionBaseLog(7);
                let std = LogStandardDev::from_log_standard_dev(-29.);

                // allocate secret keys
                let mut rlwe_sk = GlweSecretKey::generate(rlwe_dimension, polynomial_size);
                let mut lwe_sk = LweSecretKey::generate(lwe_dimension);

                let mut fourier_bsk = BootstrapKey::from_container(
                    vec![
                        Complex64::new(0., 0.);
                        rlwe_dimension.0
                            * (rlwe_dimension.0 + 1)
                            * polynomial_size.0
                            * level.0
                            * lwe_dimension.0
                            + polynomial_size.0
                                * level.0
                                * (rlwe_dimension.0 + 1)
                                * lwe_dimension.0
                    ],
                    rlwe_dimension.to_glwe_size(),
                    polynomial_size,
                    level,
                    base_log,
                );

                // msg to bootstrap
                let m0 = T::cast_from(
                    ((2. / polynomial_size.0 as f64) * f64::powi(2., <T as Numeric>::BITS as i32)),
                );
                let m0 = Plaintext(m0);
                let mut lwe_in = LweCiphertext::allocate(T::ZERO, lwe_dimension.to_lwe_size());
                let mut lwe_out = LweCiphertext::allocate(
                    T::ZERO,
                    LweSize(rlwe_dimension.0 * polynomial_size.0 + 1),
                );
                // accumulator is a trivial encryption of [0, 1/2N, 2/2N, ...]
                let mut accumulator = GlweCiphertext::allocate(
                    T::ZERO,
                    polynomial_size,
                    rlwe_dimension.to_glwe_size(),
                );

                lwe_sk.encrypt_lwe(&mut lwe_in, &m0, std);

                // fill accumulator
                for (i, elt) in accumulator
                    .get_mut_body()
                    .as_mut_tensor()
                    .iter_mut()
                    .enumerate()
                {
                    let val: u64 = (i as f64 / (2. * polynomial_size.0 as f64)
                        * f64::powi(2., <T as Numeric>::BITS as i32))
                    .round() as u64;

                    *elt = T::cast_from(val);
                }
                b.iter(|| {
                    bootstrap(&mut lwe_out, &lwe_in, &fourier_bsk, &mut accumulator);
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
