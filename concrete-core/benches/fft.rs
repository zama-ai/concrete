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

pub fn bench(c: &mut Criterion) {
    let degrees = vec![1024usize, 2048, 4096];

    let params = iproduct!(degrees);

    let mut group = c.benchmark_group("fft");
    for p in params {
        // group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("fft {}", p)),
            &p,
            |b, p| {
                // --------> all allocation
                let polynomial_size = PolynomialSize(*p);
                let first = FourierPolynomial::allocate(Complex64::new(0., 0.), polynomial_size);
                let second = FourierPolynomial::allocate(Complex64::new(0., 0.), polynomial_size);
                let mut output =
                    FourierPolynomial::allocate(Complex64::new(0., 0.), polynomial_size);

                // allocate secret keys
                b.iter(|| {
                    output.update_with_multiply_accumulate(&first, &second);
                });
            },
        );
    }
    group.finish();
}
