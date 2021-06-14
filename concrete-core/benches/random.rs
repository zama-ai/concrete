use criterion::{black_box, criterion_group, criterion_main, Benchmark, BenchmarkId, Criterion};
use itertools::iproduct;
use rand::Rng;

use concrete_commons::{CastFrom, CastInto, Numeric, UnsignedInteger};

use concrete_core::crypto::bootstrap::BootstrapKey;
use concrete_core::crypto::cross::{bootstrap, cmux, constant_sample_extract, external_product};
use concrete_core::crypto::encoding::{Plaintext, PlaintextList};
use concrete_core::crypto::glwe::{GlweCiphertext, GlweList};
use concrete_core::crypto::lwe::{LweCiphertext, LweKeyswitchKey};
use concrete_core::crypto::secret::{GlweSecretKey, LweSecretKey};
use concrete_core::crypto::{
    CiphertextCount, GlweDimension, LweDimension, LweSize, PlaintextCount, UnsignedTorus,
};
use concrete_core::math::dispersion::{DispersionParameter, LogStandardDev, Variance};
use concrete_core::math::fft::{Complex64, Fft, FourierPolynomial};
use concrete_core::math::polynomial::PolynomialSize;
use concrete_core::math::random::{
    fill_with_random_uniform, fill_with_random_uniform_boolean, random_uniform_n_msb,
    RandomGenerable, Uniform, UniformMsb,
};
use concrete_core::math::tensor::{
    AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, IntoTensor, Tensor,
};

pub fn bench<T: UnsignedInteger + RandomGenerable<Uniform>>(c: &mut Criterion) {
    let name = format!("random generate 100_000 u{}", T::BITS);
    c.bench_function(name.as_str(), |b| {
        b.iter(|| {
            let mut tensor: Tensor<Vec<T>> = Tensor::allocate(T::ZERO, black_box(100_000));
            black_box(fill_with_random_uniform(&mut tensor));
        })
    });
}

pub fn bench_8(c: &mut Criterion) {
    bench::<u8>(c);
}

pub fn bench_16(c: &mut Criterion) {
    bench::<u16>(c);
}

pub fn bench_32(c: &mut Criterion) {
    bench::<u32>(c);
}

pub fn bench_64(c: &mut Criterion) {
    bench::<u64>(c);
}

pub fn bench_128(c: &mut Criterion) {
    bench::<u128>(c);
}
