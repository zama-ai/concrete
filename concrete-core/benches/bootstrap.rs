use criterion::{BenchmarkId, Criterion};
use itertools::iproduct;

use concrete_commons::dispersion::LogStandardDev;
use concrete_commons::numeric::{CastFrom, Numeric};
use concrete_commons::parameters::{GlweDimension, LweDimension, LweSize, PolynomialSize};

use concrete_core::crypto::bootstrap::{Bootstrap, FourierBootstrapKey};
use concrete_core::crypto::encoding::Plaintext;
use concrete_core::crypto::glwe::GlweCiphertext;
use concrete_core::crypto::lwe::LweCiphertext;
use concrete_core::crypto::secret::generators::{EncryptionRandomGenerator, SecretRandomGenerator};
use concrete_core::crypto::secret::LweSecretKey;
use concrete_core::math::decomposition::{DecompositionBaseLog, DecompositionLevelCount};
use concrete_core::math::fft::Complex64;
use concrete_core::math::tensor::AsMutTensor;
use concrete_core::math::torus::UnsignedTorus;

pub fn bench<T: UnsignedTorus + CastFrom<u64>>(c: &mut Criterion) {
    let lwe_dimensions = vec![512]; // 512;
    let l_gadgets = vec![1, 3, 10];
    let rlwe_dimensions = vec![1, 2, 3];
    let degrees = vec![1024];
    let params = iproduct!(lwe_dimensions, l_gadgets, rlwe_dimensions, degrees);
    let mut group = c.benchmark_group("compilo-bootstrap");
    let mut secret_generator = SecretRandomGenerator::new(None);
    let mut encryption_generator = EncryptionRandomGenerator::new(None);
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

                let lwe_sk = LweSecretKey::generate_binary(lwe_dimension, &mut secret_generator);

                let fourier_bsk = FourierBootstrapKey::allocate(
                    Complex64::new(0., 0.),
                    rlwe_dimension.to_glwe_size(),
                    polynomial_size,
                    level,
                    base_log,
                    lwe_dimension,
                );

                // msg to bootstrap
                let m0 = T::cast_from(
                    (2. / polynomial_size.0 as f64) * f64::powi(2., <T as Numeric>::BITS as i32),
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

                lwe_sk.encrypt_lwe(&mut lwe_in, &m0, std, &mut encryption_generator);

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
                    fourier_bsk.bootstrap(&mut lwe_out, &lwe_in, &accumulator);
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
