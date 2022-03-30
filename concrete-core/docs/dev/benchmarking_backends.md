# Tutorial: How to benchmark your backend.

Once you've implemented and tested your backend, you're ready to start benchmarking it.
The `concrete-benchmark` crate has been developed for this purpose. Unfortunately, it does not rely
on the `concrete-core-fixture` crate yet, but it will in the future, so this tutorial will be even
simpler once you've followed the test tutorial.

Let's first add the GPU backend as a feature for the benchmark crate. Edit the `Cargo.toml` file
of `concrete-benchmark` to add the following lines in the dependencies and features sections:

```
[dependencies]
fhe_gpu = { version="0.0.1", optional = true }

[features]
backend_cuda = ["concrete-core/backend_gpu", "fhe_gpu"]
```

## Create a new module alongside the core one

Let's start by editing the `main.rs` file of `concrete-benchmark` to have:

```
# [cfg(feature = "backend_core")]
backends::core::bench();

# [cfg(feature = "backend_gpu")]
backends::gpu::bench();

// We launch the benchmarks.
criterion::Criterion::default ().configure_from_args()
```

and by creating a `gpu.rs` module in `concrete-benchmark/backends`, that should contain:

```rust
//! A module benchmarking the `gpu` backend of `concrete_core`.
use concrete_core::prelude::*;
use criterion::Criterion;

#[rustfmt::skip]
pub fn bench() {
    use crate::generics::*;
    let mut criterion = Criterion::default().configure_from_args();
    lwe_ciphertext_vector_conversion::bench::<CudaEngine,
        CudaLweCiphertextVector32, LweCiphertextVector32>(&mut criterion);
}
```

Edit the `concrete-benchmark/src/backends/mod.rs` file to link it with the following lines:

```rust
#[cfg(feature = "backend_gpu")]
pub mod gpu;
```

## Edit the `Synthesizer`

We now need to edit the synthesizer. The structure `Synthesizer` itself should have a `gpu_engine`
attribute, and it should be implemented in the default implementation:

```rust
/// A type containing all the necessary engines needed to generate any entity.
pub struct Synthesizer {
    #[cfg(feature = "backend_core")]
    core_engine: concrete_core::backends::core::engines::CoreEngine,
    #[cfg(feature = "backend_gpu")]
    gpu_engine: concrete_core::backends::gpu::engines::GpuEngine,
}

impl Default for Synthesizer {
    fn default() -> Self {
        Synthesizer {
            #[cfg(feature = "backend_core")]
            core_engine: concrete_core::backends::core::engines::CoreEngine::new().unwrap(),
            #[cfg(feature = "backend_gpu")]
            gpu_engine: concrete_core::backends::gpu::engines::GpuEngine::new().unwrap(),
        }
    }
}
```

Then, you can add a `gpu` module that implements `synthesize` in the following way:

```rust
#[cfg(feature = "backend_gpu")]
mod gpu {
    use super::*;
    use concrete_commons::dispersion::Variance;
    use concrete_commons::parameters::{
        LweCiphertextCount, LweDimension,
    };

    impl SynthesizableLweCiphertextVectorEntity for GpuLweCiphertextVector32 {
        fn synthesize(
            synthesizer: &mut Synthesizer,
            lwe_dimension: LweDimension,
            count: LweCiphertextCount,
            noise: Variance,
        ) -> Self {
            let lwe_sk = synthesizer
                .core_engine
                .create_lwe_secret_key(lwe_dimension)
                .unwrap();
            let ciphertext_vector = synthesizer
                .core_engine
                .zero_encrypt_lwe_ciphertext_vector(&lwe_sk, noise, count)
                .unwrap();
            synthesizer
                .gpu_engine
                .convert_lwe_ciphertext_vector(&ciphertext_vector)
                .unwrap()
        }
    }
}
```

## Create a new module in `generics`

The module for your specific engine is very likely already existing, in which case you can skip this
step. If not, here's what the conversion module should look like:

```rust
use crate::synthesizer::{
    SynthesizableLweCiphertextVectorEntity,
    Synthesizer,
};
use crate::utils::benchmark_name;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{LweCiphertextCount, LweDimension};
use concrete_core::specification::engines::LweCiphertextVectorConversionEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the lwe vector conversion operation.
pub fn bench<Engine, InputCiphertextVector, OutputCiphertextVector>(c: &mut Criterion)
    where
        Engine: LweCiphertextVectorConversionEngine<InputCiphertextVector, OutputCiphertextVector>,
        InputCiphertextVector:
        SynthesizableLweCiphertextVectorEntity,
        OutputCiphertextVector:
        SynthesizableLweCiphertextVectorEntity<KeyDistribution=InputCiphertextVector::KeyDistribution>,
{
    let mut group = c.benchmark_group(benchmark_name!(impl LweCiphertextVectorConversionEngine<
            InputCiphertextVector,
            OutputCiphertextVector
            > for Engine));

    let mut engine = Engine::new().unwrap();
    let mut synthesizer = Synthesizer::default();

    for param in PARAMETERS {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", param)),
            &param,
            |b, param| {
                let (lwe_dim, ciphertext_count) = param.to_owned();
                let input_ciphertext_vector = InputCiphertextVector::synthesize(&mut synthesizer, lwe_dim, ciphertext_count, VARIANCE);
                b.iter(|| {
                    black_box(
                        engine
                            .convert_lwe_ciphertext_vector(
                                black_box(&input_ciphertext_vector),
                            )
                            .unwrap(),
                    );
                });
            },
        );
    }
    group.finish();
}

/// The variance used to encrypt everything in the benchmark.
const VARIANCE: Variance = Variance(0.00000001);

/// The parameters the benchmark is executed against.
const PARAMETERS: [(LweDimension, LweCiphertextCount); 6] = [
    (LweDimension(100), LweCiphertextCount(100)),
    (LweDimension(300), LweCiphertextCount(100)),
    (LweDimension(600), LweCiphertextCount(100)),
    (LweDimension(1000), LweCiphertextCount(100)),
    (LweDimension(3000), LweCiphertextCount(100)),
    (LweDimension(6000), LweCiphertextCount(100)),
];
```

## Launch the benchmark

Launching the command:

```
cargo run -p concrete-benchmark --features=backend_gpu --release -- --bench LweCiphertextVectorConversion
```

Should now yield:

```
Running `target/release/concrete-benchmark --bench Conversion`
impl LweCiphertextVectorConversionEngine<CudaLweCiphertextVector32,LweCiphertextVector32> for CudaEn...                                                                             
                        time:   [46.375 us 48.507 us 51.034 us]
Found 9 outliers among 100 measurements (9.00%)
  7 (7.00%) high mild
  2 (2.00%) high severe
impl LweCiphertextVectorConversionEngine<CudaLweCiphertextVector32,LweCiphertextVector32> for CudaEn... #2                                                                            
                        time:   [108.45 us 111.37 us 114.60 us]
Found 13 outliers among 100 measurements (13.00%)
  4 (4.00%) high mild
  9 (9.00%) high severe
...
```
