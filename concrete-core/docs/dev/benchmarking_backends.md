# Tutorial: How to benchmark your backend.

Once you've implemented and tested your backend, you're ready to start benchmarking it.
The `concrete-core-bench` crate has been developed for this purpose.

Let's first add the GPU backend as a feature for the benchmark crate. Edit the `Cargo.toml` file
of `concrete-core-bench` to add the following lines in the dependencies and features sections:

```
[features]
backend_gpu = ["concrete-core/backend_gpu"]
```

## Create a new module alongside the core one

Let's start by editing the `main.rs` file of `concrete-core-bench` to have:

```rust
#[cfg(feature = "backend_gpu")]
mod gpu;

// The main entry point. Uses criterion as benchmark harness.
fn main() {
    // We instantiate the benchmarks for different backends depending on the feature flag activated.
    #[cfg(feature = "backend_core")]
        core::bench();
    #[cfg(feature = "backend_gpu")]
        gpu::bench();

    // We launch the benchmarks.
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
```

and by creating a `gpu.rs` module in `concrete-core-bench/src`, that should contain:

```rust
use crate::benchmark::BenchmarkFixture;
use concrete_core::prelude::*;
use concrete_core_fixture::fixture::*;
use concrete_core_fixture::generation::{Maker, Precision32};
use criterion::Criterion;

pub fn bench_lwe_ciphertext_vector_conversion_32() {
    let mut criterion = Criterion::default().configure_from_args();
    let mut maker = Maker::default();
    let mut engine = GpuEngine::new().unwrap();
    <LweCiphertextVectorConversionFixture as BenchmarkFixture<Precision32, GpuEngine, (
        GpuLweCiphertextVector, LweCiphertextVector),
    >>::bench_all_parameters(
        &mut maker,
        engine,
        &mut criterion,
        None
    );
}
```

That's all you need to do!

## Launch the benchmark

Launching the command:

```
cargo run -p concrete-core-bench --features=backend_gpu,backend_core --release -- --bench Conversion
```

Should now yield:

```
Running `target/release/concrete-core-bench --bench Conversion`
impl LweCiphertextVectorConversionEngine<GpuLweCiphertextVector32,LweCiphertextVector32> for GpuEn...                                                                             
                        time:   [46.375 us 48.507 us 51.034 us]
Found 9 outliers among 100 measurements (9.00%)
  7 (7.00%) high mild
  2 (2.00%) high severe
impl LweCiphertextVectorConversionEngine<GpuLweCiphertextVector32,LweCiphertextVector32> for GpuEn... #2                                                                            
                        time:   [108.45 us 111.37 us 114.60 us]
Found 13 outliers among 100 measurements (13.00%)
  4 (4.00%) high mild
  9 (9.00%) high severe
...
```
