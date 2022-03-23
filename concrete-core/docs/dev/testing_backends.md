# Tutorial: how to test your backend?

Once you've implemented your backend, the first thing you need to do is to test it.
The `concrete-core-test` crate has been developed for this purpose. It relies on
the `concrete-core-fixture` crate, that implements generic functions to sample and test engines.

Let's continue with our GPU backend example. We now have a `GpuEngine` that implements a conversion
engine for LWE ciphertext vectors from the CPU to the GPU, and back. For this engine, we can easily
check that the ciphertext copied back from the GPU is identical to the original one on the CPU.
However, for more complex engines like the keyswitch, the bootstrap, etc., we need to make sure that
the amount of noise introduced by the operation corresponds to what's expected, i.e. that
it matches the noise formula implemented in the `concrete-npe` crate. For the sake of this tutorial,
let us continue with the simple conversion engines that copy data back and forth between the CPU and
the GPU, and implement this verification.

For this, we're going to use the available fixture for LWE ciphertext vector conversion. The only
thing we need to implement in `concrete-core-fixture` is the synthesis stage, where data will be
copied to the GPU, and back. Then we'll use the existing fixture for LWE ciphertext vector
conversion to execute the test.

## Add the GPU backend in the fixtures

Let's first add the GPU backend as a feature for the fixtures: edit the `Cargo.toml` file
of `concrete-core-fixture` to add the following lines in the dependencies and features sections:

```
[dependencies]
fhe_gpu = { version="0.0.1", optional = true }

[features]
backend_cuda = ["concrete-core/backend_gpu", "fhe_gpu"]
```

Then, we need to add the `GpuEngine` to the `Maker` structure that's defined
in `concrete-core-fixture/src/generation/mod.rs`:

```rust
pub struct Maker {
    core_engine: concrete_core::backends::core::engines::CoreEngine,
    #[cfg(feature = "backend_gpu")]
    gpu_engine: concrete_core::backends::gpu::engines::GpuEngine,
}

impl Default for Maker {
    fn default() -> Self {
        Maker {
            core_engine: concrete_core::backends::core::engines::CoreEngine::new().unwrap(),
            #[cfg(feature = "backend_gpu")]
            gpu_engine: concrete_core::backends::gpu::engines::GpuEngine::new().unwrap(),
        }
    }
}
```

Now, in `concrete-core-fixture/src/generation/synthesizing/lwe_ciphertext_vector.rs`, let us
introduce the necessary implementations to copy data to the GPU, retrieve and destroy it:

```rust
#[cfg(feature = "backend_gpu")]
mod backend_gpu {
    use crate::generation::prototypes::{
        ProtoBinaryLweCiphertextVector32,
    };
    use crate::generation::synthesizing::SynthesizesLweCiphertextVector;
    use crate::generation::{Maker, Precision32};
    use concrete_core::prelude::{
        GpuLweCiphertextVector32, DestructionEngine,
        LweCiphertextVectorConversionEngine,
    };

    impl SynthesizesLweCiphertextVector<Precision32, GpuLweCiphertextVector32> for Maker {
        fn synthesize_lwe_ciphertext_vector(
            &mut self,
            prototype: &Self::LweCiphertextVectorProto,
        ) -> GpuLweCiphertextVector32 {
            self.gpu_engine
                .convert_lwe_ciphertext_vector(&prototype.0)
                .unwrap()
        }
        fn unsynthesize_lwe_ciphertext_vector(
            &mut self,
            entity: &GpuLweCiphertextVector32,
        ) -> Self::LweCiphertextVectorProto {
            let proto = self
                .gpu_engine
                .convert_lwe_ciphertext_vector(entity)
                .unwrap();
            ProtoBinaryLweCiphertextVector32(proto)
        }
        fn destroy_lwe_ciphertext_vector(&mut self, entity: GpuLweCiphertextVector32) {
            self.gpu_engine.destroy(entity).unwrap();
        }
    }
}
```

That's all we need to do on the fixtures side.

## Add the test in `concrete-core-test`

Now, let's add our test in `concrete-core-test`. Let's first edit the `Cargo.toml` to add a
dependency to our `fhe_gpu` crate, and a GPU feature:

```
[dependencies]
concrete-core = { path="../concrete-core" }
concrete-core-fixture = { path="../concrete-core-fixture" }
fhe-gpu = { version = "0.0.1", optional = true }
paste = "1.0"

[features]
backend_core = ["concrete-core/backend_core", "concrete-core-fixture/backend_core"]
backend_gpu = ["concrete-core/backend_gpu", "concrete-core-fixture/backend_gpu"]
```

Let's add a `cuda.rs` module to `concrete-core-test`. Create the file `gpu.rs`
in `concrete-core-test/src`
and edit `cocnrete-core-test/src/lib.rs` to add the following lines:

```rust
#[cfg(all(test, feature = "backend_gpu"))]
pub mod gpu;
```

The `gpu.rs` module should contain:

```rust
use crate::{REPETITIONS, SAMPLE_SIZE};
use concrete_core::prelude::*;
use concrete_core_fixture::fixture::*;
use concrete_core_fixture::generation::{Maker, Precision32};
use paste::paste;

macro_rules! test {
    ($fixture: ident, $precision: ident, ($($types:ident),+)) => {
        paste!{
            #[test]
            fn [< test_ $fixture:snake _ $precision:snake _ $($types:snake)_+ >]() {
                let mut maker = Maker::default();
                let mut engine = CudaEngine::new().unwrap();
                let test_result =
                    <$fixture as Fixture<
                        $precision,
                        CudaEngine,
                        ($($types,)+),
                    >>::stress_all_parameters(&mut maker, &mut engine, REPETITIONS, SAMPLE_SIZE);
                assert!(test_result);
            }
        }
    };
    ($(($fixture: ident, $precision: ident, ($($types:ident),+))),+) => {
        $(
            test!{$fixture, $precision, ($($types),+)}
        )+
    };
    ($(($fixture: ident, ($($types:ident),+))),+) => {
        $(
            paste!{
                test!{$fixture, Precision32, ($([< $types 32 >]),+)}
            }
        )+
    };
}

test! {
    (LweCiphertextVectorConversionFixture, (CudaLweCiphertextVector,)),
}
```

Actually this is a bit complex to just test the 32 bits implementation, but it is very easy to add
the 64 bits precision in this implementation once you have the 64 bits engines. Finally, let's run
our test!

## Execute the test

The command to run the tests for the GPU backend is:

```
cargo test -p concrete-core-test --features=backend_gpu --release
```

You can filter it to execute a specific engine only:

```
cargo test -p concrete-core-test --features=backend_gpu --release -- --test conversion
```

You should get as output:

```
     Running unittests (target/release/deps/concrete_core_test-c662f8b6b8aa1434)

running 1 test
test cuda::test_lwe_ciphertext_vector_conversion_fixture_precision32_cuda_lwe_ciphertext_vector32_lwe_ciphertext_vector32 ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 7 filtered out; finished in 42.33s

   Doc-tests concrete-core-test

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```
Next step is to benchmark your backend, for this head to the [benchmarks tutorial](benchmarking_backends.md)!
