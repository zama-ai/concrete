//! A module containing traits to generate dummy values for benchmarks.
//!
//! This module contains abstractions to create the input values needed to benchmark an engine
//! trait. For instance, if we want to benchmark an engine trait which adds two lwe ciphertexts,
//! we first have to be able to provide these ciphertexts. The benchmarked engine may not implement
//! all the traits needed to create those ciphertexts on its own though. For this reason we need a
//! different tool to create those input values.
//!
//! The [`Synthesizer`] structure contains all the necessary engines, used to generate the different
//! types needed for benchmarking. Basically, we should be able to generate any entity from the
//! engines contained in the `Synthesizer` struct.
//!
//! Then a set of `Synthesizable*Entity` traits allows to generate a value of a given entity using
//! the [`Synthesizer`] struct, passed as argument. Those traits are then used as bounds in the
//! benchmarking functions.
use concrete_commons::dispersion::*;
use concrete_commons::parameters::*;
use concrete_core::prelude::*;

/// A trait to generate a cleartext entity.
pub trait SynthesizableCleartextEntity: CleartextEntity {
    fn synthesize(synthesizer: &mut Synthesizer) -> Self;
}

/// A trait to generate a cleartext vector entity.
pub trait SynthesizableCleartextVectorEntity: CleartextVectorEntity {
    fn synthesize(synthesizer: &mut Synthesizer, count: CleartextCount) -> Self;
}

/// A trait to generate a glwe ciphertext entity.
pub trait SynthesizableGlweCiphertextEntity: GlweCiphertextEntity {
    fn synthesize(
        synthesizer: &mut Synthesizer,
        poly_size: PolynomialSize,
        glwe_dimension: GlweDimension,
        noise: Variance,
    ) -> Self;
}

/// A trait to generate a glwe ciphertext vector entity.
pub trait SynthesizableGlweCiphertextVectorEntity: GlweCiphertextVectorEntity {
    fn synthesize(
        synthesizer: &mut Synthesizer,
        poly_size: PolynomialSize,
        glwe_dimension: GlweDimension,
        count: GlweCiphertextCount,
        noise: Variance,
    ) -> Self;
}

/// A trait to generate a glwe secret key entity.
pub trait SynthesizableGlweSecretKeyEntity: GlweSecretKeyEntity {
    fn synthesize(
        synthesizer: &mut Synthesizer,
        poly_size: PolynomialSize,
        glwe_dimension: GlweDimension,
    ) -> Self;
}

/// A trait to generate an lwe bootstrap key entity.
pub trait SynthesizableLweBootstrapKeyEntity: LweBootstrapKeyEntity {
    fn synthesize(
        synthesizer: &mut Synthesizer,
        lwe_dimension: LweDimension,
        poly_size: PolynomialSize,
        glwe_dimension: GlweDimension,
        base_log: DecompositionBaseLog,
        level_count: DecompositionLevelCount,
        noise: Variance,
    ) -> Self;
}

/// A trait to generate an lwe ciphertext entity.
pub trait SynthesizableLweCiphertextEntity: LweCiphertextEntity {
    fn synthesize(
        synthesizer: &mut Synthesizer,
        lwe_dimension: LweDimension,
        noise: Variance,
    ) -> Self;
}

/// A trait to generate an lwe ciphertext vector entity.
pub trait SynthesizableLweCiphertextVectorEntity: LweCiphertextVectorEntity {
    fn synthesize(
        synthesizer: &mut Synthesizer,
        lwe_dimension: LweDimension,
        count: LweCiphertextCount,
        noise: Variance,
    ) -> Self;
}

/// A trait to generate an lwe keyswitch key entity.
pub trait SynthesizableLweKeyswitchKeyEntity: LweKeyswitchKeyEntity {
    fn synthesize(
        synthesizer: &mut Synthesizer,
        input_lwe_dimension: LweDimension,
        output_lwe_dimension: LweDimension,
        base_log: DecompositionBaseLog,
        level_count: DecompositionLevelCount,
        noise: Variance,
    ) -> Self;
}

/// A trait to generate an lwe secret key entity.
pub trait SynthesizableLweSecretKeyEntity: LweSecretKeyEntity {
    fn synthesize(synthesizer: &mut Synthesizer, lwe_dimension: LweDimension) -> Self;
}

/// A trait to generate a plaintext entity.
pub trait SynthesizablePlaintextEntity: PlaintextEntity {
    fn synthesize(synthesizer: &mut Synthesizer) -> Self;
}

/// A trait to generate a plaintext vector entity.
pub trait SynthesizablePlaintextVectorEntity: PlaintextVectorEntity {
    fn synthesize(synthesizer: &mut Synthesizer, count: PlaintextCount) -> Self;
}

/// A type containing all the necessary engines needed to generate any entity.
pub struct Synthesizer {
    #[cfg(feature = "backend_core")]
    core_engine: concrete_core::backends::core::engines::CoreEngine,
}

impl Default for Synthesizer {
    fn default() -> Self {
        Synthesizer {
            #[cfg(feature = "backend_core")]
            core_engine: concrete_core::backends::core::engines::CoreEngine::new().unwrap(),
        }
    }
}

#[cfg(feature = "backend_core")]
mod core {
    use super::*;
    use concrete_commons::dispersion::Variance;
    use concrete_commons::parameters::{
        CleartextCount, DecompositionBaseLog, DecompositionLevelCount, GlweCiphertextCount,
        GlweDimension, LweCiphertextCount, LweDimension, PlaintextCount, PolynomialSize,
    };

    impl SynthesizableCleartextEntity for Cleartext32 {
        fn synthesize(synthesize: &mut Synthesizer) -> Self {
            synthesize.core_engine.create_cleartext(&1u32).unwrap()
        }
    }

    impl SynthesizableCleartextEntity for Cleartext64 {
        fn synthesize(synthesize: &mut Synthesizer) -> Self {
            synthesize.core_engine.create_cleartext(&1u64).unwrap()
        }
    }

    impl SynthesizableCleartextVectorEntity for CleartextVector32 {
        fn synthesize(synthesize: &mut Synthesizer, count: CleartextCount) -> Self {
            synthesize
                .core_engine
                .create_cleartext_vector(vec![1u32; count.0].as_slice())
                .unwrap()
        }
    }

    impl SynthesizableCleartextVectorEntity for CleartextVector64 {
        fn synthesize(synthesize: &mut Synthesizer, count: CleartextCount) -> Self {
            synthesize
                .core_engine
                .create_cleartext_vector(vec![1u64; count.0].as_slice())
                .unwrap()
        }
    }

    impl SynthesizableGlweCiphertextEntity for GlweCiphertext32 {
        fn synthesize(
            synthesizer: &mut Synthesizer,
            poly_size: PolynomialSize,
            glwe_dimension: GlweDimension,
            noise: Variance,
        ) -> Self {
            let secret_key = synthesizer
                .core_engine
                .create_glwe_secret_key(glwe_dimension, poly_size)
                .unwrap();
            synthesizer
                .core_engine
                .zero_encrypt_glwe_ciphertext(&secret_key, noise)
                .unwrap()
        }
    }

    impl SynthesizableGlweCiphertextEntity for GlweCiphertext64 {
        fn synthesize(
            synthesizer: &mut Synthesizer,
            poly_size: PolynomialSize,
            glwe_dimension: GlweDimension,
            noise: Variance,
        ) -> Self {
            let secret_key = synthesizer
                .core_engine
                .create_glwe_secret_key(glwe_dimension, poly_size)
                .unwrap();
            synthesizer
                .core_engine
                .zero_encrypt_glwe_ciphertext(&secret_key, noise)
                .unwrap()
        }
    }

    impl SynthesizableGlweCiphertextVectorEntity for GlweCiphertextVector32 {
        fn synthesize(
            synthesizer: &mut Synthesizer,
            poly_size: PolynomialSize,
            glwe_dimension: GlweDimension,
            count: GlweCiphertextCount,
            noise: Variance,
        ) -> Self {
            let secret_key = synthesizer
                .core_engine
                .create_glwe_secret_key(glwe_dimension, poly_size)
                .unwrap();
            synthesizer
                .core_engine
                .zero_encrypt_glwe_ciphertext_vector(&secret_key, noise, count)
                .unwrap()
        }
    }

    impl SynthesizableGlweCiphertextVectorEntity for GlweCiphertextVector64 {
        fn synthesize(
            synthesizer: &mut Synthesizer,
            poly_size: PolynomialSize,
            glwe_dimension: GlweDimension,
            count: GlweCiphertextCount,
            noise: Variance,
        ) -> Self {
            let secret_key = synthesizer
                .core_engine
                .create_glwe_secret_key(glwe_dimension, poly_size)
                .unwrap();
            synthesizer
                .core_engine
                .zero_encrypt_glwe_ciphertext_vector(&secret_key, noise, count)
                .unwrap()
        }
    }

    impl SynthesizableGlweSecretKeyEntity for GlweSecretKey32 {
        fn synthesize(
            synthesizer: &mut Synthesizer,
            poly_size: PolynomialSize,
            glwe_dimension: GlweDimension,
        ) -> Self {
            synthesizer
                .core_engine
                .create_glwe_secret_key(glwe_dimension, poly_size)
                .unwrap()
        }
    }

    impl SynthesizableGlweSecretKeyEntity for GlweSecretKey64 {
        fn synthesize(
            synthesizer: &mut Synthesizer,
            poly_size: PolynomialSize,
            glwe_dimension: GlweDimension,
        ) -> Self {
            synthesizer
                .core_engine
                .create_glwe_secret_key(glwe_dimension, poly_size)
                .unwrap()
        }
    }

    impl SynthesizableLweBootstrapKeyEntity for LweBootstrapKey32 {
        fn synthesize(
            synthesizer: &mut Synthesizer,
            lwe_dimension: LweDimension,
            poly_size: PolynomialSize,
            glwe_dimension: GlweDimension,
            base_log: DecompositionBaseLog,
            level_count: DecompositionLevelCount,
            noise: Variance,
        ) -> Self {
            let lwe_sk = synthesizer
                .core_engine
                .create_lwe_secret_key(lwe_dimension)
                .unwrap();
            let glwe_sk = synthesizer
                .core_engine
                .create_glwe_secret_key(glwe_dimension, poly_size)
                .unwrap();
            synthesizer
                .core_engine
                .create_lwe_bootstrap_key(&lwe_sk, &glwe_sk, base_log, level_count, noise)
                .unwrap()
        }
    }

    impl SynthesizableLweBootstrapKeyEntity for LweBootstrapKey64 {
        fn synthesize(
            synthesizer: &mut Synthesizer,
            lwe_dimension: LweDimension,
            poly_size: PolynomialSize,
            glwe_dimension: GlweDimension,
            base_log: DecompositionBaseLog,
            level_count: DecompositionLevelCount,
            noise: Variance,
        ) -> Self {
            let lwe_sk = synthesizer
                .core_engine
                .create_lwe_secret_key(lwe_dimension)
                .unwrap();
            let glwe_sk = synthesizer
                .core_engine
                .create_glwe_secret_key(glwe_dimension, poly_size)
                .unwrap();
            synthesizer
                .core_engine
                .create_lwe_bootstrap_key(&lwe_sk, &glwe_sk, base_log, level_count, noise)
                .unwrap()
        }
    }

    impl SynthesizableLweBootstrapKeyEntity for FourierLweBootstrapKey32 {
        fn synthesize(
            synthesizer: &mut Synthesizer,
            lwe_dimension: LweDimension,
            poly_size: PolynomialSize,
            glwe_dimension: GlweDimension,
            base_log: DecompositionBaseLog,
            level_count: DecompositionLevelCount,
            noise: Variance,
        ) -> Self {
            let lwe_sk: LweSecretKey32 = synthesizer
                .core_engine
                .create_lwe_secret_key(lwe_dimension)
                .unwrap();
            let glwe_sk: GlweSecretKey32 = synthesizer
                .core_engine
                .create_glwe_secret_key(glwe_dimension, poly_size)
                .unwrap();
            let bsk: LweBootstrapKey32 = synthesizer
                .core_engine
                .create_lwe_bootstrap_key(&lwe_sk, &glwe_sk, base_log, level_count, noise)
                .unwrap();
            synthesizer
                .core_engine
                .convert_lwe_bootstrap_key(&bsk)
                .unwrap()
        }
    }

    impl SynthesizableLweBootstrapKeyEntity for FourierLweBootstrapKey64 {
        fn synthesize(
            synthesizer: &mut Synthesizer,
            lwe_dimension: LweDimension,
            poly_size: PolynomialSize,
            glwe_dimension: GlweDimension,
            base_log: DecompositionBaseLog,
            level_count: DecompositionLevelCount,
            noise: Variance,
        ) -> Self {
            let lwe_sk: LweSecretKey64 = synthesizer
                .core_engine
                .create_lwe_secret_key(lwe_dimension)
                .unwrap();
            let glwe_sk: GlweSecretKey64 = synthesizer
                .core_engine
                .create_glwe_secret_key(glwe_dimension, poly_size)
                .unwrap();
            let bsk: LweBootstrapKey64 = synthesizer
                .core_engine
                .create_lwe_bootstrap_key(&lwe_sk, &glwe_sk, base_log, level_count, noise)
                .unwrap();
            synthesizer
                .core_engine
                .convert_lwe_bootstrap_key(&bsk)
                .unwrap()
        }
    }

    impl SynthesizableLweCiphertextEntity for LweCiphertext32 {
        fn synthesize(
            synthesizer: &mut Synthesizer,
            lwe_dimension: LweDimension,
            noise: Variance,
        ) -> Self {
            let lwe_sk = synthesizer
                .core_engine
                .create_lwe_secret_key(lwe_dimension)
                .unwrap();
            synthesizer
                .core_engine
                .zero_encrypt_lwe_ciphertext(&lwe_sk, noise)
                .unwrap()
        }
    }

    impl SynthesizableLweCiphertextEntity for LweCiphertext64 {
        fn synthesize(
            synthesizer: &mut Synthesizer,
            lwe_dimension: LweDimension,
            noise: Variance,
        ) -> Self {
            let lwe_sk = synthesizer
                .core_engine
                .create_lwe_secret_key(lwe_dimension)
                .unwrap();
            synthesizer
                .core_engine
                .zero_encrypt_lwe_ciphertext(&lwe_sk, noise)
                .unwrap()
        }
    }

    impl SynthesizableLweCiphertextVectorEntity for LweCiphertextVector32 {
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
            synthesizer
                .core_engine
                .zero_encrypt_lwe_ciphertext_vector(&lwe_sk, noise, count)
                .unwrap()
        }
    }

    impl SynthesizableLweCiphertextVectorEntity for LweCiphertextVector64 {
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
            synthesizer
                .core_engine
                .zero_encrypt_lwe_ciphertext_vector(&lwe_sk, noise, count)
                .unwrap()
        }
    }

    impl SynthesizableLweKeyswitchKeyEntity for LweKeyswitchKey32 {
        fn synthesize(
            synthesizer: &mut Synthesizer,
            input_lwe_dimension: LweDimension,
            output_lwe_dimension: LweDimension,
            base_log: DecompositionBaseLog,
            level_count: DecompositionLevelCount,
            noise: Variance,
        ) -> Self {
            let input_lwe_sk = synthesizer
                .core_engine
                .create_lwe_secret_key(input_lwe_dimension)
                .unwrap();
            let output_lwe_sk = synthesizer
                .core_engine
                .create_lwe_secret_key(output_lwe_dimension)
                .unwrap();
            synthesizer
                .core_engine
                .create_lwe_keyswitch_key(
                    &input_lwe_sk,
                    &output_lwe_sk,
                    level_count,
                    base_log,
                    noise,
                )
                .unwrap()
        }
    }

    impl SynthesizableLweKeyswitchKeyEntity for LweKeyswitchKey64 {
        fn synthesize(
            synthesizer: &mut Synthesizer,
            input_lwe_dimension: LweDimension,
            output_lwe_dimension: LweDimension,
            base_log: DecompositionBaseLog,
            level_count: DecompositionLevelCount,
            noise: Variance,
        ) -> Self {
            let input_lwe_sk = synthesizer
                .core_engine
                .create_lwe_secret_key(input_lwe_dimension)
                .unwrap();
            let output_lwe_sk = synthesizer
                .core_engine
                .create_lwe_secret_key(output_lwe_dimension)
                .unwrap();
            synthesizer
                .core_engine
                .create_lwe_keyswitch_key(
                    &input_lwe_sk,
                    &output_lwe_sk,
                    level_count,
                    base_log,
                    noise,
                )
                .unwrap()
        }
    }

    impl SynthesizableLweSecretKeyEntity for LweSecretKey32 {
        fn synthesize(synthesizer: &mut Synthesizer, lwe_dimension: LweDimension) -> Self {
            synthesizer
                .core_engine
                .create_lwe_secret_key(lwe_dimension)
                .unwrap()
        }
    }

    impl SynthesizableLweSecretKeyEntity for LweSecretKey64 {
        fn synthesize(synthesizer: &mut Synthesizer, lwe_dimension: LweDimension) -> Self {
            synthesizer
                .core_engine
                .create_lwe_secret_key(lwe_dimension)
                .unwrap()
        }
    }

    impl SynthesizablePlaintextEntity for Plaintext32 {
        fn synthesize(synthesizer: &mut Synthesizer) -> Self {
            synthesizer.core_engine.create_plaintext(&1u32).unwrap()
        }
    }

    impl SynthesizablePlaintextEntity for Plaintext64 {
        fn synthesize(synthesizer: &mut Synthesizer) -> Self {
            synthesizer.core_engine.create_plaintext(&1u64).unwrap()
        }
    }

    impl SynthesizablePlaintextVectorEntity for PlaintextVector32 {
        fn synthesize(synthesizer: &mut Synthesizer, count: PlaintextCount) -> Self {
            synthesizer
                .core_engine
                .create_plaintext_vector(vec![1u32; count.0].as_slice())
                .unwrap()
        }
    }

    impl SynthesizablePlaintextVectorEntity for PlaintextVector64 {
        fn synthesize(synthesizer: &mut Synthesizer, count: PlaintextCount) -> Self {
            synthesizer
                .core_engine
                .create_plaintext_vector(vec![1u64; count.0].as_slice())
                .unwrap()
        }
    }
}
