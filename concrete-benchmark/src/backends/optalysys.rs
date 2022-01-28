//! A module benchmarking the `optalysys` backend of `concrete_core`.
use concrete_core::prelude::*;
use criterion::Criterion;

#[rustfmt::skip]
pub fn bench() {
    use crate::generics::*;
    let mut criterion = Criterion::default().configure_from_args();
    lwe_ciphertext_discarding_bootstrap::bench::<CoreEngine, FourierLweBootstrapKey32, GlweCiphertext32, LweCiphertext32, LweCiphertext32>(&mut criterion);
    lwe_ciphertext_discarding_bootstrap::bench::<CoreEngine, FourierLweBootstrapKey64, GlweCiphertext64, LweCiphertext64, LweCiphertext64>(&mut criterion);
}
