use std::time::Duration;

use concrete_cpu::c_api::types::tests::to_generic;
use concrete_cpu::implementation::keyswitch::tests::KeySet;
use concrete_cpu::implementation::types::{DecompParams, LweCiphertext};
use concrete_csprng::generators::{RandomGenerator, SoftwareRandomGenerator};
use concrete_csprng::seeders::Seed;
use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, SamplingMode};

struct KsParameters {
    out_dim: usize,
    in_dim: usize,
    level: usize,
    base_log: usize,
}

pub fn keyswitch(c: &mut Criterion) {
    let mut group = c.benchmark_group("keyswitch");
    // Configure Criterion.rs to detect smaller differences and increase sample size to improve
    // precision and counteract the resulting noise.
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(100));

    #[rustfmt::skip]
    #[allow(clippy::identity_op)]
    let parameters = [
        KsParameters {in_dim: 5 * (1 <<  8), out_dim: 582, level: 1, base_log: 15,},
        KsParameters {in_dim: 5 * (1 <<  8), out_dim: 670, level: 1, base_log: 15,},
        KsParameters {in_dim: 3 * (1 <<  9), out_dim: 708, level: 1, base_log: 18,},
        KsParameters {in_dim: 2 * (1 << 10), out_dim: 784, level: 1, base_log: 23,},
        KsParameters {in_dim: 1 * (1 << 11), out_dim: 768, level: 1, base_log: 23,},
        KsParameters {in_dim: 1 * (1 << 12), out_dim: 860, level: 1, base_log: 22,},
        KsParameters {in_dim: 1 * (1 << 13), out_dim: 894, level: 1, base_log: 22,},
        KsParameters {in_dim: 1 * (1 << 14), out_dim: 982, level: 2, base_log: 15,},
        KsParameters {in_dim: 1 * (1 << 15), out_dim:1013, level: 2, base_log: 15,},
        KsParameters {in_dim: 1 * (1 << 16), out_dim:1107, level: 2, base_log: 14,},
    ];

    for (index, parameter) in parameters.iter().enumerate() {
        let precision = index + 1;

        add_ks_bench(
            &mut group,
            &format!("optimizer_precision_{precision}_norm_0"),
            parameter,
        );
    }
    group.finish();
}

fn add_ks_bench(group: &mut BenchmarkGroup<WallTime>, name: &str, parameters: &KsParameters) {
    let mut csprng = SoftwareRandomGenerator::new(Seed(0));

    let decomp_params = DecompParams {
        level: parameters.level,
        base_log: parameters.base_log,
    };

    let in_dim = parameters.in_dim;
    let out_dim = parameters.out_dim;

    let keyset = KeySet::new(
        to_generic(&mut csprng),
        in_dim,
        out_dim,
        decomp_params,
        0.0000000000000000000001,
    );

    let mut input = LweCiphertext::zero(in_dim);

    let mut output = LweCiphertext::zero(out_dim);

    keyset.in_sk.as_view().encrypt_lwe(
        input.as_mut_view(),
        0,
        0.0000000001,
        to_generic(&mut csprng),
    );

    group.bench_function(name, |b| {
        b.iter(|| {
            keyset
                .ksk
                .as_view()
                .keyswitch_ciphertext(output.as_mut_view(), input.as_view())
        })
    });
}

criterion_group!(benches, keyswitch);
criterion_main!(benches);
