use std::time::Duration;

use concrete_cpu::c_api::types::tests::to_generic;
use concrete_cpu::implementation::bootstrap::tests::KeySet;
use concrete_cpu::implementation::types::{
    DecompParams, GlweCiphertext, GlweParams, LweCiphertext,
};
use concrete_csprng::generators::{RandomGenerator, SoftwareRandomGenerator};
use concrete_csprng::seeders::Seed;
use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, SamplingMode};

struct PbsParameters {
    glwe_dim: usize,
    log2_poly_size: usize,
    in_dim: usize,
    level: usize,
    base_log: usize,
}

pub fn bootstrap(c: &mut Criterion) {
    let mut group = c.benchmark_group("bootstrap");
    // Configure Criterion.rs to detect smaller differences and increase sample size to improve
    // precision and counteract the resulting noise.
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(100));

    #[rustfmt::skip]
    let parameters = [
        PbsParameters {glwe_dim: 5, log2_poly_size:  8, in_dim: 582, level: 1, base_log: 15,},
        PbsParameters {glwe_dim: 5, log2_poly_size:  8, in_dim: 670, level: 1, base_log: 15,},
        PbsParameters {glwe_dim: 3, log2_poly_size:  9, in_dim: 708, level: 1, base_log: 18,},
        PbsParameters {glwe_dim: 2, log2_poly_size: 10, in_dim: 784, level: 1, base_log: 23,},
        PbsParameters {glwe_dim: 1, log2_poly_size: 11, in_dim: 768, level: 1, base_log: 23,},
        PbsParameters {glwe_dim: 1, log2_poly_size: 12, in_dim: 860, level: 1, base_log: 22,},
        PbsParameters {glwe_dim: 1, log2_poly_size: 13, in_dim: 894, level: 1, base_log: 22,},
        // PbsParameters {glwe_dim: 1, log2_poly_size: 14, in_dim: 982, level: 2, base_log: 15,},
        // PbsParameters {glwe_dim: 1, log2_poly_size: 15, in_dim:1013, level: 2, base_log: 15,},
        // PbsParameters {glwe_dim: 1, log2_poly_size: 16, in_dim:1107, level: 2, base_log: 14,},
    ];

    for (index, parameter) in parameters.iter().enumerate() {
        let precision = index + 1;

        add_pbs_bench(
            &mut group,
            &format!("optimizer_precision_{precision}_norm_0"),
            parameter,
        );
    }
    group.finish();
}

fn add_pbs_bench(group: &mut BenchmarkGroup<WallTime>, name: &str, parameters: &PbsParameters) {
    let mut csprng = SoftwareRandomGenerator::new(Seed(0));

    let polynomial_size = 1 << parameters.log2_poly_size;

    let glwe_params = GlweParams {
        dimension: parameters.glwe_dim,
        polynomial_size,
    };

    let decomp_params = DecompParams {
        level: parameters.level,
        base_log: parameters.base_log,
    };

    let in_dim = parameters.in_dim;

    let mut keyset = KeySet::new(
        to_generic(&mut csprng),
        in_dim,
        glwe_params,
        decomp_params,
        0.0000000000000000000001,
    );

    let raw_lut: Vec<u64> = vec![0; (glwe_params.dimension + 1) * glwe_params.polynomial_size];

    let mut input = LweCiphertext::zero(in_dim);

    let mut output = LweCiphertext::zero(glwe_params.lwe_dimension());

    keyset.in_sk.as_view().encrypt_lwe(
        input.as_mut_view(),
        0,
        0.0000000001,
        to_generic(&mut csprng),
    );

    group.bench_function(name, |b| {
        b.iter(|| {
            keyset.bootstrap(
                input.as_view(),
                output.as_mut_view(),
                GlweCiphertext::new(&raw_lut, glwe_params),
            )
        });
    });
}

criterion_group!(benches, bootstrap);
criterion_main!(benches);
