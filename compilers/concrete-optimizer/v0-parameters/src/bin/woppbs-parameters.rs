#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![warn(clippy::style)]

use clap::Parser;
use concrete_optimizer::computing_cost::cpu::CpuComplexity;
use concrete_optimizer::config;
use concrete_optimizer::optimization::config::{Config, SearchSpace};
use concrete_optimizer::optimization::decomposition;
use concrete_optimizer::optimization::wop_atomic_pattern::optimize::{optimize_raw, WopEncoding};
use concrete_security_curves::gaussian::security::{minimal_variance_glwe, minimal_variance_lwe};
use std::io::Write;

pub const _4_SIGMA: f64 = 1.0 - 0.999_936_657_516;

/// Find parameters for new WoP-PBS with more degrees of freedom
/// than the v0-parameter tool
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct WopArgs {
    #[clap(long, default_value_t = _4_SIGMA)]
    pub p_fail: f64,

    #[clap(
        long,
        default_value_t = 128,
        help = "Supported values: 80, 96, 112, 128, 192"
    )]
    pub security_level: u64,

    #[clap(long, default_value_t = false)]
    pub cache_on_disk: bool,
}

struct ExperimentParameters {
    precision: u64,
    n_blocks: usize,
}

fn experiments_8_bits() -> Vec<ExperimentParameters> {
    // 1 LWE ciphertext with 8 bits of data
    // 2 LWE ciphertexts with 4 bits of data each
    // 4 LWE ciphertexts with 2 bits of data each
    vec![
        ExperimentParameters {
            precision: 8,
            n_blocks: 1,
        },
        ExperimentParameters {
            precision: 4,
            n_blocks: 2,
        },
        ExperimentParameters {
            precision: 2,
            n_blocks: 4,
        },
    ]
}

fn experiments_9_bits() -> Vec<ExperimentParameters> {
    // 1 LWE ciphertext with 9 bits of data
    // 3 LWE ciphertexts with 3 bits of data
    vec![
        ExperimentParameters {
            precision: 9,
            n_blocks: 1,
        },
        ExperimentParameters {
            precision: 3,
            n_blocks: 3,
        },
    ]
}

fn experiments_10_bits() -> Vec<ExperimentParameters> {
    // 1 LWE ciphertext with 10 bits of data
    // 2 LWE cihpertexts with 5 bits of data each
    vec![
        ExperimentParameters {
            precision: 10,
            n_blocks: 1,
        },
        ExperimentParameters {
            precision: 5,
            n_blocks: 2,
        },
    ]
}

fn experiments_11_bits() -> Vec<ExperimentParameters> {
    // 1 LWE ciphertext with 11 bits of data
    // 3 LWE ciphertexts with 4 bits of data each
    // 4 LWE ciphertexts with 3 bits of data each
    vec![
        ExperimentParameters {
            precision: 11,
            n_blocks: 1,
        },
        ExperimentParameters {
            precision: 4,
            n_blocks: 3,
        },
        ExperimentParameters {
            precision: 3,
            n_blocks: 4,
        },
    ]
}

fn experiments_12_bits() -> Vec<ExperimentParameters> {
    // 1 LWE ciphertext with 12 bits of data
    // 3 LWE ciphertexts with 4 bits of data each
    // 4 LWE ciphertexts with 3 bits of data each
    vec![
        ExperimentParameters {
            precision: 12,
            n_blocks: 1,
        },
        ExperimentParameters {
            precision: 4,
            n_blocks: 3,
        },
        ExperimentParameters {
            precision: 3,
            n_blocks: 4,
        },
    ]
}

fn launch_experiments(mut writer: impl Write, args: &WopArgs) {
    let processing_unit = config::ProcessingUnit::Cpu;
    let maximum_acceptable_error_probability = args.p_fail;
    let security_level = args.security_level;
    let cache_on_disk = args.cache_on_disk;
    let search_space = SearchSpace::default_cpu();

    // let guard = pprof::ProfilerGuard::new(100).unwrap();

    // let precisions_iter = CondIterator::new(precisions, !args.no_parallelize);

    let config = Config {
        security_level,
        maximum_acceptable_error_probability,
        ciphertext_modulus_log: 64,
        complexity_model: &CpuComplexity::default(),
    };

    let cache = decomposition::cache(security_level, processing_unit, None, cache_on_disk);

    writeln!(writer, "security level: {security_level}").unwrap();
    writeln!(
        writer,
        "target p_error: {maximum_acceptable_error_probability:1.1e}"
    )
    .unwrap();

    let mut launch_experiment = |experiment_parameters: Vec<ExperimentParameters>,
                                 exp_bitwidth: u64| {
        writeln!(writer, "\n// -> Experiment for {} bits", exp_bitwidth).unwrap();
        writeln!(
            writer,
            " p, n_blocks, n_inputs, k,  N, stddev,    n, stddev, br_l,br_b, ks_l,ks_b, cb_l,cb_b, pp_l,pp_b,  cost"
        )
            .unwrap();
        for n_inputs in 2..5 {
            for experiment in experiment_parameters.iter() {
                let res = optimize_raw(
                    0.,
                    config,
                    &search_space,
                    1,
                    &vec![1 << experiment.precision; experiment.n_blocks],
                    &cache,
                    &WopEncoding::RADIX,
                    n_inputs,
                )
                .best_solution;
                match res {
                    Some(solution) => {
                        let glwe_log_std_dev = minimal_variance_glwe(
                            solution.glwe_dimension,
                            solution.glwe_polynomial_size,
                            64,
                            security_level,
                        )
                        .log2()
                            * 0.5;
                        let lwe_log_std_dev = minimal_variance_lwe(
                            solution.internal_ks_output_lwe_dimension,
                            64,
                            security_level,
                        )
                        .log2()
                            * 0.5;

                        writeln!(writer,
                                 "{:2}, {:8}, {:8}, {:1}, {:2}, {:.2}, {:4}, {:.2},   {:2},  {:2},   {:2}, {:2},    {:2}, {:2},    {:2}, {:2}, {:6}",
                                 experiment.precision, experiment.n_blocks, n_inputs, solution.glwe_dimension, (solution.glwe_polynomial_size as f64).log2() as u64,
                                 glwe_log_std_dev,
                                 solution.internal_ks_output_lwe_dimension,
                                 lwe_log_std_dev,
                                 solution.br_decomposition_level_count, solution.br_decomposition_base_log,
                                 solution.ks_decomposition_level_count, solution.ks_decomposition_base_log,
                                 solution.cb_decomposition_level_count, solution.cb_decomposition_base_log,
                                 solution.pp_decomposition_level_count, solution.pp_decomposition_base_log,
                                 (solution.complexity / (1024.0 * 1024.0)) as u64,
                                 // solution.p_error
                        ).unwrap();
                        writeln!(writer,
                                 "lwe_dimension: LweDimension({:1}),\n\
                                  glwe_dimension: GlweDimension({:1}),\n\
                                  polynomial_size: PolynomialSize({:1}),\n\
                                  lwe_modular_std_dev: StandardDev({:.2}),\n\
                                  glwe_modular_std_dev: StandardDev({:.2}),\n\
                                  pbs_base_log: DecompositionBaseLog({:1}),\n\
                                  pbs_level: DecompositionLevelCount({:1}),\n\
                                  ks_base_log: DecompositionBaseLog({:1}),\n\
                                  ks_level: DecompositionLevelCount({:1}),\n\
                                  pfks_level: DecompositionLevelCount({:1}),\n\
                                  pfks_base_log: DecompositionBaseLog({:1}),\n\
                                  pfks_modular_std_dev: StandardDev({:.2}),\n\
                                  cbs_level: DecompositionLevelCount({:1}),\n\
                                  cbs_base_log: DecompositionBaseLog({:1}),\n\
                                  message_modulus: MessageModulus(-),\n\
                                  carry_modulus: CarryModulus(1),\n\
                                  ",
                                 solution.internal_ks_output_lwe_dimension,
                                 solution.glwe_dimension,
                                 (solution.glwe_polynomial_size as f64) as u64,
                                 lwe_log_std_dev,
                                 glwe_log_std_dev,
                                 solution.br_decomposition_base_log,
                                 solution.br_decomposition_level_count,
                                 solution.ks_decomposition_base_log,
                                 solution.ks_decomposition_level_count,
                                 solution.pp_decomposition_level_count,
                                 solution.pp_decomposition_base_log,
                                 glwe_log_std_dev,
                                 solution.cb_decomposition_level_count,
                                 solution.cb_decomposition_base_log,
                        ).unwrap();
                    }
                    _ => {}
                }
            }
            writeln!(writer, "\n").unwrap();
        }
    };

    launch_experiment(experiments_8_bits(), 8);
    launch_experiment(experiments_9_bits(), 9);
    launch_experiment(experiments_10_bits(), 10);
    launch_experiment(experiments_11_bits(), 11);
    launch_experiment(experiments_12_bits(), 12);
}

fn main() {
    let args = WopArgs::parse();

    launch_experiments(std::io::stdout(), &args);
}
