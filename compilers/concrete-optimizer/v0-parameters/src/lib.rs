#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![warn(clippy::style)]
#![allow(clippy::cast_precision_loss)] // u64 to f64
#![allow(clippy::cast_possible_truncation)] // u64 to usize
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]

use clap::Parser;
use concrete_optimizer::computing_cost::cpu::CpuComplexity;
use concrete_optimizer::config;
use concrete_optimizer::global_parameters::DEFAUT_DOMAINS;
use concrete_optimizer::optimization::config::{Config, SearchSpace};
use concrete_optimizer::optimization::dag::solo_key::optimize::{self as optimize_dag};
use concrete_optimizer::optimization::dag::solo_key::optimize_generic::Solution;
use concrete_optimizer::optimization::dag::solo_key::optimize_generic::Solution::{
    WopSolution, WpSolution,
};
use concrete_optimizer::optimization::wop_atomic_pattern::optimize as optimize_wop_atomic_pattern;
use concrete_optimizer::optimization::{atomic_pattern as optimize_atomic_pattern, decomposition};
use rayon_cond::CondIterator;
use std::io::Write;

pub const _4_SIGMA: f64 = 1.0 - 0.999_936_657_516;
const MIN_LOG_POLY_SIZE: u64 = DEFAUT_DOMAINS
    .glwe_pbs_constrained
    .log2_polynomial_size
    .start;
const MAX_LOG_POLY_SIZE: u64 = DEFAUT_DOMAINS.glwe_pbs_constrained.log2_polynomial_size.end - 1;
pub const MAX_GLWE_DIM: u64 = DEFAUT_DOMAINS.glwe_pbs_constrained.glwe_dimension.end - 1;
pub const MIN_LWE_DIM: u64 = DEFAUT_DOMAINS.free_glwe.glwe_dimension.start;
pub const MAX_LWE_DIM: u64 = DEFAUT_DOMAINS.free_glwe.glwe_dimension.end - 1;

/// Find parameters for classical PBS and new WoP-PBS
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
#[allow(clippy::struct_excessive_bools)]
pub struct Args {
    #[clap(long, default_value_t = 1, help = "1..16")]
    pub min_precision: u64,

    #[clap(long, default_value_t = 8, help = "1..16")]
    pub max_precision: u64,

    #[clap(long, default_value_t = _4_SIGMA)]
    pub p_error: f64,

    #[clap(
        long,
        default_value_t = 128,
        help = "Supported values: 80, 96, 112, 128, 192"
    )]
    pub security_level: u64,

    #[clap(long, default_value_t = MIN_LOG_POLY_SIZE, help = "8..16")]
    pub min_log_poly_size: u64,

    #[clap(long, default_value_t = MAX_LOG_POLY_SIZE, help = "8..16")]
    pub max_log_poly_size: u64,

    #[clap(long, default_value_t = 1, help = "1..6")]
    pub min_glwe_dim: u64,

    #[clap(long, default_value_t = MAX_GLWE_DIM, help = "1..6")]
    pub max_glwe_dim: u64,

    #[clap(long, default_value_t = MIN_LWE_DIM)]
    pub min_intern_lwe_dim: u64,

    #[clap(long, default_value_t = MAX_LWE_DIM)]
    pub max_intern_lwe_dim: u64, // 16bits needs around 1300

    #[clap(long, default_value_t = 4096)]
    pub sum_size: u64,

    #[clap(long)]
    pub no_parallelize: bool,

    #[clap(long)]
    pub wop_pbs: bool,

    #[clap(long)]
    pub simulate_dag: bool,

    #[clap(long, default_value_t = true)]
    pub cache_on_disk: bool,

    #[clap(long, default_value_t = 64)]
    pub ciphertext_modulus_log: u32,

    #[clap(long, default_value_t = 53)]
    pub fft_precision: u32,
}

pub fn all_results(args: &Args) -> Vec<Vec<Option<Solution>>> {
    let processing_unit = config::ProcessingUnit::Cpu;
    let sum_size = args.sum_size;
    let maximum_acceptable_error_probability = args.p_error;
    let security_level = args.security_level;
    let cache_on_disk = args.cache_on_disk;

    let search_space = SearchSpace {
        glwe_log_polynomial_sizes: (args.min_log_poly_size..=args.max_log_poly_size).collect(),
        glwe_dimensions: (args.min_glwe_dim..=args.max_glwe_dim).collect(),
        internal_lwe_dimensions: (args.min_intern_lwe_dim..=args.max_intern_lwe_dim).collect(),
        levelled_only_lwe_dimensions: DEFAUT_DOMAINS.free_lwe,
    };

    let precisions = args.min_precision..=args.max_precision;
    let log_norms2: Vec<_> = (0..=31).collect();

    // let guard = pprof::ProfilerGuard::new(100).unwrap();

    let precisions_iter = CondIterator::new(precisions, !args.no_parallelize);

    let config = Config {
        security_level,
        maximum_acceptable_error_probability,
        key_sharing: true,
        ciphertext_modulus_log: args.ciphertext_modulus_log,
        fft_precision: args.fft_precision,
        complexity_model: &CpuComplexity::default(),
    };

    let cache = decomposition::cache(
        security_level,
        processing_unit,
        None,
        cache_on_disk,
        args.ciphertext_modulus_log,
        args.fft_precision,
    );

    precisions_iter
        .map(|precision| {
            log_norms2
                .iter()
                .map(|&log_norm2| {
                    let noise_scale = 2_f64.powi(log_norm2);
                    if args.wop_pbs {
                        let log_norm = noise_scale.log2();
                        optimize_wop_atomic_pattern::optimize_one(
                            precision,
                            config,
                            log_norm,
                            &search_space,
                            &cache,
                        )
                        .best_solution
                        .map(WopSolution)
                    } else if args.simulate_dag {
                        optimize_dag::optimize_v0(
                            sum_size,
                            precision,
                            config,
                            noise_scale,
                            &search_space,
                            &cache,
                        )
                        .best_solution
                        .map(WpSolution)
                    } else {
                        optimize_atomic_pattern::optimize_one(
                            sum_size,
                            precision,
                            config,
                            noise_scale,
                            &search_space,
                            &cache,
                        )
                        .best_solution
                        .map(WpSolution)
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

pub fn compute_print_results(mut writer: impl Write, args: &Args) -> Result<(), std::io::Error> {
    let all_results = all_results(args);

    let p_error = args.p_error;
    let security_level = args.security_level;

    let precisions = args.min_precision..=args.max_precision;
    let manps: Vec<_> = (0..=31).collect();
    writeln!(writer, "security level: {security_level}")?;
    writeln!(writer, "target p_error: {p_error:1.1e}")?;
    writeln!(writer, "per precision and log norm2:")?;

    for (precision_i, precision) in precisions.enumerate() {
        writeln!(writer)?;
        writeln!(writer, "  - {precision}: # bits")?;
        let mut no_solution_at = None;
        for (manp_i, manp) in manps.clone().iter().enumerate() {
            if let Some(solution) = &all_results[precision_i][manp_i] {
                assert!(no_solution_at.is_none());
                match solution {
                    WpSolution(solution) => {
                        if manp_i == 0 {
                            writeln!(
                                writer,
                                "    -ln2:   k,  N,    n, br_l,br_b, ks_l,ks_b,  cost, p_error"
                            )?;
                        }
                        writeln!(writer,
                            "    - {:<2}:  {:2}, {:2}, {:4},   {:2}, {:2},    {:2}, {:2}, {:6}, {:1.1e}",
                            manp, solution.glwe_dimension, (solution.glwe_polynomial_size as f64).log2() as u64,
                            solution.internal_ks_output_lwe_dimension,
                            solution.br_decomposition_level_count, solution.br_decomposition_base_log,
                            solution.ks_decomposition_level_count, solution.ks_decomposition_base_log,
                            (solution.complexity / (1024.0 * 1024.0)) as u64,
                            solution.p_error
                        )?;
                    }
                    WopSolution(solution) => {
                        if manp_i == 0 {
                            writeln!(
                                writer,
                                "    -ln2:   k,  N,    n, br_l,br_b, ks_l,ks_b, cb_l,cb_b, pp_l,pp_b,  cost, p_error"
                            )?;
                        }
                        writeln!(writer,
                            "    - {:<2}:  {:2}, {:2}, {:4},   {:2}, {:2},    {:2}, {:2},    {:2}, {:2},    {:2}, {:2}, {:6}, {:1.1e}",
                            manp, solution.glwe_dimension, (solution.glwe_polynomial_size as f64).log2() as u64,
                            solution.internal_ks_output_lwe_dimension,
                            solution.br_decomposition_level_count, solution.br_decomposition_base_log,
                            solution.ks_decomposition_level_count, solution.ks_decomposition_base_log,
                            solution.cb_decomposition_level_count, solution.cb_decomposition_base_log,
                            solution.pp_decomposition_level_count, solution.pp_decomposition_base_log,
                            (solution.complexity / (1024.0 * 1024.0)) as u64,
                            solution.p_error
                        )?;
                    }
                }
            } else if no_solution_at.is_none() {
                no_solution_at = Some(*manp);
            }
        }
        if let Some(no_solution_at) = no_solution_at {
            if no_solution_at == 0 {
                writeln!(writer, "    # no solution at all",)?;
            } else {
                writeln!(
                    writer,
                    "    # no solution starting from log norm2 = {no_solution_at}"
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use concrete_optimizer::supported_security_levels;

    #[test]
    fn test_reference_output() {
        check_reference_output_on_levels(supported_security_levels(), false);
    }

    #[test]
    fn test_reference_output_dag() {
        check_reference_output_on_levels(supported_security_levels(), true);
    }

    fn check_reference_output_on_levels(
        security_levels: impl std::iter::Iterator<Item = u64>,
        simulate_dag: bool,
    ) {
        const CMP_LINES: &str = "\n";
        const EXACT_EQUALITY: i32 = 0;
        for security_level in security_levels {
            let ref_file: &str = &format!("ref/v0_last_{security_level}");
            let args: Args = Args {
                min_precision: 1,
                max_precision: 10,
                p_error: _4_SIGMA,
                security_level,
                min_log_poly_size: MIN_LOG_POLY_SIZE,
                max_log_poly_size: MAX_LOG_POLY_SIZE,
                min_glwe_dim: 1,
                max_glwe_dim: MAX_GLWE_DIM,
                min_intern_lwe_dim: MIN_LWE_DIM,
                max_intern_lwe_dim: MAX_LWE_DIM,
                sum_size: 4096,
                no_parallelize: false,
                wop_pbs: false,
                simulate_dag,
                cache_on_disk: true,
                ciphertext_modulus_log: 64,
                fft_precision: 53,
            };

            let mut actual_output = Vec::<u8>::new();

            compute_print_results(&mut actual_output, &args).unwrap();

            let actual_output = std::str::from_utf8(&actual_output).expect("Bad content");

            let expected_output =
                std::fs::read_to_string(ref_file).expect("Can't read reference file");

            text_diff::assert_diff(&expected_output, actual_output, CMP_LINES, EXACT_EQUALITY);
        }
    }

    #[test]
    fn test_reference_wop_output() {
        check_reference_wop_output_on_levels(supported_security_levels());
    }

    fn check_reference_wop_output_on_levels(security_levels: impl std::iter::Iterator<Item = u64>) {
        const CMP_LINES: &str = "\n";
        const EXACT_EQUALITY: i32 = 0;
        for security_level in security_levels {
            let ref_file: &str = &format!("ref/wop_pbs_last_{security_level}");

            let args = Args {
                min_precision: 1,
                max_precision: 16,
                p_error: _4_SIGMA,
                security_level,
                min_log_poly_size: 10,
                max_log_poly_size: 11,
                min_glwe_dim: 1,
                max_glwe_dim: MAX_GLWE_DIM,
                min_intern_lwe_dim: 450,
                max_intern_lwe_dim: MAX_LWE_DIM,
                sum_size: 4096,
                no_parallelize: false,
                wop_pbs: true,
                simulate_dag: false,
                cache_on_disk: true,
                ciphertext_modulus_log: 64,
                fft_precision: 53,
            };

            let mut actual_output = Vec::<u8>::new();

            compute_print_results(&mut actual_output, &args).unwrap();

            let actual_output = std::str::from_utf8(&actual_output).expect("Bad content");

            let expected_output =
                std::fs::read_to_string(ref_file).expect("Can't read reference file");

            text_diff::assert_diff(&expected_output, actual_output, CMP_LINES, EXACT_EQUALITY);
        }
    }
}
