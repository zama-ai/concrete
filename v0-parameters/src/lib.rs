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
use concrete_optimizer::global_parameters::DEFAUT_DOMAINS;
use concrete_optimizer::optimization::atomic_pattern::{
    self as optimize_atomic_pattern, OptimizationState,
};
use concrete_optimizer::optimization::wop_atomic_pattern::optimize as optimize_wop_atomic_pattern;
use rayon_cond::CondIterator;
use std::io::Write;

pub const _4_SIGMA: f64 = 1.0 - 0.999_936_657_516;
const MIN_LOG_POLY_SIZE: u64 = DEFAUT_DOMAINS
    .glwe_pbs_constrained
    .log2_polynomial_size
    .start as u64;
const MAX_LOG_POLY_SIZE: u64 =
    DEFAUT_DOMAINS.glwe_pbs_constrained.log2_polynomial_size.end as u64 - 1;
pub const MAX_GLWE_DIM: u64 = DEFAUT_DOMAINS.glwe_pbs_constrained.glwe_dimension.end - 1;
pub const MIN_LWE_DIM: u64 = DEFAUT_DOMAINS.free_glwe.glwe_dimension.start as u64;
pub const MAX_LWE_DIM: u64 = DEFAUT_DOMAINS.free_glwe.glwe_dimension.end as u64 - 1;

/// Find parameters for classical PBS and new WoP-PBS
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
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
        help = "Supported values: 80, 96, 112, 128, 144, 160, 176, 192, 256"
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
}

pub fn all_results(args: &Args) -> Vec<Vec<OptimizationState>> {
    let sum_size = args.sum_size;
    let p_error = args.p_error;
    let security_level = args.security_level;
    let glwe_log_polynomial_sizes: Vec<_> =
        (args.min_log_poly_size..=args.max_log_poly_size).collect();
    let glwe_dimensions: Vec<_> = (args.min_glwe_dim..=args.max_glwe_dim).collect();
    let internal_lwe_dimensions: Vec<_> =
        (args.min_intern_lwe_dim..=args.max_intern_lwe_dim).collect();

    let precisions = args.min_precision..=args.max_precision;
    let manps: Vec<_> = (0..=31).collect();

    // let guard = pprof::ProfilerGuard::new(100).unwrap();

    let precisions_iter = CondIterator::new(precisions, !args.no_parallelize);

    precisions_iter
        .map(|precision| {
            let mut last_solution = None;
            manps
                .iter()
                .map(|&manp| {
                    let noise_scale = 2_f64.powi(manp);
                    let result = if args.wop_pbs {
                        optimize_wop_atomic_pattern::optimize_one::<u64>(
                            sum_size,
                            precision,
                            security_level,
                            noise_scale,
                            p_error,
                            &glwe_log_polynomial_sizes,
                            &glwe_dimensions,
                            &internal_lwe_dimensions,
                        )
                    } else {
                        optimize_atomic_pattern::optimize_one::<u64>(
                            sum_size,
                            precision,
                            security_level,
                            noise_scale,
                            p_error,
                            &glwe_log_polynomial_sizes,
                            &glwe_dimensions,
                            &internal_lwe_dimensions,
                            last_solution, // 33% gains
                        )
                    };
                    last_solution = result.best_solution;
                    result
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
    writeln!(writer, "{{ /* Security level: {} */", security_level)?;
    writeln!(writer, "{{ /* {:1.1e} errors */", p_error)?;

    for (precision_i, precision) in precisions.enumerate() {
        writeln!(writer, "{{ /* precision {:2} */", precision)?;

        for (manp_i, manp) in manps.clone().iter().enumerate() {
            if let Some(solution) = all_results[precision_i][manp_i].best_solution {
                writeln!(writer,
                         "    /* {:2} */ V0Parameter({:2}, {:2}, {:4}, {:2}, {:2}, {:2}, {:2}), \t\t // {:4} mops, {:1.1e} errors",
                         manp, solution.glwe_dimension, (solution.glwe_polynomial_size as f64).log2() as u64,
                         solution.internal_ks_output_lwe_dimension,
                         solution.br_decomposition_level_count, solution.br_decomposition_base_log,
                         solution.ks_decomposition_level_count, solution.ks_decomposition_base_log,
                         (solution.complexity / (1024.0 * 1024.0)) as u64,
                         solution.p_error
                )?;
            } else {
                writeln!(
                    writer,
                    "    /* {:2} : NO SOLUTION */ V0Parameter(0,0,0,0,0,0,0),",
                    manp,
                )?;
            }
        }
        writeln!(writer, "  }},")?;
    }
    writeln!(writer, "}}")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use concrete_optimizer::security::security_weights::SECURITY_WEIGHTS_TABLE;

    #[test]
    fn test_reference_output() {
        const CMP_LINES: &str = "\n";
        const EXACT_EQUALITY: i32 = 0;
        for &security_level in SECURITY_WEIGHTS_TABLE.keys() {
            let ref_file: &str = &format!("ref/v0_2022-7-4_{}", security_level);
            let args: Args = Args {
                min_precision: 1,
                max_precision: 8,
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
        const CMP_LINES: &str = "\n";
        const EXACT_EQUALITY: i32 = 0;
        for &security_level in SECURITY_WEIGHTS_TABLE.keys() {
            let ref_file: &str = &format!("ref/wop_pbs_2022-7-10_{}", security_level);

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
