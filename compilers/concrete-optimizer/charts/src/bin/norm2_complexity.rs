use charts::{draw, Serie};
use concrete_optimizer::computing_cost::cpu::CpuComplexity;
use concrete_optimizer::config;
use concrete_optimizer::global_parameters::DEFAUT_DOMAINS;
use concrete_optimizer::optimization::atomic_pattern::{self as optimize_atomic_pattern};
use concrete_optimizer::optimization::config::{Config, SearchSpace};
use concrete_optimizer::optimization::decomposition;
use concrete_optimizer::optimization::wop_atomic_pattern::optimize as optimize_wop_atomic_pattern;

pub const _4_SIGMA: f64 = 1.0 - 0.999_936_657_516;
const MIN_LOG_POLY_SIZE: u64 = DEFAUT_DOMAINS
    .glwe_pbs_constrained
    .log2_polynomial_size
    .start;
const MAX_LOG_POLY_SIZE: u64 = DEFAUT_DOMAINS.glwe_pbs_constrained.log2_polynomial_size.end - 1;
pub const MAX_GLWE_DIM: u64 = DEFAUT_DOMAINS.glwe_pbs_constrained.glwe_dimension.end - 1;
pub const MIN_LWE_DIM: u64 = DEFAUT_DOMAINS.free_glwe.glwe_dimension.start;
pub const MAX_LWE_DIM: u64 = DEFAUT_DOMAINS.free_glwe.glwe_dimension.end - 1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sum_size = 4096;
    let p_error = _4_SIGMA;
    let security_level = 128;
    let glwe_log_polynomial_sizes: Vec<_> = (MIN_LOG_POLY_SIZE..=MAX_LOG_POLY_SIZE).collect();
    let glwe_dimensions: Vec<_> = (1..=6).collect();
    let internal_lwe_dimensions: Vec<_> = (MIN_LWE_DIM..=MAX_LWE_DIM).step_by(10).collect();

    let processing_unit = config::ProcessingUnit::Cpu;

    let search_space = SearchSpace {
        glwe_log_polynomial_sizes,
        glwe_dimensions,
        internal_lwe_dimensions,
        levelled_only_lwe_dimensions: DEFAUT_DOMAINS.free_lwe,
    };

    let precision = 8;
    let log_norm2s = 1_u64..=31;

    let ciphertext_modulus_log = 64;
    let fft_precision = 53;

    let config = Config {
        security_level,
        maximum_acceptable_error_probability: p_error,
        key_sharing: true,
        ciphertext_modulus_log,
        fft_precision,
        complexity_model: &CpuComplexity::default(),
    };

    let cache = decomposition::cache(
        security_level,
        processing_unit,
        None,
        true,
        ciphertext_modulus_log,
        53,
    );

    let solutions: Vec<_> = log_norm2s
        .clone()
        .filter_map(|log_norm2| {
            let noise_scale = 2_f64.powi(log_norm2 as i32);

            optimize_atomic_pattern::optimize_one(
                sum_size,
                precision,
                config,
                noise_scale,
                &search_space,
                &cache,
            )
            .best_solution
            .map(|a| (log_norm2, a.complexity))
        })
        .collect();

    let wop_solutions: Vec<_> = log_norm2s
        .filter_map(|log_norm2| {
            let noise_scale = 2_f64.powi(log_norm2 as i32);

            optimize_wop_atomic_pattern::optimize_one(
                precision,
                config,
                noise_scale,
                &search_space,
                &cache,
            )
            .best_solution
            .map(|a| (log_norm2, a.complexity))
        })
        .collect();

    draw(
        "comparison_cggi_bbbclot_norm2_vs_complexity.png",
        &format!("Comparison CGGI vs BBBCLOT for precision={precision}"),
        &[
            Serie {
                label: "CGGI AP Complexity".to_owned(),
                values: solutions,
            },
            Serie {
                label: "BBBCLOT AP Complexity".to_owned(),
                values: wop_solutions,
            },
        ],
        (1024, 1024),
        "Log Norm2",
        "Complexity",
    )?;

    Ok(())
}
