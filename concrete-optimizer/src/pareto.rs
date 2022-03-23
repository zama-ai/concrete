use crate::global_parameters::Range;
use crate::noise_estimator::operators::atomic_pattern::{variance_bootstrap, variance_keyswitch};
use crate::parameters::{GlweParameters, KsDecompositionParameters, PbsDecompositionParameters};
use crate::security::glwe::minimal_variance;
use concrete_commons::dispersion::Variance;
use std::collections::HashSet;

pub fn extract_br_pareto(
    security_level: u64,
    output_glwe_range: &GlweParameters<Range, Range>,
    input_lwe_range: &crate::parameters::LweDimension<Range>,
    ciphertext_modulus_log: u64,
) -> Vec<PbsDecompositionParameters<u64, u64>> {
    let mut paretos = HashSet::new();

    for glwe_dimension in &output_glwe_range.glwe_dimension {
        for log2_polynomial_size in &output_glwe_range.log2_polynomial_size {
            let polynomial_size = 1 << log2_polynomial_size;

            let variance_bsk = minimal_variance(
                polynomial_size,
                glwe_dimension,
                ciphertext_modulus_log,
                security_level,
            );

            for input_lwe_dimension in &input_lwe_range.lwe_dimension {
                let mut min_variance = Variance(f64::INFINITY);

                for level in 1..=ciphertext_modulus_log {
                    // To compute the PBS, Concrete switches from u32/u64 to f64 to represent the ciphertext
                    // which only keeps the 53 MSB of each u32/u64 (53 is the mantissa size).
                    // There is no need to decompose more bits than 53 as those ones will be erased by the conversion between u32/u64 and f64.
                    const MAX_DECOMPOSITION_DEPTH: u64 = 53;

                    let mut log_base_arg_min = None;

                    for log2_base in 1..=(MAX_DECOMPOSITION_DEPTH / level) {
                        let variance = variance_bootstrap::<u64>(
                            input_lwe_dimension,
                            polynomial_size,
                            glwe_dimension,
                            level,
                            log2_base,
                            ciphertext_modulus_log,
                            variance_bsk,
                        );

                        debug_assert!(variance.0.is_finite());

                        if variance < min_variance {
                            min_variance = variance;
                            log_base_arg_min = Some(log2_base);
                        }
                    }
                    if let Some(log2_base_arg_min) = log_base_arg_min {
                        let _ = paretos.insert(PbsDecompositionParameters {
                            level,
                            log2_base: log2_base_arg_min,
                        });
                    }
                }
            }
        }
    }

    let mut res: Vec<_> = paretos.into_iter().collect();

    res.sort_unstable();

    res
}

// We assume that the input lwe dimension is constrained
// by being the result of a sample extract
// (i.e. is the product of a glwe_dimension and a polynomial_size)
pub fn extract_ks_pareto(
    security_level: u64,
    input_glwe_range: &GlweParameters<Range, Range>,
    output_lwe_range: &crate::parameters::LweDimension<Range>,
    ciphertext_modulus_log: u64,
) -> Vec<KsDecompositionParameters<u64, u64>> {
    let mut paretos = HashSet::new();

    for output_lwe_dimension in &output_lwe_range.lwe_dimension {
        let variance_ksk = minimal_variance(
            1,
            output_lwe_dimension,
            ciphertext_modulus_log,
            security_level,
        );

        for glwe_dimension in &input_glwe_range.glwe_dimension {
            for log2_polynomial_size in &input_glwe_range.log2_polynomial_size {
                let polynomial_size = 1 << log2_polynomial_size;

                let input_lwe_dimension = polynomial_size * glwe_dimension;

                let mut min_variance = Variance(f64::INFINITY);

                for level in 1..=ciphertext_modulus_log {
                    let mut log2_base_arg_min = None;

                    for log2_base in 1..=(ciphertext_modulus_log / level) {
                        let variance = variance_keyswitch::<u64>(
                            input_lwe_dimension,
                            level,
                            log2_base,
                            ciphertext_modulus_log,
                            variance_ksk,
                        );

                        debug_assert!(variance.0.is_finite());

                        if variance < min_variance {
                            min_variance = variance;
                            log2_base_arg_min = Some(log2_base);
                        }
                    }
                    if let Some(log_base_arg_min) = log2_base_arg_min {
                        let _ = paretos.insert(KsDecompositionParameters {
                            level,
                            log2_base: log_base_arg_min,
                        });
                    }
                }
            }
        }
    }

    let mut res: Vec<_> = paretos.into_iter().collect();

    res.sort_unstable();

    res
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    // when this test fails remove function fix_1xerror and fix_2xerror
    #[test]
    fn extract_br_pareto2() {
        let start = Instant::now();

        assert_eq!(
            extract_br_pareto(
                128,
                &GlweParameters {
                    log2_polynomial_size: Range { start: 9, end: 15 },
                    glwe_dimension: Range { start: 1, end: 3 }
                },
                &crate::parameters::LweDimension {
                    lwe_dimension: Range {
                        start: 450,
                        end: 1024
                    }
                },
                64
            )
            .len(),
            44
        );

        let duration = start.elapsed();

        println!("Time elapsed in extract_br_pareto2() is: {:?}", duration);
    }

    #[test]
    fn extract_ks_pareto2() {
        let start = Instant::now();

        assert_eq!(
            extract_ks_pareto(
                128,
                &GlweParameters {
                    log2_polynomial_size: Range { start: 9, end: 15 },
                    glwe_dimension: Range { start: 1, end: 3 }
                },
                &crate::parameters::LweDimension {
                    lwe_dimension: Range {
                        start: 450,
                        end: 1024
                    }
                },
                64
            )
            .len(),
            54
        );

        let duration = start.elapsed();

        println!("Time elapsed in extract_ks_pareto2() is: {:?}", duration);
    }
}
