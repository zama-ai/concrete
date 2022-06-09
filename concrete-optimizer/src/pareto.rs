use crate::noise_estimator::operators::atomic_pattern::{variance_bootstrap, variance_keyswitch};
use crate::parameters::{
    BrDecompositionParameters, GlweParameterRanges, GlweParameters, KeyswitchParameters,
    KsDecompositionParameters, LweDimension, PbsParameters,
};
use crate::security::glwe::minimal_variance;
use concrete_commons::dispersion::Variance;
use std::collections::HashSet;

pub fn extract_br_pareto(
    security_level: u64,
    output_glwe_range: &GlweParameterRanges,
    input_lwe_range: &crate::parameters::LweDimensionRange,
    ciphertext_modulus_log: u32,
) -> Vec<BrDecompositionParameters> {
    let mut paretos = HashSet::new();

    for glwe_dimension in &output_glwe_range.glwe_dimension {
        for log2_polynomial_size in &output_glwe_range.log2_polynomial_size {
            let glwe_params = GlweParameters {
                log2_polynomial_size,
                glwe_dimension,
            };

            let variance_bsk =
                minimal_variance(glwe_params, ciphertext_modulus_log, security_level);

            for input_lwe_dimension in &input_lwe_range.lwe_dimension {
                let mut min_variance = Variance(f64::INFINITY);

                for level in 1..=(ciphertext_modulus_log as u64) {
                    // To compute the PBS, Concrete switches from u32/u64 to f64 to represent the ciphertext
                    // which only keeps the 53 MSB of each u32/u64 (53 is the mantissa size).
                    // There is no need to decompose more bits than 53 as those ones will be erased by the conversion between u32/u64 and f64.
                    const MAX_DECOMPOSITION_DEPTH: u64 = 53;

                    let mut log_base_arg_min = None;

                    for log2_base in 1..=(MAX_DECOMPOSITION_DEPTH / level) {
                        let pbs_parameters = PbsParameters {
                            internal_lwe_dimension: LweDimension(input_lwe_dimension),
                            br_decomposition_parameter: BrDecompositionParameters {
                                level,
                                log2_base,
                            },
                            output_glwe_params: GlweParameters {
                                log2_polynomial_size,
                                glwe_dimension,
                            },
                        };

                        let variance = variance_bootstrap(
                            pbs_parameters,
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
                        let _ = paretos.insert(BrDecompositionParameters {
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
    input_glwe_range: &GlweParameterRanges,
    output_lwe_range: &crate::parameters::LweDimensionRange,
    ciphertext_modulus_log: u32,
) -> Vec<KsDecompositionParameters> {
    let mut paretos = HashSet::new();

    for output_lwe_dimension in &output_lwe_range.lwe_dimension {
        let variance_ksk = minimal_variance(
            GlweParameters {
                log2_polynomial_size: 0,
                glwe_dimension: output_lwe_dimension,
            },
            ciphertext_modulus_log,
            security_level,
        );

        for glwe_dimension in &input_glwe_range.glwe_dimension {
            for log2_polynomial_size in &input_glwe_range.log2_polynomial_size {
                let polynomial_size = 1 << log2_polynomial_size;

                let input_lwe_dimension = polynomial_size * glwe_dimension;

                let mut min_variance = Variance(f64::INFINITY);

                for level in 1..=ciphertext_modulus_log as u64 {
                    let mut log2_base_arg_min = None;

                    for log2_base in 1..=(ciphertext_modulus_log as u64 / level) {
                        let keyswitch_parameters = KeyswitchParameters {
                            input_lwe_dimension: LweDimension(input_lwe_dimension),
                            output_lwe_dimension: LweDimension(output_lwe_dimension),
                            ks_decomposition_parameter: KsDecompositionParameters {
                                level,
                                log2_base,
                            },
                        };

                        let variance = variance_keyswitch(
                            keyswitch_parameters,
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

#[rustfmt::skip]
pub const BR_BL: &[(u64, u64); 41] = &[
    (12, 1),(22, 1),(23, 1),(8, 2),(15, 2),(16, 2),(6, 3),(11, 3),(12, 3),(5, 4),(9, 4),
    (4, 5),(7, 5),(8, 5),(6, 6),(7, 6),(3, 7),(6, 7),(5, 8),(5, 9),(2, 10),(4, 10),(2, 11),
    (4, 11),(3, 13),(3, 14),(3, 15),(1, 20),(2, 20),(1, 21),(2, 21),(1, 22),(2, 22),(2, 23),
    (1, 40),(1, 41),(1, 42),(1, 43),(1, 44),(1, 45),(1, 46),
];

#[rustfmt::skip]
pub const KS_BL: &[(u64, u64); 50] = &[
    (5, 1),(6, 1),(7, 1),(8, 1),(9, 1),(10, 1),(11, 1),(12, 1),(4, 2),(5, 2),(6, 2),(7, 2),
    (8, 2),(3, 3),(4, 3),(5, 3),(6, 3),(2, 4),(3, 4),(4, 4),(5, 4),(2, 5),(3, 5),(4, 5),
    (2, 6),(3, 6),(4, 6),(2, 7),(3, 7),(1, 8),(2, 8),(3, 8),(1, 9),(2, 9),(1, 10),(2, 10),
    (1, 11),(2, 11),(1, 12),(1, 13),(1, 14),(1, 15),(1, 16),(1, 17),(1, 18),(1, 19),(1, 20),
    (1, 21),(1, 22),(1, 23),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::global_parameters::DEFAUT_DOMAINS;
    use pretty_assertions::assert_eq;

    #[test]
    fn extract_br_pareto2() {
        let pareto = extract_br_pareto(
            128,
            &DEFAUT_DOMAINS.glwe_pbs_constrained,
            &DEFAUT_DOMAINS.free_glwe.into(),
            64,
        );

        let decomp_couple = |v: &BrDecompositionParameters| (v.log2_base, v.level);
        let br_bl: Vec<_> = pareto.iter().map(decomp_couple).collect();
        if br_bl != BR_BL {
            println!("---- Copy past to BR_BL");
            for (log2_base, level) in &br_bl {
                print!("({log2_base}, {level}), ");
            }
            println!();
            println!("---- End");
            assert_eq!(br_bl, BR_BL);
        }
    }

    #[test]
    fn extract_ks_pareto2() {
        let pareto = extract_ks_pareto(
            128,
            &DEFAUT_DOMAINS.glwe_pbs_constrained,
            &DEFAUT_DOMAINS.free_glwe.into(),
            64,
        );

        let decomp_couple = |v: &KsDecompositionParameters| (v.log2_base, v.level);
        let ks_bl: Vec<_> = pareto.iter().map(decomp_couple).collect();
        if ks_bl != KS_BL {
            println!("---- Copy past to KS_BL");
            for (log2_base, level) in &ks_bl {
                print!("({log2_base}, {level}), ");
            }
            println!();
            println!("---- End");
            assert_eq!(ks_bl, KS_BL);
        }
    }
}
