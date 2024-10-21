use crate::generic::{Problem, SequentialProblem};
use crate::{MyRange, Solution};
use concrete_cpu_noise_model::gaussian_noise::noise::blind_rotate::variance_blind_rotate;

use concrete_cpu_noise_model::gaussian_noise::noise::modulus_switching::estimate_modulus_switching_noise_with_binary_key;
use concrete_optimizer::computing_cost::complexity_model::ComplexityModel;
use concrete_optimizer::noise_estimator::error;
use concrete_optimizer::parameters::{
    BrDecompositionParameters, GlweParameters, LweDimension, PbsParameters,
};
use concrete_security_curves::gaussian::security::minimal_variance_glwe;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::io::Write;
// use rayon_cond::CondIterator;

#[derive(Debug, Clone, Copy)]
pub struct KSFreeParams {
    base_log_pbs: u64,
    level_pbs: u64,
    glwe_dim: u64,
    log_poly_size: u64,
}

impl KSFreeParams {
    fn big_lwe_dim(&self) -> u64 {
        let poly_size = 1 << self.log_poly_size;
        self.glwe_dim * poly_size
    }
}

struct KSFreeConstraint {
    variance_constraint: f64,
    log_norm2: u64,
    security_level: u64,
    sum_size: u64,
}

impl Problem for KSFreeConstraint {
    type Param = KSFreeParams;

    fn verify(&self, param: Self::Param) -> bool {
        let poly_size = 1 << param.log_poly_size;

        let variance_bsk =
            minimal_variance_glwe(param.glwe_dim, poly_size, 64, self.security_level);
        let v_pbs = variance_blind_rotate(
            param.big_lwe_dim(),
            param.glwe_dim,
            poly_size,
            param.base_log_pbs,
            param.level_pbs,
            64,
            53,
            variance_bsk,
        );
        let v_ms = estimate_modulus_switching_noise_with_binary_key(
            param.big_lwe_dim(),
            param.log_poly_size,
            64,
        );

        v_pbs * (1 << (2 * self.log_norm2)) as f64 + v_ms < self.variance_constraint
    }

    fn cost(&self, param: Self::Param) -> f64 {
        let complexity_model = concrete_optimizer::computing_cost::cpu::CpuComplexity::default();
        let multisum_complexity = complexity_model.levelled_complexity(
            self.sum_size,
            LweDimension(param.big_lwe_dim()),
            64,
        );
        let pbs_parameter = PbsParameters {
            internal_lwe_dimension: LweDimension(param.big_lwe_dim()),
            br_decomposition_parameter: BrDecompositionParameters {
                level: param.level_pbs,
                log2_base: param.base_log_pbs,
            },
            output_glwe_params: GlweParameters {
                log2_polynomial_size: param.log_poly_size,
                glwe_dimension: param.glwe_dim,
            },
        };
        let pbs_complexity = complexity_model.pbs_complexity(pbs_parameter, 64);

        multisum_complexity + pbs_complexity
    }
}

struct KSFreeSearchSpace {
    range_base_log_pbs: MyRange,
    range_level_pbs: MyRange,
    range_glwe_dim: MyRange,
    range_log_poly_size: MyRange,
}

impl KSFreeSearchSpace {
    #[allow(unused)]
    #[rustfmt::skip]
    fn par_iter(self) -> impl rayon::iter::ParallelIterator<Item=KSFreeParams> {
        self.range_glwe_dim
            .to_std_range()
            .into_par_iter().map(|_k| KSFreeParams {
            base_log_pbs: 0,
            level_pbs: 0,
            glwe_dim: 0,
            log_poly_size: 0,
        })
    }

    fn iter(self, _precision: u64) -> impl Iterator<Item = KSFreeParams> {
        self.range_base_log_pbs
            .to_std_range()
            .flat_map(move |base_log_pbs| {
                self.range_level_pbs
                    // .to_std_range_tight(base_log_pbs, precision)
                    .to_std_range()
                    .flat_map(move |level_pbs| {
                        self.range_glwe_dim
                            .to_std_range()
                            .flat_map(move |glwe_dim| {
                                self.range_log_poly_size
                                    .to_std_range()
                                    .map(move |log_poly_size| KSFreeParams {
                                        base_log_pbs,
                                        level_pbs,
                                        glwe_dim,
                                        log_poly_size,
                                    })
                            })
                    })
            })
    }
}

pub fn solve_all_ksfree(p_fail: f64, writer: impl Write) {
    let precisions = 1..9;
    let log_norms = vec![4, 6, 8, 10];

    let res: Vec<Solution<KSFreeParams>> = precisions
        .into_par_iter()
        .flat_map(|precision| {
            log_norms
                .clone()
                .into_par_iter()
                .map(|log_norm| {
                    let a = KSFreeSearchSpace {
                        range_base_log_pbs: MyRange(1, 53),
                        range_level_pbs: MyRange(1, 53),
                        range_glwe_dim: MyRange(1, 7),
                        range_log_poly_size: MyRange(7, 18),
                    };

                    let config = KSFreeConstraint {
                        variance_constraint: error::safe_variance_bound_2padbits(
                            precision, 64, p_fail,
                        ),
                        log_norm2: log_norm,
                        security_level: 128,
                        sum_size: 4096,
                    };

                    let intem = config.brute_force(a.iter(precision));

                    Solution {
                        precision,
                        log_norm,
                        intem,
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    write_to_file(writer, &res).unwrap();
}

pub fn write_to_file(
    mut writer: impl Write,
    res: &[Solution<KSFreeParams>],
) -> Result<(), std::io::Error> {
    writeln!(writer, "  p,log(nu), k,  N, br_l,br_b,  cost")?;

    for Solution {
        precision,
        log_norm,
        intem,
    } in res.iter()
    {
        if let Some((solution, cost)) = intem {
            writeln!(
                writer,
                " {:2},     {:2}, {:2},  {:4},   {:2},  {:2},   {:6}",
                precision,
                log_norm,
                solution.glwe_dim,
                solution.log_poly_size,
                solution.level_pbs,
                solution.base_log_pbs,
                cost
            )?;
        }
    }
    Ok(())
}
