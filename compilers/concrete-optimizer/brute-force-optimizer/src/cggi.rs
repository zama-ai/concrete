use crate::generic::{Problem, SequentialProblem};
use crate::{ExplicitRange, MyRange, Solution, STEP};
use concrete_cpu_noise_model::gaussian_noise::noise::blind_rotate::variance_blind_rotate;
use concrete_cpu_noise_model::gaussian_noise::noise::keyswitch::variance_keyswitch;
use concrete_cpu_noise_model::gaussian_noise::noise::modulus_switching::estimate_modulus_switching_noise_with_binary_key;
use concrete_optimizer::computing_cost::complexity_model::ComplexityModel;
use concrete_optimizer::noise_estimator::error;
use concrete_optimizer::parameters::{
    AtomicPatternParameters, BrDecompositionParameters, GlweParameters, KsDecompositionParameters,
    LweDimension,
};
use concrete_security_curves::gaussian::security::{minimal_variance_glwe, minimal_variance_lwe};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::io::Write;
// use rayon_cond::CondIterator;

#[derive(Debug, Clone, Copy)]
pub struct CGGIParams {
    base_log_ks: u64,
    level_ks: u64,
    base_log_pbs: u64,
    level_pbs: u64,
    glwe_dim: u64,
    log_poly_size: u64,
    small_lwe_dim: u64,
}

impl CGGIParams {
    fn big_lwe_dim(&self) -> u64 {
        let poly_size = 1 << self.log_poly_size;
        self.glwe_dim * poly_size
    }
}

struct CGGIConstraint {
    variance_constraint: f64,
    log_norm2: u64,
    security_level: u64,
    sum_size: u64,
}

impl Problem for CGGIConstraint {
    type Param = CGGIParams;

    fn verify(&self, param: Self::Param) -> bool {
        let poly_size = 1 << param.log_poly_size;

        let variance_ksk = minimal_variance_lwe(param.small_lwe_dim, 64, self.security_level);

        let v_ks = variance_keyswitch(
            param.big_lwe_dim(),
            param.base_log_ks,
            param.level_ks,
            64,
            variance_ksk,
        );

        let variance_bsk =
            minimal_variance_glwe(param.glwe_dim, poly_size, 64, self.security_level);
        let v_pbs = variance_blind_rotate(
            param.small_lwe_dim,
            param.glwe_dim,
            poly_size,
            param.base_log_pbs,
            param.level_pbs,
            64,
            53,
            variance_bsk,
        );
        let v_ms = estimate_modulus_switching_noise_with_binary_key(
            param.small_lwe_dim,
            param.log_poly_size,
            64,
        );

        (v_pbs + v_ks) * (1 << (2 * self.log_norm2)) as f64 + v_ms < self.variance_constraint
    }

    fn cost(&self, param: Self::Param) -> f64 {
        cggi_complexity(
            self.sum_size,
            AtomicPatternParameters {
                input_lwe_dimension: LweDimension(param.big_lwe_dim()),
                ks_decomposition_parameter: KsDecompositionParameters {
                    level: param.level_ks,
                    log2_base: param.base_log_ks,
                },
                internal_lwe_dimension: LweDimension(param.small_lwe_dim),
                br_decomposition_parameter: BrDecompositionParameters {
                    level: param.level_pbs,
                    log2_base: param.base_log_pbs,
                },
                output_glwe_params: GlweParameters {
                    log2_polynomial_size: param.log_poly_size,
                    glwe_dimension: param.glwe_dim,
                },
            },
            64,
        )
    }
}

#[allow(dead_code)]
pub fn cggi_complexity(
    sum_size: u64,
    params: AtomicPatternParameters,
    ciphertext_modulus_log: u32,
) -> f64 {
    let complexity_model = concrete_optimizer::computing_cost::cpu::CpuComplexity::default();
    let multisum_complexity = complexity_model.levelled_complexity(
        sum_size,
        params.input_lwe_dimension,
        ciphertext_modulus_log,
    );
    let ks_complexity =
        complexity_model.ks_complexity(params.ks_parameters(), ciphertext_modulus_log);
    let pbs_complexity =
        complexity_model.pbs_complexity(params.pbs_parameters(), ciphertext_modulus_log);

    multisum_complexity + ks_complexity + pbs_complexity
}

struct CGGISearchSpace {
    range_base_log_ks: MyRange,
    range_level_ks: MyRange,
    _range_base_log_pbs: MyRange,
    _range_level_pbs: MyRange,
    range_glwe_dim: MyRange,
    range_log_poly_size: MyRange,
    range_small_lwe_dim: MyRange,
}

impl CGGISearchSpace {
    fn to_tighten(&self, security_level: u64) -> CGGISearchSpaceTighten {
        // Keyswitch
        let mut ks_decomp = vec![];
        for log_N in self.range_log_poly_size.to_std_range() {
            for k in self.range_glwe_dim.to_std_range() {
                for n in self.range_small_lwe_dim.to_std_range() {
                    let mut current_minimal_noise = f64::INFINITY;
                    for level in self.range_level_ks.to_std_range() {
                        let mut current_minimal_noise_for_a_given_level = current_minimal_noise;
                        let mut current_pair = (0, 0);
                        for baselog in self.range_base_log_ks.to_std_range() {
                            let variance_ksk = minimal_variance_lwe(n, 64, security_level);

                            let v_ks = variance_keyswitch(
                                (1 << log_N) * k,
                                baselog,
                                level,
                                64,
                                variance_ksk,
                            );
                            if v_ks <= current_minimal_noise_for_a_given_level {
                                current_minimal_noise_for_a_given_level = v_ks;
                                current_pair = (baselog, level)
                            }
                        }
                        if current_minimal_noise_for_a_given_level < current_minimal_noise {
                            ks_decomp.push(current_pair);
                            current_minimal_noise = current_minimal_noise_for_a_given_level;
                        }
                    }
                }
            }
        }
        // PBS
        let mut pbs_decomp = vec![];
        for log_N in self.range_log_poly_size.to_std_range() {
            for k in self.range_glwe_dim.to_std_range() {
                for n in self.range_small_lwe_dim.to_std_range() {
                    let mut current_minimal_noise = f64::INFINITY;
                    for level in self.range_level_ks.to_std_range() {
                        let mut current_minimal_noise_for_a_given_level = current_minimal_noise;
                        let mut current_pair = (0, 0);
                        for baselog in self.range_base_log_ks.to_std_range() {
                            let variance_bsk =
                                minimal_variance_glwe(k, 1 << log_N, 64, security_level);
                            let v_pbs = variance_blind_rotate(
                                n,
                                k,
                                1 << log_N,
                                baselog,
                                level,
                                64,
                                53,
                                variance_bsk,
                            );
                            if v_pbs <= current_minimal_noise_for_a_given_level {
                                current_minimal_noise_for_a_given_level = v_pbs;
                                current_pair = (baselog, level)
                            }
                        }
                        if current_minimal_noise_for_a_given_level < current_minimal_noise {
                            pbs_decomp.push(current_pair);
                            current_minimal_noise = current_minimal_noise_for_a_given_level;
                        }
                    }
                }
            }
        }

        ks_decomp.sort();
        ks_decomp.dedup();
        pbs_decomp.sort();
        pbs_decomp.dedup();

        println!("Only {} couples left for keyswitch", ks_decomp.len());
        println!("Only {} couples left for bootstrap", pbs_decomp.len());

        CGGISearchSpaceTighten {
            range_base_log_level_ks: ExplicitRange(ks_decomp.clone()),
            range_base_log_level_pbs: ExplicitRange(pbs_decomp.clone()),
            range_glwe_dim: self.range_glwe_dim,
            range_log_poly_size: self.range_log_poly_size,
            range_small_lwe_dim: self.range_small_lwe_dim,
        }
    }
}

#[derive(Clone)]
struct CGGISearchSpaceTighten {
    range_base_log_level_ks: ExplicitRange,
    range_base_log_level_pbs: ExplicitRange,
    range_glwe_dim: MyRange,
    range_log_poly_size: MyRange,
    range_small_lwe_dim: MyRange,
}

impl CGGISearchSpaceTighten {
    #[allow(unused)]
    #[rustfmt::skip]
    fn par_iter(self) -> impl rayon::iter::ParallelIterator<Item=CGGIParams> {
        self.range_glwe_dim
            .to_std_range()
            .into_par_iter().map(|_k| CGGIParams {
            base_log_ks: 0,
            level_ks: 0,
            base_log_pbs: 0,
            level_pbs: 0,
            glwe_dim: 0,
            log_poly_size: 0,
            small_lwe_dim: 0,
        })
    }

    fn iter(self, _precision: u64) -> impl Iterator<Item = CGGIParams> {
        self.range_base_log_level_ks
            .into_iter()
            .flat_map(move |(base_log_ks, level_ks)| {
                self.range_base_log_level_pbs.clone().into_iter().flat_map(
                    move |(base_log_pbs, level_pbs)| {
                        self.range_glwe_dim
                            .to_std_range()
                            .flat_map(move |glwe_dim| {
                                self.range_log_poly_size.to_std_range().flat_map(
                                    move |log_poly_size| {
                                        self.range_small_lwe_dim.to_std_range().step_by(STEP).map(
                                            move |small_lwe_dim| CGGIParams {
                                                base_log_ks,
                                                level_ks,
                                                base_log_pbs,
                                                level_pbs,
                                                glwe_dim,
                                                log_poly_size,
                                                small_lwe_dim,
                                            },
                                        )
                                    },
                                )
                            })
                    },
                )
            })
    }
}

pub fn solve_all_cggi(p_fail: f64, writer: impl Write) {
    // -> Vec<(u64, u64, Option<(V0Params, f64)>)> {
    // let p_fail = 1.0 - 0.999_936_657_516;

    let precisions = 1..9;
    // let precisions_iter = ParallelIterator::new(precisions);

    let log_norms = vec![4, 6, 8, 10];
    // 0..31;
    // let mut res = vec![];

    let a = CGGISearchSpace {
        range_base_log_ks: MyRange(1, 53),
        range_level_ks: MyRange(1, 53),
        _range_base_log_pbs: MyRange(1, 53),
        _range_level_pbs: MyRange(1, 53),
        range_glwe_dim: MyRange(1, 7),
        range_log_poly_size: MyRange(8, 18),
        range_small_lwe_dim: MyRange(500, 1500),
    };

    let a_tighten = a.to_tighten(128);
    let res: Vec<Solution<CGGIParams>> = precisions
        .into_par_iter()
        .flat_map(|precision| {
            log_norms
                .clone()
                .into_par_iter()
                .map(|log_norm| {
                    let config = CGGIConstraint {
                        variance_constraint: error::safe_variance_bound_2padbits(
                            precision, 64, p_fail,
                        ), //5.960464477539063e-08, // 0.0009765625006088146,
                        log_norm2: log_norm,
                        security_level: 128,
                        sum_size: 4096,
                    };

                    let intem = config.brute_force(a_tighten.clone().iter(precision));

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
    res: &[Solution<CGGIParams>],
) -> Result<(), std::io::Error> {
    writeln!(
        writer,
        "  p,log(nu), k,  N,    n, br_l,br_b, ks_l,ks_b,  cost"
    )?;

    for Solution {
        precision,
        log_norm,
        intem,
    } in res.iter()
    {
        if let Some((solution, cost)) = intem {
            writeln!(
                writer,
                " {:2},     {:2}, {:2}, {:2}, {:4},   {:2},  {:2},   {:2},  {:2}, {:6}",
                precision,
                log_norm,
                solution.glwe_dim,
                solution.log_poly_size,
                solution.small_lwe_dim,
                solution.level_pbs,
                solution.base_log_pbs,
                solution.level_ks,
                solution.base_log_ks,
                cost
            )?;
        }
    }
    Ok(())
}
