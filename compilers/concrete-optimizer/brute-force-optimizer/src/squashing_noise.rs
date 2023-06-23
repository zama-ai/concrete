use crate::generic::{Problem, SequentialProblem};
use crate::{minimal_added_noise_by_modulus_switching, ExplicitRange, MyRange, Solution, STEP};
use concrete_cpu_noise_model::gaussian_noise::noise::blind_rotate::variance_blind_rotate;
use concrete_cpu_noise_model::gaussian_noise::noise::keyswitch::variance_keyswitch;
use concrete_cpu_noise_model::gaussian_noise::noise::modulus_switching::estimate_modulus_switching_noise_with_binary_key;
use concrete_optimizer::computing_cost::complexity_model::ComplexityModel;
use concrete_optimizer::noise_estimator::error;
use concrete_optimizer::parameters::{
    AtomicPatternParameters, BrDecompositionParameters, GlweParameters, KeyswitchParameters,
    KsDecompositionParameters, LweDimension, PbsParameters,
};
use concrete_security_curves::gaussian::security::{minimal_variance_glwe, minimal_variance_lwe};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::cggi::CGGIParams;
use std::io::Write;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct SquashingParams {
    base_log_ks: u64,
    level_ks: u64,
    base_log_pbs: u64,
    level_pbs: u64,
    glwe_dim: u64,
    log_poly_size: u64,
    small_lwe_dim: u64,
}

impl SquashingParams {
    fn big_lwe_dim(&self) -> u64 {
        let poly_size = 1 << self.log_poly_size;
        self.glwe_dim * poly_size
    }
}

struct SquashingConstraint {
    first_variance_constraint: f64,
    second_variance_constraint: f64,
    security_level: u64,
    input_variance: f64,
    input_lwe_dimension: u64,
}

impl Problem for SquashingConstraint {
    type Param = SquashingParams;

    fn verify(&self, param: Self::Param) -> bool {
        // input -> MS (constraint) -> BR + SE -> KS
        let poly_size = 1 << param.log_poly_size;
        let first_ciphertext_modulus_log = 64;
        let second_ciphertext_modulus_log = 128;
        let fft_precision = 106;

        let variance_ksk = minimal_variance_lwe(
            param.small_lwe_dim,
            second_ciphertext_modulus_log,
            self.security_level,
        );

        let v_ks = variance_keyswitch(
            param.big_lwe_dim(),
            param.base_log_ks,
            param.level_ks,
            second_ciphertext_modulus_log,
            variance_ksk,
        );

        let variance_bsk = minimal_variance_glwe(
            param.glwe_dim,
            poly_size,
            second_ciphertext_modulus_log,
            self.security_level,
        );
        let v_pbs = variance_blind_rotate(
            self.input_lwe_dimension, // param.small_lwe_dim,
            param.glwe_dim,
            poly_size,
            param.base_log_pbs,
            param.level_pbs,
            second_ciphertext_modulus_log,
            fft_precision,
            variance_bsk,
        );
        let v_first_ms = estimate_modulus_switching_noise_with_binary_key(
            self.input_lwe_dimension, // param.small_lwe_dim,
            param.log_poly_size,
            first_ciphertext_modulus_log,
        );
        // let v_second_ms = estimate_modulus_switching_noise_with_binary_key(
        //      param.small_lwe_dim,
        //     param.log_poly_size,
        //     64,
        // );

        let first_constraint = self.input_variance + v_first_ms < self.first_variance_constraint;
        let second_constraint = v_pbs + v_ks < self.second_variance_constraint;
        // self.variance_constraint * f64::exp2(-self.stat as f64);
        // if ! first_constraint{
        //     self.first_constraint += 1
        // }
        first_constraint && second_constraint
    }

    fn cost(&self, params: Self::Param) -> f64 {
        let complexity_model = concrete_optimizer::computing_cost::cpu::CpuComplexity::default();
        let first_ciphertext_modulus_log = 64;
        let second_ciphertext_modulus_log = 128;
        let fft_precision = 106;

        let ks_parameters = KeyswitchParameters {
            input_lwe_dimension: LweDimension(params.big_lwe_dim()),
            output_lwe_dimension: LweDimension(params.small_lwe_dim),
            ks_decomposition_parameter: KsDecompositionParameters {
                level: params.level_ks,
                log2_base: params.base_log_ks,
            },
        };

        let ks_complexity =
            complexity_model.ks_complexity(ks_parameters, second_ciphertext_modulus_log as u32);

        let pbs_decomposition_parameter = BrDecompositionParameters {
            level: params.level_pbs,
            log2_base: params.base_log_pbs,
        };
        let pbs_parameters = PbsParameters {
            internal_lwe_dimension: LweDimension(self.input_lwe_dimension), // LweDimension(params.small_lwe_dim),
            br_decomposition_parameter: pbs_decomposition_parameter,
            output_glwe_params: GlweParameters {
                log2_polynomial_size: params.log_poly_size,
                glwe_dimension: params.glwe_dim,
            },
        };
        let pbs_complexity =
            complexity_model.pbs_complexity(pbs_parameters, second_ciphertext_modulus_log);

        ks_complexity + pbs_complexity
    }
}

struct SquashingSearchSpace {
    range_base_log_ks: MyRange,
    range_level_ks: MyRange,
    range_base_log_pbs: MyRange,
    range_level_pbs: MyRange,
    range_glwe_dim: MyRange,
    range_log_poly_size: MyRange,
    range_small_lwe_dim: MyRange,
}

impl SquashingSearchSpace {
    fn to_tighten(
        &self,
        security_level: u64,
        input_lwe_dimension: u64,
    ) -> SquashingSearchSpaceTighten {
        let first_ciphertext_modlus_log = 64;
        let second_ciphertext_modulus_log = 128;
        let fft_precision = 106;

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
                            let variance_ksk = minimal_variance_lwe(
                                n,
                                second_ciphertext_modulus_log,
                                security_level,
                            );

                            let v_ks = variance_keyswitch(
                                (1 << log_N) * k,
                                baselog,
                                level,
                                second_ciphertext_modulus_log,
                                variance_ksk,
                            );
                            // println!("ex ks: {}", v_ks.log2() * 0.5);
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
                let mut current_minimal_noise = f64::INFINITY;
                for level in self.range_level_pbs.to_std_range() {
                    let mut current_minimal_noise_for_a_given_level = current_minimal_noise;
                    let mut current_pair = (0, 0);
                    for baselog in self.range_base_log_pbs.to_std_range() {
                        let variance_bsk = minimal_variance_glwe(
                            k,
                            1 << log_N,
                            second_ciphertext_modulus_log,
                            security_level,
                        );
                        let v_pbs = variance_blind_rotate(
                            input_lwe_dimension, // n,
                            k,
                            1 << log_N,
                            baselog,
                            level,
                            second_ciphertext_modulus_log,
                            fft_precision,
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

        ks_decomp.sort();
        ks_decomp.dedup();
        pbs_decomp.sort();
        pbs_decomp.dedup();

        println!("Only {} couples left for keyswitch", ks_decomp.len());
        println!("Only {} couples left for bootstrap", pbs_decomp.len());

        SquashingSearchSpaceTighten {
            range_base_log_level_ks: ExplicitRange(ks_decomp.clone()),
            range_base_log_level_pbs: ExplicitRange(pbs_decomp.clone()),
            range_glwe_dim: self.range_glwe_dim,
            range_log_poly_size: self.range_log_poly_size,
            range_small_lwe_dim: self.range_small_lwe_dim,
        }
    }
}

#[derive(Clone)]
struct SquashingSearchSpaceTighten {
    range_base_log_level_ks: ExplicitRange,
    range_base_log_level_pbs: ExplicitRange,
    range_glwe_dim: MyRange,
    range_log_poly_size: MyRange,
    range_small_lwe_dim: MyRange,
}

impl SquashingSearchSpaceTighten {
    #[allow(unused)]
    #[rustfmt::skip]
    fn par_iter(self) -> impl rayon::iter::ParallelIterator<Item=SquashingParams> {
        self.range_glwe_dim
            .to_std_range()
            .into_par_iter().map(|_k| SquashingParams {
            base_log_ks: 0,
            level_ks: 0,
            base_log_pbs: 0,
            level_pbs: 0,
            glwe_dim: 0,
            log_poly_size: 0,
            small_lwe_dim: 0,
        })
    }

    fn iter(self, precision: u64, minimal_ms_value: u64) -> impl Iterator<Item = SquashingParams> {
        self.range_base_log_level_ks
            .into_iter()
            .flat_map(move |(base_log_ks, level_ks)| {
                self.range_base_log_level_pbs.clone().into_iter().flat_map(
                    move |(base_log_pbs, level_pbs)| {
                        self.range_glwe_dim
                            .to_std_range()
                            .flat_map(move |glwe_dim| {
                                self.range_log_poly_size
                                    .to_std_range_poly_size(precision + minimal_ms_value)
                                    .flat_map(move |log_poly_size| {
                                        self.range_small_lwe_dim.to_std_range().step_by(STEP).map(
                                            move |small_lwe_dim| SquashingParams {
                                                base_log_ks,
                                                level_ks,
                                                base_log_pbs,
                                                level_pbs,
                                                glwe_dim,
                                                log_poly_size,
                                                small_lwe_dim,
                                            },
                                        )
                                    })
                            })
                    },
                )
            })
        //                 })
        //         })
    }
}

// struct Experiment {
//     precision: u64,
//     cggi_params: CGGIParams,
//
// }

pub fn solve_all_squash(p_fail: f64, writer: impl Write) {
    let start = Instant::now();

    // p, nu,  k,  N,    n, br_l,br_b, ks_l,ks_b,  cost
    // 1,  1,  5,  8,   10,    1,  15,    1,  12, 45615872
    // 4,  5,  1, 11,   10,    1,  21,    2,   8, 125840384

    let experiments = vec![
        (
            1,
            CGGIParams {
                base_log_ks: 12,
                level_ks: 1,
                base_log_pbs: 15,
                level_pbs: 1,
                glwe_dim: 5,
                log_poly_size: 8,
                small_lwe_dim: 1 << 10,
            },
            0,
        ),
        (
            4,
            CGGIParams {
                base_log_ks: 8,
                level_ks: 2,
                base_log_pbs: 21,
                level_pbs: 1,
                glwe_dim: 1,
                log_poly_size: 11,
                small_lwe_dim: 1 << 10,
            },
            0,
        ),
    ];

    let pow = f64::exp2(40.) * 100.;

    // find the minimal added noise by the modulus switching
    // for KS
    let a = SquashingSearchSpace {
        range_base_log_ks: MyRange(1, 100),
        range_level_ks: MyRange(1, 40),
        range_base_log_pbs: MyRange(1, 100),
        range_level_pbs: MyRange(1, 53),
        range_glwe_dim: MyRange(1, 7),
        range_log_poly_size: MyRange(8, 19),
        range_small_lwe_dim: MyRange(2000, 4000),
    };

    let minimal_ms_value = minimal_added_noise_by_modulus_switching(
        (1 << a.range_log_poly_size.0) * a.range_glwe_dim.0,
    )
    .sqrt()
    .ceil() as u64;

    let res: Vec<Solution<SquashingParams>> = experiments
        .into_par_iter()
        .map(|(precision, cggi_parameters, cggi_log_norm)| {
            // computing the input_variance
            let poly_size = 1 << cggi_parameters.log_poly_size;
            let variance_ksk = minimal_variance_lwe(cggi_parameters.small_lwe_dim, 64, 128);

            let v_ks = variance_keyswitch(
                cggi_parameters.big_lwe_dim(),
                cggi_parameters.base_log_ks,
                cggi_parameters.level_ks,
                64,
                variance_ksk,
            );

            let variance_bsk = minimal_variance_glwe(cggi_parameters.glwe_dim, poly_size, 64, 128);
            let v_pbs = variance_blind_rotate(
                cggi_parameters.small_lwe_dim,
                cggi_parameters.glwe_dim,
                poly_size,
                cggi_parameters.base_log_pbs,
                cggi_parameters.level_pbs,
                64,
                53,
                variance_bsk,
            );
            // let input_variance = (v_pbs + v_ks) * (f64::exp2(2. * cggi_log_norm as f64)) as f64;
            let input_variance = v_pbs + v_ks;
            // println!("vpbs: {}", v_pbs.log2() * 0.5);
            // println!("v_ksk var: {}", variance_ksk.log2() * 0.5);
            // println!("v_ks var: {}", v_ks.log2() * 0.5);
            // println!("input var: {}", input_variance.log2() * 0.5);

            // tightening the search space
            let a_tighten = a.to_tighten(128, cggi_parameters.small_lwe_dim);

            let config = SquashingConstraint {
                first_variance_constraint: error::safe_variance_bound_2padbits(
                    precision, 64, p_fail,
                ),
                second_variance_constraint: error::safe_variance_bound_2padbits(
                    precision, 128, p_fail,
                ) / (pow * pow), //5.960464477539063e-08, // 0.0009765625006088146,
                security_level: 128,
                input_variance,
                input_lwe_dimension: cggi_parameters.small_lwe_dim,
            };

            let intem = config.brute_force(a_tighten.clone().iter(precision, minimal_ms_value));

            Solution {
                precision,
                log_norm: cggi_log_norm,
                intem,
            }
        })
        .collect::<Vec<_>>();
    let duration = start.elapsed();
    println!(
        "Optimization took: {:?} min",
        duration.as_secs() as f64 / 60.
    );
    write_to_file(writer, &res).unwrap();
}

pub fn write_to_file(
    mut writer: impl Write,
    res: &[Solution<SquashingParams>],
) -> Result<(), std::io::Error> {
    writeln!(
        writer,
        "  p,log(nu),  k,  N,    n, br_l,br_b, ks_l,ks_b,  cost"
    )?;

    for Solution {
        precision,
        log_norm,
        intem,
    } in res.iter()
    {
        match intem {
            Some((solution, cost)) => {
                let lwe_stddev =
                    minimal_variance_lwe(solution.small_lwe_dim, 128, 128).log2() * 0.5;

                let glwe_stddev =
                    minimal_variance_glwe(solution.glwe_dim, 1 << solution.log_poly_size, 128, 128)
                        .log2()
                        * 0.5;

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
                writeln!(
                    writer,
                    "LWE: {:.4} and GLWE: {:.4}",
                    lwe_stddev, glwe_stddev
                );
            }
            None => {}
        }
    }
    Ok(())
}
