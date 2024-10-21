use crate::generic::{Problem, SequentialProblem};
use crate::{
    minimal_added_noise_by_modulus_switching, pbs_p_fail_from_global_p_fail, ExplicitRange,
    MyRange, Solution, STEP,
};
use concrete_cpu_noise_model::gaussian_noise::noise::blind_rotate::variance_blind_rotate;
use concrete_cpu_noise_model::gaussian_noise::noise::keyswitch::variance_keyswitch;
use concrete_cpu_noise_model::gaussian_noise::noise::modulus_switching::estimate_modulus_switching_noise_with_binary_key;
use concrete_optimizer::computing_cost::complexity_model::ComplexityModel;
use concrete_optimizer::noise_estimator::error;
use concrete_optimizer::parameters::{
    BrDecompositionParameters, GlweParameters, KeyswitchParameters, KsDecompositionParameters,
    LweDimension, PbsParameters,
};
use concrete_security_curves::gaussian::security::{minimal_variance_glwe, minimal_variance_lwe};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use std::io::Write;
use std::time::Instant;
// use rayon_cond::CondIterator;

#[derive(Debug, Clone, Copy)]
pub struct GBAParams {
    base_log_ks: u64,
    level_ks: u64,
    base_log_pbs: u64,
    level_pbs: u64,
    glwe_dim: u64,
    log_poly_size: u64,
    small_lwe_dim: u64,
    base_log_fpks: u64,
    level_fpks: u64,
}

impl GBAParams {
    fn big_lwe_dim(&self) -> u64 {
        let poly_size = 1 << self.log_poly_size;
        self.glwe_dim * poly_size
    }
}

struct GBAConstraint {
    variance_constraint: f64,
    log_norm2: u64,
    security_level: u64,
    sum_size: u64,
    precision: u64,
    nb_inputs: u64,
}

impl Problem for GBAConstraint {
    type Param = GBAParams;

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
        let square = |x| x * x;
        let v_after_functions =
            v_pbs * (1 << self.precision) as f64 * square((1 << self.precision) as f64 - 1.);

        let v_pp_keyswitch = poly_size as f64
            * variance_keyswitch(
                param.big_lwe_dim(),
                param.base_log_fpks,
                param.level_fpks,
                64,
                variance_bsk,
            );

        let v_tree_pbs =
            // cim noise
            v_after_functions
                // noise other layers
                + (self.nb_inputs - 1) as f64 * (v_pp_keyswitch + v_pbs);
        let v_ms = estimate_modulus_switching_noise_with_binary_key(
            param.small_lwe_dim,
            param.log_poly_size,
            64,
        );

        v_tree_pbs * (1 << (2 * self.log_norm2)) as f64 + v_ks + v_ms < self.variance_constraint
    }

    fn cost(&self, param: Self::Param) -> f64 {
        let complexity_model = concrete_optimizer::computing_cost::cpu::CpuComplexity::default();
        let multisum_complexity = self.nb_inputs as f64
            * complexity_model.levelled_complexity(
                self.sum_size,
                LweDimension(param.big_lwe_dim()),
                64,
            );
        let ks_decomposition_parameter = KsDecompositionParameters {
            level: param.level_ks,
            log2_base: param.base_log_ks,
        };
        let ks_parameter = KeyswitchParameters {
            input_lwe_dimension: LweDimension(param.big_lwe_dim()),
            output_lwe_dimension: LweDimension(param.small_lwe_dim),
            ks_decomposition_parameter,
        };
        let ks_complexity = complexity_model.ks_complexity(ks_parameter, 64);

        let pbs_decomposition_parameter = BrDecompositionParameters {
            level: param.level_pbs,
            log2_base: param.base_log_pbs,
        };

        let pbs_parameter = PbsParameters {
            internal_lwe_dimension: LweDimension(param.small_lwe_dim),
            br_decomposition_parameter: pbs_decomposition_parameter,
            output_glwe_params: GlweParameters {
                log2_polynomial_size: param.log_poly_size,
                glwe_dimension: param.glwe_dim,
            },
        };
        let pbs_complexity = complexity_model.pbs_complexity(pbs_parameter, 64);

        let fft_cost = (1 << param.log_poly_size) as f64 * param.log_poly_size as f64;
        let k = pbs_parameter.output_glwe_params.glwe_dimension as f64;
        let message_modulus = (1 << self.precision) as f64;

        let complexity_cim_pbs = multisum_complexity
            + ks_complexity
            + pbs_complexity
            // FFT
            + (k + 1.) * fft_cost +
            // element wise multplication
            (k + 1.) * self.nb_inputs as f64 * f64::powi(message_modulus, self.nb_inputs as i32 - 1) * (1 << param.log_poly_size) as f64 +
            // iFFT
            (k + 1.) * self.nb_inputs as f64 * f64::powi(message_modulus, self.nb_inputs as i32 - 1) * fft_cost;

        let pp_keyswitch_decomposition_parameter = KsDecompositionParameters {
            level: param.level_fpks,
            log2_base: param.base_log_fpks,
        };
        let pp_keyswitch_parameter = KeyswitchParameters {
            input_lwe_dimension: LweDimension(param.big_lwe_dim()),
            output_lwe_dimension: LweDimension(param.big_lwe_dim()),
            ks_decomposition_parameter: pp_keyswitch_decomposition_parameter,
        };
        let complexity_packing_ks = message_modulus
            * (complexity_model.ks_complexity(pp_keyswitch_parameter, 64)
                + (1 << param.log_poly_size) as f64 * (k + 1.));
        // addition for packing ks
        // );

        let complexity_all_ppks = self.nb_inputs as f64
            * complexity_packing_ks
            * ((f64::powi(message_modulus, self.nb_inputs as i32 - 1) - 1.)
                / (message_modulus - 1.));

        let complexity_all_pbs = self.nb_inputs as f64
            * pbs_complexity
            * ((f64::powi(message_modulus, self.nb_inputs as i32 - 1) - 1.)
                / (message_modulus - 1.));
        let complexity_all_ppks_pbs = complexity_all_pbs + complexity_all_ppks;

        complexity_cim_pbs + complexity_all_ppks_pbs + (self.nb_inputs - 1) as f64 * ks_complexity
    }
}

struct GBASearchSpace {
    range_base_log_ks: MyRange,
    range_level_ks: MyRange,
    range_base_log_pbs: MyRange,
    range_level_pbs: MyRange,
    range_glwe_dim: MyRange,
    range_log_poly_size: MyRange,
    range_small_lwe_dim: MyRange,
    range_base_log_fpks: MyRange,
    range_level_fpks: MyRange,
}

impl GBASearchSpace {
    fn to_tighten(&self, security_level: u64) -> GBASearchSpaceTighten {
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
                    for level in self.range_level_pbs.to_std_range() {
                        let mut current_minimal_noise_for_a_given_level = current_minimal_noise;
                        let mut current_pair = (0, 0);
                        for baselog in self.range_base_log_pbs.to_std_range() {
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

        // FPKS
        let mut fpks_decomp = vec![];
        for log_N in self.range_log_poly_size.to_std_range() {
            for k in self.range_glwe_dim.to_std_range() {
                for _n in self.range_small_lwe_dim.to_std_range() {
                    let mut current_minimal_noise = f64::INFINITY;
                    for level in self.range_level_fpks.to_std_range() {
                        let mut current_minimal_noise_for_a_given_level = current_minimal_noise;
                        let mut current_pair = (0, 0);
                        for baselog in self.range_base_log_fpks.to_std_range() {
                            let variance_bsk =
                                minimal_variance_glwe(k, 1 << log_N, 64, security_level);
                            let v_pp_keyswitch = (1 << log_N) as f64
                                * variance_keyswitch(
                                    (1 << log_N) * k,
                                    baselog,
                                    level,
                                    64,
                                    variance_bsk,
                                );
                            if v_pp_keyswitch <= current_minimal_noise_for_a_given_level {
                                current_minimal_noise_for_a_given_level = v_pp_keyswitch;
                                current_pair = (baselog, level)
                            }
                        }
                        if current_minimal_noise_for_a_given_level < current_minimal_noise {
                            fpks_decomp.push(current_pair);
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
        fpks_decomp.sort();
        fpks_decomp.dedup();
        println!("Only {} couples left for keyswitch", ks_decomp.len());
        println!("Only {} couples left for bootstrap", pbs_decomp.len());
        println!("Only {} couples left for fpks", fpks_decomp.len());

        GBASearchSpaceTighten {
            range_base_log_level_ks: ExplicitRange(ks_decomp.clone()),
            range_base_log_level_pbs: ExplicitRange(pbs_decomp.clone()),
            range_base_log_level_fpks: ExplicitRange(fpks_decomp.clone()),
            range_glwe_dim: self.range_glwe_dim,
            range_log_poly_size: self.range_log_poly_size,
            range_small_lwe_dim: self.range_small_lwe_dim,
        }
    }
}

#[derive(Clone)]
struct GBASearchSpaceTighten {
    range_base_log_level_ks: ExplicitRange,
    range_base_log_level_pbs: ExplicitRange,
    range_base_log_level_fpks: ExplicitRange,
    range_glwe_dim: MyRange,
    range_log_poly_size: MyRange,
    range_small_lwe_dim: MyRange,
}

impl GBASearchSpaceTighten {
    #[allow(unused)]
    #[rustfmt::skip]
    fn par_iter(self) -> impl rayon::iter::ParallelIterator<Item=GBAParams> {
        self.range_glwe_dim
            .to_std_range()
            .into_par_iter().map(|_k| GBAParams {
            base_log_ks: 0,
            level_ks: 0,
            base_log_pbs: 0,
            level_pbs: 0,
            glwe_dim: 0,
            log_poly_size: 0,
            small_lwe_dim: 0,
            base_log_fpks: 0,
            level_fpks: 0,
        })
    }

    fn iter(self, precision: u64, minimal_ms_value: u64) -> impl Iterator<Item = GBAParams> {
        self.range_base_log_level_fpks.clone().into_iter().flat_map(
            move |(base_log_fpks, level_fpks)| {
                let interm_range_base_log_level_pbs = self.range_base_log_level_pbs.clone();
                self.range_base_log_level_ks.clone().into_iter().flat_map(
                    move |(base_log_ks, level_ks)| {
                        interm_range_base_log_level_pbs
                            .clone()
                            .into_iter()
                            .flat_map(move |(base_log_pbs, level_pbs)| {
                                self.range_glwe_dim
                                    .to_std_range()
                                    .flat_map(move |glwe_dim| {
                                        self.range_log_poly_size
                                            .to_std_range_poly_size(precision + minimal_ms_value)
                                            .flat_map(move |log_poly_size| {
                                                self.range_small_lwe_dim
                                                    .to_std_range()
                                                    .step_by(STEP)
                                                    .map(move |small_lwe_dim| GBAParams {
                                                        base_log_ks,
                                                        level_ks,
                                                        base_log_pbs,
                                                        level_pbs,
                                                        glwe_dim,
                                                        log_poly_size,
                                                        small_lwe_dim,
                                                        base_log_fpks,
                                                        level_fpks,
                                                    })
                                            })
                                    })
                            })
                    },
                )
            },
        )
    }
}

pub fn solve_all_gba(p_fail: f64, writer: impl Write) {
    let nb_inputs = 2;
    let start = Instant::now();

    let precisions = 1..9;
    let log_norms = vec![4, 6, 8, 10];

    // find the minimal added noise by the modulus switching
    // for KS
    let a = GBASearchSpace {
        range_base_log_ks: MyRange(1, 40),
        range_level_ks: MyRange(1, 25),
        range_base_log_pbs: MyRange(1, 40),
        range_level_pbs: MyRange(1, 25),
        range_glwe_dim: MyRange(1, 7),
        range_log_poly_size: MyRange(8, 16),
        range_small_lwe_dim: MyRange(500, 1000),
        range_base_log_fpks: MyRange(1, 40),
        range_level_fpks: MyRange(1, 25),
    };
    let minimal_ms_value = minimal_added_noise_by_modulus_switching(
        (1 << a.range_log_poly_size.0) * a.range_glwe_dim.0,
    )
    .sqrt()
    .ceil() as u64;
    let a_tighten = a.to_tighten(128);
    let res: Vec<Solution<GBAParams>> = precisions
        .into_par_iter()
        .flat_map(|precision| {
            log_norms
                .clone()
                .into_par_iter()
                .map(|log_norm| {
                    let config = GBAConstraint {
                        variance_constraint: error::safe_variance_bound_2padbits(
                            precision,
                            64,
                            gba_p_fail_per_pbs(precision, nb_inputs, p_fail),
                        ),
                        log_norm2: log_norm,
                        security_level: 128,
                        sum_size: 4096,
                        precision,
                        nb_inputs,
                    };

                    let intem =
                        config.brute_force(a_tighten.clone().iter(precision, minimal_ms_value));

                    Solution {
                        precision,
                        log_norm,
                        intem,
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let duration = start.elapsed();
    println!(
        "Optimization took: {:?} min",
        duration.as_secs() as f64 / 60.
    );
    write_to_file(writer, &res, nb_inputs).unwrap();
}

pub fn gba_p_fail_per_pbs(precision: u64, nb_input: u64, p_fail: f64) -> f64 {
    let message_modulus = (1 << precision) as f64;
    let nb_total_pbs = (nb_input as f64)
        * ((f64::powi(message_modulus, (nb_input as i32) - 1) - 1.) / (message_modulus - 1.))
        + 1.;
    pbs_p_fail_from_global_p_fail(nb_total_pbs as u64, p_fail)
}

pub fn write_to_file(
    mut writer: impl Write,
    res: &[Solution<GBAParams>],
    nb_inputs: u64,
) -> Result<(), std::io::Error> {
    writeln!(
        writer,
        "  p,log(nu), k,  N,    n, br_l,br_b, ks_l,ks_b,fpks_l,fpks_b,  cost"
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
                    " {:2},     {:2}, {:2}, {:2}, {:4},   {:2},  {:2},   {:2},  {:2},    {:2},  {:2}, {:6}",
                    nb_inputs * precision,
                    log_norm,
                    solution.glwe_dim,
                    solution.log_poly_size,
                    solution.small_lwe_dim,
                    solution.level_pbs,
                    solution.base_log_pbs,
                    solution.level_ks,
                    solution.base_log_ks,
                    solution.level_fpks,
                    solution.base_log_fpks,
                    cost
                )?;
        }
    }
    Ok(())
}
