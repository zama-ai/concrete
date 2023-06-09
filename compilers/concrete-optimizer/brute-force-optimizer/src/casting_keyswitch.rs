use crate::generic::{Problem, SequentialProblem};
use crate::{minimal_added_noise_by_modulus_switching, ExplicitRange, MyRange, Solution, STEP};
use concrete_cpu_noise_model::gaussian_noise::noise::blind_rotate::variance_blind_rotate;
use concrete_cpu_noise_model::gaussian_noise::noise::keyswitch::variance_keyswitch;
use concrete_cpu_noise_model::gaussian_noise::noise::modulus_switching::estimate_modulus_switching_noise_with_binary_key;
use concrete_optimizer::computing_cost::complexity_model::ComplexityModel;
use concrete_optimizer::noise_estimator::error;
use concrete_optimizer::parameters::{
    AtomicPatternParameters, BrDecompositionParameters, GlweParameters, KeyswitchParameters,
    KsDecompositionParameters, LweDimension,
};
use concrete_security_curves::gaussian::security::{minimal_variance_glwe, minimal_variance_lwe};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use std::io::Write;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct CastingParams {
    base_log_ks: u64,
    level_ks: u64,
}

pub struct CastingConstraint {
    // output of part 0 i.e. input of keyswitch
    pub variance_after_dot_product: f64,
    // variance with parameter set of part 1
    pub variance_modulus_switch: f64,
    pub security_level: u64,
    pub variance_constraint: f64,
    pub glwe_dim_in: u64,
    pub log_poly_size_in: u64,
    pub small_lwe_dim_out: u64,
}

impl CastingConstraint {
    fn big_lwe_dim_in(&self) -> u64 {
        let poly_size = 1 << self.log_poly_size_in;
        self.glwe_dim_in * poly_size
    }
}

impl Problem for CastingConstraint {
    type Param = CastingParams;

    fn verify(&self, param: Self::Param) -> bool {
        let poly_size = 1 << self.log_poly_size_in;

        let variance_ksk = minimal_variance_lwe(self.small_lwe_dim_out, 64, self.security_level);

        let v_ks = variance_keyswitch(
            self.big_lwe_dim_in(),
            param.base_log_ks,
            param.level_ks,
            64,
            variance_ksk,
        );

        self.variance_after_dot_product + v_ks + self.variance_modulus_switch
            < self.variance_constraint
    }

    fn cost(&self, param: Self::Param) -> f64 {
        let complexity_model = concrete_optimizer::computing_cost::cpu::CpuComplexity::default();
        let ks_parameter = KeyswitchParameters {
            input_lwe_dimension: LweDimension(self.big_lwe_dim_in()),
            output_lwe_dimension: LweDimension(self.small_lwe_dim_out),
            ks_decomposition_parameter: KsDecompositionParameters {
                level: param.level_ks,
                log2_base: param.base_log_ks,
            },
        };
        let ks_complexity = complexity_model.ks_complexity(ks_parameter, 64);
        ks_complexity
    }
}
#[derive(Clone)]
pub struct CastingSearchSpace {
    pub range_base_log_ks: MyRange,
    pub range_level_ks: MyRange,
}

impl CastingSearchSpace {
    #[allow(unused)]
    #[rustfmt::skip]
    fn par_iter(self) -> impl rayon::iter::ParallelIterator<Item=CastingParams> {
        self.range_level_ks
            .to_std_range()
            .into_par_iter().map(|_k| CastingParams {
            base_log_ks: 0,
            level_ks: 0,
        })
    }

    pub(crate) fn iter(
        self,
        precision: u64,
        minimal_ms_value: u64,
    ) -> impl Iterator<Item = CastingParams> {
        self.range_base_log_ks
            .to_std_range()
            .into_iter()
            .flat_map(move |base_log_ks| {
                self.range_level_ks
                    .clone()
                    .to_std_range()
                    .into_iter()
                    .map(move |level_ks| CastingParams {
                        base_log_ks,
                        level_ks,
                    })
            })
    }
}

// pub fn solve_all_cjp(p_fail: f64, writer: impl Write) {
//     let start = Instant::now();
//
//     let precisions = 1..24;
//     let log_norms = vec![4, 6, 8, 10];
//
//     // find the minimal added noise by the modulus switching
//     // for KS
//     let a = CastingSearchSpace {
//         range_base_log_ks: MyRange(1, 40),
//         range_level_ks: MyRange(1, 40),
//         _range_base_log_pbs: MyRange(1, 40),
//         _range_level_pbs: MyRange(1, 53),
//         range_glwe_dim: MyRange(1, 7),
//         range_log_poly_size: MyRange(8, 19),
//         range_small_lwe_dim: MyRange(500, 1500),
//     };
//     let minimal_ms_value = minimal_added_noise_by_modulus_switching(
//         (1 << a.range_log_poly_size.0) * a.range_glwe_dim.0,
//     )
//     .sqrt()
//     .ceil() as u64;
//
//     // let a = CastingSearchSpace {
//     //     range_base_log_ks: MyRange(1, 53),
//     //     range_level_ks: MyRange(1, 53),
//     //     range_base_log_pbs: MyRange(1, 53),
//     //     range_level_pbs: MyRange(1, 53),
//     //     range_glwe_dim: MyRange(1, 7),
//     //     range_log_poly_size: MyRange(8, 16),
//     //     range_small_lwe_dim: MyRange(500, 1000),
//     // };
//     let a_tighten = a.to_tighten(128);
//     let res: Vec<Solution<CastingParams>> = precisions
//         .into_par_iter()
//         .flat_map(|precision| {
//             log_norms
//                 .clone()
//                 .into_par_iter()
//                 .map(|log_norm| {
//                     let config = CastingConstraint {
//                         variance_constraint: error::safe_variance_bound_2padbits(
//                             precision, 64, p_fail,
//                         ), //5.960464477539063e-08, // 0.0009765625006088146,
//                         norm2: 1 << log_norm,
//                         security_level: 128,
//                         sum_size: 4096,
//                     };
//
//                     let intem =
//                         config.brute_force(a_tighten.clone().iter(precision, minimal_ms_value));
//
//                     Solution {
//                         precision,
//                         log_norm,
//                         intem,
//                     }
//                 })
//                 .collect::<Vec<_>>()
//         })
//         .collect::<Vec<_>>();
//     let duration = start.elapsed();
//     println!(
//         "Optimization took: {:?} min",
//         duration.as_secs() as f64 / 60.
//     );
//     write_to_file(writer, &res).unwrap();
// }
//
// pub fn write_to_file(
//     mut writer: impl Write,
//     res: &[Solution<CastingParams>],
// ) -> Result<(), std::io::Error> {
//     writeln!(
//         writer,
//         "  p,log(nu),  k,  N,    n, br_l,br_b, ks_l,ks_b,  cost"
//     )?;
//
//     for Solution {
//         precision,
//         log_norm,
//         intem,
//     } in res.iter()
//     {
//         match intem {
//             Some((solution, cost)) => {
//                 writeln!(
//                     writer,
//                     " {:2},     {:2}, {:2}, {:2}, {:4},   {:2},  {:2},   {:2},  {:2}, {:6}",
//                     precision,
//                     log_norm,
//                     solution.glwe_dim,
//                     solution.log_poly_size,
//                     solution.small_lwe_dim,
//                     solution.level_pbs,
//                     solution.base_log_pbs,
//                     solution.level_ks,
//                     solution.base_log_ks,
//                     cost
//                 )?;
//             }
//             None => {}
//         }
//     }
//     Ok(())
// }
