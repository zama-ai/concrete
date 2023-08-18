use crate::generic::{Problem, SequentialProblem};
use crate::{ExplicitRange, MyRange, Solution, STEP};
use concrete_cpu_noise_model::gaussian_noise::noise::blind_rotate::variance_blind_rotate;
use concrete_cpu_noise_model::gaussian_noise::noise::keyswitch::variance_keyswitch;
use concrete_cpu_noise_model::gaussian_noise::noise::modulus_switching::estimate_modulus_switching_noise_with_binary_key;
use concrete_cpu_noise_model::gaussian_noise::noise::tensor_product::variance_glwe_relin;
use concrete_cpu_noise_model::gaussian_noise::noise::tensor_product::variance_tensor_product_with_glwe_relin;
use concrete_cpu_noise_model::gaussian_noise::noise::trace_packing::variance_trace_packing_keyswitch;
use concrete_optimizer::computing_cost::complexity_model::ComplexityModel;
use concrete_optimizer::noise_estimator::error;
use concrete_optimizer::parameters::{
    BrDecompositionParameters, GlweParameters, KeyswitchParameters, KsDecompositionParameters,
    LweDimension, PbsParameters, TensorProductGlweParameters, TracePackingParameters,
};
use concrete_security_curves::gaussian::security::{minimal_variance_glwe, minimal_variance_lwe};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::io::Write;
// use rayon_cond::CondIterator;

//todo fix the encoding, determine how many LWE we want to pack in an GLWE and in what positions
static INDEX_SET1: &'static [usize] = &[0, 1, 2, 3, 4];
static INDEX_SET2: &'static [usize] = &[0, 5, 10, 15, 20];

#[derive(Debug, Clone, Copy)]
pub struct TensorProductParams {
    base_log_ks: u64,
    level_ks: u64,
    base_log_pbs: u64,
    level_pbs: u64,
    base_log_pk: u64,
    level_pk: u64,
    base_log_relin: u64,
    level_relin: u64,
    glwe_dim: u64,
    log_poly_size: u64,
    small_lwe_dim: u64,
}

impl TensorProductParams {
    fn big_lwe_dim(&self) -> u64 {
        let poly_size = 1 << self.log_poly_size;
        self.glwe_dim * poly_size
    }
}

struct TensorProductConstraint {
    variance_constraint: f64,
    log_norm2: u64,
    security_level: u64,
    //sum_size: u64
}

impl Problem for TensorProductConstraint {
    type Param = TensorProductParams;

    fn verify(&self, param: Self::Param) -> bool {
        let poly_size = 1 << param.log_poly_size;

        let variance_ksk = minimal_variance_lwe(param.small_lwe_dim, 64, self.security_level);
        let variance_pk = minimal_variance_glwe(param.glwe_dim, poly_size, 64, self.security_level);
        let variance_rlk =
            minimal_variance_glwe(param.glwe_dim, poly_size, 64, self.security_level);

        let variance_bsk =
            minimal_variance_glwe(param.glwe_dim, poly_size, 64, self.security_level);
        let v_pbs = variance_blind_rotate(
            param.small_lwe_dim,
            param.glwe_dim,
            poly_size,
            param.base_log_pbs,
            param.level_pbs,
            64,
            variance_bsk,
        );

        let v_ks_trace_packing = variance_trace_packing_keyswitch(
            v_pbs * (1 << (2 * self.log_norm2)) as f64,
            param.glwe_dim,
            poly_size,
            param.base_log_pk,
            param.level_pk,
            64,
            variance_pk,
        );

        let v_tensor_ks_relin = variance_tensor_product_with_glwe_relin(
            param.glwe_dim,
            poly_size,
            64,
            v_ks_trace_packing,
            v_ks_trace_packing,
            //TODO define proper values
            1 << 6,
            1 << 4,
            20,
            20,
            param.base_log_relin,
            param.level_relin,
            variance_rlk,
        );

        let v_ks = variance_keyswitch(
            param.big_lwe_dim(),
            param.base_log_ks,
            param.level_ks,
            64,
            variance_ksk,
        );

        let v_ms = estimate_modulus_switching_noise_with_binary_key(
            param.small_lwe_dim,
            param.log_poly_size,
            64,
        );

        //println!("error of trace packing key switch {}", v_ks_trace_packing);
        //println!("error of the tensor product {}", v_tensor_ks_relin);
        //println!("error ks {}", v_ks);
        //println!("error ms {}", v_ms);
        //println!(
        //    "error of trace packing and tensor product {}",
        //    v_tensor_ks_relin + v_ks + v_ms
        //);
        //println!("error {} < constraint {}", v_tensor_ks_relin + v_ks + v_ms, self.variance_constraint);
        //if v_tensor_ks_relin + v_ks + v_ms < self.variance_constraint 
        //{
        //    println!("error {} < constraint {}", v_tensor_ks_relin + v_ks + v_ms, self.variance_constraint);
        //    println!("q_square bigger than b2l {}, ",2_f64.powi(2 * 64 as i32) > 2_f64.powi((param.base_log_relin * 2 * param.level_relin) as i32));
        //}
    
        2_f64.powi(2 * 64 as i32) > 2_f64.powi((param.base_log_relin * 2 * param.level_relin) as i32) &&
        v_tensor_ks_relin + v_ks + v_ms < self.variance_constraint
    }

    fn cost(&self, params: Self::Param) -> f64 {
        let ciphertext_modulus_log = 64 as u32;
        let complexity_model = concrete_optimizer::computing_cost::cpu::CpuComplexity::default();

        let ks_parameters = KeyswitchParameters {
            input_lwe_dimension: LweDimension(params.big_lwe_dim()),
            output_lwe_dimension: LweDimension(params.small_lwe_dim),
            ks_decomposition_parameter: KsDecompositionParameters {
                level: params.level_ks,
                log2_base: params.base_log_ks,
            },
        };

        let ks_complexity = complexity_model.ks_complexity(ks_parameters, ciphertext_modulus_log);

        let pbs_decomposition_parameter = BrDecompositionParameters {
            level: params.level_pbs,
            log2_base: params.base_log_pbs,
        };
        let pbs_parameters = PbsParameters {
            internal_lwe_dimension: LweDimension(params.small_lwe_dim),
            br_decomposition_parameter: pbs_decomposition_parameter,
            output_glwe_params: GlweParameters {
                log2_polynomial_size: params.log_poly_size,
                glwe_dimension: params.glwe_dim,
            },
        };

        let pbs_complexity =
            complexity_model.pbs_complexity(pbs_parameters, ciphertext_modulus_log);

        let glwe_packed = GlweParameters {
            log2_polynomial_size: params.log_poly_size,
            glwe_dimension: params.glwe_dim,
        };

        let packed_ks_parameters = KsDecompositionParameters {
            level: params.level_pk,
            log2_base: params.base_log_pk,
        };

        let trace_packing_parameter_1 = TracePackingParameters {
            input_lwe_dimension: params.small_lwe_dim,
            output_glwe_params: glwe_packed,
            ks_decomposition_parameter: packed_ks_parameters,
        };

        let pk_complexity_2 = complexity_model.trace_packing_complexity(
            trace_packing_parameter_1,
            ciphertext_modulus_log,
            INDEX_SET1,
        );

        let trace_packing_parameter_2 = TracePackingParameters {
            input_lwe_dimension: params.small_lwe_dim,
            output_glwe_params: glwe_packed,
            ks_decomposition_parameter: packed_ks_parameters,
        };

        let pk_complexity_1 = complexity_model.trace_packing_complexity(
            trace_packing_parameter_2,
            ciphertext_modulus_log,
            INDEX_SET2,
        );

        let relin_ks_parameters = KsDecompositionParameters {
            level: params.level_relin,
            log2_base: params.base_log_relin,
        };

        //input and output have the same parameters
        let glwe_relin = GlweParameters {
            log2_polynomial_size: params.log_poly_size,
            glwe_dimension: params.glwe_dim,
        };

        let tensor_product_parameter = TensorProductGlweParameters {
            input_glwe_params: glwe_relin,
            output_glwe_params: glwe_relin,
            ks_decomposition_parameters: relin_ks_parameters,
        };

        let tensor_product_relin_complexity = complexity_model
            .tensor_product_complexity(tensor_product_parameter, ciphertext_modulus_log);

        pk_complexity_1
            + pk_complexity_2
            + tensor_product_relin_complexity
            + ks_complexity
            + pbs_complexity
    }
}

struct TensorProductSearchSpace {
    range_base_log_ks: MyRange,
    range_level_ks: MyRange,
    range_base_log_pbs: MyRange,
    range_level_pbs: MyRange,
    range_base_log_pk: MyRange,
    range_level_pk: MyRange,
    range_base_log_relin: MyRange,
    range_level_relin: MyRange,
    range_glwe_dim: MyRange,
    range_log_poly_size: MyRange,
    range_small_lwe_dim: MyRange,
}

impl TensorProductSearchSpace {
    fn to_tighten(&self, security_level: u64) -> TensorProductSearchSpaceTighten {
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
        // Packing key switch
        let mut pks_decomp = vec![];
        for log_N in self.range_log_poly_size.to_std_range() {
            for k in self.range_glwe_dim.to_std_range() {
                for n in self.range_small_lwe_dim.to_std_range() {
                    let mut current_minimal_noise = f64::INFINITY;
                    for level in self.range_level_pk.to_std_range() {
                        let mut current_minimal_noise_for_a_given_level = current_minimal_noise;
                        let mut current_pair = (0, 0);
                        for baselog in self.range_base_log_pk.to_std_range() {
                            let variance_input = minimal_variance_lwe(n, 64, security_level);
                            let variance_pk =
                                minimal_variance_glwe(k, 1 << log_N, 64, security_level);
                            let v_pks = variance_trace_packing_keyswitch(
                                variance_input,
                                k,
                                1 << log_N,
                                baselog,
                                level,
                                64,
                                variance_pk,
                            );
                            if v_pks <= current_minimal_noise_for_a_given_level {
                                current_minimal_noise_for_a_given_level = v_pks;
                                current_pair = (baselog, level)
                            }
                        }
                        if current_minimal_noise_for_a_given_level < current_minimal_noise {
                            pks_decomp.push(current_pair);
                            current_minimal_noise = current_minimal_noise_for_a_given_level;
                        }
                    }
                }
            }
        }
        // relinearization
        let mut relin_decomp = vec![];
        for log_N in self.range_log_poly_size.to_std_range() {
            for k in self.range_glwe_dim.to_std_range() {
                //interestingly the glwe-glwe relin does not use the lwe dimension
                //for n in self.range_small_lwe_dim.to_std_range() {
                let mut current_minimal_noise = f64::INFINITY;
                for level in self.range_level_relin.to_std_range() {
                    let mut current_minimal_noise_for_a_given_level = current_minimal_noise;
                    let mut current_pair = (0, 0);
                    for baselog in self.range_base_log_relin.to_std_range() {
                        let variance_input =
                            minimal_variance_glwe(k, 1 << log_N, 64, security_level);
                        let variance_rlk = minimal_variance_glwe(k, 1 << log_N, 64, security_level);
                        let v_relin = variance_glwe_relin(
                            variance_input,
                            k,
                            1 << log_N,
                            64,
                            baselog,
                            level,
                            variance_rlk,
                        );
                        //println!("var relin {}",v_relin);
                        if v_relin > 0.0 && v_relin <= current_minimal_noise_for_a_given_level {
                            current_minimal_noise_for_a_given_level = v_relin;
                            current_pair = (baselog, level)
                        }
                    }
                    if current_minimal_noise_for_a_given_level < current_minimal_noise {
                        relin_decomp.push(current_pair);
                        current_minimal_noise = current_minimal_noise_for_a_given_level;
                    }
                }
                //}
            }
        }

        ks_decomp.sort();
        ks_decomp.dedup();
        pbs_decomp.sort();
        pbs_decomp.dedup();
        pks_decomp.sort();
        pks_decomp.dedup();
        relin_decomp.sort();
        relin_decomp.dedup();

        println!("Only {} couples left for keyswitch", ks_decomp.len());
        println!("Only {} couples left for bootstrap", pbs_decomp.len());
        println!(
            "Only {} couples left for trace packing keyswitch",
            pks_decomp.len()
        );
        println!(
            "Only {} couples left for relinearization",
            relin_decomp.len()
        );

        TensorProductSearchSpaceTighten {
            range_base_log_level_ks: ExplicitRange(ks_decomp.clone()),
            range_base_log_level_pbs: ExplicitRange(pbs_decomp.clone()),
            range_base_log_level_pks: ExplicitRange(pks_decomp.clone()),
            range_base_log_level_relin: ExplicitRange(relin_decomp.clone()),
            range_glwe_dim: self.range_glwe_dim,
            range_log_poly_size: self.range_log_poly_size,
            range_small_lwe_dim: self.range_small_lwe_dim,
        }
    }
}

#[derive(Clone)]
struct TensorProductSearchSpaceTighten {
    range_base_log_level_ks: ExplicitRange,
    range_base_log_level_pbs: ExplicitRange,
    range_base_log_level_pks: ExplicitRange,
    range_base_log_level_relin: ExplicitRange,
    range_glwe_dim: MyRange,
    range_log_poly_size: MyRange,
    range_small_lwe_dim: MyRange,
}

impl TensorProductSearchSpaceTighten {
    #[allow(unused)]
    #[rustfmt::skip]
    fn par_iter(self) -> impl rayon::iter::ParallelIterator<Item=TensorProductParams> {
        self.range_glwe_dim
            .to_std_range()
            .into_par_iter().map(|_k| TensorProductParams {
            base_log_ks: 0,
            level_ks: 0,
            base_log_pbs: 0,
            level_pbs: 0,
            base_log_pk: 0,
            level_pk: 0,
            base_log_relin: 0,
            level_relin: 0,
            glwe_dim: 0,
            log_poly_size: 0,
            small_lwe_dim: 0,
        })
    }

    fn iter(self, _precision: u64) -> impl Iterator<Item = TensorProductParams> {
        self.range_base_log_level_ks
            .into_iter()
            .flat_map(move |(base_log_ks, level_ks)| {
                let range_base_log_level_pbs_clone = self.range_base_log_level_pbs.clone();
                let range_base_log_level_pks_clone = self.range_base_log_level_pks.clone();
                let range_base_log_level_relin_clone = self.range_base_log_level_relin.clone();
                range_base_log_level_pbs_clone.into_iter().flat_map(
                    move |(base_log_pbs, level_pbs)| {
                        let range_base_log_level_relin_clone2 =
                            range_base_log_level_relin_clone.clone();
                        let range_base_log_level_pks_clone2 =
                            range_base_log_level_pks_clone.clone();
                        range_base_log_level_pks_clone2.into_iter().flat_map(
                            move |(base_log_pk, level_pk)| {
                                range_base_log_level_relin_clone2
                                    .clone()
                                    .into_iter()
                                    .flat_map(move |(base_log_relin, level_relin)| {
                                        self.range_glwe_dim.to_std_range().flat_map(
                                            move |glwe_dim| {
                                                self.range_log_poly_size.to_std_range().flat_map(
                                                    move |log_poly_size| {
                                                        self.range_small_lwe_dim
                                                            .to_std_range()
                                                            .step_by(STEP)
                                                            .map(move |small_lwe_dim| {
                                                                TensorProductParams {
                                                                    base_log_ks,
                                                                    level_ks,
                                                                    base_log_pbs,
                                                                    level_pbs,
                                                                    base_log_pk,
                                                                    level_pk,
                                                                    base_log_relin,
                                                                    level_relin,
                                                                    glwe_dim,
                                                                    log_poly_size,
                                                                    small_lwe_dim,
                                                                }
                                                            })
                                                    },
                                                )
                                            },
                                        )
                                    })
                            },
                        )
                    },
                )
            })
    }
}

pub fn solve_all_tp(p_fail: f64, writer: impl Write) {
    // -> Vec<(u64, u64, Option<(V0Params, f64)>)> {
    // let p_fail = 1.0 - 0.999_936_657_516;

    //let precisions = 1..9;
    // let precisions_iter = ParallelIterator::new(precisions);
    let precisions: Vec<u64> = vec![1];

    //let log_norms = vec![4, 6, 8, 10];
    let log_norms = precisions.clone();

    let a = TensorProductSearchSpace {
        range_base_log_ks: MyRange(1, 53),
        range_level_ks: MyRange(1, 53),
        range_base_log_pbs: MyRange(1, 53),
        range_level_pbs: MyRange(1, 53),
        range_base_log_pk: MyRange(1, 53),
        range_level_pk: MyRange(1, 53),
        range_base_log_relin: MyRange(1, 53),
        range_level_relin: MyRange(1, 53),
        range_glwe_dim: MyRange(1, 7),
        range_log_poly_size: MyRange(8, 18),
        range_small_lwe_dim: MyRange(500, 4000),
    };

    let a_tighten = a.to_tighten(128);
    let res: Vec<Solution<TensorProductParams>> = precisions
        .into_par_iter()
        .flat_map(|precision| {
            log_norms
                .clone()
                .into_par_iter()
                .map(|log_norm| {
                    let config = TensorProductConstraint {
                        variance_constraint: error::safe_variance_bound_2padbits(
                            precision, 64, p_fail,
                        ), //5.960464477539063e-08, // 0.0009765625006088146,
                        log_norm2: log_norm,
                        security_level: 128,
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
    res: &[Solution<TensorProductParams>],
) -> Result<(), std::io::Error> {
    writeln!(
        writer,
        "  p,log(nu), k,  N, n, br_l, br_b, ks_l, ks_b, pks_l, pks_b, relin_l, relin_b, cost"
    )?;

    for Solution {
        precision,
        log_norm,
        intem,
    } in res.iter()
    {
        match intem {
            Some((solution, cost)) => {
                writeln!(
                    writer,
                    " {:2}, {:2}, {:2}, {:2}, {:4}, {:2}, {:2}, {:2}, {:2}, {:2}, {:2}, {:2}, {:2}, {:6}",
                    precision,
                    log_norm,
                    solution.glwe_dim,
                    solution.log_poly_size,
                    solution.small_lwe_dim,
                    solution.level_pbs,
                    solution.base_log_pbs,
                    solution.level_ks,
                    solution.base_log_ks,
                    solution.level_pk,
                    solution.base_log_pk,
                    solution.level_relin,
                    solution.base_log_relin,
                    cost
                )?;
            }
            None => {}
        }
    }
    Ok(())
}
