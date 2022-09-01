#![allow(dead_code)]

use concrete_integer::client_key::{radix_decomposition, VecLength};
use concrete_integer::crt::{gen_key_id, CRTVecClientKey, CRTVecServerKey};
use concrete_integer::keycache::KEY_CACHE;
use concrete_integer::parameters::*;
use concrete_integer::treepbs::TreepbsKey;
use concrete_integer::wopbs::WopbsKeyV0;
use concrete_integer::{gen_keys, Ciphertext, ServerKey};
use concrete_shortint::keycache::KEY_CACHE_WOPBS;
use concrete_shortint::parameters::parameters_wopbs_message_carry::get_parameters_from_message_and_carry_wopbs;
use concrete_shortint::parameters::{get_parameters_from_message_and_carry, DEFAULT_PARAMETERS};
use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;

criterion_group!(
    to_be_reworked,
    smart_block_mul,
    radmodint_unchecked_mul,
    radmodint_unchecked_mul_many_sizes,
    crt_unchecked_mul_many_sizes,
    crt_unchecked_mul_many_sizes_parallelized,
    crt_arithmetic_many_sizes,
    crt_arithmetic_many_sizes_parallelized,
    crt,
    two_block_pbs,
    two_block_pbs,
    two_block_pbs_base,
    three_block_pbs,
    three_block_pbs_base,
    // radmodint_wopbs,
    // radmodint_wopbs_32_bits,
    // radmodint_wopbs_16bits_param_2_2_8_blocks,
    // radmodint_wopbs_16bits_param_4_4_4_blocks,
    concrete_integer_unchecked_mul_crt_16_bits,
    concrete_integer_unchecked_add_crt_16_bits,
    concrete_integer_unchecked_clean_carry_crt_16_bits,
    concrete_integer_unchecked_mul_crt_32_bits,
    concrete_integer_unchecked_add_crt_32_bits,
    concrete_integer_unchecked_clean_carry_crt_32_bits,
);

#[allow(unused_imports)]
use concrete_shortint::parameters::{
    PARAM_MESSAGE_1_CARRY_1, PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3,
    PARAM_MESSAGE_4_CARRY_4,
};

macro_rules! named_param {
    ($param:ident) => {
        (stringify!($param), $param)
    };
}

struct Parameters {
    block_parameters: concrete_shortint::Parameters,
    num_block: VecLength,
}

const BLOCK_4_MESSAGE_2_CARRY_2: Parameters = Parameters {
    block_parameters: PARAM_MESSAGE_2_CARRY_2,
    num_block: VecLength(4),
};

const BLOCK_4_MESSAGE_3_CARRY_3: Parameters = Parameters {
    block_parameters: PARAM_MESSAGE_3_CARRY_3,
    num_block: VecLength(4),
};

const SERVER_KEY_BENCH_PARAMS: [(&str, Parameters); 2] = [
    named_param!(BLOCK_4_MESSAGE_2_CARRY_2),
    named_param!(BLOCK_4_MESSAGE_3_CARRY_3),
];

fn smart_neg(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("smart_neg");

    for (param_name, param) in SERVER_KEY_BENCH_PARAMS {
        let (cks, sks) = KEY_CACHE.get_from_params(param.block_parameters, param.num_block);

        let mut rng = rand::thread_rng();

        let modulus =
            1_u64 << (param.block_parameters.message_modulus.0 * param.num_block.0) as u64;

        let clear_0 = rng.gen::<u64>() % modulus;

        let mut ct = cks.encrypt(clear_0);

        let bench_id = param_name;
        bench_group.bench_function(bench_id, |b| {
            b.iter(|| {
                sks.smart_neg(&mut ct);
            })
        });
    }

    bench_group.finish()
}

fn full_propagate(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("full_propagate");

    for (param_name, param) in SERVER_KEY_BENCH_PARAMS {
        let (cks, sks) = KEY_CACHE.get_from_params(param.block_parameters, param.num_block);

        let mut rng = rand::thread_rng();

        let modulus =
            1_u64 << (param.block_parameters.message_modulus.0 * param.num_block.0) as u64;

        let clear_0 = rng.gen::<u64>() % modulus;

        let mut ct = cks.encrypt(clear_0);

        let bench_id = param_name;
        bench_group.bench_function(bench_id, |b| {
            b.iter(|| {
                sks.full_propagate(&mut ct);
            })
        });
    }

    bench_group.finish()
}

fn bench_server_key_binary_function<F>(c: &mut Criterion, bench_name: &str, binary_op: F)
where
    F: Fn(&ServerKey, &mut Ciphertext, &mut Ciphertext),
{
    let mut bench_group = c.benchmark_group(bench_name);

    for (param_name, param) in SERVER_KEY_BENCH_PARAMS {
        let (cks, sks) = KEY_CACHE.get_from_params(param.block_parameters, param.num_block);

        let mut rng = rand::thread_rng();

        let modulus =
            1_u64 << (param.block_parameters.message_modulus.0 * param.num_block.0) as u64;

        let clear_0 = rng.gen::<u64>() % modulus;
        let clear_1 = rng.gen::<u64>() % modulus;

        let mut ct_0 = cks.encrypt(clear_0);
        let mut ct_1 = cks.encrypt(clear_1);

        let bench_id = format!("{}::{}", bench_name, param_name);
        bench_group.bench_function(&bench_id, |b| {
            b.iter(|| {
                binary_op(&sks, &mut ct_0, &mut ct_1);
            })
        });
    }

    bench_group.finish()
}

fn bench_server_key_binary_scalar_function<F>(c: &mut Criterion, bench_name: &str, binary_op: F)
where
    F: Fn(&ServerKey, &mut Ciphertext, u64),
{
    let mut bench_group = c.benchmark_group(bench_name);

    for (param_name, param) in SERVER_KEY_BENCH_PARAMS {
        let (cks, sks) = KEY_CACHE.get_from_params(param.block_parameters, param.num_block);

        let mut rng = rand::thread_rng();

        let modulus =
            1_u64 << (param.block_parameters.message_modulus.0 * param.num_block.0) as u64;

        let clear_0 = rng.gen::<u64>() % modulus;
        let clear_1 = rng.gen::<u64>() % modulus;

        let mut ct_0 = cks.encrypt(clear_0);

        let bench_id = format!("{}::{}", bench_name, param_name);
        bench_group.bench_function(&bench_id, |b| {
            b.iter(|| {
                binary_op(&sks, &mut ct_0, clear_1);
            })
        });
    }

    bench_group.finish()
}

macro_rules! define_server_key_bench_fn (
  ($server_key_method:ident) => {
      fn $server_key_method(c: &mut Criterion) {
          bench_server_key_binary_function(
              c,
              concat!("ServerKey::", stringify!($server_key_method)),
              |server_key, lhs, rhs| {
                server_key.$server_key_method(lhs, rhs);
          })
      }
  }
);

macro_rules! define_server_key_bench_scalar_fn (
  ($server_key_method:ident) => {
      fn $server_key_method(c: &mut Criterion) {
          bench_server_key_binary_scalar_function(
              c,
              concat!("ServerKey::", stringify!($server_key_method)),
              |server_key, lhs, rhs| {
                server_key.$server_key_method(lhs, rhs);
          })
      }
  }
);

define_server_key_bench_fn!(smart_add);
define_server_key_bench_fn!(smart_add_parallelized);
define_server_key_bench_fn!(smart_sub);
define_server_key_bench_fn!(smart_sub_parallelized);
define_server_key_bench_fn!(smart_mul);
define_server_key_bench_fn!(smart_mul_parallelized);
define_server_key_bench_fn!(smart_bitand);
define_server_key_bench_fn!(smart_bitand_parallelized);
define_server_key_bench_fn!(smart_bitor);
define_server_key_bench_fn!(smart_bitor_parallelized);
define_server_key_bench_fn!(smart_bitxor);
define_server_key_bench_fn!(smart_bitxor_parallelized);

define_server_key_bench_fn!(unchecked_add);
define_server_key_bench_fn!(unchecked_sub);
define_server_key_bench_fn!(unchecked_mul);
define_server_key_bench_fn!(unchecked_mul_parallelized);
define_server_key_bench_fn!(unchecked_bitand);
define_server_key_bench_fn!(unchecked_bitor);
define_server_key_bench_fn!(unchecked_bitxor);

define_server_key_bench_scalar_fn!(smart_scalar_add);
define_server_key_bench_scalar_fn!(smart_scalar_add_parallelized);
define_server_key_bench_scalar_fn!(smart_scalar_sub);
define_server_key_bench_scalar_fn!(smart_scalar_sub_parallelized);
define_server_key_bench_scalar_fn!(smart_scalar_mul);
define_server_key_bench_scalar_fn!(smart_scalar_mul_parallelized);

define_server_key_bench_scalar_fn!(unchecked_scalar_add);
define_server_key_bench_scalar_fn!(unchecked_scalar_sub);
define_server_key_bench_scalar_fn!(unchecked_small_scalar_mul);

criterion_group!(
    smart_arithmetic_operation,
    smart_neg,
    smart_add,
    smart_add_parallelized,
    smart_sub,
    smart_sub_parallelized,
    smart_mul,
    smart_mul_parallelized,
    smart_bitand,
    smart_bitand_parallelized,
    smart_bitor,
    smart_bitor_parallelized,
    smart_bitxor,
    smart_bitxor_parallelized,
);

criterion_group!(
    smart_scalar_arithmetic_operation,
    smart_scalar_add,
    smart_scalar_add_parallelized,
    smart_scalar_sub,
    smart_scalar_sub_parallelized,
    smart_scalar_mul,
    smart_scalar_mul_parallelized,
);

criterion_group!(
    unchecked_arithmetic_operation,
    unchecked_add,
    unchecked_sub,
    unchecked_mul,
    unchecked_mul_parallelized,
    unchecked_bitand,
    unchecked_bitor,
    unchecked_bitxor,
);

criterion_group!(
    unchecked_scalar_arithmetic_operation,
    unchecked_scalar_add,
    unchecked_scalar_sub,
    unchecked_small_scalar_mul,
);

criterion_group!(misc, full_propagate,);

criterion_main!(
    smart_arithmetic_operation,
    smart_scalar_arithmetic_operation,
    unchecked_arithmetic_operation,
    unchecked_scalar_arithmetic_operation,
    misc,
    to_be_reworked,
);

fn smart_block_mul(c: &mut Criterion) {
    let size = 4;

    // generate the server-client key set
    let (cks, sks) = gen_keys(&DEFAULT_PARAMETERS, size);

    //RNG
    let mut rng = rand::thread_rng();

    let block_modulus = DEFAULT_PARAMETERS.message_modulus.0 as u64;

    // message_modulus^vec_length
    let modulus = DEFAULT_PARAMETERS.message_modulus.0.pow(size as u32) as u64;

    let clear_0 = rng.gen::<u64>() % modulus;

    let clear_1 = rng.gen::<u64>() % block_modulus;

    // encryption of an integer
    let mut ct_zero = cks.encrypt(clear_0);

    // encryption of an integer
    let ct_one = cks.encrypt_one_block(clear_1);

    //scalar mul
    c.bench_function("Smart_Block_Mul", |b| {
        b.iter(|| {
            sks.smart_block_mul(&mut ct_zero, &ct_one, 0);
        })
    });
}

fn crt(c: &mut Criterion) {
    let size = 4;

    // generate the server-client key set
    let (cks, sks) = gen_keys(&DEFAULT_PARAMETERS, size);

    //RNG
    let mut rng = rand::thread_rng();

    let basis = vec![2, 3, 5];
    let modulus = 30; // 30 = 2*3*5

    // Define the cleartexts
    let clear1 = rng.gen::<u64>() % modulus;
    let clear2 = rng.gen::<u64>() % modulus;

    // Encrypt the integers
    let mut ctxt_1 = cks.encrypt_crt(clear1, basis.clone());
    let mut ctxt_2 = cks.encrypt_crt(clear2, basis);

    //scalar mul
    c.bench_function("CRT: Smart_Mul", |b| {
        b.iter(|| {
            sks.smart_mul_crt_assign(&mut ctxt_1, &mut ctxt_2);
        })
    });
    c.bench_function("CRT: Smart_Add", |b| {
        b.iter(|| {
            sks.smart_add_assign(&mut ctxt_1, &mut ctxt_2);
        })
    });
}

fn radmodint_unchecked_mul(c: &mut Criterion) {
    let size = 2;

    let param = DEFAULT_PARAMETERS;
    let (cks, sks) = KEY_CACHE.get_from_params(param, VecLength(size));

    println!("Chosen Parameter Set: {:?}", param);

    //RNG
    let mut rng = rand::thread_rng();

    // message_modulus^vec_length
    let modulus = DEFAULT_PARAMETERS.message_modulus.0.pow(size as u32) as u64;

    // Define the cleartexts
    let clear1 = rng.gen::<u64>() % modulus;
    let clear2 = rng.gen::<u64>() % modulus;

    // Encrypt the integers
    let mut ctxt_1 = cks.encrypt(clear1);
    let ctxt_2 = cks.encrypt(clear2);

    //scalar mul
    c.bench_function("Unchecked Mul + Full Propagate", |b| {
        b.iter(|| {
            sks.unchecked_mul(&mut ctxt_1, &ctxt_2);
            sks.full_propagate(&mut ctxt_1);
        })
    });
}

fn radmodint_unchecked_mul_many_sizes(c: &mut Criterion) {
    //Change the number of sample
    let mut group = c.benchmark_group("smaller-sample-count");
    group.sample_size(10);

    //At most 4bits
    let max_message_space = 4;

    for msg_space in [16] {
        let dec = radix_decomposition(msg_space, 2, max_message_space);
        println!("radix decomposition = {:?}", dec);
        for rad_decomp in dec.iter() {
            //The carry space is at least equal to the msg_space
            let carry_space = rad_decomp.msg_space;

            let param =
                get_parameters_from_message_and_carry(1 << rad_decomp.msg_space, 1 << carry_space);
            let (cks, sks) = KEY_CACHE.get_from_params(param, VecLength(rad_decomp.block_number));

            println!("Chosen Parameter Set: {:?}", param);

            //RNG
            let mut rng = rand::thread_rng();

            // Define the cleartexts
            let clear1 = rng.gen::<u64>() % msg_space as u64;
            let clear2 = rng.gen::<u64>() % msg_space as u64;

            // Encrypt the integers

            let mut ctxt_1 = cks.encrypt(clear1);
            let ctxt_2 = cks.encrypt(clear2);

            println!(
                "(Input Size {}; Carry_Space {}, Message_Space {}, Block Number {}):  \
                    Unchecked Mul\
                     + \
                    Full \
                Propagate ",
                msg_space, carry_space, rad_decomp.msg_space, rad_decomp.block_number,
            );
            let id = format!(
                "(Integer-Mul-Propagate-Message_{}_Carry_{}_Input_{}_Block_{}):",
                rad_decomp.msg_space, carry_space, msg_space, rad_decomp.block_number,
            );

            group.bench_function(&id, |b| {
                b.iter(|| {
                    sks.unchecked_mul(&mut ctxt_1, &ctxt_2);
                    sks.full_propagate(&mut ctxt_1);
                })
            });
        }
    }
}
//
fn radmodint_wopbs(c: &mut Criterion) {
    //Change the number of sample
    let mut group = c.benchmark_group("smaller-sample-count");
    group.sample_size(10);

    //At most 4bits
    let max_message_space = 4;

    for msg_space in [16] {
        let dec = radix_decomposition(msg_space, 2, max_message_space);
        println!("radix decomposition = {:?}", dec);
        //for rad_decomp in dec.iter() {
        let rad_decomp = dec[0];
        //The carry space is at least equal to the msg_space
        let carry_space = rad_decomp.msg_space;

        let param = get_parameters_from_message_and_carry_wopbs(
            1 << rad_decomp.msg_space,
            1 << carry_space,
        );
        //let (mut cks, mut sks) = KEY_CACHE.get_from_params(param);
        let keys = KEY_CACHE_WOPBS.get_from_param(param);
        let (cks, sks, wopbs_shortint) = (keys.client_key(), keys.server_key(), keys.wopbs_key());

        println!("Chosen Parameter Set: {:?}", param);

        let cks = concrete_integer::client_key::ClientKey::from_shortint(
            cks.clone(),
            VecLength(rad_decomp.block_number),
        );

        let sks = concrete_integer::server_key::ServerKey::from_shortint(&cks, sks.clone());

        let wopbs = concrete_integer::wopbs::WopbsKeyV0::new_from_shortint(wopbs_shortint);
        let mut rng = rand::thread_rng();

        let delta = 63 - f64::log2((param.message_modulus.0 * param.carry_modulus.0) as f64) as u64;
        // Define the cleartexts
        let clear1 = rng.gen::<u64>() % msg_space as u64;

        // Encrypt the integers
        let mut ctxt_1 = cks.encrypt(clear1);

        let nb_bit_to_extract = f64::log2((param.message_modulus.0 * param.carry_modulus.0) as f64)
            as usize
            * rad_decomp.block_number;

        let mut lut_size = param.polynomial_size.0;
        if (1 << nb_bit_to_extract) > wopbs_shortint.param.polynomial_size.0 {
            lut_size = 1 << nb_bit_to_extract;
        }

        let mut lut_1: Vec<u64> = vec![];
        let mut lut_2: Vec<u64> = vec![];
        for _ in 0..lut_size {
            lut_1.push(
                (rng.gen::<u64>() % (param.message_modulus.0 * param.carry_modulus.0) as u64)
                    << delta,
            );
            lut_2.push(
                (rng.gen::<u64>() % (param.message_modulus.0 * param.carry_modulus.0) as u64)
                    << delta,
            );
        }
        let big_lut = vec![lut_1, lut_2];

        println!(
            "(Input Size {}; Carry_Space {}, Message_Space {}, Block Number {}):  \
                    WoPBS",
            msg_space, carry_space, rad_decomp.msg_space, rad_decomp.block_number,
        );
        let id = format!(
            "(Integer-WoPBS-Message_{}_Carry_{}_Input_{}_Block_{}):",
            rad_decomp.msg_space, carry_space, msg_space, rad_decomp.block_number,
        );

        group.bench_function(&id, |b| {
            b.iter(|| wopbs.circuit_bootstrap_vertical_packing_v0(&sks, &mut ctxt_1, &big_lut))
        });
    }
    //}
}

fn radmodint_wopbs_16bits_param_2_2_8_blocks(c: &mut Criterion) {
    //Change the number of sample
    let param = PARAM_MESSAGE_2_CARRY_2_16_BITS;
    let nb_block = 8;
    let input = 16;

    let mut group = c.benchmark_group("smaller-sample-count");
    group.sample_size(10);

    println!(
        "Chosen Parameter Set: {:?}",
        PARAM_MESSAGE_2_CARRY_2_16_BITS
    );

    let (cks, sks) = gen_keys(&param, nb_block);
    let wopbs_key = WopbsKeyV0::new_wopbs_key(&cks, &sks);

    let mut rng = rand::thread_rng();
    let delta = 63 - f64::log2((param.message_modulus.0 * param.carry_modulus.0) as f64) as u64;
    // Define the cleartexts
    let clear1 = rng.gen::<u64>() % param.message_modulus.0 as u64;

    // Encrypt the integers
    let mut ctxt_1 = cks.encrypt(clear1);

    let nb_bit_to_extract =
        f64::log2((param.message_modulus.0 * param.carry_modulus.0) as f64) as usize * nb_block;

    let mut lut_size = param.polynomial_size.0;
    if (1 << nb_bit_to_extract) > param.polynomial_size.0 {
        lut_size = 1 << nb_block;
    }

    let mut lut_1: Vec<u64> = vec![];
    let mut lut_2: Vec<u64> = vec![];
    for _ in 0..lut_size {
        lut_1.push(
            (rng.gen::<u64>() % (param.message_modulus.0 * param.carry_modulus.0) as u64) << delta,
        );
        lut_2.push(
            (rng.gen::<u64>() % (param.message_modulus.0 * param.carry_modulus.0) as u64) << delta,
        );
    }
    let big_lut = vec![lut_1, lut_2];

    let id = format!(
        "(Integer-WoPBS-Message_{}_Carry_{}_Input_{}_Block_{}):",
        param.message_modulus.0, param.message_modulus.0, input, nb_block
    );

    group.bench_function(&id, |b| {
        b.iter(|| wopbs_key.circuit_bootstrap_vertical_packing_v0(&sks, &mut ctxt_1, &big_lut))
    });
}

fn radmodint_wopbs_16bits_param_4_4_4_blocks(c: &mut Criterion) {
    //Change the number of sample
    let param = PARAM_MESSAGE_4_CARRY_4_16_BITS;
    let nb_block = 4;
    let input = 16;

    let mut group = c.benchmark_group("smaller-sample-count");
    group.sample_size(10);

    println!("Chosen Parameter Set: {:?}", param);

    let (cks, sks) = gen_keys(&param, nb_block);
    let wopbs_key = WopbsKeyV0::new_wopbs_key(&cks, &sks);

    let mut rng = rand::thread_rng();
    let delta = 63 - f64::log2((param.message_modulus.0 * param.carry_modulus.0) as f64) as u64;
    // Define the cleartexts
    let clear1 = rng.gen::<u64>() % param.message_modulus.0 as u64;

    // Encrypt the integers
    let mut ctxt_1 = cks.encrypt(clear1);

    let nb_bit_to_extract =
        f64::log2((param.message_modulus.0 * param.carry_modulus.0) as f64) as usize * nb_block;

    let mut lut_size = param.polynomial_size.0;
    if (1 << nb_bit_to_extract) > param.polynomial_size.0 {
        lut_size = 1 << nb_block;
    }

    let mut lut_1: Vec<u64> = vec![];
    let mut lut_2: Vec<u64> = vec![];
    for _ in 0..lut_size {
        lut_1.push(
            (rng.gen::<u64>() % (param.message_modulus.0 * param.carry_modulus.0) as u64) << delta,
        );
        lut_2.push(
            (rng.gen::<u64>() % (param.message_modulus.0 * param.carry_modulus.0) as u64) << delta,
        );
    }
    let big_lut = vec![lut_1, lut_2];

    let id = format!(
        "(Integer-WoPBS-Message_{}_Carry_{}_Input_{}_Block_{}):",
        param.message_modulus.0, param.message_modulus.0, input, nb_block
    );

    group.bench_function(&id, |b| {
        b.iter(|| wopbs_key.circuit_bootstrap_vertical_packing_v0(&sks, &mut ctxt_1, &big_lut))
    });
}

fn radmodint_wopbs_32_bits(c: &mut Criterion) {
    //Change the number of sample
    let vec_param = &[
        PARAM_MESSAGE_1_CARRY_1_32_BITS,
        PARAM_MESSAGE_2_CARRY_2_32_BITS,
        PARAM_MESSAGE_4_CARRY_4_32_BITS,
    ];
    let vec_nb_block = &[32, 16, 8];
    let input = 16;

    let mut group = c.benchmark_group("smaller-sample-count");
    group.sample_size(10);

    for (param, nb_block) in vec_param.iter().zip(vec_nb_block.iter()) {
        println!("Chosen Parameter Set: {:?}", param);

        let (cks, sks) = gen_keys(param, *nb_block);
        let wopbs_key = WopbsKeyV0::new_wopbs_key(&cks, &sks);

        let mut rng = rand::thread_rng();
        let delta = 63 - f64::log2((param.message_modulus.0 * param.carry_modulus.0) as f64) as u64;
        // Define the cleartexts
        let clear1 = rng.gen::<u64>() % param.message_modulus.0 as u64;

        // Encrypt the integers
        let mut ctxt_1 = cks.encrypt(clear1);

        let nb_bit_to_extract =
            f64::log2((param.message_modulus.0 * param.carry_modulus.0) as f64) as usize * nb_block;

        let mut lut_size = param.polynomial_size.0;
        if (1 << nb_bit_to_extract) > param.polynomial_size.0 {
            lut_size = 1 << nb_block;
        }

        let mut lut_1: Vec<u64> = vec![];
        let mut lut_2: Vec<u64> = vec![];
        for _ in 0..lut_size {
            lut_1.push(
                (rng.gen::<u64>() % (param.message_modulus.0 * param.carry_modulus.0) as u64)
                    << delta,
            );
            lut_2.push(
                (rng.gen::<u64>() % (param.message_modulus.0 * param.carry_modulus.0) as u64)
                    << delta,
            );
        }
        let big_lut = vec![lut_1, lut_2];

        let id = format!(
            "(Integer-WoPBS-Message_{}_Carry_{}_Input_{}_Block_{}):",
            param.message_modulus.0, param.message_modulus.0, input, nb_block
        );

        group.bench_function(&id, |b| {
            b.iter(|| wopbs_key.circuit_bootstrap_vertical_packing_v0(&sks, &mut ctxt_1, &big_lut))
        });
    }
}

fn concrete_integer_unchecked_mul_crt_16_bits(c: &mut Criterion) {
    let mut group = c.benchmark_group("smaller-sample-count");
    group.sample_size(10);
    let param = concrete_shortint::parameters::PARAM_MESSAGE_4_CARRY_4;

    let (cks, sks) = KEY_CACHE.get_from_params(param, VecLength(4));

    println!("Chosen Parameter Set: {:?}", param);

    let basis = vec![8, 9, 11, 13, 7];
    let mut modulus = 1;
    for b in basis.iter() {
        modulus *= b;
    }

    //
    // let block_modulus = DEFAULT_PARAMETERS.message_modulus.0 as u64;
    //
    // // message_modulus^vec_length
    // let modulus = DEFAULT_PARAMETERS.message_modulus.0.pow(size as u32) as u64;

    let clear_0 = 29 % modulus;
    let clear_1 = 23 % modulus;

    // encryption of an integer
    let mut ct_zero = cks.encrypt_crt(clear_0, basis.clone());
    let ct_one = cks.encrypt_crt(clear_1, basis);

    let id = "(bench_concrete_integer_unchecked_mul_crt_16_bits):";
    // add the two ciphertexts
    group.bench_function(id, |b| {
        b.iter(|| {
            sks.unchecked_mul_crt_assign(&mut ct_zero, &ct_one);
        })
    });
}

fn concrete_integer_unchecked_add_crt_16_bits(c: &mut Criterion) {
    let mut group = c.benchmark_group("smaller-sample-count");
    group.sample_size(10);
    let param = concrete_shortint::parameters::PARAM_MESSAGE_4_CARRY_4;

    let (cks, sks) = KEY_CACHE.get_from_params(param, VecLength(4));

    println!("Chosen Parameter Set: {:?}", param);

    let basis = vec![8, 9, 11, 13, 7];
    let mut modulus = 1;
    for b in basis.iter() {
        modulus *= b;
    }

    //RN
    //
    // let block_modulus = DEFAULT_PARAMETERS.message_modulus.0 as u64;
    //
    // // message_modulus^vec_length
    // let modulus = DEFAULT_PARAMETERS.message_modulus.0.pow(size as u32) as u64;

    let clear_0 = 29 % modulus;
    let clear_1 = 23 % modulus;

    // encryption of an integer
    let mut ct_zero = cks.encrypt_crt(clear_0, basis.clone());
    let ct_one = cks.encrypt_crt(clear_1, basis);

    let id = "(bench_concrete_integer_unchecked_add_crt_16_bits):";
    // add the two ciphertexts
    group.bench_function(id, |b| {
        b.iter(|| {
            sks.unchecked_add_crt_assign(&mut ct_zero, &ct_one);
        })
    });
}

fn concrete_integer_unchecked_clean_carry_crt_16_bits(c: &mut Criterion) {
    let mut group = c.benchmark_group("smaller-sample-count");
    group.sample_size(10);
    let param = concrete_shortint::parameters::PARAM_MESSAGE_4_CARRY_4;

    // generate the server-client key set
    //let (mut cks, mut sks) =
    //gen_keys(&concrete_shortint::parameters::PARAM_MESSAGE_4_CARRY_4,
    //size);

    let (cks, sks) = KEY_CACHE.get_from_params(param, VecLength(4));

    println!("Chosen Parameter Set: {:?}", param);

    let basis = vec![8, 9, 11, 13, 7];
    let mut modulus = 1;
    for b in basis.iter() {
        modulus *= b;
    }

    //RN
    //
    // let block_modulus = DEFAULT_PARAMETERS.message_modulus.0 as u64;
    //
    // // message_modulus^vec_length
    // let modulus = DEFAULT_PARAMETERS.message_modulus.0.pow(size as u32) as u64;

    let clear_0 = 29 % modulus;

    // encryption of an integer
    let mut ct_zero = cks.encrypt_crt(clear_0, basis.clone());

    let id = "(bench_concrete_integer_clean_carry_16_bits):";
    // add the two ciphertexts
    group.bench_function(id, |b| {
        b.iter(|| {
            sks.pbs_crt_compliant_function_assign(&mut ct_zero, |x| x % basis[0]);
        })
    });
}

fn concrete_integer_unchecked_mul_crt_32_bits(c: &mut Criterion) {
    let mut group = c.benchmark_group("smaller-sample-count");
    group.sample_size(10);
    let param = concrete_shortint::parameters::PARAM_MESSAGE_4_CARRY_4;

    // generate the server-client key set
    //let (mut cks, mut sks) =
    //gen_keys(&concrete_shortint::parameters::PARAM_MESSAGE_4_CARRY_4,
    //size);

    let (cks, sks) = KEY_CACHE.get_from_params(param, VecLength(8));

    println!("Chosen Parameter Set: {:?}", param);

    let basis = vec![43, 47, 37, 49, 29, 41];
    let mut modulus = 1;
    for b in basis.iter() {
        modulus *= b;
    }

    //
    // let block_modulus = DEFAULT_PARAMETERS.message_modulus.0 as u64;
    //
    // // message_modulus^vec_length
    // let modulus = DEFAULT_PARAMETERS.message_modulus.0.pow(size as u32) as u64;

    let clear_0 = 29 % modulus;
    let clear_1 = 23 % modulus;

    // encryption of an integer
    let mut ct_zero = cks.encrypt_crt(clear_0, basis.clone());
    let ct_one = cks.encrypt_crt(clear_1, basis);

    let id = "(bench_concrete_integer_unchecked_mul_crt_32_bits):";
    // add the two ciphertexts
    group.bench_function(id, |b| {
        b.iter(|| {
            sks.unchecked_mul_crt_assign(&mut ct_zero, &ct_one);
        })
    });
}

fn concrete_integer_unchecked_add_crt_32_bits(c: &mut Criterion) {
    let mut group = c.benchmark_group("smaller-sample-count");
    group.sample_size(10);
    let param = concrete_shortint::parameters::PARAM_MESSAGE_4_CARRY_4;

    // generate the server-client key set
    //let (mut cks, mut sks) =
    //gen_keys(&concrete_shortint::parameters::PARAM_MESSAGE_4_CARRY_4,
    //size);

    let (cks, sks) = KEY_CACHE.get_from_params(param, VecLength(8));

    println!("Chosen Parameter Set: {:?}", param);

    let basis = vec![43, 47, 37, 49, 29, 41];
    let mut modulus = 1;
    for b in basis.iter() {
        modulus *= b;
    }

    //RN
    //
    // let block_modulus = DEFAULT_PARAMETERS.message_modulus.0 as u64;
    //
    // // message_modulus^vec_length
    // let modulus = DEFAULT_PARAMETERS.message_modulus.0.pow(size as u32) as u64;

    let clear_0 = 29 % modulus;
    let clear_1 = 23 % modulus;

    // encryption of an integer
    let mut ct_zero = cks.encrypt_crt(clear_0, basis.clone());
    let ct_one = cks.encrypt_crt(clear_1, basis);

    let id = "(bench_concrete_integer_unchecked_add_crt_32_bits):";
    // add the two ciphertexts
    group.bench_function(id, |b| {
        b.iter(|| {
            sks.unchecked_add_crt_assign(&mut ct_zero, &ct_one);
        })
    });
}

fn concrete_integer_unchecked_clean_carry_crt_32_bits(c: &mut Criterion) {
    let mut group = c.benchmark_group("smaller-sample-count");
    group.sample_size(10);
    let param = concrete_shortint::parameters::PARAM_MESSAGE_4_CARRY_4;

    // generate the server-client key set
    //let (mut cks, mut sks) =
    //gen_keys(&concrete_shortint::parameters::PARAM_MESSAGE_4_CARRY_4,
    //size);

    let (cks, sks) = KEY_CACHE.get_from_params(param, VecLength(8));

    println!("Chosen Parameter Set: {:?}", param);

    let basis = vec![43, 47, 37, 49, 29, 41];
    let mut modulus = 1;
    for b in basis.iter() {
        modulus *= b;
    }

    //RN
    //
    // let block_modulus = DEFAULT_PARAMETERS.message_modulus.0 as u64;
    //
    // // message_modulus^vec_length
    // let modulus = DEFAULT_PARAMETERS.message_modulus.0.pow(size as u32) as u64;

    let clear_0 = 29 % modulus;

    // encryption of an integer
    let mut ct_zero = cks.encrypt_crt(clear_0, basis.clone());

    let id = "(bench_concrete_integer_clean_carry_32_bits):";
    // add the two ciphertexts
    group.bench_function(id, |b| {
        b.iter(|| {
            sks.pbs_crt_compliant_function_assign(&mut ct_zero, |x| x % basis[0]);
        })
    });
}

fn crt_op_many_sizes_generic(
    c: &mut Criterion,
    op: &dyn Fn(&CRTVecServerKey, &mut Ciphertext, &mut Ciphertext),
    prefix: &str,
    suffix: &str,
) {
    //Change the number of sample
    let mut group = c.benchmark_group("smaller-sample-count");
    group.sample_size(10);

    //At most 4bits
    let _max_message_space = 4;

    let basis: Vec<u64> = vec![2, 3, 5];
    let carry_basis: Vec<u64> = vec![2, 3, 5];
    let mut vec_param = vec![];
    let mut vec_id = vec![];

    for ((i, base), carry_space) in basis.iter().enumerate().zip(carry_basis.iter()) {
        let tmp_param =
            get_parameters_from_message_and_carry(*base as usize, *carry_space as usize);
        let tmp_param_exists = vec_param.iter().find(|&&x| x == tmp_param);
        if tmp_param_exists != None {
            vec_id.push(vec_param.iter().position(|&x| x == tmp_param).unwrap());
        } else {
            vec_param.push(get_parameters_from_message_and_carry(
                *base as usize,
                *carry_space as usize,
            ));
            vec_id.push(i);
        }
    }

    let vec_key_id = gen_key_id(&vec_id);

    let mut vec_sks = vec![];
    let mut vec_cks = vec![];
    for param in vec_param.iter() {
        let (cks_shortint, sks_shortint) = KEY_CACHE.get_shortint_from_params(*param);
        vec_sks.push(sks_shortint);
        vec_cks.push(cks_shortint);
    }

    let cks = CRTVecClientKey::new_many_keys_from_shortint(vec_cks.as_slice());
    let sks = CRTVecServerKey::new_many_keys_from_shortint(vec_sks.as_slice());

    let mut rng = rand::thread_rng();

    // Define the cleartexts
    let mut msg_space = 1;
    for modulus in basis.iter() {
        msg_space *= modulus;
    }

    let clear1 = rng.gen::<u64>() % msg_space as u64;
    let clear2 = rng.gen::<u64>() % msg_space as u64;

    // Encrypt the integers

    let mut ctxt_1 = cks.encrypt_crt_several_keys(&clear1, &basis, &vec_key_id);
    let mut ctxt_2 = cks.encrypt_crt_several_keys(&clear2, &basis, &vec_key_id);

    println!(
        "(Input Size {}; Carry_Space {:?}, Message_Space {:?},):  \
                    Unchecked Mul\
                     + \
                    Full \
                Propagate ",
        msg_space, carry_basis, basis,
    );
    let id = format!(
        "(CRT_{}_In_{}_Carry_{:?}_Message_{:?}_{}):",
        prefix, msg_space, carry_basis, basis, suffix,
    );

    group.bench_function(&id, |b| {
        b.iter(|| {
            op(&sks, &mut ctxt_1, &mut ctxt_2);
        })
    });
}

fn crt_arithmetic_many_sizes(c: &mut Criterion) {
    crt_op_many_sizes_generic(
        c,
        &|sks, ct1, _ct2| sks.arithmetic_function_crt_many_keys_assign(ct1, |x| x + x * x),
        "Arithmetic",
        "serial",
    )
}

fn crt_arithmetic_many_sizes_parallelized(c: &mut Criterion) {
    crt_op_many_sizes_generic(
        c,
        &|sks, ct1, _ct2| {
            sks.arithmetic_function_crt_many_keys_assign_parallelized(ct1, |x| x + x * x)
        },
        "Arithmetic",
        "parallelized",
    )
}

fn crt_unchecked_mul_many_sizes(c: &mut Criterion) {
    crt_op_many_sizes_generic(
        c,
        &CRTVecServerKey::unchecked_mul_crt_many_keys_assign,
        "Mul",
        "serial",
    )
}

fn crt_unchecked_mul_many_sizes_parallelized(c: &mut Criterion) {
    crt_op_many_sizes_generic(
        c,
        &CRTVecServerKey::unchecked_mul_crt_many_keys_assign_parallelized,
        "Mul",
        "parallelized",
    )
}

fn two_block_pbs(c: &mut Criterion) {
    let size = 2;

    let (cks, sks) = gen_keys(&DEFAULT_PARAMETERS, size);
    let treepbs_key = TreepbsKey::new(&cks);

    //RNG
    let mut rng = rand::thread_rng();

    // message_modulus^vec_length
    let modulus = DEFAULT_PARAMETERS.message_modulus.0.pow(size as u32) as u64;

    let clear_0 = rng.gen::<u64>() % modulus;

    // encryption of an integer
    let ctxt_0 = cks.encrypt(clear_0);

    let f = |x: u64| x * x;

    c.bench_function("Two block PBS", |b| {
        b.iter(|| {
            treepbs_key.two_block_pbs(&sks, &ctxt_0, f);
        })
    });
}

fn two_block_pbs_base(c: &mut Criterion) {
    let size = 2;

    let (cks, sks) = gen_keys(&DEFAULT_PARAMETERS, size);
    let treepbs_key = TreepbsKey::new(&cks);

    //RNG
    let mut rng = rand::thread_rng();

    // message_modulus^vec_length
    let modulus = DEFAULT_PARAMETERS.message_modulus.0.pow(size as u32) as u64;

    let clear_0 = rng.gen::<u64>() % modulus;

    // encryption of an integer
    let ctxt_0 = cks.encrypt(clear_0);

    let f = |x: u64| x * x;

    c.bench_function("Two block PBS base", |b| {
        b.iter(|| {
            treepbs_key.two_block_pbs_base(&sks, &ctxt_0, f);
        })
    });
}

fn three_block_pbs(c: &mut Criterion) {
    let size = 3;

    let (cks, sks) = gen_keys(&DEFAULT_PARAMETERS, size);
    let treepbs_key = TreepbsKey::new(&cks);

    //RNG
    let mut rng = rand::thread_rng();

    // message_modulus^vec_length
    let modulus = DEFAULT_PARAMETERS.message_modulus.0.pow(size as u32) as u64;

    let clear_0 = rng.gen::<u64>() % modulus;

    // encryption of an integer
    let ctxt_0 = cks.encrypt(clear_0);

    let f = |x: u64| x * x;

    c.bench_function("Three block PBS", |b| {
        b.iter(|| {
            treepbs_key.three_block_pbs(&sks, &ctxt_0, f);
        })
    });
}

fn three_block_pbs_base(c: &mut Criterion) {
    let size = 3;

    let (cks, sks) = gen_keys(&DEFAULT_PARAMETERS, size);
    let treepbs_key = TreepbsKey::new(&cks);

    //RNG
    let mut rng = rand::thread_rng();

    // message_modulus^vec_length
    let modulus = DEFAULT_PARAMETERS.message_modulus.0.pow(size as u32) as u64;

    let clear_0 = rng.gen::<u64>() % modulus;

    // encryption of an integer
    let ctxt_0 = cks.encrypt(clear_0);

    let f = |x: u64| x * x;

    c.bench_function("Three block PBS base", |b| {
        b.iter(|| {
            treepbs_key.three_block_pbs_base(&sks, &ctxt_0, f);
        })
    });
}
