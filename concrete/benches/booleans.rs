use criterion::criterion_main;

trait BenchUtils: Sized {
    fn type_name() -> &'static str;
}

use concrete::prelude::*;
use concrete::{generate_keys, set_server_key, ConfigBuilder, FheBool};
use criterion::{black_box, criterion_group, Criterion};
use std::ops::{BitAnd, BitOr, BitXor, Not};

impl BenchUtils for FheBool {
    fn type_name() -> &'static str {
        "FheBool"
    }
}

macro_rules! define_trait_binary_gate_benchmark {
($trait:ident, $operator:tt => $fn_name:ident) => {
    fn $fn_name<BoolType>(c: &mut Criterion)
    where BoolType: BenchUtils + FheEncrypt<bool>,
          for<'a> &'a BoolType: $trait<&'a BoolType, Output=BoolType>
    {
        let config = ConfigBuilder::all_disabled()
            .enable_default_bool()
            .build();
        let (client_key, server_key) = generate_keys(config);
        set_server_key(server_key);

        let lhs = BoolType::encrypt(false, &client_key);
        let rhs = BoolType::encrypt(true, &client_key);

        let bench_name = format!("{}::{}", BoolType::type_name(), stringify!($trait));
        c.bench_function(&bench_name, |b| {
            b.iter(|| {
                let _: BoolType = black_box(&lhs $operator &rhs);
            })
        });
    }
};
}

define_trait_binary_gate_benchmark!(BitAnd, & => benchmark_bool_and);
define_trait_binary_gate_benchmark!(BitOr, | => benchmark_bool_or);
define_trait_binary_gate_benchmark!(BitXor, ^ => benchmark_bool_xor);

fn benchmark_bool_not<BoolType: 'static>(c: &mut Criterion)
where
    BoolType: BenchUtils + FheEncrypt<bool> + Clone,
    for<'a> &'a BoolType: Not<Output = BoolType>,
{
    let config = ConfigBuilder::all_disabled().enable_default_bool().build();
    let (client_key, server_key) = generate_keys(config);
    set_server_key(server_key);

    let lhs = BoolType::encrypt(false, &client_key);

    let bench_name = format!("{}::Not", BoolType::type_name());
    c.bench_function(&bench_name, |b| {
        b.iter(|| {
            let _: BoolType = black_box(!(&lhs));
        })
    });
}

criterion_group!(
    boolean_benches,
    benchmark_bool_and::<FheBool>,
    benchmark_bool_or::<FheBool>,
    benchmark_bool_xor::<FheBool>,
    benchmark_bool_not::<FheBool>,
);

criterion_main!(boolean_benches);
