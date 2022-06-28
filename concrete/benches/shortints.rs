use criterion::criterion_main;

trait BenchUtils: Sized {
    fn type_name() -> &'static str;
}

/// This macro defines a function that will
/// bench the given overloaded operator.
///
/// The generated function is generic over a type on which
/// the bench works.
macro_rules! define_operator_benchmark {
    ($trait:ident, $operator:tt, $scalar_type:ty => $fn_name:ident) => {
       fn $fn_name<FheType>(c: &mut Criterion)
        where
            FheType: BenchUtils + FheTryEncrypt<$scalar_type> + FheNumberConstant + Mul<$scalar_type, Output=FheType> + Clone,
            for<'a> &'a FheType: $trait<&'a FheType, Output = FheType>,
        {
            let (client_key, server_key) = generate_keys(ConfigBuilder::all_enabled());
            set_server_key(server_key);

            let lhs = FheType::try_encrypt(1, &client_key).unwrap();
            let rhs = FheType::try_encrypt(1, &client_key).unwrap();

            // This is to get the worst possible case
            let lhs = lhs * (FheType::MODULUS - 1) as $scalar_type;
            let rhs = rhs * (FheType::MODULUS - 1) as $scalar_type;

            let bench_name = format!("{}::{}", FheType::type_name(), stringify!($trait));
            c.bench_function(&bench_name, move |b| {
                b.iter_batched(
                    || (lhs.clone(), rhs.clone()),
                    |(lhs, rhs)| {
                        let _: FheType = black_box(&lhs $operator &rhs);
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    };
}

macro_rules! define_operator_assign_benchmark {
    ($trait:ident, $operator:tt, $scalar_type:ty  => $fn_name:ident) => {
       fn $fn_name<FheType>(c: &mut Criterion)
        where
            FheType: BenchUtils + FheTryEncrypt<$scalar_type> + FheNumberConstant + Mul<$scalar_type, Output=FheType> + Clone,
            FheType: for <'a> $trait<&'a FheType>,
        {
            let (client_key, server_key) = generate_keys(ConfigBuilder::all_enabled());
            set_server_key(server_key);

            let lhs = FheType::try_encrypt(1, &client_key).unwrap();
            let rhs = FheType::try_encrypt(1, &client_key).unwrap();

            // This is to get the worst possible case
            let lhs = lhs * (FheType::MODULUS - 1) as $scalar_type;
            let rhs = rhs * (FheType::MODULUS - 1) as $scalar_type;

            let bench_name = format!("{}::{}", FheType::type_name(), stringify!($trait));
            c.bench_function(&bench_name, |b| {
                 b.iter_batched(
                    || (lhs.clone(), rhs.clone()),
                    |(mut lhs, rhs)| {
                        black_box(lhs $operator &rhs);
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    };
}

macro_rules! define_scalar_operator_benchmark {
    ($trait:ident, $operator:tt, $scalar_type:ty => $fn_name:ident) => {
       fn $fn_name<FheType>(c: &mut Criterion)
        where
            FheType: BenchUtils + FheTryEncrypt<$scalar_type> + FheNumberConstant + Mul<$scalar_type, Output=FheType> + Clone,
            for<'a> &'a FheType: $trait<$scalar_type, Output = FheType>,
        {
            let (client_key, server_key) = generate_keys(ConfigBuilder::all_enabled());
            set_server_key(server_key);

            let lhs = FheType::try_encrypt(1, &client_key).unwrap();
            let scalar = (FheType::MODULUS - 1) as $scalar_type;

            // This is to get the worst possible case
            let lhs = lhs * scalar;

            let bench_name = format!("{}::Scalar{}", FheType::type_name(), stringify!($trait));
            c.bench_function(&bench_name, move |b| {
                b.iter_batched(
                    || (lhs.clone(), scalar),
                    |(lhs, scalar)| {
                        let _: FheType = black_box(&lhs $operator scalar);
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    };
}

mod shortint_benches {
    use super::BenchUtils;

    use concrete::prelude::*;
    use concrete::{generate_keys, set_server_key, ConfigBuilder, FheUint2, FheUint3, FheUint4};
    use criterion::{black_box, criterion_group, BatchSize, Criterion};
    use std::ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
        DivAssign, Mul, MulAssign, Rem, Shl, Shr, Sub, SubAssign,
    };

    impl BenchUtils for FheUint2 {
        fn type_name() -> &'static str {
            "FheUint2"
        }
    }

    impl BenchUtils for FheUint3 {
        fn type_name() -> &'static str {
            "FheUint3"
        }
    }

    impl BenchUtils for FheUint4 {
        fn type_name() -> &'static str {
            "FheUint4"
        }
    }

    define_operator_benchmark!(Add, +, u8 => benchmark_addition);
    define_operator_benchmark!(Sub, -, u8 => benchmark_subtraction);
    define_operator_benchmark!(Mul, *, u8 => benchmark_multiplication);
    define_operator_benchmark!(Div, /, u8 => benchmark_division);
    define_operator_benchmark!(BitAnd, &, u8 => benchmark_bitand);
    define_operator_benchmark!(BitOr, |, u8 => benchmark_bitor);
    define_operator_benchmark!(BitXor, ^, u8 => benchmark_bitxor);

    define_operator_assign_benchmark!(AddAssign, +=, u8 => benchmark_addition_assign);
    define_operator_assign_benchmark!(SubAssign, -=, u8 => benchmark_subtraction_assign);
    define_operator_assign_benchmark!(MulAssign, *=, u8 => benchmark_multiplication_assign);
    define_operator_assign_benchmark!(DivAssign, /=, u8 => benchmark_division_assign);
    define_operator_assign_benchmark!(BitAndAssign, &=, u8 => benchmark_bitand_assign);
    define_operator_assign_benchmark!(BitOrAssign, |=, u8 => benchmark_bitor_assign);
    define_operator_assign_benchmark!(BitXorAssign, ^=, u8 => benchmark_bitxor_assign);

    define_scalar_operator_benchmark!(Add, +, u8 => benchmark_scalar_addition);
    define_scalar_operator_benchmark!(Sub, -, u8 => benchmark_scalar_subtraction);
    define_scalar_operator_benchmark!(Mul, *, u8 => benchmark_scalar_multiplication);
    define_scalar_operator_benchmark!(Div, /, u8 => benchmark_scalar_division);
    define_scalar_operator_benchmark!(Rem, %, u8 => benchmark_scalar_bitand);
    define_scalar_operator_benchmark!(Shl, <<, u8 => benchmark_scalar_bitor);
    define_scalar_operator_benchmark!(Shr, >>, u8 => benchmark_scalar_bitxor);

    macro_rules! shortint_bench_group {
    ($group_name:ident: [ $($type:ty),* ]) => {
        criterion_group!(
            $group_name,
            $(
                benchmark_addition::<$type>,
                benchmark_subtraction::<$type>,
                benchmark_multiplication::<$type>,
                benchmark_division::<$type>,
                benchmark_bitand::<$type>,
                benchmark_bitor::<$type>,
                benchmark_bitxor::<$type>,
            )*
        );
    };
}

    macro_rules! shortint_assign_bench_group {
    ($group_name:ident: [ $($type:ty),* ]) => {
        criterion_group!(
            $group_name,
            $(
                benchmark_addition_assign::<$type>,
                benchmark_subtraction_assign::<$type>,
                benchmark_multiplication_assign::<$type>,
                benchmark_division_assign::<$type>,
                benchmark_bitand_assign::<$type>,
                benchmark_bitor_assign::<$type>,
                benchmark_bitxor_assign::<$type>,
            )*
        );
    };
}

    macro_rules! shortint_scalar_bench_group {
    ($group_name:ident: [ $($type:ty),* ]) => {
        criterion_group!(
            $group_name,
            $(
                benchmark_scalar_addition::<$type>,
                benchmark_scalar_subtraction::<$type>,
                benchmark_scalar_multiplication::<$type>,
                benchmark_scalar_division::<$type>,
                benchmark_scalar_bitand::<$type>,
                benchmark_scalar_bitor::<$type>,
                benchmark_scalar_bitxor::<$type>,
            )*
        );
    };
}

    shortint_bench_group!(shortint_benches: [FheUint2, FheUint3, FheUint4]);
    shortint_assign_bench_group!(shortint_assign_benches: [FheUint2, FheUint3, FheUint4]);
    shortint_scalar_bench_group!(shortint_scalar_benches: [FheUint2, FheUint3, FheUint4]);
}

#[cfg(feature = "integers")]
mod integer_benches {
    use concrete::prelude::*;
    use concrete::{generate_keys, set_server_key, ConfigBuilder, FheUint12, FheUint16, FheUint8};
    use criterion::{black_box, criterion_group, BatchSize, Criterion};

    use super::BenchUtils;
    use std::ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul,
        MulAssign, Sub, SubAssign,
    };

    impl BenchUtils for FheUint8 {
        fn type_name() -> &'static str {
            "FheUint8"
        }
    }

    impl BenchUtils for FheUint12 {
        fn type_name() -> &'static str {
            "FheUint12"
        }
    }

    impl BenchUtils for FheUint16 {
        fn type_name() -> &'static str {
            "FheUint16"
        }
    }

    define_operator_benchmark!(Add, +, u64 => benchmark_addition);
    define_operator_benchmark!(Sub, -, u64 => benchmark_subtraction);
    define_operator_benchmark!(Mul, *, u64 => benchmark_multiplication);
    define_operator_benchmark!(BitAnd, &, u64 => benchmark_bitand);
    define_operator_benchmark!(BitOr, |, u64 => benchmark_bitor);
    define_operator_benchmark!(BitXor, ^, u64 => benchmark_bitxor);

    define_operator_assign_benchmark!(AddAssign, +=, u64 => benchmark_addition_assign);
    define_operator_assign_benchmark!(SubAssign, -=, u64 => benchmark_subtraction_assign);
    define_operator_assign_benchmark!(MulAssign, *=, u64 => benchmark_multiplication_assign);
    define_operator_assign_benchmark!(BitAndAssign, &=, u64 => benchmark_bitand_assign);
    define_operator_assign_benchmark!(BitOrAssign, |=, u64 => benchmark_bitor_assign);
    define_operator_assign_benchmark!(BitXorAssign, ^=, u64 => benchmark_bitxor_assign);

    macro_rules! int_bench_group {
        ($group_name:ident: [ $($type:ty),* ]) => {
            criterion_group!(
                $group_name,
                $(
                    benchmark_addition::<$type>,
                    benchmark_subtraction::<$type>,
                    benchmark_multiplication::<$type>,
                    benchmark_bitand::<$type>,
                    benchmark_bitor::<$type>,
                    benchmark_bitxor::<$type>,
                )*
            );
    };
}

    macro_rules! int_assign_bench_group {
        ($group_name:ident: [ $($type:ty),* ]) => {
            criterion_group!(
                $group_name,
                $(
                    benchmark_addition_assign::<$type>,
                    benchmark_subtraction_assign::<$type>,
                    benchmark_multiplication_assign::<$type>,
                    benchmark_bitand_assign::<$type>,
                    benchmark_bitor_assign::<$type>,
                    benchmark_bitxor_assign::<$type>,
                )*
            );
    };
}

    int_bench_group!(int_benches: [FheUint8, FheUint12, FheUint16]);
    int_assign_bench_group!(int_assign_benches: [FheUint8, FheUint12, FheUint16]);
}

#[cfg(feature = "integers")]
use integer_benches::{int_assign_benches, int_benches};

use shortint_benches::{shortint_assign_benches, shortint_benches, shortint_scalar_benches};

#[cfg(not(feature = "integers"))]
criterion_main!(
    shortint_benches,
    shortint_assign_benches,
    shortint_scalar_benches
);
#[cfg(feature = "integers")]
criterion_main!(
    shortint_benches,
    shortint_assign_benches,
    shortint_scalar_benches,
    int_benches,
    int_assign_benches
);
