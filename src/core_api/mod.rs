//! Operators Module
//! * Contains material needed to manipulate tensors of Torus without any predetermined meaning

#[allow(unused_macros)]
macro_rules! assert_delta {
    ($A:expr, $B:expr, $d:expr) => {
        for (x, y) in $A.iter().zip($B) {
            if (*x as i64 - y as i64).abs() > $d {
                panic!("{} != {} ", *x, y);
            }
        }
    };
}

#[allow(unused_macros)]
macro_rules! assert_delta_scalar {
    ($A:expr, $B:expr, $d:expr) => {
        if ($A as i64 - $B as i64).abs() > $d {
            panic!("{} != {} +- {}", $A, $B, $d);
        }
    };
}

#[allow(unused_macros)]
macro_rules! assert_delta_scalar_float {
    ($A:expr, $B:expr, $d:expr) => {
        if ($A - $B).abs() > $d {
            panic!("{} != {} +- {}", $A, $B, $d);
        }
    };
}

#[allow(unused_macros)]
macro_rules! modular_distance {
    ($A:expr, $B:expr) => {
        ($A.wrapping_sub($B)).min($B.wrapping_sub($A))
    };
}

#[allow(unused_macros)]
/// takes a tensor, a value and a std_dev and panic if the gap between one of the tensor element ond the value
/// is bigger than 96% (not normal, 99% for normal) of the samples from the distribution
macro_rules! assert_delta_std_dev {
    ($A:expr, $B:expr, $d:expr) => {
        use crate::Types;
        for (x, y) in $A.iter().zip($B.iter()) {
            println!("{}, {}", *x, *y);
            println!("{}", $d);
            let distance = modular_distance!(*x, *y);
            let torus_distance =
                distance as f64 / f64::powi(2., <Torus as Types>::TORUS_BIT as i32);
            if torus_distance > 5. * $d {
                panic!("{} != {} ", x, y);
            }
        }
    };
}

#[allow(unused_macros)]
macro_rules! assert_noise_distribution {
    ($messages:expr, $new_messages: expr, $variance:expr, $op_name:expr) => {
        use crate::core_api::math::random::{vectorial_openssl_float_normal};
        use crate::core_api::math::Tensor;
        let std_dev = f64::sqrt($variance);
        let confidence = 0.95;
        let n_slots = $messages.len();
        let mut sdk_samples: Vec<f64> = vec![0.; n_slots];
        let mut theoretical_samples = vec![0.; n_slots];
        Tensor::compute_signed_modular_distance(&mut sdk_samples, &$messages, &$new_messages);
        // rng_float_normal(&mut theoretical_samples, 0., std_dev);
        vectorial_openssl_float_normal(&mut theoretical_samples, 0., std_dev);
        let result = kolmogorov_smirnov::test_f64(&sdk_samples, &theoretical_samples, confidence);
        if result.is_rejected {
            let mut mean: f64 = sdk_samples.iter().sum() ;
            mean /= sdk_samples.len() as f64 ;
            // test if theoritical_std_dev > sdk_std_dev
            let mut sdk_variance: f64 = sdk_samples.iter().map(|x| f64::powi(x - mean, 2) ).sum() ;
            sdk_variance /= (sdk_samples.len() - 1) as f64;
            let sdk_std_log2 = f64::log2(f64::sqrt(sdk_variance)).round() ;
            // sdk_variance = (sdk_variance * f64::powi(2.,2* 52)).round() *  f64::powi(2.,- 2* 52) ;
            // let variance_r = ($variance * f64::powi(2.,2 * 52)).round() *  f64::powi(2.,-2*52) ;
            let th_std_log2 = f64::log2(std_dev).round() ;
            if  (sdk_std_log2 > th_std_log2) {
                panic!("\n Statistical test failed : \n -> inputs are not from the same distribution with a probability {} \n -> sdk_std = {} ; th_std {}", result.reject_probability, sdk_std_log2, th_std_log2);
            }
        }
    };
}

macro_rules! TWIDDLES_TORUS {
    ($degree:expr) => {
        match $degree {
            256 => &crate::core_api::math::twiddles::TWIDDLES_256[..],
            512 => &crate::core_api::math::twiddles::TWIDDLES_512[..],
            1024 => &crate::core_api::math::twiddles::TWIDDLES_1024[..],
            2048 => &crate::core_api::math::twiddles::TWIDDLES_2048[..],
            4096 => &crate::core_api::math::twiddles::TWIDDLES_4096[..],
            _ => panic!("Degree {} is not yet supported", $degree),
        }
    };
}

#[allow(unused_macros)]
#[macro_export]
macro_rules! TWIDDLES_TORUS_external {
    ($degree:expr) => {
        match $degree {
            256 => &concrete_lib::core_api::math::twiddles::TWIDDLES_256[..],
            512 => &concrete_lib::core_api::math::twiddles::TWIDDLES_512[..],
            1024 => &concrete_lib::core_api::math::twiddles::TWIDDLES_1024[..],
            2048 => &concrete_lib::core_api::math::twiddles::TWIDDLES_2048[..],
            4096 => &concrete_lib::core_api::math::twiddles::TWIDDLES_4096[..],
            _ => panic!("Degree {} is not yet supported", $degree),
        }
    };
}
macro_rules! INVERSE_TWIDDLES_TORUS {
    ($degree:expr) => {
        match $degree {
            256 => &crate::core_api::math::twiddles::INVERSE_TWIDDLES_256[..],
            512 => &crate::core_api::math::twiddles::INVERSE_TWIDDLES_512[..],
            1024 => &crate::core_api::math::twiddles::INVERSE_TWIDDLES_1024[..],
            2048 => &crate::core_api::math::twiddles::INVERSE_TWIDDLES_2048[..],
            4096 => &crate::core_api::math::twiddles::INVERSE_TWIDDLES_4096[..],
            _ => panic!("Degree {} is not yet supported", $degree),
        }
    };
}

#[allow(unused_macros)]
#[macro_export]
macro_rules! INVERSE_TWIDDLES_TORUS_external {
    ($degree:expr) => {
        match $degree {
            256 => &concrete_lib::core_api::math::twiddles::INVERSE_TWIDDLES_256[..],
            512 => &concrete_lib::core_api::math::twiddles::INVERSE_TWIDDLES_512[..],
            1024 => &concrete_lib::core_api::math::twiddles::INVERSE_TWIDDLES_1024[..],
            2048 => &concrete_lib::core_api::math::twiddles::INVERSE_TWIDDLES_2048[..],
            4096 => &concrete_lib::core_api::math::twiddles::INVERSE_TWIDDLES_4096[..],
            _ => panic!("Degree {} is not yet supported", $degree),
        }
    };
}

pub mod crypto;
pub mod math;
