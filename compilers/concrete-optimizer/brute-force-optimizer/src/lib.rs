#![allow(non_snake_case)]

use std::ops::Range;
use std::vec::IntoIter;

// Useful because Range<u64> is not Copy
#[derive(Clone, Copy)]
struct MyRange(u64, u64);

impl MyRange {
    pub fn to_std_range(self) -> Range<u64> {
        Range {
            start: self.0,
            end: self.1,
        }
    }

    pub fn to_std_range_tight(self, baselog: u64, precision: u64) -> Range<u64> {
        // REMARK: in precision we take into account the min(noise MS)

        // To be used for the level range, we only need level * baselog <= 53
        // and so we have level <= 53. / baselog
        // We also need level * baselog >= precision + min(noise MS) i.e.
        // level >= (precision +  min(noise MS))/ baselog
        Range {
            start: (precision / baselog).max(self.0),
            // start: self.0,
            end: (53 / baselog).min(self.1),
        }
    }

    pub fn to_std_range_poly_size(self, precision: u64) -> Range<u64> {
        // REMARK: in precision we take into account the min(noise MS)

        // we need log2 N >= precision + 1
        Range {
            start: (precision + 1).max(self.0),
            // start: self.0,
            end: self.1,
        }
    }

    #[allow(unused)]
    pub fn to_std_range_lwe_dim(self, log_poly_size: u64, glwe_dimension: u64) -> Range<u64> {
        Range {
            start: self.0,
            end: self.1.min(glwe_dimension * (1 << log_poly_size)),
        }
    }

    #[allow(unused)]
    pub fn to_std_range_kt_zeros(
        self,
        log_poly_size: u64,
        glwe_dimension: u64,
        small_lwe_dim: u64,
    ) -> Range<u64> {
        let poly_size = 1 << (log_poly_size);
        // from 0 to min(N/-1, N-n)
        let tmp = if poly_size * glwe_dimension < small_lwe_dim {
            0
        } else {
            poly_size * glwe_dimension - small_lwe_dim + 1
        };
        Range {
            start: 0,
            end: (((1 << (log_poly_size)) - 1).min(tmp)).min(512),
        }
    }
}

pub fn minimal_added_noise_by_modulus_switching(lwe_dim: u64) -> f64 {
    (1. / 12. + lwe_dim as f64 / 24.)
        + (lwe_dim as f64 / 48. - 1. / 12.) * 4. / (f64::exp2(2. * 64.))
}

pub fn pbs_p_fail_from_global_p_fail(nb_pbs: u64, global_p_fail: f64) -> f64 {
    1. - f64::powf(1. - global_p_fail, 1. / (nb_pbs as f64))
}

#[derive(Clone)]
struct ExplicitRange(Vec<(u64, u64)>);

impl ExplicitRange {
    pub fn into_iter(self) -> IntoIter<(u64, u64)> {
        self.0.into_iter()
    }
}

const STEP: usize = 4; // 4;

pub mod cggi;
pub mod cjp;
pub mod gba;
pub mod generic;
pub mod ks_free;
pub mod lmp;
pub mod multi_bit_cjp;

pub struct Solution<T> {
    pub precision: u64,
    pub log_norm: u64,
    pub intem: Option<(T, f64)>,
}
