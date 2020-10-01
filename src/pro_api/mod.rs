//! ProAPI module

use crate::types::CTorus;
use serde::de::DeserializeOwned;
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

pub type Torus = u64;

#[allow(unused_macros)]
macro_rules! assert_eq_granularity {
    ($A:expr, $B:expr, $ENC:expr) => {
        if ($A - $B).abs() >= $ENC.get_granularity() {
            panic!(
                "{} != {} +- {} (|delta|={})-> encoder: {}",
                $A,
                $B,
                $ENC.get_granularity(),
                ($A - $B).abs(),
                $ENC
            );
        }
    };
}

#[allow(unused_macros)]
macro_rules! generate_random_interval {
    () => {{
        let mut coins = vec![0 as u32; 3];
        Random::rng_uniform(&mut coins);

        let interval_type: usize = (coins[0] % 3) as usize;
        let interval_size = ((coins[1] % (1000 * 1000)) as f64) / 1000.;
        let interval_start = ((coins[2] % (1000 * 1000)) as f64) / 1000.;
        match interval_type {
            0 => {
                // negative interval
                (-interval_start - interval_size, -interval_start)
            }
            1 => {
                // positive interval
                (interval_start, interval_size + interval_start)
            }
            2 => {
                // zero in the interval
                let tmp = ((coins[2] % (1000 * 1000)) as f64) / (1000. * 1000.) * interval_size;
                (-interval_size + tmp, tmp)
            }
            _ => (0., 0.),
        }
    }};
}

#[allow(unused_macros)]
macro_rules! generate_random_centered_interval {
    () => {{
        let mut coins = vec![0 as u32; 2];
        Random::rng_uniform(&mut coins);

        let interval_size = ((coins[0] % (1000 * 1000)) as f64) / 1000.;

        // zero in the interval
        let tmp = ((coins[1] % (1000 * 1000)) as f64) / (1000. * 1000.) * interval_size;
        (-interval_size + tmp, tmp)
    }};
}

#[allow(unused_macros)]
macro_rules! generate_precision_padding {
    ($max_precision: expr, $max_padding: expr) => {{
        let mut rs = vec![0 as u32; 2];
        Random::rng_uniform(&mut rs);
        (
            ((rs[0] % $max_precision) as usize) + 1,
            (rs[1] % $max_padding) as usize,
        )
    }};
}

#[allow(unused_macros)]
macro_rules! random_index {
    ($max: expr) => {{
        if $max == 0 {
            (0 as usize)
        } else {
            let mut rs = vec![0 as u32; 1];
            Random::rng_uniform(&mut rs);
            (rs[0] % ($max as u32)) as usize
        }
    }};
}

#[allow(unused_macros)]
macro_rules! random_message {
    ($min: expr, $max: expr) => {{
        let mut rs = vec![0 as u64; 1];
        Random::rng_uniform(&mut rs);
        (rs[0] as f64) / f64::powi(2., 64) * ($max - $min) + $min
    }};
}

#[allow(unused_macros)]
macro_rules! random_messages {
    ($min: expr, $max: expr, $nb: expr) => {{
        let mut res = vec![0 as f64; $nb];
        for r in res.iter_mut() {
            *r = random_message!($min, $max);
        }
        res
    }};
}

macro_rules! deltas_eq {
    ($delta1: expr, $delta2: expr) => {{
        let tmp = ($delta1 - $delta2).abs() / $delta1.max($delta2);
        let limit = f64::powi(2., -45);
        tmp < limit
    }};
}

pub fn write_to_file<P: AsRef<Path>, U: Serialize>(path: P, u: &U) -> Result<(), Box<dyn Error>> {
    // Create the file
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, u)?;
    Ok(())
}

fn read_from_file<P: AsRef<Path>, U: DeserializeOwned>(path: P) -> Result<U, Box<dyn Error>> {
    // Open the file in read-only mode with buffer.
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    // Read the JSON contents of the file
    let u = serde_json::from_reader(reader)?;
    Ok(u)
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "CTorus")]
struct SerdeCtorus {
    re: f64,
    im: f64,
}

fn deserialize_vec_ctorus<'de, D>(deserializer: D) -> Result<Vec<CTorus>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    struct Wrapper(#[serde(with = "SerdeCtorus")] CTorus);
    let v = Vec::deserialize(deserializer)?;
    Ok(v.into_iter().map(|Wrapper(a)| a).collect())
}

fn serialize_vec_ctorus<S>(x: &Vec<CTorus>, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    #[derive(Serialize)]
    struct Wrapper(#[serde(with = "SerdeCtorus")] CTorus);
    let mut seq = s.serialize_seq(Some(x.len()))?;
    for e in x.iter() {
        seq.serialize_element(&Wrapper(*e))?;
    }
    seq.end()
}

#[allow(unused_macros)]
macro_rules! assert_with_granularity {
    ($before: expr, $after: expr, $encoders: expr) => {
        for (b, (a, encoder)) in $before.iter().zip($after.iter().zip($encoders.iter())) {
            if (b - a).abs() > encoder.get_granularity() / 2. {
                panic!("a = {} ; b = {} ; granularity = ", a, b);
            }
        }
    };
}

macro_rules! pub_mod_use {
    ($I:ident) => {
        pub mod $I;
        pub use $I::*;
    };
}
#[macro_use]
pub mod error;

pub_mod_use!(lwe_params);
pub_mod_use!(encoder);
pub_mod_use!(lwe);
pub_mod_use!(plaintext);
pub_mod_use!(rlwe);
pub_mod_use!(lwe_ksk);
pub_mod_use!(lwe_bsk);
pub_mod_use!(lwe_secret_key);
pub_mod_use!(rlwe_params);
pub_mod_use!(rlwe_secret_key);

#[cfg(test)]
mod tests_serde;
