#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![warn(clippy::style)]

use chrono::{Datelike, Utc};
use clap::Parser;
use concrete_optimizer::security::security_weights::SECURITY_WEIGHTS_TABLE;
use std::fs::File;
use v0_parameters::{compute_print_results, Args};

fn main() {
    let mut args = Args::parse();

    let now = Utc::now();

    let year = now.year();
    let month = now.month();
    let day = now.day();

    let filename = if args.wop_pbs {
        format!("ref/wop_pbs_{year}-{month}-{day}")
    } else {
        format!("ref/v0_{year}-{month}-{day}")
    };

    if args.wop_pbs {
        args.min_intern_lwe_dim = 450;
        args.min_log_poly_size = 10;
        args.max_log_poly_size = 11;
        args.max_precision = 16;
    }

    for &security_level in SECURITY_WEIGHTS_TABLE.keys() {
        args.security_level = security_level;

        let filename = format!("{filename}_{security_level}");

        let file = File::create(filename).unwrap();

        compute_print_results(file, &args).unwrap();
    }
}
