#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![warn(clippy::style)]

use chrono::{Datelike, Utc};
use clap::Parser;
use concrete_optimizer::supported_security_levels;
use std::fs::File;
use v0_parameters::{compute_print_results, Args};

fn main() {
    let mut args = Args::parse();

    let now = Utc::now();

    let year = now.year();
    let month = now.month();
    let day = now.day();

    if args.wop_pbs {
        args.min_intern_lwe_dim = 450;
        args.min_log_poly_size = 10;
        args.max_log_poly_size = 11;
        args.max_precision = 16;
    } else {
        args.max_precision = 10;
        args.max_log_poly_size = 17;
    }

    for security_level in supported_security_levels() {
        let filename_date = if args.wop_pbs {
            format!("ref/wop_pbs_{year}-{month}-{day}_{security_level}")
        } else {
            format!("ref/v0_{year}-{month}-{day}_{security_level}")
        };
        let filename_last = if args.wop_pbs {
            format!("ref/wop_pbs_last_{security_level}")
        } else {
            format!("ref/v0_last_{security_level}")
        };
        args.security_level = security_level;
        let file = File::create(&filename_date).unwrap();
        compute_print_results(file, &args).unwrap();
        std::fs::copy(&filename_date, filename_last).expect("Copy to last failed");
    }
}
