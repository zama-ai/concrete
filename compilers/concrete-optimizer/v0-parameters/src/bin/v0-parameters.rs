#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![warn(clippy::style)]

use clap::Parser;
use v0_parameters::{compute_print_results, Args};

fn main() {
    let args = Args::parse();

    compute_print_results(std::io::stdout(), &args).unwrap();
}
