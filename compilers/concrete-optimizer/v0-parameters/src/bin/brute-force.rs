use brute_force_optimizer::cggi::solve_all_cggi;
use brute_force_optimizer::cjp::solve_all_cjp;
use brute_force_optimizer::ks_free::solve_all_ksfree;

use brute_force_optimizer::gba::solve_all_gba;
use brute_force_optimizer::lmp::solve_all_lmp;
use brute_force_optimizer::multi_bit_cjp::solve_all_multi_bit_cjp;
use clap::Parser;
use std::fs::File;
use v0_parameters::_4_SIGMA;

/// Find parameters for a variety of atomic patterns using a brute force algorithm
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
#[allow(clippy::struct_excessive_bools)]
pub struct BruteForceArgs {
    #[clap(long, default_value_t = _4_SIGMA)]
    pub p_fail: f64,

    #[clap(
        long,
        help = "Supported atomic patterns: CJP, KSfree, CGGI, GBA and LMP and MBCJP"
    )]
    pub atomic_pattern: String,
}

fn main() {
    let args = BruteForceArgs::parse();

    let filename = format!(
        "exp/{}-pfail-{}.txt",
        args.atomic_pattern,
        args.p_fail.log2().round()
    );
    let file = File::create(filename).unwrap();

    match args.atomic_pattern.as_str() {
        "CJP" => solve_all_cjp(args.p_fail, file),
        "CGGI" => solve_all_cggi(args.p_fail, file),
        "KSfree" => solve_all_ksfree(args.p_fail, file),
        "LMP" => solve_all_lmp(args.p_fail, file),
        "GBA" => solve_all_gba(args.p_fail, file),
        "MBCJP" => solve_all_multi_bit_cjp(args.p_fail, file),
        _ => {
            panic!(
                "The resquested AP is not supported ({})",
                args.atomic_pattern
            )
        }
    };
}
