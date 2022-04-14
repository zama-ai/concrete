//! This program uses the concrete csprng to generate an infinite stream of random bytes on
//! the program stdout. For testing purpose.
use concrete_csprng::generators::{AesniRandomGenerator, RandomGenerator};
use concrete_csprng::seeders::{RdseedSeeder, Seeder};
use std::io::prelude::*;
use std::io::stdout;

pub fn main() {
    let mut seeder = RdseedSeeder;
    let mut generator = AesniRandomGenerator::new(seeder.seed());
    let mut stdout = stdout();
    let mut buffer = [0u8; 16];
    loop {
        buffer
            .iter_mut()
            .zip(&mut generator)
            .for_each(|(b, g)| *b = g);
        stdout.write_all(&buffer).unwrap();
    }
}
