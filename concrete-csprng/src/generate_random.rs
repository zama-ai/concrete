//! This program uses the concrete csprng to generate an infinite stream of random bytes on
//! the program stdout. For testing purpose.
use std::io::prelude::*;
use std::io::stdout;

use concrete_csprng::RandomGenerator;

pub fn main() {
    let mut generator = RandomGenerator::new(None);
    let mut stdout = stdout();
    let mut buffer = [0u8; 16];
    loop {
        buffer
            .iter_mut()
            .for_each(|a| *a = generator.generate_next());
        stdout.write_all(&buffer).unwrap();
    }
}
