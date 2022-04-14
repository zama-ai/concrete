use crate::seeders::{Seed, Seeder};
use std::fs::File;
use std::io::Read;

/// A seeder which uses the linux `/dev/random` source.
pub struct LinuxSeeder {
    counter: u128,
    secret: u128,
    file: File,
}

impl LinuxSeeder {
    /// Creates a new seeder from a user defined secret.
    ///
    /// Important:
    /// ----------
    ///
    /// This secret is used to ensure the quality of the seed in scenarios where `/dev/random` may
    /// be compromised.
    pub fn new(secret: u128) -> LinuxSeeder {
        let file = std::fs::File::open("/dev/random").expect("Failed to open /dev/random .");
        let counter = std::time::UNIX_EPOCH
            .elapsed()
            .expect("Failed to initialized software rdseed counter.")
            .as_nanos();
        LinuxSeeder {
            secret,
            counter,
            file,
        }
    }
}

impl Seeder for LinuxSeeder {
    fn seed(&mut self) -> Seed {
        let output = self.secret ^ self.counter ^ dev_random(&mut self.file);
        self.counter = self.counter.wrapping_add(1);
        Seed(output)
    }
}

fn dev_random(random: &mut File) -> u128 {
    let mut buf = [0u8; 16];
    random
        .read_exact(&mut buf[..])
        .expect("Failed to read from /dev/random .");
    u128::from_ne_bytes(buf)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::seeders::generic_tests::check_seeder_fixed_sequences_different;

    #[test]
    fn check_bounded_sequence_difference() {
        check_seeder_fixed_sequences_different(LinuxSeeder::new);
    }
}
