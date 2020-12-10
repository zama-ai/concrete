compile_error! {
        "The current implementation of `concrete-csprng` uses the `aes` and `sse2` instruction \
        sets. Either add `RUSTFLAGS=\"-C target-cpu=native\"` or \
        `RUSTFLAGS=\"-C target-feature=+aes -C target-feature+sse2\"` to your cargo command, or \
        modify your `.cargo/config` to include the proper compilation flags."
}

// This is a fake generator to please the compiler. This module is not implemented.
pub struct RandomGenerator;

impl RandomGenerator {
    pub fn new() -> RandomGenerator {
        unimplemented!()
    }
    pub fn generate_next(&mut self) -> u8 {
        unimplemented!()
    }
}
