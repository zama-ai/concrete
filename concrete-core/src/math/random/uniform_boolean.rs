use super::*;

/// A distribution type representing uniform sampling for boolean type.
#[derive(Clone, Copy)]
pub struct UniformBoolean;

impl RandomGenerable<UniformBoolean> for bool {
    #[allow(unused)]
    fn generate_one(generator: &mut RandomGenerator, distribution: UniformBoolean) -> Self {
        generator.generate_next() & 1 == 1
    }
}
