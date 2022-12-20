const SECURITY_WEIGHTS_ARRAY: [(u64, SecurityWeights); 4] = include!("./curves.gen.rs");

#[derive(Clone, Copy)]
pub struct SecurityWeights {
    slope: f64,
    bias: f64,
    minimal_lwe_dimension: u64,
}

impl SecurityWeights {
    pub fn secure_log2_std(&self, lwe_dimension: u64, ciphertext_modulus_log: f64) -> f64 {
        // ensure to have a minimal on std deviation covering the 2 lowest bits on modular scale
        let epsilon_log2_std_modular = 2.0;
        let epsilon_log2_std = epsilon_log2_std_modular - (ciphertext_modulus_log);
        // ensure the requested lwe_dimension is bigger than the minimal lwe dimension
        if self.minimal_lwe_dimension <= lwe_dimension {
            f64::max(
                self.slope * lwe_dimension as f64 + self.bias,
                epsilon_log2_std,
            )
        } else {
            ciphertext_modulus_log
        }
    }
}

pub fn supported_security_levels() -> impl std::iter::Iterator<Item = u64> {
    SECURITY_WEIGHTS_ARRAY
        .iter()
        .map(|(security_level, _)| *security_level)
}

pub fn security_weight(security_level: u64) -> Option<SecurityWeights> {
    let index = SECURITY_WEIGHTS_ARRAY
        .binary_search_by_key(&security_level, |(security_level, _)| *security_level)
        .ok()?;

    Some(SECURITY_WEIGHTS_ARRAY[index].1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let weight = security_weight(128).unwrap();

        let secure_log_2_std = weight.secure_log2_std(512, 64.);

        assert!((-12.0..-10.0).contains(&secure_log_2_std));
    }
}
