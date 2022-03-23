pub fn sigma_scale_of_error_probability(p_error: f64) -> f64 {
    // https://en.wikipedia.org/wiki/Error_function#Applications
    let p_in = 1.0 - p_error;
    statrs::function::erf::erf_inv(p_in) * 2_f64.sqrt()
}

pub fn error_probability_of_sigma_scale(sigma_scale: f64) -> f64 {
    // https://en.wikipedia.org/wiki/Error_function#Applications
    1.0 - statrs::function::erf::erf(sigma_scale / 2_f64.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_sigmas() {
        // https://en.wikipedia.org/wiki/Normal_distribution#Standard_deviation_and_coverage
        let reference = &[
            0.682_689_492_137, // +- 1 sigma
            0.954_499_736_104, // 2
            0.997_300_203_937, // ...
            0.999_936_657_516,
            0.999_999_426_697,
        ];
        for (i, &p_in) in reference.iter().enumerate() {
            let p_out = 1.0 - p_in;
            let expected_scale = (i + 1) as f64;
            approx::assert_relative_eq!(
                expected_scale,
                sigma_scale_of_error_probability(p_out),
                max_relative = 1e-8
            );
            approx::assert_relative_eq!(
                p_out,
                error_probability_of_sigma_scale(sigma_scale_of_error_probability(p_out)),
                max_relative = 1e-8
            );
        }
    }
}
