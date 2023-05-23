pub fn mask_bytes_per_coef() -> usize {
    u64::BITS as usize / 8
}

pub fn mask_bytes_per_polynomial(polynomial_size: usize) -> usize {
    polynomial_size * mask_bytes_per_coef()
}

pub fn mask_bytes_per_glwe(dimension: usize, polynomial_size: usize) -> usize {
    dimension * mask_bytes_per_polynomial(polynomial_size)
}

pub fn mask_bytes_per_ggsw_level(dimension: usize, polynomial_size: usize) -> usize {
    (dimension + 1) * mask_bytes_per_glwe(dimension, polynomial_size)
}

pub fn mask_bytes_per_lwe(lwe_dimension: usize) -> usize {
    lwe_dimension * mask_bytes_per_coef()
}

pub fn mask_bytes_per_gsw_level(lwe_dimension: usize) -> usize {
    (lwe_dimension + 1) * mask_bytes_per_lwe(lwe_dimension)
}

pub fn mask_bytes_per_ggsw(
    decomposition_level_count: usize,
    dimension: usize,
    polynomial_size: usize,
) -> usize {
    decomposition_level_count * mask_bytes_per_ggsw_level(dimension, polynomial_size)
}

pub fn mask_bytes_per_pfpksk_chunk(
    decomposition_level_count: usize,
    dimension: usize,
    polynomial_size: usize,
) -> usize {
    decomposition_level_count * mask_bytes_per_glwe(dimension, polynomial_size)
}

pub fn mask_bytes_per_pfpksk(
    decomposition_level_count: usize,
    dimension: usize,
    polynomial_size: usize,
    lwe_dimension: usize,
) -> usize {
    (lwe_dimension + 1)
        * mask_bytes_per_pfpksk_chunk(decomposition_level_count, dimension, polynomial_size)
}

pub fn noise_bytes_per_coef() -> usize {
    // We use f64 to sample the noise from a normal distribution with the polar form of the
    // Box-Muller algorithm. With this algorithm, the input pair of uniform values will be rejected
    // with a probability of  pi/4 which means that in average, we need ~4/pi pair of uniform
    // values for one pair of normal values. To have a safety margin, we require 32 uniform inputs
    // (>> 4/pi) for one pair of normal values
    8 * 32
}
pub fn noise_bytes_per_polynomial(polynomial_size: usize) -> usize {
    polynomial_size * noise_bytes_per_coef()
}

pub fn noise_bytes_per_glwe(polynomial_size: usize) -> usize {
    noise_bytes_per_polynomial(polynomial_size)
}

pub fn noise_bytes_per_ggsw_level(dimension: usize, polynomial_size: usize) -> usize {
    (dimension + 1) * noise_bytes_per_glwe(polynomial_size)
}

pub fn noise_bytes_per_lwe() -> usize {
    // Here we take 3 to keep a safety margin
    noise_bytes_per_coef() * 3
}

pub fn noise_bytes_per_gsw_level(lwe_dimension: usize) -> usize {
    (lwe_dimension + 1) * noise_bytes_per_lwe()
}

pub fn noise_bytes_per_ggsw(
    decomposition_level_count: usize,
    dimension: usize,
    polynomial_size: usize,
) -> usize {
    decomposition_level_count * noise_bytes_per_ggsw_level(dimension, polynomial_size)
}

pub fn noise_bytes_per_pfpksk_chunk(
    decomposition_level_count: usize,
    polynomial_size: usize,
) -> usize {
    decomposition_level_count * noise_bytes_per_glwe(polynomial_size)
}

pub fn noise_bytes_per_pfpksk(
    decomposition_level_count: usize,
    polynomial_size: usize,
    lwe_dimension: usize,
) -> usize {
    (lwe_dimension + 1) * noise_bytes_per_pfpksk_chunk(decomposition_level_count, polynomial_size)
}
