pub trait ComplexityNoise {
    fn noise(&self) -> f64;
    fn complexity(&self) -> f64;
}

pub fn cut_complexity_noise<E>(cut_complexity: f64, cut_noise: f64, decomps: &[E]) -> &[E]
where
    E: ComplexityNoise,
{
    let mut min_index = None;
    // Search first valid noise
    for (i, decomp) in decomps.iter().enumerate() {
        if decomp.noise() <= cut_noise {
            min_index = Some(i);
            break; // noise is decreasing
        }
    }
    let min_index = min_index.unwrap_or(decomps.len());
    // Search first invalid complexity
    let mut max_index = None;
    for (i, decomp) in decomps.iter().enumerate().skip(min_index) {
        if cut_complexity < decomp.complexity() {
            max_index = Some(i);
            break; // complexity is increasing
        }
    }
    let max_index = max_index.unwrap_or(decomps.len());
    if min_index == max_index {
        return &[];
    }
    assert!(min_index < max_index);
    &decomps[min_index..max_index]
}
