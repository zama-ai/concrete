pub fn f64_max(values: &[f64], default: f64) -> f64 {
    values.iter().copied().reduce(f64::max).unwrap_or(default)
}
