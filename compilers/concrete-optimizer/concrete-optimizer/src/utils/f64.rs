pub fn f64_max(values: &[f64], default: f64) -> f64 {
    values.iter().copied().reduce(f64::max).unwrap_or(default)
}

pub fn f64_dot(a: &[f64], b: &[f64]) -> f64 {
    assert!(
        a.len() == b.len(),
        "Dot incompatible size: {} vs {}",
        a.len(),
        b.len()
    );
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}
