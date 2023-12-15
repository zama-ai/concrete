pub mod wop_simulation;

#[inline]
fn debug_assert_same_len(a: (usize, Option<usize>), b: (usize, Option<usize>)) {
    debug_assert_eq!(a.1, Some(a.0));
    debug_assert_eq!(b.1, Some(b.0));
    debug_assert_eq!(a.0, b.0);
}

pub fn zip_eq<T, U>(
    a: impl IntoIterator<Item = T>,
    b: impl IntoIterator<Item = U>,
) -> impl Iterator<Item = (T, U)> {
    let a = a.into_iter();
    let b = b.into_iter();
    debug_assert_same_len(a.size_hint(), b.size_hint());
    core::iter::zip(a, b)
}

pub fn from_torus(input: f64) -> u64 {
    let mut fract = input - f64::round(input);
    fract *= 2.0_f64.powi(u64::BITS as i32);
    fract = f64::round(fract);

    fract as i64 as u64
}
