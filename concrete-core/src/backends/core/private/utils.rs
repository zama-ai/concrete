//! Utilities for the library.

/// This macro is used in tandem with the [`zip_args`] macro, to allow to zip iterators and access
/// them in an non-nested fashion. This makes large zip iterators easier to write, but also,
/// makes the code faster, as zipped-flatten iterators are hard to optimize for the compiler.
///
/// # Example
///
/// ```rust
/// use concrete_core::backends::core::private::{zip, zip_args};
/// let a = [1, 2, 3];
/// let b = [2, 2, 3];
/// let c = [4, 5, 6];
///
/// // Normally we would do:
/// for (a, (b, c)) in a.iter().zip(b.iter().zip(c.iter())) {
///     println!("{}{}{}", a, b, c);
/// }
///
/// // Now we can do:
/// for zip_args!(a, b, c) in zip!(a.iter(), b.iter(), c.iter()) {
///     println!("{}{}{}", a, b, c);
/// }
/// ```
macro_rules! zip {
    ($($iterator:expr),*)  => {
        $crate::backends::core::private::utils::zip!(@zip $($iterator),*)
    };
    (@zip $first:expr, $($iterator:expr),* ) => {
        $first.zip($crate::backends::core::private::utils::zip!(@zip $($iterator),*))
    };
    (@zip $first:expr) => {
        $first
    };
}
pub(crate) use zip;

/// Companion macro to flatten the iterators made with the [`zip`]
macro_rules! zip_args {
    ($($iterator:pat),*)  => {
        $crate::backends::core::private::utils::zip_args!(@zip $($iterator),*)
    };
    (@zip $first:pat, $second:pat) => {
        ($first, $second)
    };
    (@zip $first:pat, $($iterator:pat),*) => {
        ($first, $crate::backends::core::private::utils::zip_args!(@zip $($iterator),*))
    };
}
pub(crate) use zip_args;

#[cfg(test)]
mod test {
    #![allow(clippy::many_single_char_names)]

    #[test]
    fn test_zip() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        let c = vec![7, 8, 9];
        let d = vec![10, 11, 12];
        let e = vec![13, 14, 15];
        let f = vec![16, 17, 18];
        let g = vec![19, 20, 21];
        for zip_args!(a, b, c) in zip!(a.iter(), b.iter(), c.iter()) {
            println!("{},{},{}", a, b, c);
        }
        let mut iterator = zip!(
            a.into_iter(),
            b.into_iter(),
            c.into_iter(),
            d.into_iter(),
            e.into_iter(),
            f.into_iter(),
            g.into_iter()
        );
        assert_eq!(
            iterator.next().unwrap(),
            (1, (4, (7, (10, (13, (16, 19))))))
        );
        assert_eq!(
            iterator.next().unwrap(),
            (2, (5, (8, (11, (14, (17, 20))))))
        );
        assert_eq!(
            iterator.next().unwrap(),
            (3, (6, (9, (12, (15, (18, 21))))))
        );
    }
}
