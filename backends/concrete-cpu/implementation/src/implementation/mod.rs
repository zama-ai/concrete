use core::mem::MaybeUninit;

use aligned_vec::{ABox, AVec, CACHELINE_ALIGN};

#[allow(unused_macros)]
macro_rules! izip {
    // no one should need to zip more than 16 iterators, right?
    (@ __closure @ ($a:expr)) => { |a| (a,) };
    (@ __closure @ ($a:expr, $b:expr)) => { |(a, b)| (a, b) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr)) => { |((a, b), c)| (a, b, c) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr)) => { |(((a, b), c), d)| (a, b, c, d) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr)) => { |((((a, b), c), d), e)| (a, b, c, d, e) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr)) => { |(((((a, b), c), d), e), f)| (a, b, c, d, e, f) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr)) => { |((((((a, b), c), d), e), f), g)| (a, b, c, d, e, f, e) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr)) => { |(((((((a, b), c), d), e), f), g), h)| (a, b, c, d, e, f, g, h) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr)) => { |((((((((a, b), c), d), e), f), g), h), i)| (a, b, c, d, e, f, g, h, i) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr, $j: expr)) => { |(((((((((a, b), c), d), e), f), g), h), i), j)| (a, b, c, d, e, f, g, h, i, j) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr, $j: expr, $k: expr)) => { |((((((((((a, b), c), d), e), f), g), h), i), j), k)| (a, b, c, d, e, f, g, h, i, j, k) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr, $j: expr, $k: expr, $l: expr)) => { |(((((((((((a, b), c), d), e), f), g), h), i), j), k), l)| (a, b, c, d, e, f, g, h, i, j, k, l) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr, $j: expr, $k: expr, $l: expr, $m:expr)) => { |((((((((((((a, b), c), d), e), f), g), h), i), j), k), l), m)| (a, b, c, d, e, f, g, h, i, j, k, l, m) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr, $j: expr, $k: expr, $l: expr, $m:expr, $n:expr)) => { |(((((((((((((a, b), c), d), e), f), g), h), i), j), k), l), m), n)| (a, b, c, d, e, f, g, h, i, j, k, l, m, n) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr, $j: expr, $k: expr, $l: expr, $m:expr, $n:expr, $o:expr)) => { |((((((((((((((a, b), c), d), e), f), g), h), i), j), k), l), m), n), o)| (a, b, c, d, e, f, g, h, i, j, k, l, m, n, o) };

    ( $first:expr $(,)?) => {
        {
            #[allow(unused_imports)]
            use $crate::implementation::ZipEq;
            ::core::iter::IntoIterator::into_iter($first)
        }
    };
    ( $first:expr, $($rest:expr),+ $(,)?) => {
        {
            #[allow(unused_imports)]
            use $crate::implementation::ZipEq;
            ::core::iter::IntoIterator::into_iter($first)
                $(.zip_eq($rest))*
                .map(izip!(@ __closure @ ($first, $($rest),*)))
        }
    };
}

mod convert;
mod decomposition;
pub mod fft;

pub mod encrypt;
pub mod wop;

/// Convert a mutable slice reference to an uninitialized mutable slice reference.
///
/// # Safety
///
/// No uninitialized values must be written into the output slice by the time the borrow ends
#[inline]
pub unsafe fn as_mut_uninit<T>(slice: &mut [T]) -> &mut [MaybeUninit<T>] {
    let len = slice.len();
    let ptr = slice.as_mut_ptr();
    // SAFETY: T and MaybeUninit<T> have the same layout
    unsafe { core::slice::from_raw_parts_mut(ptr as *mut _, len) }
}

/// Convert an uninitialized mutable slice reference to an initialized mutable slice reference.
///
/// # Safety
///
/// All the elements of the input slice must be initialized and in a valid state.
#[inline]
pub unsafe fn assume_init_mut<T>(slice: &mut [MaybeUninit<T>]) -> &mut [T] {
    let len = slice.len();
    let ptr = slice.as_mut_ptr();
    // SAFETY: T and MaybeUninit<T> have the same layout
    unsafe { core::slice::from_raw_parts_mut(ptr as *mut _, len) }
}

#[inline]
fn debug_assert_same_len(a: (usize, Option<usize>), b: (usize, Option<usize>)) {
    debug_assert_eq!(a.1, Some(a.0));
    debug_assert_eq!(b.1, Some(b.0));
    debug_assert_eq!(a.0, b.0);
}

/// Returns a Zip iterator, but checks that the two components have the same length.
trait ZipEq: IntoIterator + Sized {
    #[inline]
    fn zip_eq<B: IntoIterator>(
        self,
        b: B,
    ) -> core::iter::Zip<<Self as IntoIterator>::IntoIter, <B as IntoIterator>::IntoIter> {
        let a = self.into_iter();
        let b = b.into_iter();
        debug_assert_same_len(a.size_hint(), b.size_hint());
        core::iter::zip(a, b)
    }
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

impl<A: IntoIterator> ZipEq for A {}

pub trait Container: Sized + AsRef<[Self::Item]> {
    type Item;

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

pub trait ContainerMut: Container + AsMut<[Self::Item]> {}

pub trait ContainerOwned: Container + AsMut<[Self::Item]> {
    fn collect(iter: impl Iterator<Item = Self::Item>) -> Self;
}

impl<'a, T> Container for &'a [T] {
    type Item = T;
}

impl<'a, T> Container for &'a mut [T] {
    type Item = T;
}
impl<'a, T> ContainerMut for &'a mut [T] {}

impl<T> Container for ABox<[T]> {
    type Item = T;
}
impl<T> ContainerMut for ABox<[T]> {}
impl<T> ContainerOwned for ABox<[T]> {
    fn collect(iter: impl Iterator<Item = Self::Item>) -> Self {
        AVec::from_iter(CACHELINE_ALIGN, iter).into_boxed_slice()
    }
}

impl<T> Container for AVec<T> {
    type Item = T;
}
impl<T> ContainerMut for AVec<T> {}
impl<T> ContainerOwned for AVec<T> {
    fn collect(iter: impl Iterator<Item = Self::Item>) -> Self {
        AVec::from_iter(CACHELINE_ALIGN, iter)
    }
}

impl<T> Container for Vec<T> {
    type Item = T;
}
impl<T> ContainerMut for Vec<T> {}
impl<T> ContainerOwned for Vec<T> {
    fn collect(iter: impl Iterator<Item = Self::Item>) -> Self {
        iter.collect()
    }
}

pub trait Split: Container {
    type Pointer: Copy;
    type Chunks: DoubleEndedIterator<Item = Self> + ExactSizeIterator<Item = Self>;

    unsafe fn from_raw_parts(data: Self::Pointer, len: usize) -> Self;
    fn split_at(self, mid: usize) -> (Self, Self);
    fn chunk(self, start: usize, end: usize) -> Self {
        self.split_at(end).0.split_at(start).1
    }
    fn into_chunks(self, chunk_size: usize) -> Self::Chunks;
    fn split_into(self, chunk_count: usize) -> Self::Chunks {
        debug_assert_ne!(chunk_count, 0);
        let len = self.len();
        debug_assert_eq!(len % chunk_count, 0);
        self.into_chunks(len / chunk_count)
    }
}

impl<'a, T> Split for &'a [T] {
    type Pointer = *const T;
    type Chunks = core::slice::ChunksExact<'a, T>;

    unsafe fn from_raw_parts(data: Self::Pointer, len: usize) -> Self {
        unsafe { core::slice::from_raw_parts(data, len) }
    }

    fn split_at(self, mid: usize) -> (Self, Self) {
        (*self).split_at(mid)
    }

    fn into_chunks(self, chunk_size: usize) -> Self::Chunks {
        debug_assert_ne!(chunk_size, 0);
        debug_assert_eq!(self.len() % chunk_size, 0);
        self.chunks_exact(chunk_size)
    }
}

impl<'a, T> Split for &'a mut [T] {
    type Pointer = *mut T;
    type Chunks = core::slice::ChunksExactMut<'a, T>;

    unsafe fn from_raw_parts(data: Self::Pointer, len: usize) -> Self {
        unsafe { core::slice::from_raw_parts_mut(data, len) }
    }

    fn split_at(self, mid: usize) -> (Self, Self) {
        (*self).split_at_mut(mid)
    }

    fn into_chunks(self, chunk_size: usize) -> Self::Chunks {
        debug_assert_ne!(chunk_size, 0);
        debug_assert_eq!(self.len() % chunk_size, 0);
        self.chunks_exact_mut(chunk_size)
    }
}

#[cfg(feature = "parallel")]
pub mod parallel {
    use super::*;
    use rayon::prelude::*;

    pub trait ParSplit: Split + Send {
        type ParChunks: IndexedParallelIterator<Item = Self>;

        fn into_par_chunks(self, chunk_size: usize) -> Self::ParChunks;

        fn par_split_into(self, chunk_count: usize) -> Self::ParChunks {
            if chunk_count == 0 {
                self.split_at(0).0.into_par_chunks(1)
            } else {
                let len = self.len();
                debug_assert_eq!(len % chunk_count, 0);
                self.into_par_chunks(len / chunk_count)
            }
        }
    }

    impl<'a, T: Sync> ParSplit for &'a [T] {
        type ParChunks = rayon::slice::ChunksExact<'a, T>;

        fn into_par_chunks(self, chunk_size: usize) -> Self::ParChunks {
            self.par_chunks_exact(chunk_size)
        }
    }

    impl<'a, T: Send> ParSplit for &'a mut [T] {
        type ParChunks = rayon::slice::ChunksExactMut<'a, T>;

        fn into_par_chunks(self, chunk_size: usize) -> Self::ParChunks {
            self.par_chunks_exact_mut(chunk_size)
        }
    }
}

pub fn from_torus(input: f64) -> u64 {
    let mut fract = input - f64::round(input);
    fract *= 2.0_f64.powi(u64::BITS as i32);
    fract = f64::round(fract);

    fract as i64 as u64
}
