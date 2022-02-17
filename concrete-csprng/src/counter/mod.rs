use crate::{aesni, software};
#[cfg(feature = "multithread")]
use rayon::{iter::IndexedParallelIterator, prelude::*};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[cfg(test)]
mod test;

#[cfg(all(
    test,
    target_arch = "x86_64",
    target_feature = "aes",
    target_feature = "sse2",
    target_feature = "rdseed"
))]
mod test_aes;

mod state;
pub use state::*;

/// Represents a key used in the AES ciphertext.
#[derive(Clone, Copy)]
pub struct AesKey(pub u128);

/// A trait for batched generators, i.e. generators that creates 128 bytes of random values at a
/// time.
pub trait AesBatchedGenerator: Clone {
    /// Instantiate a new generator from a secret key.
    fn new(key: Option<AesKey>) -> Self;
    /// Generates the batch corresponding to the given counter.
    fn generate_batch(&mut self, index: AesIndex) -> [u8; 128];
}

/// A generator that uses the software implementation.
pub type SoftAesCtrGenerator = AesCtrGenerator<software::Generator>;

/// A generator that uses the hardware implementation.
pub type HardAesCtrGenerator = AesCtrGenerator<aesni::Generator>;

/// An error occuring during a generator fork.
#[derive(Debug)]
pub enum ForkError {
    ForkTooLarge,
    ZeroChildrenCount,
    ZeroBytesPerChild,
}

impl Display for ForkError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ForkError::ForkTooLarge => {
                write!(
                    f,
                    "The children generators would output bytes after the parent bound. "
                )
            }
            ForkError::ZeroChildrenCount => {
                write!(
                    f,
                    "The number of children in the fork must be greater than zero."
                )
            }
            ForkError::ZeroBytesPerChild => {
                write!(
                    f,
                    "The number of bytes per child must be greater than zero."
                )
            }
        }
    }
}
impl Error for ForkError {}

/// A csprng which operates in batch mode.
#[derive(Clone)]
pub struct AesCtrGenerator<G: AesBatchedGenerator> {
    generator: G,
    state: State,
    bound: TableIndex,
    last: TableIndex,
    buffer: [u8; 128],
}

impl<G: AesBatchedGenerator> AesCtrGenerator<G> {
    /// Generates a new csprng.
    ///
    /// If not given, the key is automatically selected, and the state is set to zero.
    ///
    /// Note :
    /// ------
    ///
    /// The state given in input, points to the first byte that will be outputted by the generator.
    /// The bound points to the first byte that can not be outputted by the generator.
    pub fn new(
        key: Option<AesKey>,
        start_index: Option<TableIndex>,
        bound_index: Option<TableIndex>,
    ) -> AesCtrGenerator<G> {
        AesCtrGenerator::from_generator(
            G::new(key),
            start_index.unwrap_or(TableIndex::SECOND),
            bound_index.unwrap_or(TableIndex::LAST),
        )
    }

    /// Generates a csprng from an existing generator.
    pub fn from_generator(
        generator: G,
        start_index: TableIndex,
        bound_index: TableIndex,
    ) -> AesCtrGenerator<G> {
        assert!(start_index < bound_index);
        let last = bound_index.decremented();
        let buffer = [0u8; 128];
        AesCtrGenerator {
            generator,
            state: State::new(start_index),
            bound: bound_index,
            last,
            buffer,
        }
    }

    /// Returns the table index related to the last yielded byte.
    pub fn last_table_index(&self) -> TableIndex {
        self.state.table_index()
    }

    /// Returns the bound of the generator if any.
    ///
    /// The bound is the table index of the first byte that can not be outputted by the generator.
    pub fn get_bound(&self) -> TableIndex {
        self.bound
    }

    /// Returns whether the generator is bounded or not.
    pub fn is_bounded(&self) -> bool {
        self.bound != TableIndex::LAST
    }

    /// Computes the number of bytes that can still be outputted by the generator.
    ///
    /// Note :
    /// ------
    ///
    /// Note that `ByteCount` uses the `u128` datatype to store the byte count. Unfortunately, the
    /// number of remaining bytes is in ⟦0;2¹³² -1⟧. When the number is greater than 2¹²⁸ - 1,
    /// we saturate the count at 2¹²⁸ - 1.
    pub fn remaining_bytes(&self) -> ByteCount {
        TableIndex::distance(&self.last, &self.state.table_index()).unwrap()
    }

    /// Yields the next random byte.
    pub fn generate_next(&mut self) -> u8 {
        assert!(self.state.table_index() < self.last,);
        match self.state.increment() {
            ShiftAction::YieldByte(BufferPointer(ptr)) => self.buffer[ptr],
            ShiftAction::RefreshBatchAndYieldByte(aes_index, BufferPointer(ptr)) => {
                self.buffer = self.generator.generate_batch(aes_index);
                self.buffer[ptr]
            }
        }
    }

    /// Tries to fork the current generator into `n_child` generators each able to yield
    /// `child_bytes` random bytes.
    pub fn try_fork(
        &mut self,
        n_child: ChildrenCount,
        child_bytes: BytesPerChild,
    ) -> Result<impl Iterator<Item = AesCtrGenerator<G>>, ForkError> {
        if n_child.0 == 0 {
            return Err(ForkError::ZeroChildrenCount);
        }
        if child_bytes.0 == 0 {
            return Err(ForkError::ZeroBytesPerChild);
        }
        if !self.is_fork_in_bound(n_child, child_bytes) {
            return Err(ForkError::ForkTooLarge);
        }

        let generator = self.generator.clone();
        // The state currently stored in the parent generator points to the table index of the last
        // generated byte. The first index to be generated is the next one :
        let first_index = self.state.table_index().incremented();
        let output = (0..n_child.0).map(move |i| {
            // The first index to be outputted by the child is the `first_index` shifted by the
            // proper amount of `child_bytes`.
            let child_first_index = first_index.increased(child_bytes.0 * i);
            // The bound of the child is the first index of its next sibling.
            let child_bound_index = first_index.increased(child_bytes.0 * (i + 1));
            AesCtrGenerator::from_generator(generator.clone(), child_first_index, child_bound_index)
        });
        // The parent next index is the bound of the last child.
        let next_index = first_index.increased(child_bytes.0 * n_child.0);
        self.state = State::new(next_index);

        Ok(output)
    }

    /// Tries to fork the current generator into `n_child` generators each able to yield
    /// `child_bytes` random bytes as a parallel iterator.
    ///
    /// # Notes
    ///
    /// This method necessitate the "multithread" feature.
    #[cfg(feature = "multithread")]
    pub fn par_try_fork(
        &mut self,
        n_child: ChildrenCount,
        child_bytes: BytesPerChild,
    ) -> Result<impl IndexedParallelIterator<Item = AesCtrGenerator<G>>, ForkError>
    where
        G: Send + Sync,
    {
        if n_child.0 == 0 {
            return Err(ForkError::ZeroChildrenCount);
        }
        if child_bytes.0 == 0 {
            return Err(ForkError::ZeroBytesPerChild);
        }
        if !self.is_fork_in_bound(n_child, child_bytes) {
            return Err(ForkError::ForkTooLarge);
        }

        let generator = self.generator.clone();
        // The state currently stored in the parent generator points to the table index of the last
        // generated byte. The first index to be generated is the next one :
        let first_index = self.state.table_index().incremented();
        let output = (0..n_child.0).into_par_iter().map(move |i| {
            // The first index to be outputted by the child is the `first_index` shifted by the
            // proper amount of `child_bytes`.
            let child_first_index = first_index.increased(child_bytes.0 * i);
            // The bound of the child is the first index of its next sibling.
            let child_bound_index = first_index.increased(child_bytes.0 * (i + 1));
            AesCtrGenerator::from_generator(generator.clone(), child_first_index, child_bound_index)
        });
        // The parent next index is the bound of the last child.
        let next_index = first_index.increased(child_bytes.0 * n_child.0);
        self.state = State::new(next_index);

        Ok(output)
    }

    fn is_fork_in_bound(&self, n_child: ChildrenCount, child_bytes: BytesPerChild) -> bool {
        let mut end = self.state.table_index();
        end.increase(n_child.0 * child_bytes.0);
        end < self.bound
    }
}

impl<G: AesBatchedGenerator> Iterator for AesCtrGenerator<G> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.state.table_index() < self.last {
            None
        } else {
            Some(self.generate_next())
        }
    }
}

/// The number of children created when a generator is forked.
#[derive(Debug, Copy, Clone)]
pub struct ChildrenCount(pub usize);

/// The number of bytes each children can generate, when a generator is forked.
#[derive(Debug, Copy, Clone)]
pub struct BytesPerChild(pub usize);
