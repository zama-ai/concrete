//! A module to manipulate aes counter states.
//!
//! Coarse-grained pseudo-random table lookup
//! =========================================
//!
//! To generate random values, we use the AES block cipher in counter mode. If we denote f the aes
//! encryption function, we have:
//! ```ascii
//!     f: ⟦0;2¹²⁸ -1⟧ X ⟦0;2¹²⁸ -1⟧ ↦ ⟦0;2¹²⁸ -1⟧
//!     f(secret_key, input) ↦ output
//! ```
//!
//! If we fix the secret key to a value k, we have a pure function fₖ from ⟦0;2¹²⁸ -1⟧ to
//! ⟦0;2¹²⁸-1⟧, transforming the state of the counter into a pseudo random value. Essentially, this
//! fₖ function can be considered as a lookup function into a table of 2¹²⁸ pseudo-random values:
//! ```ascii  
//!     ╭──────────────┬──────────────┬─────┬──────────────╮
//!     │     fₖ(0)    │     fₖ(1)    │     │  fₖ(2¹²⁸ -1) │
//!     ╔═══════↧══════╦═══════↧══════╦═════╦═══════↧══════╗
//!     ║┏━━━━━━━━━━━━┓║┏━━━━━━━━━━━━┓║     ║┏━━━━━━━━━━━━┓║
//!     ║┃    u128    ┃║┃    u128    ┃║ ... ║┃    u128    ┃║
//!     ║┗━━━━━━━━━━━━┛║┗━━━━━━━━━━━━┛║     ║┗━━━━━━━━━━━━┛║
//!     ╚══════════════╩══════════════╩═════╩══════════════╝
//! ```
//!
//! We call this input to the fₖ function, an _aes index_ of the pseudo-random table. The
//! [`AesIndex`] structure defined in this module represents such an index in the code.
//!
//! Fine-grained pseudo-random table lookup
//! =======================================
//!
//! Unfortunately this is not enough to handle our situation, since we want to deliver the
//! pseudo-random bytes one by one. Fortunately, each `u128` value yielded by fₖ can be seen as a
//! table of 16 `u8`:
//! ```ascii
//!     ╭──────────────┬──────────────┬─────┬──────────────╮
//!     │     fₖ(0)    │     fₖ(1)    │     │  fₖ(2¹²⁸ -1) │
//!     ╔═══════↧══════╦═══════↧══════╦═════╦═══════↧══════╗
//!     ║┏━━━━━━━━━━━━┓║┏━━━━━━━━━━━━┓║     ║┏━━━━━━━━━━━━┓║
//!     ║┃    u128    ┃║┃    u128    ┃║     ║┃    u128    ┃║
//!     ║┣━━┯━━┯━━━┯━━┫║┣━━┯━━┯━━━┯━━┫║ ... ║┣━━┯━━┯━━━┯━━┫║
//!     ║┃u8│u8│...│u8┃║┃u8│u8│...│u8┃║     ║┃u8│u8│...│u8┃║
//!     ║┗━━┷━━┷━━━┷━━┛║┗━━┷━━┷━━━┷━━┛║     ║┗━━┷━━┷━━━┷━━┛║
//!     ╚══════════════╩══════════════╩═════╩══════════════╝
//! ```
//!
//! We introduce a second function to index into this table of small integers:
//! ```ascii
//!     g: ⟦0;2¹²⁸ -1⟧ X ⟦0;15⟧ ↦ ⟦0;2⁸ -1⟧
//!     g(big_int, index) ↦ byte
//! ```
//!
//! If we fix the `u128` value to a value e, we have a pure function gₑ from ⟦0;15⟧ to ⟦0;2⁸ -1⟧
//! transforming an index into a pseudo-random byte:
//! ```ascii
//!     ┏━━━━━━━━┯━━━━━━━━┯━━━┯━━━━━━━━┓
//!     ┃   u8   │   u8   │...│   u8   ┃
//!     ┗━━━━━━━━┷━━━━━━━━┷━━━┷━━━━━━━━┛
//!     │  gₑ(0) │  gₑ(1) │   │ gₑ(15) │
//!     ╰────────┴─────-──┴───┴────────╯
//! ```
//!
//! We call this input to the gₑ function, a _byte index_ of the pseudo-random table. The
//! [`ByteIndex`] structure defined in this module represents such an index in the code.
//!
//! By using both the g and the fₖ functions, we can define a new function l which allows to index
//! any byte of the pseudo-random table:
//! ```ascii
//!     l: ⟦0;2¹²⁸ -1⟧ X ⟦0;15⟧ ↦ ⟦0;2⁸ -1⟧
//!     l(aes_index, byte_index) ↦ g(fₖ(aes_index), byte_index)
//! ```
//!
//! In this sense, any member of ⟦0;2¹²⁸ -1⟧ X ⟦0;15⟧ uniquely defines a byte in this pseudo-random
//! table:
//! ```ascii
//!     ╭──────────────────────────────────────────────────╮
//!     │                    e = fₖ(a)                     │
//!     ╔══════════════╦═══════↧══════╦═════╦══════════════╗
//!     ║┏━━━━━━━━━━━━┓║┏━━━━━━━━━━━━┓║     ║┏━━━━━━━━━━━━┓║
//!     ║┃    u128    ┃║┃    u128    ┃║     ║┃    u128    ┃║
//!     ║┣━━┯━━┯━━━┯━━┫║┣━━┯━━┯━━━┯━━┫║ ... ║┣━━┯━━┯━━━┯━━┫║
//!     ║┃u8│u8│...│u8┃║┃u8│u8│...│u8┃║     ║┃u8│u8│...│u8┃║
//!     ║┗━━┷━━┷━━━┷━━┛║┗━━┷↥━┷━━━┷━━┛║     ║┗━━┷━━┷━━━┷━━┛║
//!     ║              ║│    gₑ(b)   │║     ║              ║
//!     ║              ║╰───-────────╯║     ║              ║
//!     ╚══════════════╩══════════════╩═════╩══════════════╝
//! ```
//!
//! We call this input to the l function, a _table index_ of the pseudo-random table. The
//! [`TableIndex`] structure defined in this module represents such an index in the code.
//!
//! Prngs current table index
//! =========================
//!
//! When created, a prng is given an initial _table index_, denoted (a₀, b₀), which identifies the
//! first byte of the table to be outputted by the prng. Then, each time the prng is queried for a
//! new value, the byte corresponding to the current _table index_ is returned, and the current
//! _table index_ is incremented:
//! ```ascii
//!     ╭─────────────────────────────────────────╮     ╭─────────────────────────────────────────╮
//!     │ e = fₖ(a₀)                              │     │             e = fₖ(a₁)                  │
//!     ╔═════↧═════╦═══════════╦═════╦═══════════╗     ╔═══════════╦═════↧═════╦═════╦═══════════╗
//!     ║┏━┯━┯━━━┯━┓║┏━┯━┯━━━┯━┓║ ... ║┏━┯━┯━━━┯━┓║     ║┏━┯━┯━━━┯━┓║┏━┯━┯━━━┯━┓║ ... ║┏━┯━┯━━━┯━┓║
//!     ║┃ │ │...│ ┃║┃ │ │...│ ┃║     ║┃ │ │...│ ┃║     ║┃ │ │...│ ┃║┃ │ │...│ ┃║     ║┃ │ │...│ ┃║
//!     ║┗━┷━┷━━━┷↥┛║┗━┷━┷━━━┷━┛║     ║┗━┷━┷━━━┷━┛║  →  ║┗━┷━┷━━━┷━┛║┗↥┷━┷━━━┷━┛║     ║┗━┷━┷━━━┷━┛║
//!     ║│  gₑ(b₀) │║           ║     ║           ║     ║           ║│  gₑ(b₁) │║     ║           ║
//!     ║╰─────────╯║           ║     ║           ║     ║           ║╰─────────╯║     ║           ║
//!     ╚═══════════╩═══════════╩═════╩═══════════╝     ╚═══════════╩═══════════╩═════╩═══════════╝
//! ```
//!
//! Prng bound
//! ==========
//!
//! When created, a prng is also given a _bound_ (aₘ, bₘ) , that is a table index which it is not
//! allowed to exceed:
//! ```ascii
//!     ╭─────────────────────────────────────────╮
//!     │ e = fₖ(a₀)                              │
//!     ╔═════↧═════╦═══════════╦═════╦═══════════╗
//!     ║┏━┯━┯━━━┯━┓║┏━┯━┯━━━┯━┓║ ... ║┏━┯━┯━━━┯━┓║
//!     ║┃ │ │...│ ┃║┃ │╳│...│╳┃║     ║┃╳│╳│...│╳┃║
//!     ║┗━┷━┷━━━┷↥┛║┗━┷━┷━━━┷━┛║     ║┗━┷━┷━━━┷━┛║ The current byte can be returned.
//!     ║│  gₑ(b₀) │║           ║     ║           ║
//!     ║╰─────────╯║           ║     ║           ║
//!     ╚═══════════╩═══════════╩═════╩═══════════╝
//!     
//!     ╭─────────────────────────────────────────╮
//!     │             e = fₖ(aₘ)                  │
//!     ╔═══════════╦═════↧═════╦═════╦═══════════╗
//!     ║┏━┯━┯━━━┯━┓║┏━┯━┯━━━┯━┓║ ... ║┏━┯━┯━━━┯━┓║
//!     ║┃ │ │...│ ┃║┃ │╳│...│╳┃║     ║┃╳│╳│...│╳┃║ The table index reached the bound,
//!     ║┗━┷━┷━━━┷━┛║┗━┷↥┷━━━┷━┛║     ║┗━┷━┷━━━┷━┛║ the current byte can not be
//!     ║           ║│  gₑ(bₘ) │║     ║           ║ returned.
//!     ║           ║╰─────────╯║     ║           ║
//!     ╚═══════════╩═══════════╩═════╩═══════════╝
//! ```
//!
//! Buffering
//! =========
//!
//! Calling the aes function every time we need to yield a single byte would be a huge waste of
//! resources. In practice, we call aes 8 times in a row, for 8 successive values of aes index, and
//! store the results in a buffer. For platforms which have a dedicated aes chip, this allows to
//! fill the unit pipeline and reduces the amortized cost of the aes function.
//!
//! Together with the current table index of the prng, we also store a pointer p (initialized at
//! p₀=b₀) to the current byte in the buffer. If we denote v the lookup function we have :
//! ```ascii
//!     ╭───────────────────────────────────────────────╮
//!     │                  e = fₖ(a₀)                   │     Buffer(length=128)
//!     ╔═════╦═══════════╦═════↧═════╦═══════════╦═════╗  ┏━┯━┯━┯━┯━┯━┯━┯━┯━━━┯━┓
//!     ║ ... ║┏━┯━┯━━━┯━┓║┏━┯━┯━━━┯━┓║┏━┯━┯━━━┯━┓║ ... ║  ┃▓│▓│▓│▓│▓│▓│▓│▓│...│▓┃
//!     ║     ║┃ │ │...│ ┃║┃▓│▓│...│▓┃║┃▓│▓│...│▓┃║     ║  ┗━┷↥┷━┷━┷━┷━┷━┷━┷━━━┷━┛
//!     ║     ║┗━┷━┷━━━┷━┛║┗━┷↥┷━━━┷━┛║┗━┷━┷━━━┷━┛║     ║  │ v(p₀)               │
//!     ║     ║           ║│  gₑ(b₀) │║           ║     ║  ╰─────────────────────╯
//!     ║     ║           ║╰─────────╯║           ║     ║
//!     ╚═════╩═══════════╩═══════════╩═══════════╩═════╝
//! ```
//!
//! We call this input to the v function, a _buffer pointer_. The [`BufferPointer`] structure
//! defined in this module represents such a pointer in the code.
//!
//! When the table index is incremented, the buffer pointer is incremented alongside:
//! ```ascii
//!     ╭───────────────────────────────────────────────╮
//!     │                  e = fₖ(a)                    │     Buffer(length=128)
//!     ╔═════╦═══════════╦═════↧═════╦═══════════╦═════╗  ┏━┯━┯━┯━┯━┯━┯━┯━┯━━━┯━┓
//!     ║ ... ║┏━┯━┯━━━┯━┓║┏━┯━┯━━━┯━┓║┏━┯━┯━━━┯━┓║ ... ║  ┃▓│▓│▓│▓│▓│▓│▓│▓│...│▓┃
//!     ║     ║┃ │ │...│ ┃║┃▓│▓│...│▓┃║┃▓│▓│...│▓┃║     ║  ┗━┷━┷↥┷━┷━┷━┷━┷━┷━━━┷━┛
//!     ║     ║┗━┷━┷━━━┷━┛║┗━┷━┷↥━━┷━┛║┗━┷━┷━━━┷━┛║     ║  │   v(p)              │
//!     ║     ║           ║│  gₑ(b)  │║           ║     ║  ╰─────────────────────╯
//!     ║     ║           ║╰─────────╯║           ║     ║
//!     ╚═════╩═══════════╩═══════════╩═══════════╩═════╝
//! ```
//!
//! When the buffer pointer is incremented it is checked against the size of the buffer, and if
//! necessary, a new batch of aes index values.
use std::cmp::Ordering;

/// A structure representing an [aes index](#coarse-grained-pseudo-random-table-lookup).
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub struct AesIndex(pub u128);

/// A structuure representing a [byte index](#fine-grained-pseudo-random-table-lookup).
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub struct ByteIndex(pub usize);

/// A structure representing the number of bytes between two table indices.
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub struct ByteCount(pub u128);

/// A structure representing a [table index](#fine-grained-pseudo-random-table-lookup)
#[derive(Clone, Copy, Debug)]
pub struct TableIndex {
    pub(crate) aes_index: AesIndex,
    pub(crate) byte_index: ByteIndex,
}

impl TableIndex {
    /// The first table index.
    pub const FIRST: TableIndex = TableIndex {
        aes_index: AesIndex(0),
        byte_index: ByteIndex(0),
    };

    /// The second table index.
    pub const SECOND: TableIndex = TableIndex {
        aes_index: AesIndex(0),
        byte_index: ByteIndex(1),
    };

    /// The last table index.
    pub const LAST: TableIndex = TableIndex {
        aes_index: AesIndex(u128::MAX),
        byte_index: ByteIndex(15),
    };

    /// Creates a table index from an aes index and a byte index.
    pub fn new(aes_index: AesIndex, byte_index: ByteIndex) -> Self {
        assert!(byte_index.0 <= 15);
        TableIndex {
            aes_index,
            byte_index,
        }
    }

    /// Shifts the table index forward of `shift` bytes.
    pub fn increase(&mut self, shift: usize) {
        let total = self.byte_index.0 + shift;
        self.byte_index.0 = total % 16;
        self.aes_index.0 = self.aes_index.0.wrapping_add(total as u128 / 16);
    }

    /// Shifts the table index backward of `shift` bytes.
    pub fn decrease(&mut self, shift: usize) {
        let remainder = shift % 16;
        if remainder <= self.byte_index.0 {
            self.aes_index.0 = self.aes_index.0.wrapping_sub((shift / 16) as u128);
            self.byte_index.0 -= remainder;
        } else {
            self.aes_index.0 = self.aes_index.0.wrapping_sub((shift / 16) as u128 + 1);
            self.byte_index.0 += 16 - remainder;
        }
    }

    /// Shifts the table index forward of one byte.
    pub fn increment(&mut self) {
        self.increase(1)
    }

    /// Shifts the table index backward of one byte.
    pub fn decrement(&mut self) {
        self.decrease(1)
    }

    /// Returns the table index shifted forward by `shift` bytes.
    pub fn increased(mut self, shift: usize) -> Self {
        self.increase(shift);
        self
    }

    /// Returns the table index shifted backward by `shift` bytes.
    pub fn decreased(mut self, shift: usize) -> Self {
        self.decrease(shift);
        self
    }

    /// Returns the table index to the next byte.
    pub fn incremented(mut self) -> Self {
        self.increment();
        self
    }

    /// Returns the table index to the previous byte.
    pub fn decremented(mut self) -> Self {
        self.decrement();
        self
    }

    /// Returns the distance between two table indices in bytes.
    ///
    /// Note:
    /// -----
    ///
    /// This method assumes that the `larger` input is, well, larger than the `smaller` input. If
    /// this is not the case, the method returns `None`. Also, note that `ByteCount` uses the
    /// `u128` datatype to store the byte count. Unfortunately, the number of bytes between two
    /// table indices is in ⟦0;2¹³² -1⟧. When the distance is greater than 2¹²⁸ - 1, we saturate
    /// the count at 2¹²⁸ - 1.
    pub fn distance(larger: &Self, smaller: &Self) -> Option<ByteCount> {
        match std::cmp::Ord::cmp(larger, smaller) {
            Ordering::Less => None,
            Ordering::Equal => Some(ByteCount(0)),
            Ordering::Greater => {
                let mut result = larger.aes_index.0 - smaller.aes_index.0;
                result = result.saturating_mul(16);
                result = result.saturating_add(larger.byte_index.0 as u128);
                result = result.saturating_sub(smaller.byte_index.0 as u128);
                Some(ByteCount(result))
            }
        }
    }
}

impl Eq for TableIndex {}

impl PartialEq<Self> for TableIndex {
    fn eq(&self, other: &Self) -> bool {
        matches!(self.partial_cmp(other), Some(Ordering::Equal))
    }
}

impl PartialOrd<Self> for TableIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.aes_index.partial_cmp(&other.aes_index) {
            Some(Ordering::Equal) => self.byte_index.partial_cmp(&other.byte_index),
            other => other,
        }
    }
}

impl Ord for TableIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// A pointer to the next byte to be outputted by the generator.
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub struct BufferPointer(pub usize);

/// A structure representing the current state of generator using batched aes-ctr approach.
#[derive(Debug, Clone, Copy)]
pub struct State {
    table_index: TableIndex,
    buffer_pointer: BufferPointer,
}

/// A structure representing the action to be taken by the generator after shifting its state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShiftAction {
    /// Yield the byte pointed to by the 0-th field.
    YieldByte(BufferPointer),
    /// Refresh the buffer starting from the 0-th field, and yield the byte pointed to by the 0-th
    /// field.
    RefreshBatchAndYieldByte(AesIndex, BufferPointer),
}

impl State {
    /// Creates a new state from the initial table index.
    pub fn new(table_index: TableIndex) -> Self {
        State {
            table_index: table_index.decremented(),
            buffer_pointer: BufferPointer(127),
        }
    }

    /// Shifts the state forward of `shift` bytes.
    pub fn increase(&mut self, shift: usize) -> ShiftAction {
        self.table_index.increase(shift);
        let total_batch_index = self.buffer_pointer.0 + shift;
        if total_batch_index > 127 {
            self.buffer_pointer.0 = self.table_index.byte_index.0;
            ShiftAction::RefreshBatchAndYieldByte(self.table_index.aes_index, self.buffer_pointer)
        } else {
            self.buffer_pointer.0 = total_batch_index;
            ShiftAction::YieldByte(self.buffer_pointer)
        }
    }

    /// Shifts the state forward of one byte.
    pub fn increment(&mut self) -> ShiftAction {
        self.increase(1)
    }

    /// Returns the current table index.
    pub fn table_index(&self) -> TableIndex {
        self.table_index
    }
}

impl Default for State {
    fn default() -> Self {
        State::new(TableIndex::FIRST)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{thread_rng, Rng};

    const REPEATS: usize = 1_000_000;

    fn any_table_index() -> impl Iterator<Item = TableIndex> {
        std::iter::repeat_with(|| {
            TableIndex::new(
                AesIndex(thread_rng().gen()),
                ByteIndex(thread_rng().gen::<usize>() % 16),
            )
        })
    }

    fn any_usize() -> impl Iterator<Item = usize> {
        std::iter::repeat_with(|| thread_rng().gen())
    }

    #[test]
    #[should_panic]
    /// Verifies that the constructor of `TableIndex` panics when the byte index is too large.
    fn test_table_index_new_panic() {
        TableIndex::new(AesIndex(12), ByteIndex(144));
    }

    #[test]
    /// Verifies that the `TableIndex` wraps nicely with predecessor
    fn test_table_index_predecessor_edge() {
        assert_eq!(TableIndex::FIRST.decremented(), TableIndex::LAST);
    }

    #[test]
    /// Verifies that the `TableIndex` wraps nicely with successor
    fn test_table_index_successor_edge() {
        assert_eq!(TableIndex::LAST.incremented(), TableIndex::FIRST);
    }

    #[test]
    /// Check that the table index distance saturates nicely.
    fn prop_table_index_distance_saturates() {
        assert_eq!(
            TableIndex::distance(&TableIndex::LAST, &TableIndex::FIRST)
                .unwrap()
                .0,
            u128::MAX
        )
    }

    #[test]
    /// Check the property:
    ///     For all table indices t,
    ///         distance(t, t) = Some(0).
    fn prop_table_index_distance_zero() {
        for _ in 0..REPEATS {
            let t = any_table_index().next().unwrap();
            assert_eq!(TableIndex::distance(&t, &t), Some(ByteCount(0)));
        }
    }

    #[test]
    /// Check the property:
    ///     For all table indices t1, t2 such that t1 < t2,
    ///         distance(t1, t2) = None.
    fn prop_table_index_distance_wrong_order_none() {
        for _ in 0..REPEATS {
            let (t1, t2) = any_table_index()
                .zip(any_table_index())
                .find(|(t1, t2)| t1 < t2)
                .unwrap();
            assert_eq!(TableIndex::distance(&t1, &t2), None);
        }
    }

    #[test]
    /// Check the property:
    ///     For all table indices t1, t2 such that t1 > t2,
    ///         distance(t1, t2) = Some(v) where v is strictly positive.
    fn prop_table_index_distance_some_positive() {
        for _ in 0..REPEATS {
            let (t1, t2) = any_table_index()
                .zip(any_table_index())
                .find(|(t1, t2)| t1 > t2)
                .unwrap();
            assert!(matches!(TableIndex::distance(&t1, &t2), Some(ByteCount(v)) if v > 0));
        }
    }

    #[test]
    /// Check the property:
    ///     For all table indices t, positive i such that i < distance (MAX, t) with MAX the largest
    ///     table index,
    ///         distance(t.increased(i), t) = Some(i).
    fn prop_table_index_distance_increase() {
        for _ in 0..REPEATS {
            let (t, inc) = any_table_index()
                .zip(any_usize())
                .find(|(t, inc)| {
                    (*inc as u128) < TableIndex::distance(&TableIndex::LAST, t).unwrap().0
                })
                .unwrap();
            assert_eq!(
                TableIndex::distance(&t.increased(inc), &t).unwrap().0 as usize,
                inc
            );
        }
    }

    #[test]
    /// Check the property:
    ///     For all table indices t, t =? t = true.
    fn prop_table_index_equality() {
        for _ in 0..REPEATS {
            let t = any_table_index().next().unwrap();
            assert_eq!(
                std::cmp::PartialOrd::partial_cmp(&t, &t),
                Some(std::cmp::Ordering::Equal)
            );
        }
    }

    #[test]
    /// Check the property:
    ///     For all table indices t, positive i such that i < distance (MAX, t) with MAX the largest
    ///     table index,
    ///         t.increased(i) >? t = true.
    fn prop_table_index_greater() {
        for _ in 0..REPEATS {
            let (t, inc) = any_table_index()
                .zip(any_usize())
                .find(|(t, inc)| {
                    (*inc as u128) < TableIndex::distance(&TableIndex::LAST, t).unwrap().0
                })
                .unwrap();
            assert_eq!(
                std::cmp::PartialOrd::partial_cmp(&t.increased(inc), &t),
                Some(std::cmp::Ordering::Greater),
            );
        }
    }

    #[test]
    /// Check the property:
    ///     For all table indices t, positive i such that i < distance (t, 0) with MAX the largest
    ///     table index,
    ///         t.decreased(i) <? t = true.
    fn prop_table_index_less() {
        for _ in 0..REPEATS {
            let (t, inc) = any_table_index()
                .zip(any_usize())
                .find(|(t, inc)| {
                    (*inc as u128) < TableIndex::distance(t, &TableIndex::FIRST).unwrap().0
                })
                .unwrap();
            assert_eq!(
                std::cmp::PartialOrd::partial_cmp(&t.decreased(inc), &t),
                Some(std::cmp::Ordering::Less)
            );
        }
    }

    #[test]
    /// Check the property:
    ///     For all table indices t,
    ///         successor(predecessor(t)) = t.
    fn prop_table_index_decrement_increment() {
        for _ in 0..REPEATS {
            let t = any_table_index().next().unwrap();
            assert_eq!(t.decremented().incremented(), t);
        }
    }

    #[test]
    /// Check the property:
    ///     For all table indices t,
    ///         predecessor(successor(t)) = t.
    fn prop_table_index_increment_decrement() {
        for _ in 0..REPEATS {
            let t = any_table_index().next().unwrap();
            assert_eq!(t.incremented().decremented(), t);
        }
    }

    #[test]
    /// Check the property:
    ///     For all table indices t, positive integer i,
    ///         increase(decrease(t, i), i) = t.
    fn prop_table_index_increase_decrease() {
        for _ in 0..REPEATS {
            let (t, i) = any_table_index().zip(any_usize()).next().unwrap();
            assert_eq!(t.increased(i).decreased(i), t);
        }
    }

    #[test]
    /// Check the property:
    ///     For all table indices t, positive integer i,
    ///         decrease(increase(t, i), i) = t.
    fn prop_table_index_decrease_increase() {
        for _ in 0..REPEATS {
            let (t, i) = any_table_index().zip(any_usize()).next().unwrap();
            assert_eq!(t.decreased(i).increased(i), t);
        }
    }

    #[test]
    /// Check the property:
    ///     For all table indices t,
    ///         State::new(t).increment() = RefreshBatchAndYield(t.aes_index, t.byte_index)
    fn prop_state_new_increment() {
        for _ in 0..REPEATS {
            let (t, mut s) = any_table_index()
                .map(|t| (t, State::new(t)))
                .next()
                .unwrap();
            assert!(matches!(
                s.increment(),
                ShiftAction::RefreshBatchAndYieldByte(t_, BufferPointer(p_)) if t_ == t.aes_index && p_ == t.byte_index.0
            ))
        }
    }

    #[test]
    /// Check the property:
    ///     For all states s, table indices t, positive integer i
    ///         if s = State::new(t), then t.increased(i) = s.increased(i-1).table_index().
    fn prop_state_increase_table_index() {
        for _ in 0..REPEATS {
            let (t, mut s, i) = any_table_index()
                .zip(any_usize())
                .map(|(t, i)| (t, State::new(t), i))
                .next()
                .unwrap();
            s.increase(i);
            assert_eq!(s.table_index(), t.increased(i - 1))
        }
    }

    #[test]
    /// Check the property:
    ///     For all table indices t, positive integer i such as t.byte_index + i < 127,
    ///         if s = State::new(t), and s.increment() was executed, then
    ///         s.increase(i) = YieldByte(t.byte_index + i).  
    fn prop_state_increase_small() {
        for _ in 0..REPEATS {
            let (t, mut s, i) = any_table_index()
                .zip(any_usize())
                .map(|(t, i)| (t, State::new(t), i % 128))
                .find(|(t, _, i)| t.byte_index.0 + i < 127)
                .unwrap();
            s.increment();
            assert!(matches!(
                s.increase(i),
                ShiftAction::YieldByte(BufferPointer(p_)) if p_ == t.byte_index.0 + i
            ));
        }
    }

    #[test]
    /// Check the property:
    ///     For all table indices t, positive integer i such as t.byte_index + i >= 127,
    ///         if s = State::new(t), and s.increment() was executed, then
    ///         s.increase(i) = RefreshBatchAndYield(
    ///             t.increased(i).aes_index,
    ///             t.increased(i).byte_index).
    fn prop_state_increase_large() {
        for _ in 0..REPEATS {
            let (t, mut s, i) = any_table_index()
                .zip(any_usize())
                .map(|(t, i)| (t, State::new(t), i))
                .find(|(t, _, i)| t.byte_index.0 + i >= 127)
                .unwrap();
            s.increment();
            assert!(matches!(
                s.increase(i),
                ShiftAction::RefreshBatchAndYieldByte(t_, BufferPointer(p_))
                    if t_ == t.increased(i).aes_index && p_ == t.increased(i).byte_index.0
            ));
        }
    }
}
