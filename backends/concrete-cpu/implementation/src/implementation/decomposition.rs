use core::iter::Map;
use core::slice::IterMut;
use dyn_stack::{DynArray, DynStack};

/// An iterator that yields the terms of the signed decomposition of an integer.
///
/// # Warning
///
/// This iterator yields the decomposition in reverse order. That means that the highest level
/// will be yielded first.
pub struct SignedDecompositionIter {
    // The value being decomposed
    input: u64,
    level: usize,
    base_log: usize,
    // The internal state of the decomposition
    state: u64,
    // The current level
    current_level: usize,
    // A mask which allows to compute the mod B of a value. For B=2^4, this guy is of the form:
    // ...0001111
    mod_b_mask: u64,
    // A flag which store whether the iterator is a fresh one (for the recompose method)
    fresh: bool,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct DecompositionTerm {
    level: usize,
    base_log: usize,
    value: u64,
}

impl DecompositionTerm {
    // Creates a new decomposition term.
    pub(crate) fn new(level: usize, base_log: usize, value: u64) -> Self {
        Self {
            level,
            base_log,
            value,
        }
    }

    /// Turns this term into a summand.
    ///
    /// If our member represents one $\tilde{\theta}\_i$ of the decomposition, this method returns
    /// $\tilde{\theta}\_i\frac{q}{B^i}$.
    pub fn as_recomposition_summand(&self) -> u64 {
        let shift: usize = u64::BITS as usize - self.base_log * self.level;
        self.value << shift
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_recomposition_summand(&self) -> u64 {
        let shift: usize = u64::BITS as usize - self.base_log * self.level;
        self.value << shift
    }

    /// Returns the value of the term.
    ///
    /// If our member represents one $\tilde{\theta}\_i$, this returns its actual value.
    pub fn value(&self) -> u64 {
        self.value
    }

    /// Returns the level of the term.
    ///
    /// If our member represents one $\tilde{\theta}\_i$, this returns the value of $i$.
    pub fn level(&self) -> usize {
        self.level
    }
}

impl SignedDecompositionIter {
    pub(crate) fn new(input: u64, level: usize, base_log: usize) -> Self {
        Self {
            input,
            level,
            base_log,
            state: input >> (u64::BITS as usize - base_log * level),
            current_level: level,
            mod_b_mask: (1 << base_log) - 1,
            fresh: true,
        }
    }
}

impl Iterator for SignedDecompositionIter {
    type Item = DecompositionTerm;

    fn next(&mut self) -> Option<Self::Item> {
        // The iterator is not fresh anymore
        self.fresh = false;
        // We check if the decomposition is over
        if self.current_level == 0 {
            return None;
        }
        // We decompose the current level
        let output = decompose_one_level(self.base_log, &mut self.state, self.mod_b_mask);
        self.current_level -= 1;
        // We return the output for this level
        Some(DecompositionTerm::new(
            self.current_level + 1,
            self.base_log,
            output,
        ))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.current_level, Some(self.current_level))
    }
}

pub struct TensorSignedDecompositionLendingIter<'buffers> {
    // The base log of the decomposition
    base_log: usize,
    // The current level
    current_level: usize,
    // A mask which allows to compute the mod B of a value. For B=2^4, this guy is of the form:
    // ...0001111
    mod_b_mask: u64,
    // The internal states of each decomposition
    states: DynArray<'buffers, u64>,
    // A flag which stores whether the iterator is a fresh one (for the recompose method).
    fresh: bool,
}

impl<'buffers> TensorSignedDecompositionLendingIter<'buffers> {
    #[inline]
    pub(crate) fn new(
        input: impl Iterator<Item = u64>,
        base_log: usize,
        level: usize,
        stack: DynStack<'buffers>,
    ) -> (Self, DynStack<'buffers>) {
        let shift = u64::BITS as usize - base_log * level;
        let (states, stack) =
            stack.collect_aligned(aligned_vec::CACHELINE_ALIGN, input.map(|i| i >> shift));
        (
            TensorSignedDecompositionLendingIter {
                base_log,
                current_level: level,
                mod_b_mask: (1_u64 << base_log) - 1_u64,
                states,
                fresh: true,
            },
            stack,
        )
    }

    // inlining this improves perf of external product by about 25%, even in LTO builds
    #[inline]
    pub fn next_term<'short>(
        &'short mut self,
    ) -> Option<(
        usize,
        usize,
        Map<IterMut<'short, u64>, impl FnMut(&'short mut u64) -> u64>,
    )> {
        // The iterator is not fresh anymore.
        self.fresh = false;
        // We check if the decomposition is over
        if self.current_level == 0 {
            return None;
        }
        let current_level = self.current_level;
        let base_log = self.base_log;
        let mod_b_mask = self.mod_b_mask;
        self.current_level -= 1;

        Some((
            current_level,
            self.base_log,
            self.states
                .iter_mut()
                .map(move |state| decompose_one_level(base_log, state, mod_b_mask)),
        ))
    }
}

#[inline]
fn decompose_one_level(base_log: usize, state: &mut u64, mod_b_mask: u64) -> u64 {
    let res = *state & mod_b_mask;
    *state >>= base_log;
    let mut carry = (res.wrapping_sub(1_u64) | *state) & res;
    carry >>= base_log - 1;
    *state += carry;
    res.wrapping_sub(carry << base_log)
}
