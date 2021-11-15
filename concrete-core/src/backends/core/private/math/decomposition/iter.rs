use crate::backends::core::private::math::decomposition::{
    DecompositionLevel, DecompositionTerm, DecompositionTermTensor,
};
use crate::backends::core::private::math::tensor::Tensor;
use crate::backends::core::private::utils::{zip, zip_args};
use concrete_commons::numeric::{SignedInteger, UnsignedInteger};
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};

/// An iterator-like object that yields the terms of the signed decomposition of a tensor of values.
///
/// # Note
///
/// On each call to [`TensorSignedDecompositionIter::next_term`], this structure yields a new
/// [`DecompositionTermTensor`], backed by a `Vec` owned by the structure. This vec is mutated at
/// each call of the `next_term` method, and as such the term must be dropped before `next_term` is
/// called again.
///
/// Such a pattern can not be implemented with iterators yet (without GATs), which is why this
/// iterator must be explicitly called.
///
/// # Warning
///
/// This iterator yields the decomposition in reverse order. That means that the highest level
/// will be yielded first.
pub struct TensorSignedDecompositionIter<Scalar>
where
    Scalar: UnsignedInteger,
{
    // The base log of the decomposition
    base_log: usize,
    // The number of levels of the decomposition
    level_count: usize,
    // The current level
    current_level: usize,
    // A mask which allows to compute the mod B of a value. For B=2^4, this guy is of the form:
    // ...0001111
    mod_b_mask: Scalar,
    // A mask which allows to test whether the value is larger than B/2. For B=2^4, this guy is
    // of the form:
    // ...0001000
    carry_mask: Scalar,
    // The values being decomposed
    inputs: Vec<Scalar>,
    // The carries from the previous level
    previous_carries: Vec<Scalar>,
    // In order to avoid allocating a new Vec every time we yield a decomposition term, we store
    // a Vec inside the structure and yield slices pointing to it.
    outputs: Vec<Scalar>,
    // A flag which stores whether the iterator is a fresh one (for the recompose method).
    fresh: bool,
}

impl<Scalar> TensorSignedDecompositionIter<Scalar>
where
    Scalar: UnsignedInteger,
{
    // Creates a new tensor decomposition iterator.
    pub(crate) fn new(
        input: Tensor<Vec<Scalar>>,
        base_log: DecompositionBaseLog,
        level: DecompositionLevelCount,
    ) -> TensorSignedDecompositionIter<Scalar> {
        let len = input.len();
        TensorSignedDecompositionIter {
            base_log: base_log.0,
            level_count: level.0,
            current_level: level.0,
            mod_b_mask: (Scalar::ONE << base_log.0) - Scalar::ONE,
            carry_mask: Scalar::ONE << (base_log.0 - 1),
            inputs: input.into_container(),
            outputs: vec![Scalar::ZERO; len],
            previous_carries: vec![Scalar::ZERO; len],
            fresh: true,
        }
    }

    pub(crate) fn is_fresh(&self) -> bool {
        self.fresh
    }

    /// Returns the logarithm in base two of the base of this decomposition.
    ///
    /// If the decomposition uses a base $B=2^b$, this returns $b$.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
    /// use concrete_core::backends::core::private::math::decomposition::SignedDecomposer;
    /// use concrete_core::backends::core::private::math::tensor::Tensor;
    /// let decomposer =
    ///     SignedDecomposer::<u32>::new(DecompositionBaseLog(4), DecompositionLevelCount(3));
    /// let decomposable = Tensor::allocate(1_340_987_234_u32, 2);
    /// let decomp = decomposer.decompose_tensor(&decomposable);
    /// assert_eq!(decomp.base_log(), DecompositionBaseLog(4));
    /// ```
    pub fn base_log(&self) -> DecompositionBaseLog {
        DecompositionBaseLog(self.base_log)
    }

    /// Returns the number of levels of this decomposition.
    ///
    /// If the decomposition uses $l$ levels, this returns $l$.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
    /// use concrete_core::backends::core::private::math::decomposition::SignedDecomposer;
    /// use concrete_core::backends::core::private::math::tensor::Tensor;
    /// let decomposer =
    ///     SignedDecomposer::<u32>::new(DecompositionBaseLog(4), DecompositionLevelCount(3));
    /// let decomposable = Tensor::allocate(1_340_987_234_u32, 2);
    /// let decomp = decomposer.decompose_tensor(&decomposable);
    /// assert_eq!(decomp.level_count(), DecompositionLevelCount(3));
    /// ```
    pub fn level_count(&self) -> DecompositionLevelCount {
        DecompositionLevelCount(self.level_count)
    }

    /// Yield the next term of the decomposition, if any.
    ///
    /// # Note
    ///
    /// Because this function returns a borrowed tensor, owned by the iterator, the term must be
    /// dropped before `next_term` is called again.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
    /// use concrete_core::backends::core::private::math::decomposition::{
    ///     DecompositionLevel, SignedDecomposer,
    /// };
    /// use concrete_core::backends::core::private::math::tensor::Tensor;
    /// let decomposer =
    ///     SignedDecomposer::<u32>::new(DecompositionBaseLog(4), DecompositionLevelCount(3));
    /// let decomposable = Tensor::allocate(1_340_987_234_u32, 1);
    /// let mut decomp = decomposer.decompose_tensor(&decomposable);
    /// let term = decomp.next_term().unwrap();
    /// assert_eq!(term.level(), DecompositionLevel(3));
    /// assert_eq!(*term.as_tensor().get_element(0), 4294967295);
    /// ```
    pub fn next_term(&mut self) -> Option<DecompositionTermTensor<'_, Scalar>> {
        // The iterator is not fresh anymore.
        self.fresh = false;
        // We check if the decomposition is over
        if self.current_level == 0 {
            return None;
        }
        // We iterate over the elements of the outputs and decompose
        for zip_args!(output_i, carry_i, input_i) in zip!(
            self.outputs.iter_mut(),
            self.previous_carries.iter_mut(),
            self.inputs.iter()
        ) {
            let (dec, carry) = decompose_one_level(
                self.base_log,
                self.current_level,
                *input_i,
                *carry_i,
                self.mod_b_mask,
                self.carry_mask,
            );
            *carry_i = carry;
            *output_i = dec;
        }
        self.current_level -= 1;
        // We return the term tensor.
        Some(DecompositionTermTensor::new(
            DecompositionLevel(self.current_level + 1),
            DecompositionBaseLog(self.base_log),
            Tensor::from_container(self.outputs.as_slice()),
        ))
    }
}

/// An iterator that yields the terms of the signed decomposition of an integer.
///
/// # Warning
///
/// This iterator yields the decomposition in reverse order. That means that the highest level
/// will be yielded first.
pub struct SignedDecompositionIter<T>
where
    T: UnsignedInteger,
{
    // The value being decomposed
    input: T,
    // The base log of the decomposition
    base_log: usize,
    // The number of levels of the decomposition
    level_count: usize,
    // The carry from the previous level
    previous_carry: T,
    // The current level
    current_level: usize,
    // A mask which allows to compute the mod B of a value. For B=2^4, this guy is of the form:
    // ...0001111
    mod_b_mask: T,
    // A mask which allows to test whether the value is larger than B/2. For B=2^4, this guy is
    // of the form:
    // ...0001000
    carry_mask: T,
    // A flag which store whether the iterator is a fresh one (for the recompose method)
    fresh: bool,
}

impl<T> SignedDecompositionIter<T>
where
    T: UnsignedInteger,
{
    pub(crate) fn new(
        input: T,
        base_log: DecompositionBaseLog,
        level: DecompositionLevelCount,
    ) -> SignedDecompositionIter<T> {
        SignedDecompositionIter {
            input,
            base_log: base_log.0,
            level_count: level.0,
            previous_carry: T::ZERO,
            current_level: level.0,
            mod_b_mask: (T::ONE << base_log.0) - T::ONE,
            carry_mask: T::ONE << (base_log.0 - 1),
            fresh: true,
        }
    }

    pub(crate) fn is_fresh(&self) -> bool {
        self.fresh
    }

    /// Returns the logarithm in base two of the base of this decomposition.
    ///
    /// If the decomposition uses a base $B=2^b$, this returns $b$.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
    /// use concrete_core::backends::core::private::math::decomposition::SignedDecomposer;
    /// let decomposer =
    ///     SignedDecomposer::<u32>::new(DecompositionBaseLog(4), DecompositionLevelCount(3));
    /// let val = 1_340_987_234_u32;
    /// let decomp = decomposer.decompose(val);
    /// assert_eq!(decomp.base_log(), DecompositionBaseLog(4));
    /// ```
    pub fn base_log(&self) -> DecompositionBaseLog {
        DecompositionBaseLog(self.base_log)
    }

    /// Returns the number of levels of this decomposition.
    ///
    /// If the decomposition uses $l$ levels, this returns $l$.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
    /// use concrete_core::backends::core::private::math::decomposition::SignedDecomposer;
    /// let decomposer =
    ///     SignedDecomposer::<u32>::new(DecompositionBaseLog(4), DecompositionLevelCount(3));
    /// let val = 1_340_987_234_u32;
    /// let decomp = decomposer.decompose(val);
    /// assert_eq!(decomp.level_count(), DecompositionLevelCount(3));
    /// ```
    pub fn level_count(&self) -> DecompositionLevelCount {
        DecompositionLevelCount(self.level_count)
    }
}

impl<T> Iterator for SignedDecompositionIter<T>
where
    T: UnsignedInteger,
{
    type Item = DecompositionTerm<T>;

    fn next(&mut self) -> Option<Self::Item> {
        // The iterator is not fresh anymore
        self.fresh = false;
        // We check if the decomposition is over
        if self.current_level == 0 {
            return None;
        }
        // We decompose the current level
        let (output, carry) = decompose_one_level(
            self.base_log,
            self.current_level,
            self.input,
            self.previous_carry,
            self.mod_b_mask,
            self.carry_mask,
        );
        self.previous_carry = carry;
        self.current_level -= 1;
        // We return the output for this level
        Some(DecompositionTerm::new(
            DecompositionLevel(self.current_level + 1),
            DecompositionBaseLog(self.base_log),
            output,
        ))
    }
}

fn decompose_one_level<S: UnsignedInteger>(
    base_log: usize,
    current_level: usize,
    input: S,
    previous_carry: S,
    mod_b_mask: S,
    carry_mask: S,
) -> (S, S) {
    // We perform the division of the input by q/B^i
    let res = input >> (S::BITS - base_log * current_level);
    // We reduce the result modulo B
    let res = res & mod_b_mask;
    // The result may already be greater or equal to B/2.
    let carry = res & carry_mask;
    // We propagate the carry from the previous level
    let res = res.wrapping_add(previous_carry);
    // The previous carry may have made the result larger or equal to B/2.
    let carry = carry | (res & carry_mask);
    // If the result is greater or equal to B/2, we subtract B from the result (viewed as a
    // signed integer)
    let res = (res.into_signed() - (carry << 1).into_signed()).into_unsigned();
    let carry = carry >> (base_log - 1);
    // We return the decomposition and the carry
    (res, carry)
}
