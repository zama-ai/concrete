use crate::backends::core::private::math::decomposition::{
    SignedDecompositionIter, TensorSignedDecompositionIter,
};
use crate::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor, Tensor};
use concrete_commons::numeric::{Numeric, UnsignedInteger};
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
use std::marker::PhantomData;

/// A structure which allows to decompose unsigned integers into a set of smaller terms.
///
/// See the [module level](super) documentation for a description of the signed decomposition.
#[derive(Debug)]
pub struct SignedDecomposer<Scalar>
where
    Scalar: UnsignedInteger,
{
    pub(super) base_log: usize,
    pub(super) level_count: usize,
    integer_type: PhantomData<Scalar>,
}

impl<Scalar> SignedDecomposer<Scalar>
where
    Scalar: UnsignedInteger,
{
    /// Creates a new decomposer.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
    /// use concrete_core::backends::core::private::math::decomposition::SignedDecomposer;
    /// let decomposer =
    ///     SignedDecomposer::<u32>::new(DecompositionBaseLog(4), DecompositionLevelCount(3));
    /// assert_eq!(decomposer.level_count(), DecompositionLevelCount(3));
    /// assert_eq!(decomposer.base_log(), DecompositionBaseLog(4));
    /// ```
    pub fn new(
        base_log: DecompositionBaseLog,
        level_count: DecompositionLevelCount,
    ) -> SignedDecomposer<Scalar> {
        debug_assert!(
            Scalar::BITS > base_log.0 * level_count.0,
            "Decomposed bits exceeds the size of the integer to be decomposed"
        );
        SignedDecomposer {
            base_log: base_log.0,
            level_count: level_count.0,
            integer_type: PhantomData,
        }
    }

    /// Returns the logarithm in base two of the base of this decomposer.
    ///
    /// If the decomposer uses a base $B=2^b$, this returns $b$.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
    /// use concrete_core::backends::core::private::math::decomposition::SignedDecomposer;
    /// let decomposer =
    ///     SignedDecomposer::<u32>::new(DecompositionBaseLog(4), DecompositionLevelCount(3));
    /// assert_eq!(decomposer.base_log(), DecompositionBaseLog(4));
    /// ```
    pub fn base_log(&self) -> DecompositionBaseLog {
        DecompositionBaseLog(self.base_log)
    }

    /// Returns the number of levels of this decomposer.
    ///
    /// If the decomposer uses $l$ levels, this returns $l$.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
    /// use concrete_core::backends::core::private::math::decomposition::SignedDecomposer;
    /// let decomposer =
    ///     SignedDecomposer::<u32>::new(DecompositionBaseLog(4), DecompositionLevelCount(3));
    /// assert_eq!(decomposer.level_count(), DecompositionLevelCount(3));
    /// ```
    pub fn level_count(&self) -> DecompositionLevelCount {
        DecompositionLevelCount(self.level_count)
    }

    /// Returns the closet value representable by the decomposition.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
    /// use concrete_core::backends::core::private::math::decomposition::SignedDecomposer;
    /// let decomposer =
    ///     SignedDecomposer::<u32>::new(DecompositionBaseLog(4), DecompositionLevelCount(3));
    /// let closest = decomposer.closest_representable(1_340_987_234_u32);
    /// assert_eq!(closest, 1_341_128_704_u32);
    /// ```
    pub fn closest_representable(&self, input: Scalar) -> Scalar {
        // The closest number representable by the decomposition can be computed by performing
        // the rounding at the appropriate bit.

        // We compute the number of least significant bits which can not be represented by the
        // decomposition
        let non_rep_bit_count: usize = <Scalar as Numeric>::BITS - self.level_count * self.base_log;
        // We generate a mask which captures the non representable bits
        let non_rep_mask = Scalar::ONE << (non_rep_bit_count - 1);
        // We retrieve the non representable bits
        let non_rep_bits = input & non_rep_mask;
        // We extract the msb of the  non representable bits to perform the rounding
        let non_rep_msb = non_rep_bits >> (non_rep_bit_count - 1);
        // We remove the non-representable bits and perform the rounding
        let res = input >> non_rep_bit_count;
        let res = res + non_rep_msb;
        res << non_rep_bit_count
    }

    /// Fills a mutable tensor-like objects with the closest representable values from another
    /// tensor-like object.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
    /// use concrete_core::backends::core::private::math::decomposition::SignedDecomposer;
    /// use concrete_core::backends::core::private::math::tensor::Tensor;
    /// let decomposer =
    ///     SignedDecomposer::<u32>::new(DecompositionBaseLog(4), DecompositionLevelCount(3));
    ///
    /// let input = Tensor::allocate(1_340_987_234_u32, 1);
    /// let mut closest = Tensor::allocate(0u32, 1);
    /// decomposer.fill_tensor_with_closest_representable(&mut closest, &input);
    /// assert_eq!(*closest.get_element(0), 1_341_128_704_u32);
    /// ```
    pub fn fill_tensor_with_closest_representable<I, O>(&self, output: &mut O, input: &I)
    where
        I: AsRefTensor<Element = Scalar>,
        O: AsMutTensor<Element = Scalar>,
    {
        output
            .as_mut_tensor()
            .fill_with_one(input.as_tensor(), |elmt| self.closest_representable(*elmt))
    }

    /// Generates an iterator over the terms of the decomposition of the input.
    ///
    /// # Warning
    ///
    /// The returned iterator yields the terms $\tilde{\theta}_i$ in order of decreasing $i$.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::numeric::UnsignedInteger;
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
    /// use concrete_core::backends::core::private::math::decomposition::SignedDecomposer;
    /// let decomposer =
    ///     SignedDecomposer::<u32>::new(DecompositionBaseLog(4), DecompositionLevelCount(3));
    /// for term in decomposer.decompose(1_340_987_234_u32) {
    ///     assert!(1 <= term.level().0);
    ///     assert!(term.level().0 <= 3);
    ///     let signed_term = term.value().into_signed();
    ///     let half_basis = 2i32.pow(4) / 2i32;
    ///     assert!(-half_basis <= signed_term);
    ///     assert!(signed_term < half_basis);
    /// }
    /// assert_eq!(decomposer.decompose(1).count(), 3);
    /// ```
    pub fn decompose(&self, input: Scalar) -> SignedDecompositionIter<Scalar> {
        // Note that there would be no sense of making the decomposition on an input which was
        // not rounded to the closest representable first. We then perform it before decomposing.
        SignedDecompositionIter::new(
            self.closest_representable(input),
            DecompositionBaseLog(self.base_log),
            DecompositionLevelCount(self.level_count),
        )
    }

    /// Recomposes a decomposed value by summing all the terms.
    ///
    /// If the input iterator yields $\tilde{\theta}_i$, this returns
    /// $\sum_{i=1}^l\tilde{\theta}_i\frac{q}{B^i}$.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
    /// use concrete_core::backends::core::private::math::decomposition::SignedDecomposer;
    /// let decomposer =
    ///     SignedDecomposer::<u32>::new(DecompositionBaseLog(4), DecompositionLevelCount(3));
    /// let val = 1_340_987_234_u32;
    /// let dec = decomposer.decompose(val);
    /// let rec = decomposer.recompose(dec);
    /// assert_eq!(decomposer.closest_representable(val), rec.unwrap());
    /// ```
    pub fn recompose(&self, decomp: SignedDecompositionIter<Scalar>) -> Option<Scalar> {
        if decomp.is_fresh() {
            Some(decomp.fold(Scalar::ZERO, |acc, term| {
                acc.wrapping_add(term.to_recomposition_summand())
            }))
        } else {
            None
        }
    }

    /// Generates an iterator-like object over tensors of terms of the decomposition of the input
    /// tensor.
    ///
    /// # Warning
    ///
    /// The returned iterator yields the terms $(\tilde{\theta}^{(a)}_i)_{a\in\mathbb{N}}$ in order
    /// of decreasing $i$.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::numeric::UnsignedInteger;
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
    /// use concrete_core::backends::core::private::math::decomposition::SignedDecomposer;
    /// use concrete_core::backends::core::private::math::tensor::Tensor;
    /// let decomposer =
    ///     SignedDecomposer::<u32>::new(DecompositionBaseLog(4), DecompositionLevelCount(3));
    /// let decomposable = Tensor::from_container(vec![1_340_987_234_u32, 1_340_987_234_u32]);
    /// let mut decomp = decomposer.decompose_tensor(&decomposable);
    /// ///
    /// let mut count = 0;
    /// while let Some(term) = decomp.next_term() {
    ///     assert!(1 <= term.level().0);
    ///     assert!(term.level().0 <= 3);
    ///     for elmt in term.as_tensor().iter() {
    ///         let signed_term = elmt.into_signed();
    ///         let half_basis = 2i32.pow(4) / 2i32;
    ///         assert!(-half_basis <= signed_term);
    ///         assert!(signed_term < half_basis);
    ///     }
    ///     count += 1;
    /// }
    /// assert_eq!(count, 3);
    /// ```
    pub fn decompose_tensor<I>(&self, input: &I) -> TensorSignedDecompositionIter<Scalar>
    where
        I: AsRefTensor<Element = Scalar>,
    {
        // Note that there would be no sense of making the decomposition on an input which was
        // not rounded to the closest representable first. We then perform it before decomposing.
        let mut rounded = Tensor::allocate(Scalar::ZERO, input.as_tensor().len());
        self.fill_tensor_with_closest_representable(&mut rounded, input);
        TensorSignedDecompositionIter::new(
            rounded,
            DecompositionBaseLog(self.base_log),
            DecompositionLevelCount(self.level_count),
        )
    }

    /// Fills the output tensor with the recomposition of an other tensor.
    ///
    /// Returns `Some(())` if the decomposition was fresh, and the output was filled with a
    /// recomposition, and `None`, if not.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
    /// use concrete_core::backends::core::private::math::decomposition::SignedDecomposer;
    /// use concrete_core::backends::core::private::math::tensor::Tensor;
    /// let decomposer =
    ///     SignedDecomposer::<u32>::new(DecompositionBaseLog(4), DecompositionLevelCount(3));
    /// let decomposable = Tensor::allocate(1_340_987_234_u32, 1);
    /// let mut rounded = Tensor::allocate(0u32, 1);
    /// decomposer.fill_tensor_with_closest_representable(&mut rounded, &decomposable);
    /// let mut decomp = decomposer.decompose_tensor(&rounded);
    /// let mut recomposition = Tensor::allocate(0u32, 1);
    /// decomposer
    ///     .fill_tensor_with_recompose(decomp, &mut recomposition)
    ///     .unwrap();
    /// assert_eq!(recomposition, rounded);
    /// ```
    pub fn fill_tensor_with_recompose<TLike>(
        &self,
        decomp: TensorSignedDecompositionIter<Scalar>,
        output: &mut TLike,
    ) -> Option<()>
    where
        TLike: AsMutTensor<Element = Scalar>,
    {
        let mut decomp = decomp;
        if decomp.is_fresh() {
            while let Some(term) = decomp.next_term() {
                term.update_tensor_with_recomposition_summand_wrapping_addition(output);
            }
            Some(())
        } else {
            None
        }
    }
}
