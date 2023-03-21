use super::decomposition::SignedDecompositionIter;
use super::types::DecompParams;

#[derive(Copy, Clone, Debug)]
#[readonly::make]
pub struct SignedDecomposer {
    pub decomp_params: DecompParams,
}

impl SignedDecomposer {
    /// Creates a new decomposer.
    pub fn new(decomp_params: DecompParams) -> SignedDecomposer {
        debug_assert!(
            u64::BITS as usize > decomp_params.base_log * decomp_params.level,
            "Decomposed bits exceeds the size of the integer to be decomposed"
        );
        SignedDecomposer { decomp_params }
    }

    /// Returns the closet value representable by the decomposition.
    #[inline]
    pub fn closest_representable(&self, input: u64) -> u64 {
        // The closest number representable by the decomposition can be computed by performing
        // the rounding at the appropriate bit.

        // We compute the number of least significant bits which can not be represented by the
        // decomposition
        let non_rep_bit_count: usize =
            u64::BITS as usize - self.decomp_params.level * self.decomp_params.base_log;
        // We generate a mask which captures the non representable bits
        let non_rep_mask = 1_u64 << (non_rep_bit_count - 1);
        // We retrieve the non representable bits
        let non_rep_bits = input & non_rep_mask;
        // We extract the msb of the  non representable bits to perform the rounding
        let non_rep_msb = non_rep_bits >> (non_rep_bit_count - 1);
        // We remove the non-representable bits and perform the rounding
        let res = input >> non_rep_bit_count;
        let res = res + non_rep_msb;
        res << non_rep_bit_count
    }

    pub fn decompose(&self, input: u64) -> SignedDecompositionIter {
        // Note that there would be no sense of making the decomposition on an input which was
        // not rounded to the closest representable first. We then perform it before decomposing.
        SignedDecompositionIter::new(self.closest_representable(input), self.decomp_params)
    }
}
