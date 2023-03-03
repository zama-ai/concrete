use crate::implementation::Container;

#[derive(Debug, Clone)]
pub struct PolynomialList<C: Container> {
    pub data: C,
    pub count: usize,
    pub polynomial_size: usize,
}

impl<C: Container> PolynomialList<C> {
    pub fn new(data: C, polynomial_size: usize, count: usize) -> Self {
        debug_assert_eq!(data.len(), polynomial_size * count);
        Self {
            data,
            count,
            polynomial_size,
        }
    }

    fn container_len(&self) -> usize {
        self.data.len()
    }
}

impl PolynomialList<&[u64]> {
    pub fn iter_polynomial(&self) -> impl DoubleEndedIterator<Item = &'_ [u64]> {
        self.data.chunks_exact(self.polynomial_size)
    }

    // Creates an iterator over borrowed sub-lists.
    pub fn sublist_iter(
        &self,
        count: usize,
    ) -> impl DoubleEndedIterator<Item = PolynomialList<&[u64]>> {
        let polynomial_size = self.polynomial_size;

        debug_assert_eq!(self.count % count, 0);

        self.data
            .chunks_exact(count * polynomial_size)
            .map(move |sub| PolynomialList {
                data: sub,
                polynomial_size,
                count,
            })
    }
}
