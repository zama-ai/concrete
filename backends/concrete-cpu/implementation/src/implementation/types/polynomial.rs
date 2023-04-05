use crate::implementation::Container;

#[derive(Debug, Clone, Copy)]
pub struct Polynomial<C: Container> {
    data: C,
    pub polynomial_size: usize,
}

impl<C: Container> Polynomial<C> {
    pub fn new(data: C, polynomial_size: usize) -> Self {
        debug_assert_eq!(data.len(), polynomial_size);
        Self {
            data,
            polynomial_size,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn into_data(self) -> C {
        self.data
    }
}

impl<'a> Polynomial<&'a [u64]> {
    pub fn iter(self) -> impl DoubleEndedIterator<Item = &'a u64> {
        self.data.iter()
    }

    pub fn as_ref(&'a self) -> Self {
        Self {
            data: self.data,
            polynomial_size: self.polynomial_size,
        }
    }
}

impl<'a> Polynomial<&'a mut [u64]> {
    fn iter(self) -> impl DoubleEndedIterator<Item = &'a mut u64> {
        self.data.iter_mut()
    }

    pub fn as_mut_view(&mut self) -> Polynomial<&mut [u64]> {
        Polynomial {
            data: self.data,
            polynomial_size: self.polynomial_size,
        }
    }
}
