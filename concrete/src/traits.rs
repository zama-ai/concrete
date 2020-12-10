pub trait HomomorphicAdd<T, U> {
    fn add(&self, left: &U, right: T) -> U;
    fn add_inplace(&self, left: &mut U, right: T);
}

pub trait HomomorphicSub<T, U> {
    fn sub(&self, left: &U, right: T) -> U;
    fn sub_inplace(&self, left: &mut U, right: T);
}

pub trait HomomorphicMul<T, U> {
    fn mul(&self, left: &U, right: T) -> U;
    fn mul_inplace(&self, left: &mut U, right: T);
}

pub trait GenericAdd<T, E>: Sized {
    fn add(&self, right: T) -> Result<Self, E>;
    fn add_inplace(&mut self, right: T) -> Result<(), E>;
}
