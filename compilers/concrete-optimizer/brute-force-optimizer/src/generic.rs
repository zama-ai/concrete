use rayon::prelude::{IntoParallelIterator, ParallelIterator};

pub trait Problem {
    type Param;

    fn verify(&self, param: Self::Param) -> bool;
    fn cost(&self, param: Self::Param) -> f64;
}

pub trait ParallelBruteForcableProblem: Problem + Sync
where
    Self::Param: Send + Copy,
{
    fn brute_force_parallel(
        &self,
        params: impl rayon::iter::ParallelIterator<Item = Self::Param>,
    ) -> Option<(Self::Param, f64)>;
}

impl<T> ParallelBruteForcableProblem for T
where
    T: Problem + Sync,
    T::Param: Send + Copy,
{
    fn brute_force_parallel(
        &self,
        params: impl rayon::iter::ParallelIterator<Item = Self::Param>,
    ) -> Option<(Self::Param, f64)> {
        params
            .into_par_iter()
            .filter_map(|param| {
                if self.verify(param) {
                    Some((param, self.cost(param)))
                } else {
                    None
                }
            })
            .min_by(|(_, cost1), (_, cost2)| cost1.partial_cmp(cost2).unwrap())
    }
}

pub trait SequentialProblem: Problem + Sync
where
    Self::Param: Send + Copy,
{
    fn brute_force(&self, params: impl Iterator<Item = Self::Param>) -> Option<(Self::Param, f64)>;
}

impl<T> SequentialProblem for T
where
    T: Problem + Sync,
    T::Param: Send + Copy,
{
    fn brute_force(&self, params: impl Iterator<Item = Self::Param>) -> Option<(Self::Param, f64)> {
        params
            .into_iter()
            .filter_map(|param| {
                if self.verify(param) {
                    Some((param, self.cost(param)))
                } else {
                    None
                }
            })
            .min_by(|(_, cost1), (_, cost2)| cost1.partial_cmp(cost2).unwrap())
    }
}
