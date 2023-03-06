use std::iter::Sum;
use std::ops::Mul;

use crate::utils::square_ref;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shape {
    pub dimensions_size: Vec<u64>,
}

impl Shape {
    pub fn first_dim_size(&self) -> u64 {
        self.dimensions_size[0]
    }

    pub fn rank(&self) -> usize {
        self.dimensions_size.len()
    }

    pub fn flat_size(&self) -> u64 {
        let mut product = 1;
        for dim_size in &self.dimensions_size {
            product *= dim_size;
        }
        product
    }

    pub fn number() -> Self {
        Self {
            dimensions_size: vec![],
        }
    }

    pub fn is_number(&self) -> bool {
        self.rank() == 0
    }

    pub fn vector(size: u64) -> Self {
        Self {
            dimensions_size: vec![size],
        }
    }

    pub fn is_vector(&self) -> bool {
        self.rank() == 1
    }

    pub fn duplicated(out_dim_size: u64, other: &Self) -> Self {
        let mut dimensions_size = Vec::with_capacity(other.rank() + 1);
        dimensions_size.push(out_dim_size);
        dimensions_size.extend_from_slice(&other.dimensions_size);
        Self { dimensions_size }
    }

    pub fn erase_first_dim(&self) -> Self {
        Self {
            dimensions_size: self.dimensions_size[1..].to_vec(),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ClearTensor<W> {
    pub shape: Shape,
    pub values: Vec<W>,
}

impl<W> ClearTensor<W>
where
    W: Copy + Mul<Output = W> + Sum<W>,
{
    pub fn number(value: W) -> Self {
        Self {
            shape: Shape::number(),
            values: vec![value],
        }
    }

    pub fn vector(values: impl Into<Vec<W>>) -> Self {
        let values = values.into();
        Self {
            shape: Shape::vector(values.len() as u64),
            values,
        }
    }

    pub fn is_number(&self) -> bool {
        self.shape.is_number()
    }

    pub fn is_vector(&self) -> bool {
        self.shape.is_vector()
    }

    pub fn flat_size(&self) -> u64 {
        self.shape.flat_size()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn square_norm2(&self) -> W {
        self.values.iter().map(square_ref).sum()
    }
}

// helps using shared shapes
impl From<&Self> for Shape {
    fn from(item: &Self) -> Self {
        item.clone()
    }
}

// helps using shared weights
impl<W> From<&Self> for ClearTensor<W>
where
    W: Copy + Mul<Output = W> + Sum<W>,
{
    fn from(item: &Self) -> Self {
        item.clone()
    }
}

// helps using array as weights
impl<const N: usize, W> From<[W; N]> for ClearTensor<W>
where
    W: Copy + Mul<Output = W> + Sum<W>,
{
    fn from(items: [W; N]) -> Self {
        Self::vector(items)
    }
}
