use delegate::delegate;
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shape {
    pub dimensions_size: Vec<u64>,
}

impl Shape {
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
        dimensions_size.push(out_dim_size as u64);
        dimensions_size.extend_from_slice(&other.dimensions_size);
        Self { dimensions_size }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ClearTensor {
    pub shape: Shape,
    pub values: Vec<u64>,
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn square(v: &u64) -> u64 {
    v * v
}

impl ClearTensor {
    pub fn number(value: u64) -> Self {
        Self {
            shape: Shape::number(),
            values: vec![value],
        }
    }

    pub fn vector(values: &[u64]) -> Self {
        Self {
            shape: Shape::vector(values.len() as u64),
            values: values.to_vec(),
        }
    }

    delegate! {
        to self.shape {
            pub fn is_number(&self) -> bool;
            pub fn is_vector(&self) -> bool;
            pub fn flat_size(&self) -> u64;
            pub fn rank(&self) -> usize;
        }
    }

    pub fn square_norm2(&self) -> u64 {
        self.values.iter().map(square).sum()
    }
}
