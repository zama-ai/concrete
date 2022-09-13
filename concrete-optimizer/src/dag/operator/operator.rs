use crate::dag::operator::tensor::{ClearTensor, Shape};
use derive_more::{Add, AddAssign};

pub type Weights = ClearTensor<i64>;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FunctionTable {
    pub values: Vec<u64>,
}

impl FunctionTable {
    pub const UNKWOWN: Self = Self { values: vec![] };
}

#[derive(Clone, Copy, PartialEq, Add, AddAssign, Debug)]
pub struct LevelledComplexity {
    pub lwe_dim_cost_factor: f64,
    pub fixed_cost: f64,
}

impl LevelledComplexity {
    pub const ZERO: Self = Self {
        lwe_dim_cost_factor: 0.0,
        fixed_cost: 0.0,
    };
    pub const ADDITION: Self = Self {
        lwe_dim_cost_factor: 1.0,
        fixed_cost: 0.0,
    };
}

impl LevelledComplexity {
    pub fn cost(&self, lwe_dimension: u64) -> f64 {
        self.lwe_dim_cost_factor * (lwe_dimension as f64) + self.fixed_cost
    }
}

impl std::ops::Mul<u64> for LevelledComplexity {
    type Output = Self;
    fn mul(self, factor: u64) -> Self {
        Self {
            lwe_dim_cost_factor: self.lwe_dim_cost_factor * factor as f64,
            fixed_cost: self.fixed_cost * factor as f64,
        }
    }
}
pub type Precision = u8;
pub const MIN_PRECISION: Precision = 1;

#[derive(Clone, PartialEq, Debug)]
pub enum Operator<InputExtraData, LutExtraData, DotExtraData, LevelledOpExtraData> {
    Input {
        out_precision: Precision,
        out_shape: Shape,
        extra_data: InputExtraData,
    },
    Lut {
        input: OperatorIndex,
        table: FunctionTable,
        out_precision: Precision,
        extra_data: LutExtraData,
    },
    Dot {
        inputs: Vec<OperatorIndex>,
        weights: Weights,
        extra_data: DotExtraData,
    },
    LevelledOp {
        inputs: Vec<OperatorIndex>,
        complexity: LevelledComplexity,
        manp: f64,
        out_shape: Shape,
        comment: String,
        extra_data: LevelledOpExtraData,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OperatorIndex {
    pub i: usize,
}
