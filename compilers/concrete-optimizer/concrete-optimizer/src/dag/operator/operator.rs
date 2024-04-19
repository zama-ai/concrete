use std::fmt;
use std::iter::{empty, once};
use std::ops::Deref;

use crate::dag::operator::tensor::{ClearTensor, Shape};

use super::DotKind;

pub type Weights = ClearTensor<i64>;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FunctionTable {
    pub values: Vec<u64>,
}

impl FunctionTable {
    pub const UNKWOWN: Self = Self { values: vec![] };
}

#[derive(Clone, Copy, PartialEq, Debug)]
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

impl std::ops::Add for LevelledComplexity {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            lwe_dim_cost_factor: self.lwe_dim_cost_factor + rhs.lwe_dim_cost_factor,
            fixed_cost: self.fixed_cost + rhs.fixed_cost,
        }
    }
}

impl std::ops::AddAssign for LevelledComplexity {
    fn add_assign(&mut self, rhs: Self) {
        self.lwe_dim_cost_factor += rhs.lwe_dim_cost_factor;
        self.fixed_cost += rhs.fixed_cost;
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
pub enum Operator {
    Input {
        out_precision: Precision,
        out_shape: Shape,
    },
    Lut {
        input: OperatorIndex,
        table: FunctionTable,
        out_precision: Precision,
    },
    Dot {
        inputs: Vec<OperatorIndex>,
        weights: Weights,
        kind: DotKind,
    },
    LevelledOp {
        inputs: Vec<OperatorIndex>,
        complexity: LevelledComplexity,
        manp: f64,
        out_shape: Shape,
        comment: String,
    },
    // Used to reduced or increase precision when the ciphertext is compatible with different precision
    // This is done without any checking
    UnsafeCast {
        input: OperatorIndex,
        out_precision: Precision, // precision is changed without modifying the input, can be increase or decrease
    },
    // Round is expanded to sub-graph on direct representation or fused in lut for Radix and Crt representation.
    Round {
        input: OperatorIndex,
        out_precision: Precision,
    },
}

impl Operator {
    // Returns an iterator on the indices of the operator inputs.
    pub(crate) fn get_inputs_iter(&self) -> Box<dyn Iterator<Item = &OperatorIndex> + '_> {
        match self {
            Self::Input { .. } => Box::new(empty()),
            Self::LevelledOp { inputs, .. } | Self::Dot { inputs, .. } => Box::new(inputs.iter()),
            Self::UnsafeCast { input, .. }
            | Self::Lut { input, .. }
            | Self::Round { input, .. } => Box::new(once(input)),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct OperatorIndex(pub usize);

impl Deref for OperatorIndex {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for OperatorIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "")?;
        match self {
            Self::Input {
                out_precision,
                out_shape,
            } => {
                write!(f, "Input : u{out_precision} x {out_shape:?}")?;
            }
            Self::Dot {
                inputs, weights, ..
            } => {
                for (i, (input, weight)) in inputs.iter().zip(weights.values.iter()).enumerate() {
                    if i > 0 {
                        write!(f, " + ")?;
                    }
                    write!(f, "{weight} x %{}", input.0)?;
                }
            }
            Self::UnsafeCast {
                input,
                out_precision,
            } => {
                write!(f, "%{} : u{out_precision}", input.0)?;
            }
            Self::Lut {
                input,
                out_precision,
                ..
            } => {
                write!(f, "LUT[%{}] : u{out_precision}", input.0)?;
            }
            Self::LevelledOp {
                inputs,
                manp,
                out_shape,
                ..
            } => {
                write!(f, "LINEAR[")?;
                for (i, input) in inputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "%{}", input.0)?;
                }
                write!(f, "] : manp={manp} x {out_shape:?}")?;
            }
            Self::Round {
                input,
                out_precision,
            } => {
                write!(f, "ROUND[%{}] : u{out_precision}", input.0)?;
            }
        }
        Ok(())
    }
}
