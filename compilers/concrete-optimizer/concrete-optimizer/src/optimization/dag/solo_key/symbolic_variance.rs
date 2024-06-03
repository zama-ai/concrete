use std::iter::Sum;

/**
 * A variance that is represented as a linear combination of base variances.
 * Only the linear coefficient are known.
 * The base variances are unknown.
 *
 * Only 2 base variances are possible in the solo key setup:
 *  - from input,
 *  - or from lut output.
 *
 * We only kown that the first one is lower or equal to the second one.
 * Each linear coefficient is a variance factor.
 * There are homogenious to squared weight (or summed square weights or squared norm2).
 */
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct SymbolicVariance {
    pub lut_coeff: f64,
    pub input_coeff: f64,
    // variance = vf.lut_coeff * lut_out_noise
    //          + vf.input_coeff * input_out_noise
    // E.g. variance(dot([lut, input], [3, 4])) = VariancesFactors {lut_coeff:9, input_coeff: 16}

    // NOTE: lut_base_noise is the first field since it has higher impact,
    // see pareto sorting and dominate_or_equal
}

impl std::ops::Add for SymbolicVariance {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            lut_coeff: self.lut_coeff + rhs.lut_coeff,
            input_coeff: self.input_coeff + rhs.input_coeff,
        }
    }
}

impl std::ops::AddAssign for SymbolicVariance {
    fn add_assign(&mut self, rhs: Self) {
        self.lut_coeff += rhs.lut_coeff;
        self.input_coeff += rhs.input_coeff;
    }
}

impl Sum for SymbolicVariance {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut accumulator = Self::ZERO;

        for item in iter {
            accumulator += item;
        }

        accumulator
    }
}

impl std::ops::Mul<f64> for SymbolicVariance {
    type Output = Self;
    fn mul(self, sq_weight: f64) -> Self {
        Self {
            input_coeff: self.input_coeff * sq_weight,
            lut_coeff: self.lut_coeff * sq_weight,
        }
    }
}

impl std::ops::Mul<u64> for SymbolicVariance {
    type Output = Self;
    fn mul(self, sq_weight: u64) -> Self {
        self * sq_weight as f64
    }
}

impl SymbolicVariance {
    pub const ZERO: Self = Self {
        input_coeff: 0.0,
        lut_coeff: 0.0,
    };
    pub const INPUT: Self = Self {
        input_coeff: 1.0,
        lut_coeff: 0.0,
    };
    pub const LUT: Self = Self {
        input_coeff: 0.0,
        lut_coeff: 1.0,
    };

    pub fn dominate_or_equal(&self, other: &Self) -> bool {
        let extra_other_minimal_base_noise = 0.0_f64.max(other.input_coeff - self.input_coeff);
        other.lut_coeff + extra_other_minimal_base_noise <= self.lut_coeff
    }

    pub fn eval(&self, minimal_base_noise: f64, lut_base_noise: f64) -> f64 {
        minimal_base_noise * self.input_coeff + lut_base_noise * self.lut_coeff
    }

    pub fn reduce_to_pareto_front(mut vfs: Vec<Self>) -> Vec<Self> {
        if vfs.is_empty() {
            return vec![];
        }
        vfs.sort_by(
            // bigger first
            |a, b| b.partial_cmp(a).unwrap(),
        );
        // Due to the special domination nature, this can be done in one pass
        let mut dominator = vfs[0];
        let mut pareto = vec![dominator];
        for &vf in vfs.iter().skip(1) {
            if dominator.dominate_or_equal(&vf) {
                continue;
            }
            dominator = vf;
            pareto.push(vf);
        }
        pareto
    }
}
