use std::{
    fmt::Display,
    ops::{Add, AddAssign, Mul, MulAssign},
};

use super::{
    partitions::PartitionIndex,
    symbolic::{Symbol, SymbolArray, SymbolMap, SymbolScheme},
};

/// An ensemble of noise values for fhe operations.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseValues(SymbolArray<f64>);

impl NoiseValues {
    /// Returns an empty set of noise values.
    pub fn from_scheme(scheme: &SymbolScheme) -> NoiseValues {
        NoiseValues(SymbolArray::from_scheme(scheme))
    }

    /// Sets the noise variance associated with a noise source.
    pub fn set_variance(&mut self, source: NoiseSource, value: f64) {
        self.0.set(&source.0, value);
    }

    /// Returns the variance associated with a noise source
    pub fn variance(&self, source: NoiseSource) -> f64 {
        *self.0.get(&source.0)
    }
}

impl Display for NoiseValues {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt_with(f, ";", ":=")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NoiseEvaluator(SymbolArray<f64>);

impl NoiseEvaluator {
    /// Returns a zero noise expression
    pub fn from_scheme_and_expression(
        scheme: &SymbolScheme,
        expr: &NoiseExpression,
    ) -> NoiseEvaluator {
        NoiseEvaluator(SymbolArray::from_scheme_and_map(scheme, &expr.0))
    }

    /// Returns the coefficient associated with a noise source.
    pub fn coeff(&self, source: NoiseSource) -> f64 {
        *self.0.get(&source.0)
    }

    /// Evaluate the noise expression on a set of noise values.
    pub fn evaluate(&self, values: &NoiseValues) -> f64 {
        self.0
            .iter()
            .zip(values.0.iter())
            .fold(0.0, |acc, (coef, var)| acc + coef * var)
    }
}

/// A noise expression, i.e. a sum of noise terms associating a noise source,
/// with a multiplicative coefficient.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseExpression(pub SymbolMap<f64>);

impl NoiseExpression {
    /// Returns a zero noise expression
    pub fn zero() -> Self {
        NoiseExpression(SymbolMap::new())
    }

    /// Returns an iterator over noise terms.
    pub fn terms_iter(&self) -> impl Iterator<Item = NoiseTerm> + '_ {
        self.0.iter().map(|(s, c)| NoiseTerm {
            source: NoiseSource(s),
            coefficient: c,
        })
    }

    /// Returns the coefficient associated with a noise source.
    pub fn coeff(&self, source: NoiseSource) -> f64 {
        self.0.get(source.0)
    }

    /// Builds a noise expression with the largest coefficients of the two expressions.
    pub fn max(lhs: &Self, rhs: &Self) -> Self {
        let mut lhs = lhs.to_owned();
        for (k, v) in rhs.0.iter() {
            let coef = f64::max(lhs.0.get(k), v);
            lhs.0.set(k, coef);
        }
        lhs
    }

    // /// Evaluate the noise expression on a set of noise values.
    // pub fn evaluate(&self, values: &NoiseValues) -> f64 {
    //     self.terms_iter().fold(0.0, |acc, term| {
    //         acc + term.coefficient * values.variance(term.source)
    //     })
    // }
}

impl Display for NoiseExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt_with(f, "+", "σ²")
    }
}

impl From<NoiseTerm> for NoiseExpression {
    fn from(v: NoiseTerm) -> NoiseExpression {
        NoiseExpression::zero() + v
    }
}

impl MulAssign<f64> for NoiseExpression {
    fn mul_assign(&mut self, rhs: f64) {
        if rhs == 0. {
            self.0.clear();
        }
        self.0
            .iter()
            .for_each(|(sym, coef)| self.0.set(sym, coef * rhs));
    }
}

impl Mul<f64> for NoiseExpression {
    type Output = NoiseExpression;

    fn mul(mut self, rhs: f64) -> Self::Output {
        self *= rhs;
        self
    }
}

impl Mul<NoiseExpression> for f64 {
    type Output = NoiseExpression;

    fn mul(self, mut rhs: NoiseExpression) -> Self::Output {
        rhs *= self;
        rhs
    }
}

impl AddAssign<NoiseTerm> for NoiseExpression {
    fn add_assign(&mut self, rhs: NoiseTerm) {
        self.0.update(rhs.source.0, |a| a + rhs.coefficient);
    }
}

impl Add<NoiseTerm> for NoiseExpression {
    type Output = NoiseExpression;

    fn add(mut self, rhs: NoiseTerm) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<NoiseExpression> for NoiseTerm {
    type Output = NoiseExpression;

    fn add(self, mut rhs: NoiseExpression) -> Self::Output {
        rhs += self;
        rhs
    }
}

impl Add<NoiseExpression> for NoiseExpression {
    type Output = NoiseExpression;

    fn add(mut self, rhs: NoiseExpression) -> Self::Output {
        for term in rhs.terms_iter() {
            self += term;
        }
        self
    }
}

impl Add<NoiseTerm> for NoiseTerm {
    type Output = NoiseExpression;

    fn add(self, rhs: NoiseTerm) -> Self::Output {
        let mut output = NoiseExpression::zero();
        output += self;
        output += rhs;
        output
    }
}

/// A symbolic noise term, or a multiplicative coefficient associated with a noise source.
#[derive(Debug)]
pub struct NoiseTerm {
    pub source: NoiseSource,
    pub coefficient: f64,
}

impl Mul<f64> for NoiseSource {
    type Output = NoiseTerm;

    fn mul(self, rhs: f64) -> Self::Output {
        NoiseTerm {
            source: self,
            coefficient: rhs,
        }
    }
}

impl Mul<NoiseSource> for f64 {
    type Output = NoiseTerm;

    fn mul(self, rhs: NoiseSource) -> Self::Output {
        NoiseTerm {
            source: rhs,
            coefficient: self,
        }
    }
}

/// A symbolic source of noise, or a noise source variable.
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct NoiseSource(pub Symbol);

/// Returns an input noise source symbol.
pub fn input_noise(partition: PartitionIndex) -> NoiseSource {
    NoiseSource(Symbol::Input(partition))
}

/// Returns a keyswitch noise source symbol.
pub fn keyswitch_noise(from: PartitionIndex, to: PartitionIndex) -> NoiseSource {
    NoiseSource(Symbol::Keyswitch(from, to))
}

/// Returns a fast keyswitch noise source symbol.
pub fn fast_keyswitch_noise(from: PartitionIndex, to: PartitionIndex) -> NoiseSource {
    NoiseSource(Symbol::FastKeyswitch(from, to))
}

/// Returns a pbs noise source symbol.
pub fn bootstrap_noise(partition: PartitionIndex) -> NoiseSource {
    NoiseSource(Symbol::Bootstrap(partition))
}

/// Returns a modulus switching noise source symbol.
pub fn modulus_switching_noise(partition: PartitionIndex) -> NoiseSource {
    NoiseSource(Symbol::ModulusSwitch(partition))
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_noise_expression() {
        let mut expr = 1. * input_noise(PartitionIndex(0))
            + bootstrap_noise(PartitionIndex(0)) * 2.0
            + 5. * (3. * input_noise(PartitionIndex(1)) + bootstrap_noise(PartitionIndex(1)) * 4.0);
        assert_eq!(expr.coeff(input_noise(PartitionIndex(0))), 1.0);
        assert_eq!(expr.coeff(bootstrap_noise(PartitionIndex(0))), 2.0);
        assert_eq!(expr.coeff(input_noise(PartitionIndex(1))), 15.0);
        assert_eq!(expr.coeff(bootstrap_noise(PartitionIndex(1))), 20.0);
        expr *= 4.;
        println!("{expr}");
        assert_eq!(expr.coeff(input_noise(PartitionIndex(0))), 4.0);
        assert_eq!(expr.coeff(bootstrap_noise(PartitionIndex(0))), 8.0);
        assert_eq!(expr.coeff(input_noise(PartitionIndex(1))), 60.0);
        assert_eq!(expr.coeff(bootstrap_noise(PartitionIndex(1))), 80.0);
        expr *= 0.;
        assert_eq!(expr.0.len(), 0);
    }
}
