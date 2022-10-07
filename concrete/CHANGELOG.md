# Unreleased (Target 0.2.0)

## Added
 
 - Crt representation for integers (8 bits +) 
   (**Breakinkg Change** for the parameters of Dynamic Integers).
 - Univariate function evaluation for integers (8 bits +)
 - Initial support for aarch64 targets (requires nightly rust)

## Changed

 - Replaced fftw with concrete-fft

 - Updated concrete-boolean dependency to 0.2.0
 - Updated concrete-shortint dependency to 0.1.0
 - Updated concrete-integer dependency to 0.1.0

---

# 0.2.0-beta.2

## Added

- KeyCacher struct, to help avoiding generating the keys for each run.
- Implementation of `std::iter::{Sum, Product}` for shortints
- Trivial encryption for shortints
- `concrete::Config` is now a public type.
- `FheEq` and `FheOrd` traits to enable comparisons of shortints with scalars or shortints.

---

# 0.2.0-beta.1

## Misc.

- Updated `Cargo.toml` metadata for `doc.rs`

---

# 0.2.0-beta.0

## Changed

- Totally new API, focused on ease of use for non-cryptographers
