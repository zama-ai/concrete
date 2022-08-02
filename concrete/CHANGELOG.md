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
