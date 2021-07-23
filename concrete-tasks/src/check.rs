use crate::{cmd, ENV_DOC_KATEX};
use std::io::Error;

pub fn doc() -> Result<(), Error> {
    cmd!(<ENV_DOC_KATEX> "cargo doc --no-deps")
}

pub fn clippy() -> Result<(), Error> {
    cmd!("cargo +nightly clippy --all-targets --all-features -- -D warnings")
}

pub fn fmt() -> Result<(), Error> {
    cmd!("cargo +nightly fmt --all -- --check")
}
