use crate::{cmd, ENV_COVERAGE, ENV_TARGET_NATIVE};
use std::io::Error;

pub fn toplevel() -> Result<(), Error> {
    cmd!(<ENV_TARGET_NATIVE> "cargo test --release --no-fail-fast --all-features -p concrete")
}

pub fn commons() -> Result<(), Error> {
    cmd!(<ENV_TARGET_NATIVE> "cargo test --release --no-fail-fast --all-features -p concrete-commons")
}

pub fn core() -> Result<(), Error> {
    cmd!(<ENV_TARGET_NATIVE> "cargo test --release --no-fail-fast --all-features -p concrete-core")
}

pub fn csprng() -> Result<(), Error> {
    cmd!(<ENV_TARGET_NATIVE> "cargo test --release --no-fail-fast --all-features -p concrete-csprng")
}

pub fn npe() -> Result<(), Error> {
    cmd!(<ENV_TARGET_NATIVE> "cargo test --release --no-fail-fast --all-features -p concrete-npe")
}

pub fn crates() -> Result<(), Error> {
    cmd!(<ENV_TARGET_NATIVE> "cargo test --release --no-fail-fast --all-features")
}

pub fn cov_crates() -> Result<(), Error> {
    cmd!(<ENV_COVERAGE> "cargo +nightly test --release --no-fail-fast --all-features")
}
