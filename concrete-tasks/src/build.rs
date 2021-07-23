use crate::{cmd, ENV_TARGET_NATIVE, ENV_TARGET_SIMD};
use std::io::Error;

pub fn debug_crates() -> Result<(), Error> {
    cmd!("cargo build --all-features")
}

pub fn release_crates() -> Result<(), Error> {
    cmd!("cargo build --release --all-features")
}

pub fn simd_crates() -> Result<(), Error> {
    if cfg!(target_os = "linux") {
        cmd!(<ENV_TARGET_SIMD> "cargo build --release --all-features")
    } else if cfg!(target_os = "macos") {
        cmd!(<ENV_TARGET_NATIVE> "cargo build --release --all-features")
    } else {
        unreachable!()
    }
}

pub fn benches() -> Result<(), Error> {
    cmd!(<ENV_TARGET_NATIVE> "cargo build --release --benches")
}
