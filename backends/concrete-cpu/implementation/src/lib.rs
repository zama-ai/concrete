#![allow(clippy::missing_safety_doc, dead_code)]
#![cfg_attr(feature = "nightly", feature(avx512_target_feature))]

extern crate alloc;

pub mod c_api;
mod implementation;
