
mod ffi;
mod utils;

#[doc(hidden)]
pub mod compiler{
    pub use crate::ffi::CompilationOptions;
    pub use crate::ffi::Library;
    pub use crate::ffi::compilation_options_new;
    pub use crate::ffi::compile;
}

pub mod protocol;
