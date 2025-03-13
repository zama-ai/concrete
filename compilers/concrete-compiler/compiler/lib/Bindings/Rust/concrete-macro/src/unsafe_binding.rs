// (void (*)(void *, ...))

use concrete::protocol::ProgramInfo;
use quote::{format_ident, quote};


pub fn generate_unsafe_binding(pi: &ProgramInfo) -> proc_macro2::TokenStream {
    let func_defs = pi.circuits.iter().map(|circuit| {
        #[cfg(target_os = "macos")]
        let name = format_ident!("concrete_{}", circuit.name);
        #[cfg(target_os = "linux")]
        let name = format_ident!("concrete_{}", circuit.name);
        quote!{
            fn #name(arg: *mut std::ffi::c_void, ...) -> *mut std::ffi::c_void;
        }
    }).collect::<Vec<_>>();
    quote!{
        extern "C" {
            #(#func_defs)*
        }
    }
}
