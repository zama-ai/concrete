use proc_macro::TokenStream;
use proc_macro::{self};
use quote::quote;
use std::path::PathBuf;
use syn::LitStr;

#[proc_macro]
pub fn from_concrete_python_export_zip(input: TokenStream) -> TokenStream {
    let pt: Result<LitStr, _> = syn::parse(input);
    let Ok(path_litteral) = pt else {
        return quote!(compile_error!("Unexpected input. Expected path string litteral.");).into();
    };

    let path = PathBuf::from(path_litteral.value());
    if !path.exists() {
        return quote!(compile_error!("Input path must point to an existing export zip.");).into();
    }

    quote! {}.into()
}
