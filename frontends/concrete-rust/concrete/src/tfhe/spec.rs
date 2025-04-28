use super::IntegerType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ModuleSpec {
    pub input_types_per_func: HashMap<String, Vec<Option<IntegerType>>>,
    pub output_types_per_func: HashMap<String, Vec<Option<IntegerType>>>,
    pub input_shapes_per_func: HashMap<String, Vec<Option<Vec<usize>>>>,
    pub output_shapes_per_func: HashMap<String, Vec<Option<Vec<usize>>>>,
}

impl ModuleSpec {
    pub fn get_func(&self, name: &str) -> Option<FunctionSpec> {
        if !self.input_types_per_func.contains_key(name) {
            return None;
        }
        Some(FunctionSpec {
            input_types: self.input_types_per_func.get(name).unwrap(),
            output_types: self.output_types_per_func.get(name).unwrap(),
            input_shapes: self.input_shapes_per_func.get(name).unwrap(),
            output_shapes: self.output_shapes_per_func.get(name).unwrap(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct FunctionSpec<'a> {
    pub input_types: &'a Vec<Option<IntegerType>>,
    pub output_types: &'a Vec<Option<IntegerType>>,
    pub input_shapes: &'a Vec<Option<Vec<usize>>>,
    pub output_shapes: &'a Vec<Option<Vec<usize>>>,
}

#[cfg(feature = "compiler")]
mod to_tokens {
    //! This module contains `ToTokens` implementations. This allows protocol
    //! values to be interpolated in the `quote!` macro as constructors of the values.
    //! Useful to construct static protocol values.

    use super::*;
    use proc_macro2::TokenStream;
    use quote::{quote, ToTokens};

    impl ToTokens for ModuleSpec {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let input_types_per_func = &self
                .input_types_per_func
                .iter()
                .map(|(key, value)| {
                    (
                        key,
                        value.iter().map(|maybe_type| match maybe_type {
                            Some(typ) => quote! {Some(#typ)},
                            None => quote! { None },
                        }),
                    )
                })
                .map(|(key, value)| quote! { (#key.to_string(), vec![#(#value),*]) })
                .collect::<Vec<_>>();
            let output_types_per_func = &self
                .output_types_per_func
                .iter()
                .map(|(key, value)| {
                    (
                        key,
                        value.iter().map(|maybe_type| match maybe_type {
                            Some(typ) => quote! {Some(#typ)},
                            None => quote! { None },
                        }),
                    )
                })
                .map(|(key, value)| quote! { (#key.to_string(), vec![#(#value),*]) })
                .collect::<Vec<_>>();
            let input_shapes_per_func = &self
                .input_shapes_per_func
                .iter()
                .map(|(key, value)| {
                    (
                        key,
                        value.iter().map(|maybe_shape| match maybe_shape {
                            Some(shape) => quote! {Some(vec![#(#shape),*])},
                            None => quote! { None },
                        }),
                    )
                })
                .map(|(key, value)| quote! { (#key.to_string(), vec![#(#value),*]) })
                .collect::<Vec<_>>();
            let output_shapes_per_func = &self
                .output_shapes_per_func
                .iter()
                .map(|(key, value)| {
                    (
                        key,
                        value.iter().map(|maybe_shape| match maybe_shape {
                            Some(shape) => quote! {Some(vec![#(#shape),*])},
                            None => quote! { None },
                        }),
                    )
                })
                .map(|(key, value)| quote! { (#key.to_string(), vec![#(#value),*]) })
                .collect::<Vec<_>>();

            tokens.extend(quote! {
                    ::concrete::tfhe::ModuleSpec {
                        input_types_per_func: vec![#(#input_types_per_func),*].into_iter().collect(),
                        output_types_per_func: vec![#(#output_types_per_func),*].into_iter().collect(),
                        input_shapes_per_func: vec![#(#input_shapes_per_func),*].into_iter().collect(),
                        output_shapes_per_func: vec![#(#output_shapes_per_func),*].into_iter().collect(),
                    }
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    #[test]
    fn test_deserialize_spec() {
        let string = r#"{
          "input_types_per_func": {
            "my_func": [
              {
                "is_signed": false,
                "bit_width": 16,
                "carry_width": 2,
                "msg_width": 2,
                "params": {
                  "lwe_dimension": 909,
                  "glwe_dimension": 1,
                  "polynomial_size": 4096,
                  "pbs_base_log": 15,
                  "pbs_level": 2,
                  "lwe_noise_distribution": 0,
                  "glwe_noise_distribution": 2.168404344971009e-19,
                  "encryption_key_choice": 0
                }
              },
              {
                "is_signed": false,
                "bit_width": 16,
                "carry_width": 2,
                "msg_width": 2,
                "params": {
                  "lwe_dimension": 909,
                  "glwe_dimension": 1,
                  "polynomial_size": 4096,
                  "pbs_base_log": 15,
                  "pbs_level": 2,
                  "lwe_noise_distribution": 0,
                  "glwe_noise_distribution": 2.168404344971009e-19,
                  "encryption_key_choice": 0
                }
              }
            ]
          },
          "output_types_per_func": {
            "my_func": [
              {
                "is_signed": false,
                "bit_width": 16,
                "carry_width": 2,
                "msg_width": 2,
                "params": {
                  "lwe_dimension": 909,
                  "glwe_dimension": 1,
                  "polynomial_size": 4096,
                  "pbs_base_log": 15,
                  "pbs_level": 2,
                  "lwe_noise_distribution": 0,
                  "glwe_noise_distribution": 2.168404344971009e-19,
                  "encryption_key_choice": 0
                }
              }
            ]
          },
          "input_shapes_per_func": { "my_func": [[], []] },
          "output_shapes_per_func": { "my_func": [[]] }
        }
        "#;
        let cp: ModuleSpec = serde_json::from_str(string).unwrap();
        let a = quote! {#cp};

        dbg!(a.to_string());
    }
}
