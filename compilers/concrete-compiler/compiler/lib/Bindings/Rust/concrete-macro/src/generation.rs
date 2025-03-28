use concrete::protocol::{CircuitInfo, ProgramInfo};
use quote::{format_ident, quote};

pub fn generate_unsafe_binding(pi: &ProgramInfo) -> proc_macro2::TokenStream {
    let func_defs = pi
        .circuits
        .iter()
        .map(|circuit| {
            let name = format_ident!("_mlir_concrete_{}", circuit.name);
            quote! {
                pub fn #name(arg: *mut std::ffi::c_void, ...) -> *mut std::ffi::c_void;
            }
        })
        .collect::<Vec<_>>();
    quote! {
        extern "C" {
            #(#func_defs)*
        }
    }
}

pub fn generate_infos(pi: &ProgramInfo) -> proc_macro2::TokenStream {
    quote! {
        pub static PROGRAM_INFO: std::sync::LazyLock<::concrete::protocol::ProgramInfo> = std::sync::LazyLock::new(|| {
            #pi
        });
    }
}

pub fn generate_keyset() -> proc_macro2::TokenStream {
    quote! {
        pub fn new_keyset(
            secret_csprng: std::pin::Pin<&mut ::concrete::common::SecretCsprng>,
            encryption_csprng: std::pin::Pin<&mut ::concrete::common::EncryptionCsprng>
        ) -> ::concrete::UniquePtr<::concrete::common::Keyset> {
            ::concrete::common::Keyset::new(
                &PROGRAM_INFO.keyset,
                secret_csprng,
                encryption_csprng
            )
        }
    }
}

pub(crate) fn generate_client(program_info: &ProgramInfo) -> proc_macro2::TokenStream {
    let client_functions = program_info
        .circuits
        .iter()
        .map(|ci| generate_client_function(ci));
    quote! {
        pub mod client {
            #(#client_functions)*
        }
    }
}

fn generate_client_function_prepare_inputs(circuit_info: &CircuitInfo) -> proc_macro2::TokenStream {
    let ith = (0..circuit_info.inputs.len()).collect::<Vec<_>>();
    let input_idents = circuit_info.inputs.iter().enumerate().map(|(ith, _)| format_ident!("arg_{ith}")).collect::<Vec<_>>();
    let input_types = circuit_info.inputs.iter().map(|gi| {
        match (gi.rawInfo.integerPrecision, gi.rawInfo.isSigned) {
            (8, true) => quote! {::concrete::common::Tensor<i8>},
            (8, false) => quote! {::concrete::common::Tensor<u8>},
            (16, true) => quote! {::concrete::common::Tensor<i16>},
            (16, false) => quote! {::concrete::common::Tensor<u16>},
            (32, true) => quote! {::concrete::common::Tensor<i32>},
            (32, false) => quote! {::concrete::common::Tensor<u32>},
            (64, true) => quote! {::concrete::common::Tensor<i64>},
            (64, false) => quote! {::concrete::common::Tensor<u64>},
            _ => unreachable!(),
        }
    }).collect::<Vec<_>>();
    let input_dims = circuit_info.inputs.iter().map(|gi| {
        let a = gi.rawInfo.shape.dimensions.iter().map(|a| *a as usize);
        quote!{[#(#a),*]}
    }).collect::<Vec<_>>();
    let output_types = circuit_info.inputs.iter().map(|_| {
        quote! {::concrete::UniquePtr<::concrete::common::TransportValue>}
    }).collect::<Vec<_>>();

    quote!{
        pub fn prepare_inputs(&mut self, #(#input_idents: #input_types),*) -> (#(#output_types),*) {
            // #(
            //     assert_eq!(#input_idents .dimensions(), #input_dims, "Argument {} has unexpected dimension.", stringify!(#input_idents));
            // )*
            (
                #(
                    self.0.pin_mut().prepare_input(::concrete::common::Value::from_tensor(#input_idents), #ith)
                ),*
            )
        }
    }
}

fn generate_client_function_process_outputs(circuit_info: &CircuitInfo) -> proc_macro2::TokenStream {
    let ith = (0..circuit_info.outputs.len()).collect::<Vec<_>>();
    let input_idents = circuit_info.outputs.iter().enumerate().map(|(ith, _)| format_ident!("res_{ith}")).collect::<Vec<_>>();
    let input_types = circuit_info.outputs.iter().map(|_| {
        quote! {::concrete::UniquePtr<::concrete::common::TransportValue>}
    }).collect::<Vec<_>>();
    let output_types = circuit_info.outputs.iter().map(|gi| {
        match (gi.rawInfo.integerPrecision, gi.rawInfo.isSigned) {
            (8, true) => quote! {::concrete::common::Tensor<i8>},
            (8, false) => quote! {::concrete::common::Tensor<u8>},
            (16, true) => quote! {::concrete::common::Tensor<i16>},
            (16, false) => quote! {::concrete::common::Tensor<u16>},
            (32, true) => quote! {::concrete::common::Tensor<i32>},
            (32, false) => quote! {::concrete::common::Tensor<u32>},
            (64, true) => quote! {::concrete::common::Tensor<i64>},
            (64, false) => quote! {::concrete::common::Tensor<u64>},
            _ => unreachable!(),
        }
    }).collect::<Vec<_>>();
    let output_unwrap = circuit_info.outputs.iter().map(|gi| {
        match (gi.rawInfo.integerPrecision, gi.rawInfo.isSigned) {
            (8, true) => quote! {get_tensor::<i8>().unwrap()},
            (8, false) => quote! {get_tensor::<u8>().unwrap()},
            (16, true) => quote! {get_tensor::<i16>().unwrap()},
            (16, false) => quote! {get_tensor::<u16>().unwrap()},
            (32, true) => quote! {get_tensor::<i32>().unwrap()},
            (32, false) => quote! {get_tensor::<u32>().unwrap()},
            (64, true) => quote! {get_tensor::<i64>().unwrap()},
            (64, false) => quote! {get_tensor::<u64>().unwrap()},
            _ => unreachable!(),
        }
    }).collect::<Vec<_>>();


    quote!{
        pub fn process_outputs(&mut self, #(#input_idents: #input_types),*) -> (#(#output_types),*) {
            (
                #(
                    self.0.pin_mut().process_output(#input_idents, #ith).#output_unwrap
                ),*
            )
        }
    }
}

fn generate_client_function(circuit_info: &CircuitInfo) -> proc_macro2::TokenStream {
    let function_identifier = format_ident!("{}", circuit_info.name);

    let prepare_inputs = generate_client_function_prepare_inputs(circuit_info);
    let process_outputs = generate_client_function_process_outputs(circuit_info);

    quote! {
        pub mod #function_identifier{
            pub static CIRCUIT_INFO: std::sync::LazyLock<::concrete::protocol::CircuitInfo> = std::sync::LazyLock::new(|| {
                #circuit_info
            });

            pub struct ClientFunction(::concrete::UniquePtr<::concrete::client::ClientFunction>);

            impl ClientFunction{
                pub fn new(
                    client_keyset: &::concrete::common::ClientKeyset,
                    encryption_csprng: ::concrete::UniquePtr<::concrete::common::EncryptionCsprng>
                ) -> Self {
                    ClientFunction(
                        ::concrete::client::ClientFunction::new_encrypted(
                            & CIRCUIT_INFO,
                            client_keyset,
                            encryption_csprng
                        )
                    )
                }

                #prepare_inputs

                #process_outputs
            }
        }
    }
}

pub(crate) fn generate_server(program_info: &ProgramInfo) -> proc_macro2::TokenStream {
    let server_functions = program_info
        .circuits
        .iter()
        .map(|ci| generate_server_function(ci));
    quote! {
        pub mod server {
            #(#server_functions)*
        }
    }
}

fn generate_server_function(circuit_info: &CircuitInfo) -> proc_macro2::TokenStream {

    let function_identifier = format_ident!("{}", circuit_info.name);
    let binding_identifier = format_ident!("_mlir_concrete_{}", circuit_info.name);
    let invoke =  generate_server_function_invoke(circuit_info);

    quote! {
        pub mod #function_identifier{
            pub static CIRCUIT_INFO: std::sync::LazyLock<::concrete::protocol::CircuitInfo> = std::sync::LazyLock::new(|| {
                #circuit_info
            });

            pub struct ServerFunction(::concrete::UniquePtr<::concrete::server::ServerFunction>);

            impl ServerFunction{
                pub fn new() -> Self {
                    ServerFunction(
                        ::concrete::server::ServerFunction::new(
                            & CIRCUIT_INFO,
                            super::super::_binding::#binding_identifier as *mut ::concrete::c_void,
                            false
                        )
                    )
                }

                #invoke
            }
        }
    }

}

fn generate_server_function_invoke(circuit_info: &CircuitInfo) -> proc_macro2::TokenStream {
    let args_idents = (0..circuit_info.inputs.len()).map(|a| format_ident!("arg_{a}")).collect::<Vec<_>>();
    let args_types = (0..circuit_info.inputs.len()).map(|_| quote!{::concrete::UniquePtr<::concrete::common::TransportValue>}).collect::<Vec<_>>();
    let results_idents = (0..circuit_info.outputs.len()).map(|a| format_ident!("res_{a}")).collect::<Vec<_>>();
    let results_types = (0..circuit_info.outputs.len()).map(|_| quote!{::concrete::UniquePtr<::concrete::common::TransportValue>}).collect::<Vec<_>>();
    let output_len = circuit_info.outputs.len();

    quote!{
        pub fn invoke(&mut self, server_keyset: &::concrete::common::ServerKeyset, #(#args_idents: #args_types),*) -> (#(#results_types),*) {
            let inputs = vec![
                #(#args_idents),*
            ];
            let output = self.0.pin_mut().call(server_keyset, inputs);
            let [#(#results_idents),*] = <[::concrete::UniquePtr<::concrete::common::TransportValue>; #output_len]>::try_from(output).unwrap();
            (
                #(#results_idents),*
            )
        }
    }
}
