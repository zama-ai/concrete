use concrete::protocol::{CircuitInfo, GateInfo, ProgramInfo};
use concrete::tfhe::{FunctionSpec, IntegerType};
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
        .map(|ci| {
            (
                ci,
                program_info
                    .tfhers_specs
                    .as_ref()
                    .map(|s| s.get_func(&ci.name).unwrap()),
            )
        })
        .map(|(ci, ts)| generate_client_function(ci, ts));
    quote! {
        pub mod client {
            #(#client_functions)*
        }
    }
}

fn generate_types(gi: &GateInfo, ts: &Option<IntegerType>) -> proc_macro2::TokenStream {
    match (ts, gi.rawInfo.integerPrecision, gi.rawInfo.isSigned) {
        (
            Some(IntegerType {
                bit_width: 2,
                is_signed: true,
                ..
            }),
            _,
            _,
        ) => quote! {::tfhe::FheInt2},
        (
            Some(IntegerType {
                bit_width: 2,
                is_signed: false,
                ..
            }),
            _,
            _,
        ) => quote! {::tfhe::FheUint2},
        (
            Some(IntegerType {
                bit_width: 4,
                is_signed: true,
                ..
            }),
            _,
            _,
        ) => quote! {::tfhe::FheInt4},
        (
            Some(IntegerType {
                bit_width: 4,
                is_signed: false,
                ..
            }),
            _,
            _,
        ) => quote! {::tfhe::FheUint4},
        (
            Some(IntegerType {
                bit_width: 6,
                is_signed: true,
                ..
            }),
            _,
            _,
        ) => quote! {::tfhe::FheInt6},
        (
            Some(IntegerType {
                bit_width: 6,
                is_signed: false,
                ..
            }),
            _,
            _,
        ) => quote! {::tfhe::FheUint6},
        (
            Some(IntegerType {
                bit_width: 8,
                is_signed: true,
                ..
            }),
            _,
            _,
        ) => quote! {::tfhe::FheInt8},
        (
            Some(IntegerType {
                bit_width: 8,
                is_signed: false,
                ..
            }),
            _,
            _,
        ) => quote! {::tfhe::FheUint8},
        (
            Some(IntegerType {
                bit_width: 10,
                is_signed: true,
                ..
            }),
            _,
            _,
        ) => quote! {::tfhe::FheInt10},
        (
            Some(IntegerType {
                bit_width: 10,
                is_signed: false,
                ..
            }),
            _,
            _,
        ) => quote! {::tfhe::FheUint10},
        (
            Some(IntegerType {
                bit_width: 12,
                is_signed: true,
                ..
            }),
            _,
            _,
        ) => quote! {::tfhe::FheInt12},
        (
            Some(IntegerType {
                bit_width: 12,
                is_signed: false,
                ..
            }),
            _,
            _,
        ) => quote! {::tfhe::FheUint12},
        (
            Some(IntegerType {
                bit_width: 14,
                is_signed: true,
                ..
            }),
            _,
            _,
        ) => quote! {::tfhe::FheInt14},
        (
            Some(IntegerType {
                bit_width: 14,
                is_signed: false,
                ..
            }),
            _,
            _,
        ) => quote! {::tfhe::FheUint14},
        (
            Some(IntegerType {
                bit_width: 16,
                is_signed: true,
                ..
            }),
            _,
            _,
        ) => quote! {::tfhe::FheInt16},
        (
            Some(IntegerType {
                bit_width: 16,
                is_signed: false,
                ..
            }),
            _,
            _,
        ) => quote! {::tfhe::FheUint16},
        (_, 8, true) => quote! {::concrete::common::Tensor<i8>},
        (_, 8, false) => quote! {::concrete::common::Tensor<u8>},
        (_, 16, true) => quote! {::concrete::common::Tensor<i16>},
        (_, 16, false) => quote! {::concrete::common::Tensor<u16>},
        (_, 32, true) => quote! {::concrete::common::Tensor<i32>},
        (_, 32, false) => quote! {::concrete::common::Tensor<u32>},
        (_, 64, true) => quote! {::concrete::common::Tensor<i64>},
        (_, 64, false) => quote! {::concrete::common::Tensor<u64>},
        _ => unreachable!(),
    }
}

fn generate_client_function_prepare_inputs(
    circuit_info: &CircuitInfo,
    tfhers_spec: Option<FunctionSpec>,
) -> proc_macro2::TokenStream {
    let ith = (0..circuit_info.inputs.len()).collect::<Vec<_>>();
    let input_specs = tfhers_spec.map_or(vec![None; circuit_info.inputs.len()], |v| {
        v.input_types.to_owned()
    });
    let input_idents = circuit_info
        .inputs
        .iter()
        .enumerate()
        .map(|(ith, _)| format_ident!("arg_{ith}"))
        .collect::<Vec<_>>();
    let input_types = circuit_info
        .inputs
        .iter()
        .zip(input_specs.iter())
        .map(|(gi, ts)| generate_types(gi, ts))
        .collect::<Vec<_>>();
    let output_types = circuit_info
        .inputs
        .iter()
        .map(|_| {
            quote! {::concrete::UniquePtr<::concrete::common::TransportValue>}
        })
        .collect::<Vec<_>>();
    quote! {
        pub fn prepare_inputs(&mut self, #(#input_idents: #input_types),*) -> (#(#output_types),*) {
            (
                #(
                    self.0.pin_mut().prepare_input(<#input_types as ::concrete::utils::into_value::IntoValue>::into_value(#input_idents), #ith)
                ),*
            )
        }
    }
}

fn generate_client_function_process_outputs(
    circuit_info: &CircuitInfo,
    tfhers_spec: Option<FunctionSpec>,
) -> proc_macro2::TokenStream {
    let ith = (0..circuit_info.outputs.len()).collect::<Vec<_>>();
    let output_specs = tfhers_spec.map_or(vec![None; circuit_info.outputs.len()], |v| {
        v.output_types.to_owned()
    });
    let input_idents = circuit_info
        .outputs
        .iter()
        .enumerate()
        .map(|(ith, _)| format_ident!("res_{ith}"))
        .collect::<Vec<_>>();
    let input_types = circuit_info
        .outputs
        .iter()
        .map(|_| {
            quote! {::concrete::UniquePtr<::concrete::common::TransportValue>}
        })
        .collect::<Vec<_>>();
    let output_types = circuit_info
        .outputs
        .iter()
        .zip(output_specs.iter())
        .map(|(gi, ts)| generate_types(gi, ts))
        .collect::<Vec<_>>();
    let output_specs = output_specs.iter()
        .map(|a| match a {
            Some(v) => quote! {#v},
            None => quote! { () },
        }).collect::<Vec<_>>();
    quote! {
        pub fn process_outputs(&mut self, #(#input_idents: #input_types),*) -> (#(#output_types),*) {
            (
                #(
                    <#output_types as ::concrete::utils::from_value::FromValue>::from_value(#output_specs, self.0.pin_mut().process_output(#input_idents, #ith))
                ),*
            )
        }
    }
}

fn generate_client_function(
    circuit_info: &CircuitInfo,
    tfhers_spec: Option<FunctionSpec>,
) -> proc_macro2::TokenStream {
    let function_identifier = format_ident!("{}", circuit_info.name);

    let prepare_inputs = generate_client_function_prepare_inputs(circuit_info, tfhers_spec.clone());
    let process_outputs =
        generate_client_function_process_outputs(circuit_info, tfhers_spec.clone());

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
    let invoke = generate_server_function_invoke(circuit_info);

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
    let args_idents = (0..circuit_info.inputs.len())
        .map(|a| format_ident!("arg_{a}"))
        .collect::<Vec<_>>();
    let args_types = (0..circuit_info.inputs.len())
        .map(|_| quote! {::concrete::UniquePtr<::concrete::common::TransportValue>})
        .collect::<Vec<_>>();
    let results_idents = (0..circuit_info.outputs.len())
        .map(|a| format_ident!("res_{a}"))
        .collect::<Vec<_>>();
    let results_types = (0..circuit_info.outputs.len())
        .map(|_| quote! {::concrete::UniquePtr<::concrete::common::TransportValue>})
        .collect::<Vec<_>>();
    let output_len = circuit_info.outputs.len();

    quote! {
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
