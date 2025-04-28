use concrete::protocol::{CircuitInfo, ProgramInfo, TypeInfo};
use concrete::tfhe::{FunctionSpec, IntegerType};
use itertools::multizip;
use quote::{format_ident, quote};

pub fn generate(pi: &ProgramInfo, hash: u64) -> proc_macro2::TokenStream {
    let lib_name = format!("concrete-artifact-{hash}");
    let unsafe_binding = generate_unsafe_binding(&pi);
    let infos = generate_infos(&pi);
    let keyset = generate_keyset(&pi);
    let client = generate_client(&pi);
    let server = generate_server(&pi);

    let links = if cfg!(target_os = "macos") {
        quote! {
            #[link(name = "ConcretelangRuntime")]
            #[link(name = "omp")]
        }
    } else if cfg!(target_os = "linux") {
        quote! {
            #[link(name = "ConcretelangRuntime")]
            #[link(name = "hpx_iostreams")]
            #[link(name = "hpx_core")]
            #[link(name = "hpx")]
            #[link(name = "omp")]
        }
    } else {
        panic!("Unsupported platform");
    };

    quote! {
        #infos
        #keyset
        #client
        #server

        #[doc(hidden)]
        pub mod _binding {
            #[link(name = #lib_name, kind="static")]
            #links
            #unsafe_binding
        }
    }
}

fn generate_unsafe_binding(pi: &ProgramInfo) -> proc_macro2::TokenStream {
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

fn generate_infos(pi: &ProgramInfo) -> proc_macro2::TokenStream {
    quote! {
        pub static PROGRAM_INFO: std::sync::LazyLock<::concrete::protocol::ProgramInfo> = std::sync::LazyLock::new(|| {
            #pi
        });
    }
}

fn generate_keyset(pi: &ProgramInfo) -> proc_macro2::TokenStream {
    let mut need_providing = pi
        .circuits
        .iter()
        .flat_map(|ci| {
            multizip((
                std::iter::repeat(&ci.name),
                std::iter::successors(Some(0), |a| Some(a + 1)),
                ci.inputs.iter(),
                pi.tfhers_specs
                    .as_ref()
                    .unwrap()
                    .get_func(&ci.name)
                    .unwrap()
                    .input_types
                    .iter(),
            ))
        })
        .filter(|(_, _, _, spec)| spec.is_some())
        .collect::<Vec<_>>();

    need_providing.dedup_by_key(|a| {
        let TypeInfo::lweCiphertext(ref ct) = a.2.typeInfo else {
            unreachable!()
        };
        ct.encryption.keyId
    });

    let fields_idents = need_providing
        .iter()
        .map(|(func_name, ith, ..)| format_ident!("{func_name}_{ith}"))
        .collect::<Vec<_>>();

    let fields_types = need_providing
        .iter()
        .map(|_| {
            quote! { Option<concrete::UniquePtr<concrete::common::LweSecretKey>>}
        })
        .collect::<Vec<_>>();

    let methods = need_providing.iter().map(|(func_name, ith, gi, ..)| {
        let method_name = format_ident!("with_key_for_{func_name}_{ith}_arg");
        let ident = format_ident!("{func_name}_{ith}");
        let TypeInfo::lweCiphertext(ref ct) = gi.typeInfo else {
            unreachable!()
        };
        let kid = ct.encryption.keyId;
        quote! {
            pub fn #method_name(mut self, key: &::tfhe::ClientKey) -> Self {
                let mut key = Some(<::tfhe::ClientKey as concrete::tfhe::IntoLweSecretKey>::into_lwe_secret_key(key, Some(#kid)));
                if self.#ident.is_some() {
                    assert_eq!(self.#ident.as_mut().unwrap().pin_mut().get_buffer(), key.as_mut().unwrap().pin_mut().get_buffer(), "Tried to set the same underlying key twice, with a different key. Something must be wrong...");
                    return self;
                }
                self.#ident = key;
                self
            }
        }
    });

    quote! {
        #[derive(Default)]
        pub struct KeysetBuilder{
            #(#fields_idents: #fields_types),*
        }

        impl KeysetBuilder {
            pub fn new() -> Self {
                return Self::default();
            }

            #(#methods),*

            pub fn generate(
                self,
                secret_csprng: std::pin::Pin<&mut ::concrete::common::SecretCsprng>,
                encryption_csprng: std::pin::Pin<&mut ::concrete::common::EncryptionCsprng>
            ) -> ::concrete::UniquePtr<::concrete::common::Keyset> {
                ::concrete::common::Keyset::new(
                    &PROGRAM_INFO.keyset,
                    secret_csprng,
                    encryption_csprng,
                    vec![
                        #(
                            self.#fields_idents.expect(concat!("Missing tfhers key ", stringify!(#fields_idents)))
                        ),*
                    ]
                )
            }
        }
    }
}

fn generate_client(program_info: &ProgramInfo) -> proc_macro2::TokenStream {
    let client_functions = program_info
        .circuits
        .iter()
        .map(|ci| {
            (
                ci,
                program_info
                    .tfhers_specs
                    .as_ref()
                    .unwrap()
                    .get_func(&ci.name)
                    .unwrap(),
            )
        })
        .map(|(ci, ts)| generate_client_function(ci, Some(ts)));
    quote! {
        pub mod client {
            #(#client_functions)*
        }
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
    let input_types = multizip((input_specs.iter(), circuit_info.inputs.iter()))
        .map(|(spec, gate_info)| {
            match (
                spec,
                gate_info.rawInfo.integerPrecision,
                gate_info.rawInfo.isSigned,
            ) {
                (Some(_), _, _) => quote! {()},
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
        })
        .collect::<Vec<_>>();
    let output_types = input_specs
        .iter()
        .map(|spec| match spec {
            Some(_) => quote! {()},
            None => quote! {::concrete::UniquePtr<::concrete::common::TransportValue>},
        })
        .collect::<Vec<_>>();
    let preparations = multizip((input_specs.iter(), input_types.iter(), input_idents.iter(), ith.iter()))
        .map(|(spec, typ, ident, ith)|{
            match spec {
                Some(..) => quote!{()},
                None => quote!{self.0.pin_mut().prepare_input(<#typ as ::concrete::utils::into_value::IntoValue>::into_value(#ident), #ith)}
            }
        });
    quote! {
        pub fn prepare_inputs(&mut self, #(#input_idents: #input_types),*) -> (#(#output_types),*) {
            (#(#preparations),*)
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
    let input_types = output_specs
        .iter()
        .map(|spec| match spec {
            Some(_) => quote! {()},
            None => quote! {::concrete::UniquePtr<::concrete::common::TransportValue>},
        })
        .collect::<Vec<_>>();
    let output_types = multizip((circuit_info.outputs.iter(), output_specs.iter()))
        .map(
            |(gi, ts)| match (ts, gi.rawInfo.integerPrecision, gi.rawInfo.isSigned) {
                (Some(_), _, _) => quote! {()},
                (_, 8, true) => quote! {::concrete::common::Tensor<i8>},
                (_, 8, false) => quote! {::concrete::common::Tensor<u8>},
                (_, 16, true) => quote! {::concrete::common::Tensor<i16>},
                (_, 16, false) => quote! {::concrete::common::Tensor<u16>},
                (_, 32, true) => quote! {::concrete::common::Tensor<i32>},
                (_, 32, false) => quote! {::concrete::common::Tensor<u32>},
                (_, 64, true) => quote! {::concrete::common::Tensor<i64>},
                (_, 64, false) => quote! {::concrete::common::Tensor<u64>},
                _ => unreachable!(),
            },
        )
        .collect::<Vec<_>>();
    let unwrappers = multizip((output_specs.iter(), output_types.iter(), input_idents.iter(), ith.iter()))
        .map(|(spec, typ, ident, ith)|{
            match spec {
                Some(_) => quote!{()},
                None => quote!{<#typ as ::concrete::utils::from_value::FromValue>::from_value((), self.0.pin_mut().process_output(#ident, #ith))},
            }
        });
    quote! {
        pub fn process_outputs(&mut self, #(#input_idents: #input_types),*) -> (#(#output_types),*) {
            (#(#unwrappers),*)
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

fn generate_server(program_info: &ProgramInfo) -> proc_macro2::TokenStream {
    let server_functions = program_info
        .circuits
        .iter()
        .map(|ci| {
            (
                ci,
                program_info
                    .tfhers_specs
                    .as_ref()
                    .unwrap()
                    .get_func(&ci.name)
                    .unwrap(),
            )
        })
        .map(|(ci, spec)| generate_server_function(ci, Some(spec)));
    quote! {
        pub mod server {
            #(#server_functions)*
        }
    }
}

fn generate_server_function(
    circuit_info: &CircuitInfo,
    tfhers_spec: Option<FunctionSpec>,
) -> proc_macro2::TokenStream {
    let function_identifier = format_ident!("{}", circuit_info.name);
    let binding_identifier = format_ident!("_mlir_concrete_{}", circuit_info.name);
    let invoke = generate_server_function_invoke(circuit_info, tfhers_spec);

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

fn generate_types(ts: &Option<IntegerType>) -> proc_macro2::TokenStream {
    match ts {
        Some(IntegerType {
            bit_width: 2,
            is_signed: true,
            ..
        }) => quote! {::tfhe::FheInt2},
        Some(IntegerType {
            bit_width: 2,
            is_signed: false,
            ..
        }) => quote! {::tfhe::FheUint2},
        Some(IntegerType {
            bit_width: 4,
            is_signed: true,
            ..
        }) => quote! {::tfhe::FheInt4},
        Some(IntegerType {
            bit_width: 4,
            is_signed: false,
            ..
        }) => quote! {::tfhe::FheUint4},
        Some(IntegerType {
            bit_width: 6,
            is_signed: true,
            ..
        }) => quote! {::tfhe::FheInt6},
        Some(IntegerType {
            bit_width: 6,
            is_signed: false,
            ..
        }) => quote! {::tfhe::FheUint6},
        Some(IntegerType {
            bit_width: 8,
            is_signed: true,
            ..
        }) => quote! {::tfhe::FheInt8},
        Some(IntegerType {
            bit_width: 8,
            is_signed: false,
            ..
        }) => quote! {::tfhe::FheUint8},
        Some(IntegerType {
            bit_width: 10,
            is_signed: true,
            ..
        }) => quote! {::tfhe::FheInt10},
        Some(IntegerType {
            bit_width: 10,
            is_signed: false,
            ..
        }) => quote! {::tfhe::FheUint10},
        Some(IntegerType {
            bit_width: 12,
            is_signed: true,
            ..
        }) => quote! {::tfhe::FheInt12},
        Some(IntegerType {
            bit_width: 12,
            is_signed: false,
            ..
        }) => quote! {::tfhe::FheUint12},
        Some(IntegerType {
            bit_width: 14,
            is_signed: true,
            ..
        }) => quote! {::tfhe::FheInt14},
        Some(IntegerType {
            bit_width: 14,
            is_signed: false,
            ..
        }) => quote! {::tfhe::FheUint14},
        Some(IntegerType {
            bit_width: 16,
            is_signed: true,
            ..
        }) => quote! {::tfhe::FheInt16},
        Some(IntegerType {
            bit_width: 16,
            is_signed: false,
            ..
        }) => quote! {::tfhe::FheUint16},
        None => quote! {::concrete::UniquePtr<::concrete::common::TransportValue>},
        _ => unreachable!(),
    }
}

fn generate_server_function_invoke(
    circuit_info: &CircuitInfo,
    tfhers_spec: Option<FunctionSpec>,
) -> proc_macro2::TokenStream {
    let input_specs = tfhers_spec
        .clone()
        .map_or(vec![None; circuit_info.inputs.len()], |v| {
            v.input_types.to_owned()
        });
    let output_specs = tfhers_spec
        .clone()
        .map_or(vec![None; circuit_info.outputs.len()], |v| {
            v.output_types.to_owned()
        });
    let args_idents = (0..circuit_info.inputs.len())
        .map(|a| format_ident!("arg_{a}"))
        .collect::<Vec<_>>();
    let args_types = input_specs
        .iter()
        .map(|s| generate_types(s))
        .collect::<Vec<_>>();
    let results_idents = (0..circuit_info.outputs.len())
        .map(|a| format_ident!("res_{a}"))
        .collect::<Vec<_>>();
    let results_types = output_specs
        .iter()
        .map(|s| generate_types(s))
        .collect::<Vec<_>>();
    let output_len = circuit_info.outputs.len();

    let preludes = multizip((input_specs.iter(), args_idents.iter(), args_types.iter(), circuit_info.inputs.iter()))
        .map(|(spec, ident, typ, gi)| {
            match spec{
                Some(_) => {
                    let type_info_json_string = serde_json::to_string(&gi.typeInfo).unwrap();
                    quote!{ <#typ as ::concrete::utils::into_value::IntoValue>::into_value(#ident).into_transport_value(#type_info_json_string)}
                }
                None => quote!{#ident}
            }
        });

    let postludes = multizip((output_specs.iter(), results_idents.iter(), results_types.iter()))
        .map(|(spec, ident, typ)|{
            match spec {
                Some(s) => quote!{ <#typ as ::concrete::utils::from_value::FromValue>::from_value(#s, #ident.to_value())},
                None => quote!{#ident}
            }
        });

    quote! {
        pub fn invoke(&mut self, server_keyset: &::concrete::common::ServerKeyset, #(#args_idents: #args_types),*) -> (#(#results_types),*) {
            let inputs = vec![#(#preludes),*];
            let output = self.0.pin_mut().call(server_keyset, inputs);
            let [#(#results_idents),*] = <[::concrete::UniquePtr<::concrete::common::TransportValue>; #output_len]>::try_from(output).unwrap();
            (#(#postludes),*)
        }
    }
}
