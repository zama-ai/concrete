#![allow(unused_imports, unused)]
use std::any::Any;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::pin::Pin;

use crate::protocol::{
    CircuitInfo, KeysetInfo, LweBootstrapKeyInfo, LweKeyswitchKeyInfo, LweSecretKeyInfo,
    PackingKeyswitchKeyInfo, ProgramInfo,
};
use crate::utils::into_value::IntoValue;
use cxx::{CxxVector, SharedPtr, UniquePtr};

#[cxx::bridge(namespace = "concrete_rust")]
mod ffi {

    unsafe extern "C++" {
        include!("ffi.h");
        type c_void;

        // ------------------------------------------------------------------------------------------- Compiler

        /// Holds different flags and options of the compilation process.
        type CompilationOptions;

        #[doc(hidden)]
        fn _compilation_options_new() -> UniquePtr<CompilationOptions>;

        /// Set display flag of optimizer choices.
        fn set_display_optimizer_choice(self: Pin<&mut CompilationOptions>, val: bool);

        /// Set option for loop parallelization.
        fn set_loop_parallelize(self: Pin<&mut CompilationOptions>, val: bool);

        /// Set option for dataflow parallelization.
        fn set_dataflow_parallelize(self: Pin<&mut CompilationOptions>, val: bool);

        /// Set option for auto parallelization.
        fn set_auto_parallelize(self: Pin<&mut CompilationOptions>, val: bool);

        /// Set option for compression of evaluation keys.
        fn set_compress_evaluation_keys(self: Pin<&mut CompilationOptions>, val: bool);

        /// Set option for compression of input ciphertexts.
        fn set_compress_input_ciphertexts(self: Pin<&mut CompilationOptions>, val: bool);

        /// Set error probability for shared by each pbs.
        fn set_p_error(self: Pin<&mut CompilationOptions>, val: f64);

        /// Set global error probability for the full circuit.
        fn set_global_p_error(self: Pin<&mut CompilationOptions>, val: f64);

        /// Set the strategy of the optimizer.
        fn set_optimizer_strategy(self: Pin<&mut CompilationOptions>, val: u8);

        /// Set the strategy of the optimizer for multi-parameter.
        fn set_optimizer_multi_parameter_strategy(self: Pin<&mut CompilationOptions>, val: u8);

        /// Enable or disable tlu fusing.
        fn set_enable_tlu_fusing(self: Pin<&mut CompilationOptions>, val: bool);

        /// Enable or disable printing tlu fusing.
        fn set_print_tlu_fusing(self: Pin<&mut CompilationOptions>, val: bool);

        /// Enable or disable simulation.
        fn set_simulate(self: Pin<&mut CompilationOptions>, val: bool);

        /// Enable or disable overflow detection during simulation.
        fn set_enable_overflow_detection_in_simulation(
            self: Pin<&mut CompilationOptions>,
            val: bool,
        );

        /// Set composable flag.
        fn set_composable(self: Pin<&mut CompilationOptions>, val: bool);

        /// Add a range restriction.
        fn set_range_restriction(self: Pin<&mut CompilationOptions>, json: &str);

        /// Add a keyset restriction.
        fn set_keyset_restriction(self: Pin<&mut CompilationOptions>, json: &str);

        /// Set security level.
        fn set_security_level(self: Pin<&mut CompilationOptions>, val: u64);

        /// Add a composition rule.
        fn add_composition_rule(
            self: Pin<&mut CompilationOptions>,
            from_func: &str,
            from_pos: usize,
            to_func: &str,
            to_pos: usize,
        );

        /// Library object representing the output of a compilation.
        type Library;
        /// Return the path to the static library.
        fn get_static_library_path(self: &Library) -> String;
        #[doc(hidden)]
        fn _get_program_info_json(self: &Library) -> String;

        /// Compile the `mlir` source string to a static library available at `output_dir_path` using the `options`.
        fn compile(
            sources: &str,
            options: &CompilationOptions,
            output_dir_path: &str,
        ) -> Result<UniquePtr<Library>>;

        // ------------------------------------------------------------------------------------------- Commons

        /// A CSPRNG used to generate keys.
        type SecretCsprng;
        #[doc(hidden)]
        fn _secret_csprng_new(high: u64, low: u64) -> UniquePtr<SecretCsprng>;

        /// A CSPRNG used to encrypt ciphertexts.
        type EncryptionCsprng;
        #[doc(hidden)]
        fn _encryption_csprng_new(high: u64, low: u64) -> UniquePtr<EncryptionCsprng>;

        /// Represents an LWE Bootstrap Key.
        type LweBootstrapKey;
        /// Get the buffer of the LWE Bootstrap Key.
        fn get_buffer(self: Pin<&mut LweBootstrapKey>) -> &[u64];
        #[doc(hidden)]
        fn _get_info_json(self: &LweBootstrapKey) -> String;

        /// Represents an LWE Keyswitch Key.
        type LweKeyswitchKey;
        /// Get the buffer of the LWE Keyswitch Key.
        fn get_buffer(self: Pin<&mut LweKeyswitchKey>) -> &[u64];
        #[doc(hidden)]
        fn _get_info_json(self: &LweKeyswitchKey) -> String;

        /// Represents a Packing Keyswitch Key.
        type PackingKeyswitchKey;
        /// Get the buffer of the Packing Keyswitch Key.
        fn get_buffer(self: Pin<&mut PackingKeyswitchKey>) -> &[u64];
        #[doc(hidden)]
        fn _get_info_json(self: &PackingKeyswitchKey) -> String;

        /// Represents an LWE Secret Key.
        type LweSecretKey;
        /// Get the buffer of the LWE Secret Key.
        fn get_buffer(self: Pin<&mut LweSecretKey>) -> &[u64];
        #[doc(hidden)]
        fn _get_info_json(self: &LweSecretKey) -> String;
        #[doc(hidden)]
        fn _lwe_secret_key_from_buffer_and_info(
            buffer: &[u64],
            info: &str,
        ) -> UniquePtr<LweSecretKey>;

        /// A Keyset object holding both the [`ClientKeyset`]  and the [`ServerKeyset`].
        type Keyset;
        #[doc(hidden)]
        fn _keyset_new(
            keyset_info_json: &str,
            secret_csprng: Pin<&mut SecretCsprng>,
            encryption_csprng: Pin<&mut EncryptionCsprng>,
            initial_keys: &mut [UniquePtr<LweSecretKey>],
        ) -> UniquePtr<Keyset>;
        /// Return the associated server keyset.
        fn get_server(self: &Keyset) -> UniquePtr<ServerKeyset>;
        /// Return the associated client keyset.
        fn get_client(self: &Keyset) -> UniquePtr<ClientKeyset>;

        /// A server keyset holding the keys necessary to __execute__ an FHE program on encrypted data.
        ///
        /// Note:
        /// -----
        /// This is public material, and as such can be sent safely to the execution platform.
        type ServerKeyset;
        #[doc(hidden)]
        fn _lwe_bootstrap_keys_len(self: &ServerKeyset) -> usize;
        #[doc(hidden)]
        fn _lwe_bootstrap_keys_nth(self: &ServerKeyset, nth: usize) -> &LweBootstrapKey;
        #[doc(hidden)]
        fn _lwe_keyswitch_keys_len(self: &ServerKeyset) -> usize;
        #[doc(hidden)]
        fn _lwe_keyswitch_keys_nth(self: &ServerKeyset, nth: usize) -> &LweKeyswitchKey;
        #[doc(hidden)]
        fn _packing_keyswitch_keys_len(self: &ServerKeyset) -> usize;
        #[doc(hidden)]
        fn _packing_keyswitch_keys_nth(self: &ServerKeyset, nth: usize) -> &PackingKeyswitchKey;
        /// Serialize a server keyset to bytes.
        fn serialize(self: &ServerKeyset) -> Vec<u8>;
        #[doc(hidden)]
        fn _deserialize_server_keyset(bytes: &[u8]) -> UniquePtr<ServerKeyset>;

        /// A client keyset holding the keys necessary to __encrypt__ input data (and decrypt outputs).
        ///
        /// Warning:
        /// --------
        /// This is private material, and as such SHOULD NOT BE SHARED with untrusted third-party.
        type ClientKeyset;
        #[doc(hidden)]
        fn _lwe_secret_keys_len(self: &ClientKeyset) -> usize;
        #[doc(hidden)]
        fn _lwe_secret_keys_nth(self: &ClientKeyset, nth: usize) -> &LweSecretKey;
        /// Serialize a client keyset to bytes.
        fn serialize(self: &ClientKeyset) -> Vec<u8>;
        #[doc(hidden)]
        fn _deserialize_client_keyset(bytes: &[u8]) -> UniquePtr<ClientKeyset>;

        #[doc(hidden)]
        type TensorU8;
        #[doc(hidden)]
        fn _tensor_u8_new(values: &[u8], dimensions: &[usize]) -> UniquePtr<TensorU8>;
        #[doc(hidden)]
        fn _get_dimensions(self: &TensorU8) -> &[usize];
        #[doc(hidden)]
        fn _get_values(self: &TensorU8) -> &[u8];

        #[doc(hidden)]
        type TensorU16;
        #[doc(hidden)]
        fn _tensor_u16_new(values: &[u16], dimensions: &[usize]) -> UniquePtr<TensorU16>;
        #[doc(hidden)]
        fn _get_dimensions(self: &TensorU16) -> &[usize];
        #[doc(hidden)]
        fn _get_values(self: &TensorU16) -> &[u16];

        #[doc(hidden)]
        type TensorU32;
        #[doc(hidden)]
        fn _tensor_u32_new(values: &[u32], dimensions: &[usize]) -> UniquePtr<TensorU32>;
        #[doc(hidden)]
        fn _get_dimensions(self: &TensorU32) -> &[usize];
        #[doc(hidden)]
        fn _get_values(self: &TensorU32) -> &[u32];

        #[doc(hidden)]
        type TensorU64;
        #[doc(hidden)]
        fn _tensor_u64_new(values: &[u64], dimensions: &[usize]) -> UniquePtr<TensorU64>;
        #[doc(hidden)]
        fn _get_dimensions(self: &TensorU64) -> &[usize];
        #[doc(hidden)]
        fn _get_values(self: &TensorU64) -> &[u64];

        #[doc(hidden)]
        type TensorI8;
        #[doc(hidden)]
        fn _tensor_i8_new(values: &[i8], dimensions: &[usize]) -> UniquePtr<TensorI8>;
        #[doc(hidden)]
        fn _get_dimensions(self: &TensorI8) -> &[usize];
        #[doc(hidden)]
        fn _get_values(self: &TensorI8) -> &[i8];

        #[doc(hidden)]
        type TensorI16;
        #[doc(hidden)]
        fn _tensor_i16_new(values: &[i16], dimensions: &[usize]) -> UniquePtr<TensorI16>;
        #[doc(hidden)]
        fn _get_dimensions(self: &TensorI16) -> &[usize];
        #[doc(hidden)]
        fn _get_values(self: &TensorI16) -> &[i16];

        #[doc(hidden)]
        type TensorI32;
        #[doc(hidden)]
        fn _tensor_i32_new(values: &[i32], dimensions: &[usize]) -> UniquePtr<TensorI32>;
        #[doc(hidden)]
        fn _get_dimensions(self: &TensorI32) -> &[usize];
        #[doc(hidden)]
        fn _get_values(self: &TensorI32) -> &[i32];

        #[doc(hidden)]
        type TensorI64;
        #[doc(hidden)]
        fn _tensor_i64_new(values: &[i64], dimensions: &[usize]) -> UniquePtr<TensorI64>;
        #[doc(hidden)]
        fn _get_dimensions(self: &TensorI64) -> &[usize];
        #[doc(hidden)]
        fn _get_values(self: &TensorI64) -> &[i64];

        /// A generic value type carrying a `Tensor` of elements `(u|i)(8|16|32|64)`.
        type Value;
        #[doc(hidden)]
        fn _value_from_tensor_u8(tensor: UniquePtr<TensorU8>) -> UniquePtr<Value>;
        #[doc(hidden)]
        fn _value_from_tensor_u16(tensor: UniquePtr<TensorU16>) -> UniquePtr<Value>;
        #[doc(hidden)]
        fn _value_from_tensor_u32(tensor: UniquePtr<TensorU32>) -> UniquePtr<Value>;
        #[doc(hidden)]
        fn _value_from_tensor_u64(tensor: UniquePtr<TensorU64>) -> UniquePtr<Value>;
        #[doc(hidden)]
        fn _value_from_tensor_i8(tensor: UniquePtr<TensorI8>) -> UniquePtr<Value>;
        #[doc(hidden)]
        fn _value_from_tensor_i16(tensor: UniquePtr<TensorI16>) -> UniquePtr<Value>;
        #[doc(hidden)]
        fn _value_from_tensor_i32(tensor: UniquePtr<TensorI32>) -> UniquePtr<Value>;
        #[doc(hidden)]
        fn _value_from_tensor_i64(tensor: UniquePtr<TensorI64>) -> UniquePtr<Value>;
        #[doc(hidden)]
        fn _has_element_type_u8(self: &Value) -> bool;
        #[doc(hidden)]
        fn _has_element_type_u16(self: &Value) -> bool;
        #[doc(hidden)]
        fn _has_element_type_u32(self: &Value) -> bool;
        #[doc(hidden)]
        fn _has_element_type_u64(self: &Value) -> bool;
        #[doc(hidden)]
        fn _has_element_type_i8(self: &Value) -> bool;
        #[doc(hidden)]
        fn _has_element_type_i16(self: &Value) -> bool;
        #[doc(hidden)]
        fn _has_element_type_i32(self: &Value) -> bool;
        #[doc(hidden)]
        fn _has_element_type_i64(self: &Value) -> bool;
        #[doc(hidden)]
        fn _get_tensor_u8(self: &Value) -> UniquePtr<TensorU8>;
        #[doc(hidden)]
        fn _get_tensor_u16(self: &Value) -> UniquePtr<TensorU16>;
        #[doc(hidden)]
        fn _get_tensor_u32(self: &Value) -> UniquePtr<TensorU32>;
        #[doc(hidden)]
        fn _get_tensor_u64(self: &Value) -> UniquePtr<TensorU64>;
        #[doc(hidden)]
        fn _get_tensor_i8(self: &Value) -> UniquePtr<TensorI8>;
        #[doc(hidden)]
        fn _get_tensor_i16(self: &Value) -> UniquePtr<TensorI16>;
        #[doc(hidden)]
        fn _get_tensor_i32(self: &Value) -> UniquePtr<TensorI32>;
        #[doc(hidden)]
        fn _get_tensor_i64(self: &Value) -> UniquePtr<TensorI64>;
        fn get_dimensions(self: &Value) -> &[usize];
        fn into_transport_value(self: &Value, type_info_json: &str) -> UniquePtr<TransportValue>;

        /// A serialized value which can be transported between the server and the client.
        type TransportValue;
        /// Build an owned copy from a reference.
        fn to_owned(self: &TransportValue) -> UniquePtr<TransportValue>;
        /// Serialize a transport value to bytes.
        fn serialize(self: &TransportValue) -> Vec<u8>;
        #[doc(hidden)]
        fn _deserialize_transport_value(bytes: &[u8]) -> UniquePtr<TransportValue>;
        #[doc(hidden)]
        fn _transport_value_to_value(tv: &TransportValue) -> UniquePtr<Value>;

        // ------------------------------------------------------------------------------------------- Client

        /// Client-side interface to an FHE module.
        type ClientModule;
        #[doc(hidden)]
        fn _client_module_new_encrypted(
            program_info_json: &str,
            client_keyset: &ClientKeyset,
            encryption_prng: UniquePtr<EncryptionCsprng>,
        ) -> UniquePtr<ClientModule>;
        #[doc(hidden)]
        fn _client_module_new_simulated(
            program_info_json: &str,
            encryption_prng: UniquePtr<EncryptionCsprng>,
        ) -> UniquePtr<ClientModule>;
        #[doc(hidden)]
        fn _get_client_function(
            self: &ClientModule,
            name: &str,
        ) -> Result<UniquePtr<ClientFunction>>;

        /// Client-side interface to an FHE function.
        ///
        /// Note :
        /// ------
        /// Among other things, this object is responsible for encrypting the function inputs before they can be sent to the server.
        type ClientFunction;
        #[doc(hidden)]
        fn _client_function_new_encrypted(
            circuit_info_json: &str,
            client_keyset: &ClientKeyset,
            encryption_prng: UniquePtr<EncryptionCsprng>,
        ) -> UniquePtr<ClientFunction>;
        #[doc(hidden)]
        fn _client_function_new_simulated(
            circuit_info_json: &str,
            encryption_prng: UniquePtr<EncryptionCsprng>,
        ) -> UniquePtr<ClientFunction>;
        /// Prepare one function input to be sent to the server.
        ///
        /// Note:
        /// -----
        /// This include encoding -> encryption -> conversion to serializable value.
        fn prepare_input(
            self: Pin<&mut ClientFunction>,
            arg: UniquePtr<Value>,
            pos: usize,
        ) -> UniquePtr<TransportValue>;
        /// Process one function output retrieved from the server.
        ///
        /// Note:
        /// -----
        /// This include conversion from deserializable value -> decryption -> decoding.
        fn process_output(
            self: Pin<&mut ClientFunction>,
            result: UniquePtr<TransportValue>,
            pos: usize,
        ) -> UniquePtr<Value>;
        #[doc(hidden)]
        fn simulate_prepare_input(
            self: Pin<&mut ClientFunction>,
            arg: &Value,
            pos: usize,
        ) -> UniquePtr<TransportValue>;
        #[doc(hidden)]
        fn simulate_process_output(
            self: Pin<&mut ClientFunction>,
            result: &TransportValue,
            pos: usize,
        ) -> UniquePtr<Value>;

        // ------------------------------------------------------------------------------------------- Server
        //
        /// Server-side interface to an FHE function.
        ///
        /// Note:
        /// -----
        /// This object allows to invoke the FHE function on the encrypted inputs coming from the client.
        type ServerFunction;
        #[doc(hidden)]
        unsafe fn _server_function_new(
            circuit_info_json: &str,
            func: *mut c_void,
            use_simulation: bool,
        ) -> UniquePtr<ServerFunction>;
        #[doc(hidden)]
        fn _call(
            self: Pin<&mut ServerFunction>,
            keys: &ServerKeyset,
            args: &mut [UniquePtr<TransportValue>],
        ) -> UniquePtr<CxxVector<TransportValue>>;
        #[doc(hidden)]
        fn _simulate(
            self: Pin<&mut ServerFunction>,
            args: &mut [UniquePtr<TransportValue>],
        ) -> UniquePtr<CxxVector<TransportValue>>;
    }
}

pub use ffi::*;
use serde_json::map::IntoValues;

impl ServerKeyset {
    /// Deserialize a server keyset from bytes.
    pub fn deserialize(bytes: &[u8]) -> UniquePtr<ServerKeyset> {
        _deserialize_server_keyset(bytes)
    }
}

impl ClientKeyset {
    /// Deserialize a client keyset from bytes.
    pub fn deserialize(bytes: &[u8]) -> UniquePtr<ClientKeyset> {
        _deserialize_client_keyset(bytes)
    }
}

impl TransportValue {
    /// Deserialize a `TransportValue` from bytes.
    pub fn deserialize(bytes: &[u8]) -> UniquePtr<TransportValue> {
        _deserialize_transport_value(bytes)
    }

    pub fn to_value(&self) -> UniquePtr<Value> {
        _transport_value_to_value(self)
    }
}

impl ServerFunction {
    #[doc(hidden)]
    pub fn new(
        circuit_info: &CircuitInfo,
        func: *mut c_void,
        use_simulation: bool,
    ) -> UniquePtr<ServerFunction> {
        unsafe {
            _server_function_new(
                &serde_json::to_string(circuit_info).unwrap(),
                func,
                use_simulation,
            )
        }
    }

    /// Performs a call to the FHE function using the `keys` server keyset and the `args` arguments.
    pub fn call(
        self: Pin<&mut ServerFunction>,
        keys: &ServerKeyset,
        mut args: Vec<UniquePtr<TransportValue>>,
    ) -> Vec<UniquePtr<TransportValue>> {
        let output = self._call(keys, args.as_mut_slice());
        output.iter().map(|v| v.to_owned()).collect()
    }

    #[doc(hidden)]
    pub fn simulate(
        self: Pin<&mut ServerFunction>,
        mut args: Vec<UniquePtr<TransportValue>>,
    ) -> Vec<UniquePtr<TransportValue>> {
        let output = self._simulate(args.as_mut_slice());
        output.iter().map(|v| v.to_owned()).collect()
    }
}

pub trait FromElements {
    type Element;
    fn from_elements(elements: Vec<Self::Element>, dimensions: Vec<usize>) -> Self;
}

impl FromElements for Tensor<u8> {
    type Element = u8;
    fn from_elements(values: Vec<Self::Element>, dimensions: Vec<usize>) -> Tensor<u8> {
        Tensor {
            inner: InnerTensor::U8(_tensor_u8_new(&values, &dimensions)),
            phantom: PhantomData,
        }
    }
}

impl FromElements for Tensor<u16> {
    type Element = u16;
    fn from_elements(values: Vec<Self::Element>, dimensions: Vec<usize>) -> Tensor<u16> {
        Tensor {
            inner: InnerTensor::U16(_tensor_u16_new(&values, &dimensions)),
            phantom: PhantomData,
        }
    }
}

impl FromElements for Tensor<u32> {
    type Element = u32;
    fn from_elements(values: Vec<Self::Element>, dimensions: Vec<usize>) -> Tensor<u32> {
        Tensor {
            inner: InnerTensor::U32(_tensor_u32_new(&values, &dimensions)),
            phantom: PhantomData,
        }
    }
}

impl FromElements for Tensor<u64> {
    type Element = u64;
    fn from_elements(values: Vec<Self::Element>, dimensions: Vec<usize>) -> Tensor<u64> {
        Tensor {
            inner: InnerTensor::U64(_tensor_u64_new(&values, &dimensions)),
            phantom: PhantomData,
        }
    }
}

impl FromElements for Tensor<i8> {
    type Element = i8;
    fn from_elements(values: Vec<Self::Element>, dimensions: Vec<usize>) -> Tensor<i8> {
        Tensor {
            inner: InnerTensor::I8(_tensor_i8_new(&values, &dimensions)),
            phantom: PhantomData,
        }
    }
}

impl FromElements for Tensor<i16> {
    type Element = i16;
    fn from_elements(values: Vec<Self::Element>, dimensions: Vec<usize>) -> Tensor<i16> {
        Tensor {
            inner: InnerTensor::I16(_tensor_i16_new(&values, &dimensions)),
            phantom: PhantomData,
        }
    }
}

impl FromElements for Tensor<i32> {
    type Element = i32;
    fn from_elements(values: Vec<Self::Element>, dimensions: Vec<usize>) -> Tensor<i32> {
        Tensor {
            inner: InnerTensor::I32(_tensor_i32_new(&values, &dimensions)),
            phantom: PhantomData,
        }
    }
}

impl FromElements for Tensor<i64> {
    type Element = i64;
    fn from_elements(values: Vec<Self::Element>, dimensions: Vec<usize>) -> Tensor<i64> {
        Tensor {
            inner: InnerTensor::I64(_tensor_i64_new(&values, &dimensions)),
            phantom: PhantomData,
        }
    }
}

pub trait GetValues {
    type Element;
    fn get_values(&self) -> &[Self::Element];
}

impl GetValues for Tensor<u8> {
    type Element = u8;
    fn get_values(&self) -> &[Self::Element] {
        let InnerTensor::U8(inner) = &self.inner else {
            unreachable!()
        };
        inner._get_values()
    }
}

impl GetValues for Tensor<u16> {
    type Element = u16;
    fn get_values(&self) -> &[Self::Element] {
        let InnerTensor::U16(inner) = &self.inner else {
            unreachable!()
        };
        inner._get_values()
    }
}

impl GetValues for Tensor<u32> {
    type Element = u32;
    fn get_values(&self) -> &[Self::Element] {
        let InnerTensor::U32(inner) = &self.inner else {
            unreachable!()
        };
        inner._get_values()
    }
}

impl GetValues for Tensor<u64> {
    type Element = u64;
    fn get_values(&self) -> &[Self::Element] {
        let InnerTensor::U64(inner) = &self.inner else {
            unreachable!()
        };
        inner._get_values()
    }
}

impl GetValues for Tensor<i8> {
    type Element = i8;
    fn get_values(&self) -> &[Self::Element] {
        let InnerTensor::I8(inner) = &self.inner else {
            unreachable!()
        };
        inner._get_values()
    }
}

impl GetValues for Tensor<i16> {
    type Element = i16;
    fn get_values(&self) -> &[Self::Element] {
        let InnerTensor::I16(inner) = &self.inner else {
            unreachable!()
        };
        inner._get_values()
    }
}

impl GetValues for Tensor<i32> {
    type Element = i32;
    fn get_values(&self) -> &[Self::Element] {
        let InnerTensor::I32(inner) = &self.inner else {
            unreachable!()
        };
        inner._get_values()
    }
}

impl GetValues for Tensor<i64> {
    type Element = i64;
    fn get_values(&self) -> &[Self::Element] {
        let InnerTensor::I64(inner) = &self.inner else {
            unreachable!()
        };
        inner._get_values()
    }
}

struct TensorPrinter<'a, T> {
    vals: &'a [T],
    dims: &'a [usize],
}

impl<'a, T> Debug for TensorPrinter<'a, T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = self
            .vals
            .iter()
            .map(|v| format!("{:?}", v).len())
            .max()
            .unwrap();
        fn format_recursive<J: Debug>(
            f: &mut std::fmt::Formatter<'_>,
            vals: &[J],
            dims: &[usize],
            indent: usize,
            width: usize,
        ) -> std::fmt::Result {
            match dims.len() {
                1 => {
                    if f.alternate() {
                        write!(f, "{:indent$}", "", indent = indent)?;
                        write!(f, "{:width$?}", vals)?;
                    } else {
                        write!(f, "{:?}", vals)?;
                    }
                }
                _ => {
                    let stride = dims[1..].iter().product::<usize>();
                    if f.alternate() {
                        write!(f, "{:indent$}", "", indent = indent)?;
                    }
                    write!(f, "[")?;
                    if f.alternate() {
                        writeln!(f)?;
                    }
                    for i in 0..dims[0] {
                        format_recursive(
                            f,
                            &vals[i * stride..(i + 1) * stride],
                            &dims[1..],
                            indent + 2,
                            width,
                        )?;
                        write!(f, ",")?;
                        if f.alternate() {
                            writeln!(f)?;
                        }
                    }
                    if f.alternate() {
                        write!(f, "{:indent$}", "", indent = indent)?;
                    }
                    write!(f, "]")?;
                }
            }
            Ok(())
        }

        if self.dims.len() == 0 {
            self.vals[0].fmt(f)
        } else {
            format_recursive(f, &self.vals, &self.dims, 0, width)
        }
    }
}

/// A generic tensor type.
pub struct Tensor<T> {
    inner: InnerTensor,
    phantom: PhantomData<T>,
}

impl<T> Debug for Tensor<T>
where
    Self: GetValues,
    <Self as GetValues>::Element: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field(
                "values",
                &TensorPrinter {
                    vals: self.get_values(),
                    dims: self.dimensions(),
                },
            )
            .field("dimensions", &self.dimensions())
            .field(
                "dtype",
                match self.inner {
                    InnerTensor::U8(_) => &"u8",
                    InnerTensor::U16(_) => &"u16",
                    InnerTensor::U32(_) => &"u32",
                    InnerTensor::U64(_) => &"u64",
                    InnerTensor::I8(_) => &"i8",
                    InnerTensor::I16(_) => &"i16",
                    InnerTensor::I32(_) => &"i32",
                    InnerTensor::I64(_) => &"i64",
                },
            )
            .finish()
    }
}

impl<T> Tensor<T> {
    /// Create a new tensor from values and dimensions (shape).
    pub fn new(values: Vec<T>, dimensions: Vec<usize>) -> Tensor<T>
    where
        Tensor<T>: FromElements<Element = T>,
    {
        assert_eq!(
            values.len(),
            dimensions.iter().product::<usize>(),
            "Wrong number of dimensions provided"
        );
        Self::from_elements(values, dimensions)
    }

    /// Return the dimensions of the tensor.
    pub fn dimensions(&self) -> &[usize] {
        match self.inner {
            InnerTensor::U8(ref unique_ptr) => unique_ptr._get_dimensions(),
            InnerTensor::U16(ref unique_ptr) => unique_ptr._get_dimensions(),
            InnerTensor::U32(ref unique_ptr) => unique_ptr._get_dimensions(),
            InnerTensor::U64(ref unique_ptr) => unique_ptr._get_dimensions(),
            InnerTensor::I8(ref unique_ptr) => unique_ptr._get_dimensions(),
            InnerTensor::I16(ref unique_ptr) => unique_ptr._get_dimensions(),
            InnerTensor::I32(ref unique_ptr) => unique_ptr._get_dimensions(),
            InnerTensor::I64(ref unique_ptr) => unique_ptr._get_dimensions(),
        }
    }

    /// Return the values of the tensor.
    pub fn values(&self) -> &[T]
    where
        Tensor<T>: GetValues<Element = T>,
    {
        self.get_values()
    }
}

enum InnerTensor {
    U8(UniquePtr<TensorU8>),
    U16(UniquePtr<TensorU16>),
    U32(UniquePtr<TensorU32>),
    U64(UniquePtr<TensorU64>),
    I8(UniquePtr<TensorI8>),
    I16(UniquePtr<TensorI16>),
    I32(UniquePtr<TensorI32>),
    I64(UniquePtr<TensorI64>),
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self._has_element_type_u8() {
            f.debug_tuple("Value")
                .field(&Tensor::<u8> {
                    inner: InnerTensor::U8(self._get_tensor_u8()),
                    phantom: PhantomData,
                })
                .finish()
        } else if self._has_element_type_u16() {
            f.debug_tuple("Value")
                .field(&Tensor::<u16> {
                    inner: InnerTensor::U16(self._get_tensor_u16()),
                    phantom: PhantomData,
                })
                .finish()
        } else if self._has_element_type_u32() {
            f.debug_tuple("Value")
                .field(&Tensor::<u32> {
                    inner: InnerTensor::U32(self._get_tensor_u32()),
                    phantom: PhantomData,
                })
                .finish()
        } else if self._has_element_type_u64() {
            f.debug_tuple("Value")
                .field(&Tensor::<u64> {
                    inner: InnerTensor::U64(self._get_tensor_u64()),
                    phantom: PhantomData,
                })
                .finish()
        } else if self._has_element_type_i8() {
            f.debug_tuple("Value")
                .field(&Tensor::<i8> {
                    inner: InnerTensor::I8(self._get_tensor_i8()),
                    phantom: PhantomData,
                })
                .finish()
        } else if self._has_element_type_i16() {
            f.debug_tuple("Value")
                .field(&Tensor::<i16> {
                    inner: InnerTensor::I16(self._get_tensor_i16()),
                    phantom: PhantomData,
                })
                .finish()
        } else if self._has_element_type_i32() {
            f.debug_tuple("Value")
                .field(&Tensor::<i32> {
                    inner: InnerTensor::I32(self._get_tensor_i32()),
                    phantom: PhantomData,
                })
                .finish()
        } else if self._has_element_type_i64() {
            f.debug_tuple("Value")
                .field(&Tensor::<i64> {
                    inner: InnerTensor::I64(self._get_tensor_i64()),
                    phantom: PhantomData,
                })
                .finish()
        } else {
            unreachable!()
        }
    }
}

pub trait GetTensor<T> {
    fn _get_tensor(&self) -> Option<Tensor<T>>;
}

impl GetTensor<u8> for Value {
    fn _get_tensor(&self) -> Option<Tensor<u8>> {
        self._has_element_type_u8().then(|| Tensor {
            inner: InnerTensor::U8(self._get_tensor_u8()),
            phantom: PhantomData,
        })
    }
}

impl GetTensor<u16> for Value {
    fn _get_tensor(&self) -> Option<Tensor<u16>> {
        self._has_element_type_u16().then(|| Tensor {
            inner: InnerTensor::U16(self._get_tensor_u16()),
            phantom: PhantomData,
        })
    }
}

impl GetTensor<u32> for Value {
    fn _get_tensor(&self) -> Option<Tensor<u32>> {
        self._has_element_type_u32().then(|| Tensor {
            inner: InnerTensor::U32(self._get_tensor_u32()),
            phantom: PhantomData,
        })
    }
}

impl GetTensor<u64> for Value {
    fn _get_tensor(&self) -> Option<Tensor<u64>> {
        self._has_element_type_u64().then(|| Tensor {
            inner: InnerTensor::U64(self._get_tensor_u64()),
            phantom: PhantomData,
        })
    }
}

impl GetTensor<i8> for Value {
    fn _get_tensor(&self) -> Option<Tensor<i8>> {
        self._has_element_type_i8().then(|| Tensor {
            inner: InnerTensor::I8(self._get_tensor_i8()),
            phantom: PhantomData,
        })
    }
}

impl GetTensor<i16> for Value {
    fn _get_tensor(&self) -> Option<Tensor<i16>> {
        self._has_element_type_i16().then(|| Tensor {
            inner: InnerTensor::I16(self._get_tensor_i16()),
            phantom: PhantomData,
        })
    }
}

impl GetTensor<i32> for Value {
    fn _get_tensor(&self) -> Option<Tensor<i32>> {
        self._has_element_type_i32().then(|| Tensor {
            inner: InnerTensor::I32(self._get_tensor_i32()),
            phantom: PhantomData,
        })
    }
}

impl GetTensor<i64> for Value {
    fn _get_tensor(&self) -> Option<Tensor<i64>> {
        self._has_element_type_i64().then(|| Tensor {
            inner: InnerTensor::I64(self._get_tensor_i64()),
            phantom: PhantomData,
        })
    }
}

impl Value {
    /// Create a `Value` from a `Tensor`.
    pub fn from_tensor<T>(input: Tensor<T>) -> UniquePtr<Value> {
        match input.inner {
            InnerTensor::U8(unique_ptr) => _value_from_tensor_u8(unique_ptr),
            InnerTensor::U16(unique_ptr) => _value_from_tensor_u16(unique_ptr),
            InnerTensor::U32(unique_ptr) => _value_from_tensor_u32(unique_ptr),
            InnerTensor::U64(unique_ptr) => _value_from_tensor_u64(unique_ptr),
            InnerTensor::I8(unique_ptr) => _value_from_tensor_i8(unique_ptr),
            InnerTensor::I16(unique_ptr) => _value_from_tensor_i16(unique_ptr),
            InnerTensor::I32(unique_ptr) => _value_from_tensor_i32(unique_ptr),
            InnerTensor::I64(unique_ptr) => _value_from_tensor_i64(unique_ptr),
        }
    }

    /// Unwrap the value to a tensor of the given type (if it indeed holds a tensor with elements of this type).
    pub fn get_tensor<T>(&self) -> Option<Tensor<T>>
    where
        Self: GetTensor<T>,
    {
        self._get_tensor()
    }
}

impl ClientModule {
    /// Create a new client module.
    pub fn new_encrypted(
        program_info: &ProgramInfo,
        client_keyset: &ClientKeyset,
        csprng: UniquePtr<EncryptionCsprng>,
    ) -> UniquePtr<ClientModule> {
        _client_module_new_encrypted(
            &serde_json::to_string(program_info).unwrap(),
            client_keyset,
            csprng,
        )
    }

    #[doc(hidden)]
    pub fn new_simulated(
        program_info: &ProgramInfo,
        csprng: UniquePtr<EncryptionCsprng>,
    ) -> UniquePtr<ClientModule> {
        _client_module_new_simulated(&serde_json::to_string(program_info).unwrap(), csprng)
    }
}

impl ClientFunction {
    /// Create a new client function.
    pub fn new_encrypted(
        circuit_info: &CircuitInfo,
        client_keyset: &ClientKeyset,
        csprng: UniquePtr<EncryptionCsprng>,
    ) -> UniquePtr<ClientFunction> {
        _client_function_new_encrypted(
            &serde_json::to_string(circuit_info).unwrap(),
            client_keyset,
            csprng,
        )
    }

    #[doc(hidden)]
    pub fn new_simulated(
        circuit_info: &CircuitInfo,
        csprng: UniquePtr<EncryptionCsprng>,
    ) -> UniquePtr<ClientFunction> {
        _client_function_new_simulated(&serde_json::to_string(circuit_info).unwrap(), csprng)
    }
}

impl ServerKeyset {
    /// Return references to the lwe bootstrap keys of this server keyset.
    pub fn lwe_bootstrap_keys(&self) -> Vec<&LweBootstrapKey> {
        (0..self._lwe_bootstrap_keys_len())
            .map(|i| self._lwe_bootstrap_keys_nth(i))
            .collect()
    }

    /// Return references to the lwe kewysitch keys of this server keyset.
    pub fn lwe_keyswitch_keys(&self) -> Vec<&LweKeyswitchKey> {
        (0..self._lwe_keyswitch_keys_len())
            .map(|i| self._lwe_keyswitch_keys_nth(i))
            .collect()
    }

    /// Return references to the packing keyswitch keys of this server keyset.
    pub fn packing_keyswitch_keys(&self) -> Vec<&PackingKeyswitchKey> {
        (0..self._packing_keyswitch_keys_len())
            .map(|i| self._packing_keyswitch_keys_nth(i))
            .collect()
    }
}

impl ClientKeyset {
    /// Return references to the lwe secret keys of this client keyset.
    pub fn lwe_secret_keys(&self) -> Vec<&LweSecretKey> {
        (0..self._lwe_secret_keys_len())
            .map(|i| self._lwe_secret_keys_nth(i))
            .collect()
    }
}

impl Keyset {
    /// Generate a new keyset.
    pub fn new(
        keyset_info: &KeysetInfo,
        secret_csprng: Pin<&mut SecretCsprng>,
        encryption_csprng: Pin<&mut EncryptionCsprng>,
        mut initial_keys: Vec<UniquePtr<LweSecretKey>>,
    ) -> UniquePtr<Keyset> {
        _keyset_new(
            &serde_json::to_string(keyset_info).unwrap(),
            secret_csprng,
            encryption_csprng,
            initial_keys.as_mut_slice(),
        )
    }
}

impl SecretCsprng {
    /// Create a new secret CSPRNG.
    pub fn new(seed: u128) -> UniquePtr<SecretCsprng> {
        let words: [u64; 2] = unsafe { std::mem::transmute::<u128, [u64; 2]>(seed) };
        _secret_csprng_new(words[0], words[1])
    }
}

impl EncryptionCsprng {
    /// Create a new encryption CSPRNG.
    pub fn new(seed: u128) -> UniquePtr<EncryptionCsprng> {
        let words: [u64; 2] = unsafe { std::mem::transmute::<u128, [u64; 2]>(seed) };
        _encryption_csprng_new(words[0], words[1])
    }
}

impl CompilationOptions {
    /// Create a new set of compilation options.
    pub fn new() -> UniquePtr<CompilationOptions> {
        _compilation_options_new()
    }
}

impl Library {
    /// Return the associated program info.
    pub fn get_program_info(&self) -> ProgramInfo {
        serde_json::from_str(&self._get_program_info_json()).unwrap()
    }
}

impl LweSecretKey {
    /// Return the associated info.
    pub fn get_info(&self) -> LweSecretKeyInfo {
        serde_json::from_str(&self._get_info_json()).unwrap()
    }
}

impl LweKeyswitchKey {
    /// Return the associated info.
    pub fn get_info(&self) -> LweKeyswitchKeyInfo {
        serde_json::from_str(&self._get_info_json()).unwrap()
    }
}

impl PackingKeyswitchKey {
    /// Return the associated info.
    pub fn get_info(&self) -> PackingKeyswitchKeyInfo {
        serde_json::from_str(&self._get_info_json()).unwrap()
    }
}

impl LweBootstrapKey {
    /// Return the associated info.
    pub fn get_info(&self) -> LweBootstrapKeyInfo {
        serde_json::from_str(&self._get_info_json()).unwrap()
    }
}

impl std::fmt::Debug for TransportValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TransportValue")
    }
}

#[cfg(test)]
mod test {
    use super::*;

    const TEST_FOLDER: &str = "/tmp/test_concrete";

    #[test]
    fn test_compile() {
        let sources = "
                func.func @dec(%arg0: !FHE.eint<3>) -> !FHE.eint<3> {
                  %cst_1 = arith.constant 1 : i4
                  %1 = \"FHE.sub_eint_int\"(%arg0, %cst_1) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
                  return %1: !FHE.eint<3>
                }
            ";
        let options = _compilation_options_new();
        let _ = std::fs::remove_dir(TEST_FOLDER);
        let _ = std::fs::create_dir_all(TEST_FOLDER);
        let mut output = compile(sources, &options, TEST_FOLDER).unwrap();
        let static_library_path = output.as_mut().unwrap().get_static_library_path();
        let mut archive = ar::Archive::new(std::fs::File::open(static_library_path).unwrap());
        let symbols = archive
            .symbols()
            .expect("failed to parse symbols")
            .map(|a| String::from_utf8_lossy(a).to_string())
            .collect::<Vec<_>>();
        #[cfg(target_os = "macos")]
        assert!(symbols.contains(&"_concrete_dec".to_string()));
        #[cfg(target_os = "linux")]
        assert!(symbols.contains(&"concrete_dec".to_string()));
    }
}
