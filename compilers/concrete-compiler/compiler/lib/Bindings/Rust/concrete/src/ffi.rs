#![allow(unused_imports)]
use std::{any::Any, marker::PhantomData, pin::Pin};

use crate::protocol::{
    KeysetInfo, LweBootstrapKeyInfo, LweKeyswitchKeyInfo, LweSecretKeyInfo,
    PackingKeyswitchKeyInfo, ProgramInfo,
};
use cxx::{SharedPtr, UniquePtr};

#[cxx::bridge(namespace = "concrete_rust")]
mod ffi {
    unsafe extern "C++" {
        include!("ffi.h");

        // ------------------------------------------------------------------------------------------- Compiler
        type CompilationOptions;
        #[doc(hidden)]
        fn _compilation_options_new() -> UniquePtr<CompilationOptions>;
        fn set_display_optimizer_choice(self: Pin<&mut CompilationOptions>, val: bool);
        fn set_loop_parallelize(self: Pin<&mut CompilationOptions>, val: bool);
        fn set_dataflow_parallelize(self: Pin<&mut CompilationOptions>, val: bool);
        fn set_auto_parallelize(self: Pin<&mut CompilationOptions>, val: bool);
        fn set_compress_evaluation_keys(self: Pin<&mut CompilationOptions>, val: bool);
        fn set_compress_input_ciphertexts(self: Pin<&mut CompilationOptions>, val: bool);
        fn set_p_error(self: Pin<&mut CompilationOptions>, val: f64);
        fn set_global_p_error(self: Pin<&mut CompilationOptions>, val: f64);
        fn set_optimizer_strategy(self: Pin<&mut CompilationOptions>, val: u8);
        fn set_optimizer_multi_parameter_strategy(self: Pin<&mut CompilationOptions>, val: u8);
        fn set_enable_tlu_fusing(self: Pin<&mut CompilationOptions>, val: bool);
        fn set_print_tlu_fusing(self: Pin<&mut CompilationOptions>, val: bool);
        fn set_simulate(self: Pin<&mut CompilationOptions>, val: bool);
        fn set_enable_overflow_detection_in_simulation(
            self: Pin<&mut CompilationOptions>,
            val: bool,
        );
        fn set_composable(self: Pin<&mut CompilationOptions>, val: bool);
        fn set_range_restriction(self: Pin<&mut CompilationOptions>, json: &str);
        fn set_keyset_restriction(self: Pin<&mut CompilationOptions>, json: &str);
        fn set_security_level(self: Pin<&mut CompilationOptions>, val: u64);
        fn add_composition_rule(
            self: Pin<&mut CompilationOptions>,
            from_func: &str,
            from_pos: usize,
            to_func: &str,
            to_pos: usize,
        );

        type Library;
        fn get_static_library_path(self: &Library) -> String;
        #[doc(hidden)]
        fn _get_program_info_json(self: &Library) -> String;

        fn compile(
            sources: &str,
            options: &CompilationOptions,
            output_dir_path: &str,
        ) -> Result<UniquePtr<Library>>;

        // ------------------------------------------------------------------------------------------- Commons
        type SecretCsprng;
        #[doc(hidden)]
        fn _secret_csprng_new(high: u64, low: u64) -> SharedPtr<SecretCsprng>;

        type EncryptionCsprng;
        #[doc(hidden)]
        fn _encryption_csprng_new(high: u64, low: u64) -> SharedPtr<EncryptionCsprng>;

        type LweBootstrapKey;
        fn get_buffer(self: Pin<&mut LweBootstrapKey>) -> &[u64];
        #[doc(hidden)]
        fn _get_info_json(self: &LweBootstrapKey) -> String;

        type LweKeyswitchKey;
        fn get_buffer(self: Pin<&mut LweKeyswitchKey>) -> &[u64];
        #[doc(hidden)]
        fn _get_info_json(self: &LweKeyswitchKey) -> String;

        type PackingKeyswitchKey;
        fn get_buffer(self: Pin<&mut PackingKeyswitchKey>) -> &[u64];
        #[doc(hidden)]
        fn _get_info_json(self: &PackingKeyswitchKey) -> String;

        type LweSecretKey;
        fn get_buffer(self: Pin<&mut LweSecretKey>) -> &[u64];
        #[doc(hidden)]
        fn _get_info_json(self: &LweSecretKey) -> String;

        type Keyset;
        #[doc(hidden)]
        fn _keyset_new(
            keyset_info_json: &str,
            secret_csprng: Pin<&mut SecretCsprng>,
            encryption_csprng: Pin<&mut EncryptionCsprng>,
        ) -> UniquePtr<Keyset>;
        fn get_server(self: &Keyset) -> UniquePtr<ServerKeyset>;
        fn get_client(self: &Keyset) -> UniquePtr<ClientKeyset>;

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

        type ClientKeyset;
        #[doc(hidden)]
        fn _lwe_secret_keys_len(self: &ClientKeyset) -> usize;
        #[doc(hidden)]
        fn _lwe_secret_keys_nth(self: &ClientKeyset, nth: usize) -> &LweSecretKey;

        type TensorU8;
        #[doc(hidden)]
        fn _tensor_u8_new(values: &[u8], dimensions: &[usize]) -> UniquePtr<TensorU8>;

        type TensorU16;
        #[doc(hidden)]
        fn _tensor_u16_new(values: &[u16], dimensions: &[usize]) -> UniquePtr<TensorU16>;

        type TensorU32;
        #[doc(hidden)]
        fn _tensor_u32_new(values: &[u32], dimensions: &[usize]) -> UniquePtr<TensorU32>;

        type TensorU64;
        #[doc(hidden)]
        fn _tensor_u64_new(values: &[u64], dimensions: &[usize]) -> UniquePtr<TensorU64>;

        type TensorI8;
        #[doc(hidden)]
        fn _tensor_i8_new(values: &[i8], dimensions: &[usize]) -> UniquePtr<TensorI8>;

        type TensorI16;
        #[doc(hidden)]
        fn _tensor_i16_new(values: &[i16], dimensions: &[usize]) -> UniquePtr<TensorI16>;

        type TensorI32;
        #[doc(hidden)]
        fn _tensor_i32_new(values: &[i32], dimensions: &[usize]) -> UniquePtr<TensorI32>;

        type TensorI64;
        #[doc(hidden)]
        fn _tensor_i64_new(values: &[i64], dimensions: &[usize]) -> UniquePtr<TensorI64>;

        type Value;
        #[doc(hidden)]
        fn _value_from_tensor_u8(tensor: &TensorU8) -> UniquePtr<Value>;
        #[doc(hidden)]
        fn _value_from_tensor_u16(tensor: &TensorU16) -> UniquePtr<Value>;
        #[doc(hidden)]
        fn _value_from_tensor_u32(tensor: &TensorU32) -> UniquePtr<Value>;
        #[doc(hidden)]
        fn _value_from_tensor_u64(tensor: &TensorU64) -> UniquePtr<Value>;
        #[doc(hidden)]
        fn _value_from_tensor_i8(tensor: &TensorI8) -> UniquePtr<Value>;
        #[doc(hidden)]
        fn _value_from_tensor_i16(tensor: &TensorI16) -> UniquePtr<Value>;
        #[doc(hidden)]
        fn _value_from_tensor_i32(tensor: &TensorI32) -> UniquePtr<Value>;
        #[doc(hidden)]
        fn _value_from_tensor_i64(tensor: &TensorI64) -> UniquePtr<Value>;
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
        fn _get_tensor_u8(self: &Value) -> &TensorU8;
        #[doc(hidden)]
        fn _get_tensor_u16(self: &Value) -> &TensorU16;
        #[doc(hidden)]
        fn _get_tensor_u32(self: &Value) -> &TensorU32;
        #[doc(hidden)]
        fn _get_tensor_u64(self: &Value) -> &TensorU64;
        #[doc(hidden)]
        fn _get_tensor_i8(self: &Value) -> &TensorI8;
        #[doc(hidden)]
        fn _get_tensor_i16(self: &Value) -> &TensorI16;
        #[doc(hidden)]
        fn _get_tensor_i32(self: &Value) -> &TensorI32;
        #[doc(hidden)]
        fn _get_tensor_i64(self: &Value) -> &TensorI64;
        fn get_dimensions(self: &Value) -> &[usize];

        type TransportValue;


        // ------------------------------------------------------------------------------------------- Client
        type ClientProgram;
        #[doc(hidden)]
        fn _client_program_new_encrypted(
            program_info_json: &str,
            client_keyset: &ClientKeyset,
            encryption_prng: SharedPtr<EncryptionCsprng>,
        ) -> UniquePtr<ClientProgram>;
        #[doc(hidden)]
        fn _client_program_new_simulated(
            program_info_json: &str,
            encryption_prng: SharedPtr<EncryptionCsprng>,
        ) -> UniquePtr<ClientProgram>;
        #[doc(hidden)]
        fn _get_client_circuit(self: &ClientProgram, name: &str) -> Result<UniquePtr<ClientCircuit>>;

        type ClientCircuit;
        #[doc(hidden)]
        fn _client_circuit_new_encrypted(
            circuit_info_json: &str,
            client_keyset: &ClientKeyset,
            encryption_prng: SharedPtr<EncryptionCsprng>
        ) -> UniquePtr<ClientCircuit>;
        #[doc(hidden)]
        fn _client_circuit_new_simulated(
            circuit_info_json: &str,
            encryption_prng: SharedPtr<EncryptionCsprng>
        ) -> UniquePtr<ClientCircuit>;



    }
}

pub use ffi::*;

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


pub struct Tensor<T>{
    inner: InnerTensor,
    phantom: PhantomData<T>
}

impl<T> Tensor<T> {
    pub fn new(values: Vec<T>, dimensions: Vec<usize>) -> Tensor<T> where Tensor<T>:FromElements<Element=T> {
        Self::from_elements(values, dimensions)
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

impl ClientProgram{
    pub fn new_encrypted(program_info: &ProgramInfo, client_keyset: &ClientKeyset, csprng: SharedPtr<EncryptionCsprng>) -> UniquePtr<ClientProgram>{
        _client_program_new_encrypted(
            &serde_json::to_string(program_info).unwrap(),
            client_keyset,
            csprng,
        )
    }

    pub fn new_simulated(program_info: &ProgramInfo, csprng: SharedPtr<EncryptionCsprng>) -> UniquePtr<ClientProgram>{
        _client_program_new_simulated(
            &serde_json::to_string(program_info).unwrap(),
            csprng,
        )
    }
}

impl ServerKeyset {
    pub fn lwe_bootstrap_keys(&self) -> Vec<&LweBootstrapKey> {
        (0..self._lwe_bootstrap_keys_len())
            .map(|i| self._lwe_bootstrap_keys_nth(i))
            .collect()
    }

    pub fn lwe_keyswitch_keys(&self) -> Vec<&LweKeyswitchKey> {
        (0..self._lwe_keyswitch_keys_len())
            .map(|i| self._lwe_keyswitch_keys_nth(i))
            .collect()
    }

    pub fn packing_keyswitch_keys(&self) -> Vec<&PackingKeyswitchKey> {
        (0..self._packing_keyswitch_keys_len())
            .map(|i| self._packing_keyswitch_keys_nth(i))
            .collect()
    }
}

impl ClientKeyset {
    pub fn lwe_secret_keys(&self) -> Vec<&LweSecretKey> {
        (0..self._lwe_secret_keys_len())
            .map(|i| self._lwe_secret_keys_nth(i))
            .collect()
    }
}

impl Keyset {
    pub fn new(
        keyset_info: &KeysetInfo,
        secret_csprng: Pin<&mut SecretCsprng>,
        encryption_csprng: Pin<&mut EncryptionCsprng>,
    ) -> UniquePtr<Keyset> {
        _keyset_new(
            &serde_json::to_string(keyset_info).unwrap(),
            secret_csprng,
            encryption_csprng,
        )
    }
}

impl SecretCsprng {
    pub fn new(seed: u128) -> SharedPtr<SecretCsprng> {
        let words: [u64; 2] = unsafe { std::mem::transmute::<u128, [u64; 2]>(seed) };
        _secret_csprng_new(words[0], words[1])
    }
}

impl EncryptionCsprng {
    pub fn new(seed: u128) -> SharedPtr<EncryptionCsprng> {
        let words: [u64; 2] = unsafe { std::mem::transmute::<u128, [u64; 2]>(seed) };
        _encryption_csprng_new(words[0], words[1])
    }
}

impl CompilationOptions {
    pub fn new() -> UniquePtr<CompilationOptions> {
        _compilation_options_new()
    }
}

impl Library {
    pub fn get_program_info(&self) -> ProgramInfo {
        serde_json::from_str(&self._get_program_info_json()).unwrap()
    }
}

impl LweSecretKey {
    pub fn get_info(&self) -> LweSecretKeyInfo {
        serde_json::from_str(&self._get_info_json()).unwrap()
    }
}

impl LweKeyswitchKey {
    pub fn get_info(&self) -> LweKeyswitchKeyInfo {
        serde_json::from_str(&self._get_info_json()).unwrap()
    }
}

impl PackingKeyswitchKey {
    pub fn get_info(&self) -> PackingKeyswitchKeyInfo {
        serde_json::from_str(&self._get_info_json()).unwrap()
    }
}

impl LweBootstrapKey {
    pub fn get_info(&self) -> LweBootstrapKeyInfo {
        serde_json::from_str(&self._get_info_json()).unwrap()
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
