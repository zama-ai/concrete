//! Compiler module

use crate::mlir::ffi;
use std::os::raw::c_char;
use std::{ffi::CStr, path::Path};

pub struct CompilerError(String);

// Manual implementation to use pretty formatting of line-breaks
// contained in the String.
impl std::fmt::Debug for CompilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(f, "CompilerError {{")?;
        writeln!(f, "{:#}", self.0)?;
        writeln!(f, "}}")
    }
}

impl std::fmt::Display for CompilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        <Self as std::fmt::Debug>::fmt(self, f)
    }
}

impl std::error::Error for CompilerError {}

/// Retrieve buffer of the error message from a C struct.
trait CStructErrorMsg {
    fn error_msg(&self) -> *const i8;
}

/// All C struct can return a pointer to the allocated error message.
macro_rules! impl_CStructErrorMsg {
    ([$($t:ty),+]) => {
        $(impl CStructErrorMsg for $t {
            fn error_msg(&self) -> *const i8 {
                self.error
            }
        })*
    }
}
impl_CStructErrorMsg! {[
    ffi::BufferRef,
    ffi::CompilationOptions,
    ffi::OptimizerConfig,
    ffi::CompilerEngine,
    ffi::CompilationResult,
    ffi::Library,
    ffi::LibraryCompilationResult,
    ffi::LibrarySupport,
    ffi::ServerLambda,
    ffi::CircuitGate,
    ffi::EncryptionGate,
    ffi::Encoding,
    ffi::ClientParameters,
    ffi::KeySet,
    ffi::KeySetCache,
    ffi::EvaluationKeys,
    ffi::LambdaArgument,
    ffi::PublicArguments,
    ffi::PublicResult,
    ffi::CompilationFeedback
]}

/// Construct a rust error message from a buffer in the C struct.
fn get_error_msg_from_ctype<T: CStructErrorMsg>(c_struct: &T) -> String {
    unsafe {
        let error_msg_cstr = CStr::from_ptr(c_struct.error_msg());
        String::from(error_msg_cstr.to_str().unwrap())
    }
}

/// Wrapper to own MlirStringRef coming from the compiler and destroy them on drop
struct MlirStringRef(ffi::MlirStringRef);

impl MlirStringRef {
    pub fn to_string(&self) -> Result<String, CompilerError> {
        unsafe {
            if self.0.data.is_null() {
                return Err(CompilerError("string ref points to null".to_string()));
            }
            let result = String::from_utf8_lossy(std::slice::from_raw_parts(
                self.0.data as *const u8,
                self.0.length as usize,
            ))
            .to_string();
            Ok(result)
        }
    }

    /// Create an ffi MlirStringRef for a rust str.
    ///
    /// The reason behind not returning a wrapper is that it would lead to freeing rust memory
    /// using a custom destructor in C.
    ///
    /// # SAFETY
    /// The caller has to make sure the &str outlive the ffi::MlirStringRef
    pub unsafe fn from_rust_str(s: &str) -> ffi::MlirStringRef {
        ffi::MlirStringRef {
            data: s.as_ptr() as *const c_char,
            length: s.len() as ffi::size_t,
        }
    }
}

impl Drop for MlirStringRef {
    fn drop(&mut self) {
        unsafe { ffi::mlirStringRefDestroy(self.0) }
    }
}

trait CStructWrapper<T> {
    // wrap a c-struct inside a rust-struct
    fn wrap(c_struct: T) -> Self;
    // check if the wrapped c-struct is null
    fn is_null(&self) -> bool;
    // get error message
    fn error_msg(&self) -> String;
    // drop
    fn destroy(&mut self);
}

/// Wrapper of CStruct.
///
/// We want to have a Rust wrapper for every CStruct that will take care of owning
/// it, and freeing memory when it's no longer used.
macro_rules! def_CStructWrapper {
    (
        $name:ident => {
            $ffi_is_null_fn:ident,
            $ffi_destroy_fn:ident
            $(,)?
        }
    ) => {

        pub struct $name{ _c: ffi::$name }

        impl CStructWrapper<ffi::$name> for $name {
            // wrap a c-struct inside a rust-struct
            fn wrap(c_struct: ffi::$name) -> Self {
                Self{_c: c_struct}
            }
            // check if the wrapped C-struct is null
            fn is_null(&self) -> bool {
                unsafe {
                    ffi::$ffi_is_null_fn(self._c)
                }
            }
            // get error message
            fn error_msg(&self) -> String {
                get_error_msg_from_ctype(&self._c)
            }
            // free memory allocated for the C-struct
            fn destroy(&mut self) {
                unsafe {
                    ffi::$ffi_destroy_fn(self._c)
                }
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                self.destroy();
            }
        }
    };

    (
        $(
            $name:ident => {
                $ffi_is_null_fn:ident,
                $ffi_destroy_fn:ident
                $(,)?
            }
        ),+
        $(,)?
    ) =>  {
        $(
            def_CStructWrapper!{
                $name => {
                    $ffi_is_null_fn,
                    $ffi_destroy_fn
                }
            }
        )+
    };
}
def_CStructWrapper! {
    BufferRef => {
        bufferRefIsNull,
        bufferRefDestroy
    },
    CompilationOptions => {
        compilationOptionsIsNull,
        compilationOptionsDestroy,
    },
    OptimizerConfig => {
        optimizerConfigIsNull,
        optimizerConfigDestroy,
    },
    CompilerEngine => {
        compilerEngineIsNull,
        compilerEngineDestroy,
    },
    CompilationResult => {
        compilationResultIsNull,
        compilationResultDestroy,
    },
    Library => {
        libraryIsNull,
        libraryDestroy,
    },
    LibraryCompilationResult => {
        libraryCompilationResultIsNull,
        libraryCompilationResultDestroy,
    },
    LibrarySupport => {
        librarySupportIsNull,
        librarySupportDestroy,
    },
    ServerLambda => {
        serverLambdaIsNull,
        serverLambdaDestroy,
    },
    CircuitGate => {
        circuitGateIsNull,
        circuitGateDestroy,
    },
    EncryptionGate => {
        encryptionGateIsNull,
        encryptionGateDestroy,
    },
    Encoding => {
        encodingIsNull,
        encodingDestroy,
    },
    ClientParameters => {
        clientParametersIsNull,
        clientParametersDestroy,
    },
    KeySetCache => {
        keySetCacheIsNull,
        keySetCacheDestroy,
    },
    EvaluationKeys => {
        evaluationKeysIsNull,
        evaluationKeysDestroy,
    },
    LambdaArgument => {
        lambdaArgumentIsNull,
        lambdaArgumentDestroy,
    },
    PublicArguments => {
        publicArgumentsIsNull,
        publicArgumentsDestroy,
    },
    PublicResult => {
        publicResultIsNull,
        publicResultDestroy,
    },
    CompilationFeedback => {
        compilationFeedbackIsNull,
        compilationFeedbackDestroy,
    }
}

impl BufferRef {
    /// Create a reference to a buffer in memory.
    ///
    /// # SAFETY
    ///
    /// - The pointed memory will not get owned.
    /// - The caller must make sure the pointer points
    ///   to a valid memory region of the provided length
    /// - The caller must make sure that the pointed memory outlive
    ///   the buffer reference.
    unsafe fn new(
        ptr: *const c_char,
        length: ffi::size_t,
    ) -> Result<ffi::BufferRef, CompilerError> {
        let buffer_ref = ffi::bufferRefCreate(ptr, length);
        if ffi::bufferRefIsNull(buffer_ref) {
            let error_msg = get_error_msg_from_ctype(&buffer_ref);
            ffi::bufferRefDestroy(buffer_ref);
            return Err(CompilerError(error_msg));
        }
        return Ok(buffer_ref);
    }

    /// Copy the content of the buffer into a new vector of bytes.
    ///
    /// Returns an empty vector if the buffer reference is a null pointer.
    fn to_bytes(&self) -> Vec<c_char> {
        if self.is_null() {
            return Vec::new();
        }
        let buffer_ref_c = self._c;
        unsafe {
            let result = std::slice::from_raw_parts(
                buffer_ref_c.data as *const c_char,
                buffer_ref_c.length as usize,
            )
            .to_vec();
            result
        }
    }
}

impl CompilationOptions {
    pub fn new(
        func_name: &str,
        auto_parallelize: bool,
        batch_concrete_ops: bool,
        dataflow_parallelize: bool,
        emit_gpu_ops: bool,
        loop_parallelize: bool,
        optimize_concrete: bool,
        optimizer_config: &OptimizerConfig,
        verify_diagnostics: bool,
    ) -> Result<CompilationOptions, CompilerError> {
        unsafe {
            let options = CompilationOptions::wrap(ffi::compilationOptionsCreate(
                // Its safe to give a string ref to the rust str
                // as the `compilationOptionsCreate` function is going to copy the content.
                MlirStringRef::from_rust_str(func_name),
                auto_parallelize,
                batch_concrete_ops,
                dataflow_parallelize,
                emit_gpu_ops,
                loop_parallelize,
                optimize_concrete,
                optimizer_config._c,
                verify_diagnostics,
            ));
            if options.is_null() {
                return Err(CompilerError(options.error_msg()));
            }
            Ok(options)
        }
    }

    pub fn default() -> Result<CompilationOptions, CompilerError> {
        unsafe {
            let options = CompilationOptions::wrap(ffi::compilationOptionsCreateDefault());
            if options.is_null() {
                return Err(CompilerError(options.error_msg()));
            }
            Ok(options)
        }
    }
}

impl OptimizerConfig {
    pub fn new(
        display: bool,
        fallback_log_norm_woppbs: f64,
        global_p_error: f64,
        p_error: f64,
        security: u64,
        strategy_v0: bool,
        use_gpu_constraints: bool,
    ) -> Result<OptimizerConfig, CompilerError> {
        unsafe {
            let config = OptimizerConfig::wrap(ffi::optimizerConfigCreate(
                display,
                fallback_log_norm_woppbs,
                global_p_error,
                p_error,
                security,
                strategy_v0,
                use_gpu_constraints,
            ));
            if config.is_null() {
                return Err(CompilerError(config.error_msg()));
            }
            Ok(config)
        }
    }

    pub fn default() -> Result<OptimizerConfig, CompilerError> {
        unsafe {
            let config = OptimizerConfig::wrap(ffi::optimizerConfigCreateDefault());
            if config.is_null() {
                return Err(CompilerError(config.error_msg()));
            }
            Ok(config)
        }
    }
}
impl CompilerEngine {
    pub fn new(options: Option<&CompilationOptions>) -> Result<CompilerEngine, CompilerError> {
        unsafe {
            let engine = CompilerEngine::wrap(ffi::compilerEngineCreate());
            if engine.is_null() {
                return Err(CompilerError(engine.error_msg()));
            }
            if let Some(o) = options {
                engine.set_options(o)
            }
            Ok(engine)
        }
    }

    pub fn set_options(&self, options: &CompilationOptions) {
        unsafe {
            ffi::compilerEngineCompileSetOptions(self._c, options._c);
        }
    }

    pub fn compile(
        &self,
        module: &str,
        target: ffi::CompilationTarget,
    ) -> Result<CompilationResult, CompilerError> {
        unsafe {
            let module_string_ref = MlirStringRef::from_rust_str(module);
            let result = CompilationResult::wrap(ffi::compilerEngineCompile(
                self._c,
                module_string_ref,
                target,
            ));
            if result.is_null() {
                return Err(CompilerError(format!(
                    "Error in compiler (check logs for more info): {}",
                    result.error_msg()
                )));
            }
            Ok(result)
        }
    }
}
impl CompilationResult {
    pub fn module_string(&self) -> Result<String, CompilerError> {
        unsafe { MlirStringRef(ffi::compilationResultGetModuleString(self._c)).to_string() }
    }
}
impl Library {
    pub fn new(
        output_dir_path: &str,
        runtime_library_path: Option<&str>,
        clean_up: bool,
    ) -> Result<Library, CompilerError> {
        unsafe {
            let lib = Library::wrap(ffi::libraryCreate(
                MlirStringRef::from_rust_str(output_dir_path),
                MlirStringRef::from_rust_str(runtime_library_path.unwrap_or("")),
                clean_up,
            ));
            if lib.is_null() {
                return Err(CompilerError(lib.error_msg()));
            }
            Ok(lib)
        }
    }
}

impl LibraryCompilationResult {}

/// Support for compiling and executing libraries.
impl LibrarySupport {
    /// LibrarySupport manages build files generated by the compiler under the `output_dir_path`.
    ///
    /// The compiled library needs to link to the runtime for proper execution.
    pub fn new(
        output_dir_path: &str,
        runtime_library_path: Option<String>,
    ) -> Result<LibrarySupport, CompilerError> {
        unsafe {
            let runtime_library_path = match runtime_library_path {
                Some(val) => val.to_string(),
                None => "".to_string(),
            };
            let runtime_library_path_buffer = runtime_library_path.as_str();
            let support = LibrarySupport::wrap(ffi::librarySupportCreateDefault(
                MlirStringRef::from_rust_str(output_dir_path),
                MlirStringRef::from_rust_str(runtime_library_path_buffer),
            ));
            if support.is_null() {
                return Err(CompilerError(format!(
                    "Error in compiler (check logs for more info): {}",
                    support.error_msg()
                )));
            }
            Ok(support)
        }
    }

    /// Compile an MLIR into a library.
    pub fn compile(
        &self,
        mlir_code: &str,
        options: Option<CompilationOptions>,
    ) -> Result<LibraryCompilationResult, CompilerError> {
        unsafe {
            let options = options.unwrap_or_else(|| CompilationOptions::default().unwrap());
            let result = LibraryCompilationResult::wrap(ffi::librarySupportCompile(
                self._c,
                MlirStringRef::from_rust_str(mlir_code),
                options._c,
            ));
            if result.is_null() {
                return Err(CompilerError(format!(
                    "Error in compiler (check logs for more info): {}",
                    result.error_msg()
                )));
            }
            Ok(result)
        }
    }

    /// Load server lambda from a compilation result.
    ///
    /// This can be used for executing the compiled function.
    pub fn load_server_lambda(
        &self,
        result: &LibraryCompilationResult,
    ) -> Result<ServerLambda, CompilerError> {
        unsafe {
            let server =
                ServerLambda::wrap(ffi::librarySupportLoadServerLambda(self._c, result._c));
            if server.is_null() {
                return Err(CompilerError(format!(
                    "Error in compiler (check logs for more info): {}",
                    server.error_msg()
                )));
            }
            Ok(server)
        }
    }

    /// Load client parameters from a compilation result.
    ///
    /// This can be used for creating keys for the compiled library.
    pub fn load_client_parameters(
        &self,
        result: &LibraryCompilationResult,
    ) -> Result<ClientParameters, CompilerError> {
        unsafe {
            let params =
                ClientParameters::wrap(ffi::librarySupportLoadClientParameters(self._c, result._c));
            if params.is_null() {
                return Err(CompilerError(format!(
                    "Error in compiler (check logs for more info): {}",
                    params.error_msg()
                )));
            }
            Ok(params)
        }
    }

    /// Load compilation result from the library support's output directory.
    ///
    /// This should be used when the output directory already has artefacts from a previous compilation.
    pub fn load_compilation_result(&self) -> Result<LibraryCompilationResult, CompilerError> {
        unsafe {
            let result =
                LibraryCompilationResult::wrap(ffi::librarySupportLoadCompilationResult(self._c));
            if result.is_null() {
                return Err(CompilerError(format!(
                    "Error in compiler (check logs for more info): {}",
                    result.error_msg()
                )));
            }
            Ok(result)
        }
    }

    /// Run a compiled circuit.
    pub fn server_lambda_call(
        &self,
        server_lambda: &ServerLambda,
        args: &PublicArguments,
        eval_keys: &EvaluationKeys,
    ) -> Result<PublicResult, CompilerError> {
        unsafe {
            let result = PublicResult::wrap(ffi::librarySupportServerCall(
                self._c,
                server_lambda._c,
                args._c,
                eval_keys._c,
            ));
            if result.is_null() {
                return Err(CompilerError(format!(
                    "Error in compiler (check logs for more info): {}",
                    result.error_msg()
                )));
            }
            Ok(result)
        }
    }

    /// Get path to the compiled shared library
    pub fn shared_lib_path(&self) -> String {
        unsafe {
            MlirStringRef(ffi::librarySupportGetSharedLibPath(self._c))
                .to_string()
                .unwrap()
        }
    }

    /// Get path to the client parameters
    pub fn client_parameters_path(&self) -> String {
        unsafe {
            MlirStringRef(ffi::librarySupportGetClientParametersPath(self._c))
                .to_string()
                .unwrap()
        }
    }
}

impl ServerLambda {}

impl CircuitGate {
    pub fn encryption_gate(self) -> Option<EncryptionGate> {
        let inner = unsafe { ffi::circuitGateEncryptionGate(self._c) };
        let gate = EncryptionGate::wrap(inner);
        if gate.is_null() {
            None
        } else {
            Some(gate)
        }
    }
}

impl EncryptionGate {
    pub fn encoding(self) -> Encoding {
        let inner = unsafe { ffi::encryptionGateEncoding(self._c) };

        Encoding::wrap(inner)
    }

    pub fn variance(&self) -> f64 {
        unsafe { ffi::encryptionGateVariance(self._c) }
    }
}

impl Encoding {
    pub fn precision(&self) -> u64 {
        unsafe { ffi::encodingPrecision(self._c) }
    }
}

impl ClientParameters {
    pub fn num_inputs(&self) -> usize {
        unsafe { ffi::clientParametersInputsSize(self._c) }
            .try_into()
            .unwrap()
    }

    pub fn input(&self, index: usize) -> Option<CircuitGate> {
        if index >= self.num_inputs() {
            None
        } else {
            let gate = unsafe {
                ffi::clientParametersInputCircuitGate(self._c, index.try_into().unwrap())
            };
            Some(CircuitGate::wrap(gate))
        }
    }

    pub fn num_outputs(&self) -> usize {
        unsafe { ffi::clientParametersOutputsSize(self._c) }
            .try_into()
            .unwrap()
    }

    pub fn output(&self, index: usize) -> Option<CircuitGate> {
        if index >= self.num_outputs() {
            None
        } else {
            let gate = unsafe {
                ffi::clientParametersOutputCircuitGate(self._c, index.try_into().unwrap())
            };
            Some(CircuitGate::wrap(gate))
        }
    }

    pub fn serialize(&self) -> Result<Vec<c_char>, CompilerError> {
        unsafe {
            let serialized_ref = BufferRef::wrap(ffi::clientParametersSerialize(self._c));
            if serialized_ref.is_null() {
                return Err(CompilerError(serialized_ref.error_msg()));
            }
            Ok(serialized_ref.to_bytes())
        }
    }
    pub fn unserialize(serialized: &Vec<c_char>) -> Result<ClientParameters, CompilerError> {
        unsafe {
            let serialized_ref = BufferRef::new(
                serialized.as_ptr() as *const c_char,
                serialized.len().try_into().unwrap(),
            )
            .unwrap();
            let params = ClientParameters::wrap(ffi::clientParametersUnserialize(serialized_ref));
            if params.is_null() {
                return Err(CompilerError(params.error_msg()));
            }
            Ok(params)
        }
    }
}

impl Clone for ClientParameters {
    fn clone(&self) -> Self {
        unsafe { ClientParameters::wrap(ffi::clientParametersCopy(self._c)) }
    }
}

struct KeySet_ {
    _c: ffi::KeySet,
}

impl CStructWrapper<ffi::KeySet> for KeySet_ {
    // wrap a c-struct inside a rust-struct
    fn wrap(c_struct: ffi::KeySet) -> KeySet_ {
        KeySet_ { _c: c_struct }
    }
    // check if the wrapped C-struct is null
    fn is_null(&self) -> bool {
        unsafe { ffi::keySetIsNull(self._c) }
    }
    // get error message
    fn error_msg(&self) -> String {
        get_error_msg_from_ctype(&self._c)
    }
    // free memory allocated for the C-struct
    fn destroy(&mut self) {
        unsafe { ffi::keySetDestroy(self._c) }
    }
}

impl Drop for KeySet_ {
    fn drop(&mut self) {
        self.destroy();
    }
}
pub struct KeySet {
    key_set: KeySet_,
    client_params: ClientParameters,
}

impl KeySet {
    /// Get a keyset based on the client parameters, and the different seeds.
    ///
    /// If a cache is set, this operation would first try to load an existing key,
    /// otherwise, a new keyset will be generated.
    pub fn new(
        client_params: &ClientParameters,
        seed_msb: Option<u64>,
        seed_lsb: Option<u64>,
        key_set_cache: Option<&KeySetCache>,
    ) -> Result<KeySet, CompilerError> {
        unsafe {
            let key_set = match key_set_cache {
                Some(cache) => KeySet_::wrap(ffi::keySetCacheLoadOrGenerateKeySet(
                    cache._c,
                    client_params._c,
                    seed_msb.unwrap_or(0),
                    seed_lsb.unwrap_or(0),
                )),
                None => KeySet_::wrap(ffi::keySetGenerate(
                    client_params._c,
                    seed_msb.unwrap_or(0),
                    seed_lsb.unwrap_or(0),
                )),
            };
            if key_set.is_null() {
                return Err(CompilerError(format!(
                    "Error in compiler (check logs for more info): {}",
                    key_set.error_msg()
                )));
            }
            Ok(KeySet {
                key_set,
                client_params: client_params.clone(),
            })
        }
    }

    pub fn evaluation_keys(&self) -> Result<EvaluationKeys, CompilerError> {
        unsafe {
            let eval_keys = EvaluationKeys::wrap(ffi::keySetGetEvaluationKeys(self.key_set._c));
            if eval_keys.is_null() {
                return Err(CompilerError(eval_keys.error_msg()));
            }
            Ok(eval_keys)
        }
    }

    /// Encrypt arguments of a compiled circuit.
    pub fn encrypt_args(&self, args: &[LambdaArgument]) -> Result<PublicArguments, CompilerError> {
        LambdaArgument::encrypt_args(args, self)
    }

    pub fn decrypt_result(&self, result: &PublicResult) -> Result<LambdaArgument, CompilerError> {
        result.decrypt(self)
    }
}

impl KeySetCache {
    pub fn new(path: &Path) -> Result<KeySetCache, CompilerError> {
        unsafe {
            let cache_path_buffer = path.to_str().unwrap();
            let cache = KeySetCache::wrap(ffi::keySetCacheCreate(MlirStringRef::from_rust_str(
                cache_path_buffer,
            )));
            if cache.is_null() {
                return Err(CompilerError(format!(
                    "Error in compiler (check logs for more info): {}",
                    cache.error_msg()
                )));
            }
            Ok(cache)
        }
    }
}

impl EvaluationKeys {
    pub fn serialize(&self) -> Result<Vec<c_char>, CompilerError> {
        unsafe {
            let serialized_ref = BufferRef::wrap(ffi::evaluationKeysSerialize(self._c));
            if serialized_ref.is_null() {
                return Err(CompilerError(serialized_ref.error_msg()));
            }
            Ok(serialized_ref.to_bytes())
        }
    }
    pub fn unserialize(serialized: &Vec<c_char>) -> Result<EvaluationKeys, CompilerError> {
        unsafe {
            let serialized_ref = BufferRef::new(
                serialized.as_ptr() as *const c_char,
                serialized.len().try_into().unwrap(),
            )
            .unwrap();
            let eval_keys = EvaluationKeys::wrap(ffi::evaluationKeysUnserialize(serialized_ref));
            if eval_keys.is_null() {
                return Err(CompilerError(eval_keys.error_msg()));
            }
            Ok(eval_keys)
        }
    }
}

impl LambdaArgument {
    pub fn encrypt_args(
        args: &[LambdaArgument],
        key_set: &KeySet,
    ) -> Result<PublicArguments, CompilerError> {
        unsafe {
            let args: Vec<ffi::LambdaArgument> = args.into_iter().map(|a| a._c).collect();
            let public_args = PublicArguments::wrap(ffi::lambdaArgumentEncrypt(
                args.as_ptr(),
                args.len() as u64,
                key_set.client_params._c,
                key_set.key_set._c,
            ));
            if public_args.is_null() {
                return Err(CompilerError(format!(
                    "Error in compiler (check logs for more info): {}",
                    public_args.error_msg()
                )));
            }
            Ok(public_args)
        }
    }

    pub fn from_scalar(scalar: u64) -> Result<LambdaArgument, CompilerError> {
        unsafe {
            let arg = LambdaArgument::wrap(ffi::lambdaArgumentFromScalar(scalar));
            if arg.is_null() {
                return Err(CompilerError(arg.error_msg()));
            }
            Ok(arg)
        }
    }

    pub fn is_scalar(&self) -> bool {
        unsafe { ffi::lambdaArgumentIsScalar(self._c) }
    }

    pub fn scalar(&self) -> Result<u64, CompilerError> {
        unsafe {
            if !self.is_scalar() {
                return Err(CompilerError("argument is not a scalar".to_string()));
            }
            Ok(ffi::lambdaArgumentGetScalar(self._c))
        }
    }

    pub fn from_tensor_u8(data: &[u8], dims: &[i64]) -> Result<LambdaArgument, CompilerError> {
        unsafe {
            let arg = LambdaArgument::wrap(ffi::lambdaArgumentFromTensorU8(
                data.as_ptr(),
                dims.as_ptr(),
                dims.len().try_into().unwrap(),
            ));
            if arg.is_null() {
                return Err(CompilerError(arg.error_msg()));
            }
            Ok(arg)
        }
    }

    pub fn from_tensor_u16(data: &[u16], dims: &[i64]) -> Result<LambdaArgument, CompilerError> {
        unsafe {
            let arg = LambdaArgument::wrap(ffi::lambdaArgumentFromTensorU16(
                data.as_ptr(),
                dims.as_ptr(),
                dims.len().try_into().unwrap(),
            ));
            if arg.is_null() {
                return Err(CompilerError(arg.error_msg()));
            }
            Ok(arg)
        }
    }

    pub fn from_tensor_u32(data: &[u32], dims: &[i64]) -> Result<LambdaArgument, CompilerError> {
        unsafe {
            let arg = LambdaArgument::wrap(ffi::lambdaArgumentFromTensorU32(
                data.as_ptr(),
                dims.as_ptr(),
                dims.len().try_into().unwrap(),
            ));
            if arg.is_null() {
                return Err(CompilerError(arg.error_msg()));
            }
            Ok(arg)
        }
    }

    pub fn from_tensor_u64(data: &[u64], dims: &[i64]) -> Result<LambdaArgument, CompilerError> {
        unsafe {
            let arg = LambdaArgument::wrap(ffi::lambdaArgumentFromTensorU64(
                data.as_ptr(),
                dims.as_ptr(),
                dims.len().try_into().unwrap(),
            ));
            if arg.is_null() {
                return Err(CompilerError(arg.error_msg()));
            }
            Ok(arg)
        }
    }

    pub fn is_tensor(&self) -> bool {
        unsafe { ffi::lambdaArgumentIsTensor(self._c) }
    }

    pub fn data_size(&self) -> Result<i64, CompilerError> {
        unsafe {
            if !self.is_tensor() {
                return Err(CompilerError("argument is not a tensor".to_string()));
            }
            Ok(ffi::lambdaArgumentGetTensorDataSize(self._c))
        }
    }

    pub fn rank(&self) -> Result<ffi::size_t, CompilerError> {
        unsafe {
            if !self.is_tensor() {
                return Err(CompilerError("argument is not a tensor".to_string()));
            }
            Ok(ffi::lambdaArgumentGetTensorRank(self._c))
        }
    }

    pub fn dims(&self) -> Result<Vec<i64>, CompilerError> {
        unsafe {
            let rank = self.rank().unwrap();
            let mut dims = Vec::new();
            dims.resize(rank.try_into().unwrap(), 0);
            if !ffi::lambdaArgumentGetTensorDims(self._c, dims.as_mut_ptr()) {
                return Err(CompilerError("couldn't get dims".to_string()));
            }
            Ok(dims)
        }
    }

    pub fn data(&self) -> Result<Vec<u64>, CompilerError> {
        unsafe {
            let size = self.data_size().unwrap();
            let mut data = Vec::new();
            data.resize(size.try_into().unwrap(), 0);
            if !ffi::lambdaArgumentGetTensorData(self._c, data.as_mut_ptr()) {
                return Err(CompilerError("couldn't get data".to_string()));
            }
            Ok(data)
        }
    }
}

impl PublicArguments {
    pub fn serialize(&self) -> Result<Vec<c_char>, CompilerError> {
        unsafe {
            let serialized_ref = BufferRef::wrap(ffi::publicArgumentsSerialize(self._c));
            if serialized_ref.is_null() {
                return Err(CompilerError(serialized_ref.error_msg()));
            }
            Ok(serialized_ref.to_bytes())
        }
    }
    pub fn unserialize(
        serialized: &Vec<c_char>,
        client_parameters: &ClientParameters,
    ) -> Result<PublicArguments, CompilerError> {
        unsafe {
            let serialized_ref = BufferRef::new(
                serialized.as_ptr() as *const c_char,
                serialized.len().try_into().unwrap(),
            )
            .unwrap();
            let public_args = PublicArguments::wrap(ffi::publicArgumentsUnserialize(
                serialized_ref,
                client_parameters._c,
            ));
            if public_args.is_null() {
                return Err(CompilerError(public_args.error_msg()));
            }
            Ok(public_args)
        }
    }
}

impl PublicResult {
    pub fn serialize(&self) -> Result<Vec<c_char>, CompilerError> {
        unsafe {
            let serialized_ref = BufferRef::wrap(ffi::publicResultSerialize(self._c));
            if serialized_ref.is_null() {
                return Err(CompilerError(serialized_ref.error_msg()));
            }
            Ok(serialized_ref.to_bytes())
        }
    }
    pub fn unserialize(
        serialized: &Vec<c_char>,
        client_parameters: &ClientParameters,
    ) -> Result<PublicResult, CompilerError> {
        unsafe {
            let serialized_ref = BufferRef::new(
                serialized.as_ptr() as *const c_char,
                serialized.len().try_into().unwrap(),
            )
            .unwrap();
            let public_result = PublicResult::wrap(ffi::publicResultUnserialize(
                serialized_ref,
                client_parameters._c,
            ));
            if public_result.is_null() {
                return Err(CompilerError(public_result.error_msg()));
            }
            Ok(public_result)
        }
    }

    pub fn decrypt(&self, key_set: &KeySet) -> Result<LambdaArgument, CompilerError> {
        unsafe {
            let arg = LambdaArgument::wrap(ffi::publicResultDecrypt(self._c, key_set.key_set._c));
            if arg.is_null() {
                return Err(CompilerError(format!(
                    "Error in compiler (check logs for more info): {}",
                    arg.error_msg()
                )));
            }
            Ok(arg)
        }
    }
}

impl CompilationFeedback {
    pub fn complexity(&self) -> f64 {
        unsafe { ffi::compilationFeedbackGetComplexity(self._c) }
    }

    pub fn p_error(&self) -> f64 {
        unsafe { ffi::compilationFeedbackGetPError(self._c) }
    }

    pub fn global_p_error(&self) -> f64 {
        unsafe { ffi::compilationFeedbackGetGlobalPError(self._c) }
    }

    pub fn total_secret_keys_size(&self) -> u64 {
        unsafe { ffi::compilationFeedbackGetTotalSecretKeysSize(self._c) }
    }

    pub fn total_bootstrap_keys_size(&self) -> u64 {
        unsafe { ffi::compilationFeedbackGetTotalBootstrapKeysSize(self._c) }
    }

    pub fn total_keyswitch_keys_size(&self) -> u64 {
        unsafe { ffi::compilationFeedbackGetTotalKeyswitchKeysSize(self._c) }
    }

    pub fn total_inputs_size(&self) -> u64 {
        unsafe { ffi::compilationFeedbackGetTotalInputsSize(self._c) }
    }

    pub fn total_outputs_size(&self) -> u64 {
        unsafe { ffi::compilationFeedbackGetTotalOutputsSize(self._c) }
    }
}

/// Parse the MLIR code and returns it.
///
/// The function parse the provided MLIR textual representation and returns it. It would fail with
/// an error message to stderr reporting what's bad with the parsed IR.
///
/// # Examples
/// ```
/// use concrete_compiler::compiler::*;
///
/// let module_to_compile = "
///     func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
///         %0 = \"FHE.add_eint\"(%arg0, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
///         return %0 : !FHE.eint<5>
///     }";
/// let result_str = round_trip(module_to_compile);
/// ```
///
pub fn round_trip(mlir_code: &str) -> Result<String, CompilerError> {
    let engine = CompilerEngine::new(None).unwrap();
    let compilation_result = engine.compile(mlir_code, ffi::CompilationTarget_ROUND_TRIP)?;
    compilation_result.module_string()
}

#[cfg(test)]
mod test {
    use std::env;
    use tempdir::TempDir;

    use super::*;

    fn runtime_lib_path() -> Option<String> {
        match env::var("CONCRETE_COMPILER_INSTALL_DIR") {
            Ok(val) => Some(val + "/lib/libConcretelangRuntime.so"),
            Err(_e) => None,
        }
    }

    #[test]
    fn test_compiler_round_trip() {
        let module_to_compile = "
                func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
                    %0 = \"FHE.add_eint\"(%arg0, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
                    return %0 : !FHE.eint<5>
                }";
        let result_str = round_trip(module_to_compile).unwrap();
        let expected_module = "module {
  func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
    %0 = \"FHE.add_eint\"(%arg0, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
    return %0 : !FHE.eint<5>
  }
}
";
        assert_eq!(expected_module, result_str);
    }

    #[test]
    fn test_compiler_round_trip_invalid_mlir() {
        let module_to_compile = "bla bla bla";
        let result_str = round_trip(module_to_compile);
        assert!(
            matches!(result_str, Err(CompilerError(err)) if err == "Error in compiler (check logs for more info): Could not parse source\n")
        );
    }

    #[test]
    fn test_compiler_compile_lib() {
        let module_to_compile = "
            func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
                    %0 = \"FHE.add_eint\"(%arg0, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
                    return %0 : !FHE.eint<5>
                }";
        let runtime_library_path = runtime_lib_path();
        let temp_dir = TempDir::new("concrete_compiler_test").unwrap();
        let support =
            LibrarySupport::new(temp_dir.path().to_str().unwrap(), runtime_library_path).unwrap();
        let lib = support.compile(module_to_compile, None).unwrap();
        assert!(!lib.is_null());
        // the sharedlib should be enough as a sign that the compilation worked
        assert!(Path::new(support.shared_lib_path().as_str()).exists());
        assert!(Path::new(support.client_parameters_path().as_str()).exists());
    }

    /// We want to make sure setting a pointer to null in rust passes the nullptr check in C/Cpp
    #[test]
    fn test_compiler_null_ptr_compatibility() {
        unsafe {
            let lib = ffi::Library {
                ptr: std::ptr::null_mut(),
                error: std::ptr::null_mut(),
            };
            assert!(ffi::libraryIsNull(lib));
        }
    }

    #[test]
    fn test_compiler_load_server_lambda_and_client_parameters() {
        let module_to_compile = "
            func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
                    %0 = \"FHE.add_eint\"(%arg0, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
                    return %0 : !FHE.eint<5>
                }";
        let runtime_library_path = runtime_lib_path();
        let temp_dir = TempDir::new("concrete_compiler_test").unwrap();
        let support =
            LibrarySupport::new(temp_dir.path().to_str().unwrap(), runtime_library_path).unwrap();
        let result = support.compile(module_to_compile, None).unwrap();
        let server = support.load_server_lambda(&result).unwrap();
        assert!(!server.is_null());
        let client_params = support.load_client_parameters(&result).unwrap();
        assert!(!client_params.is_null());

        assert_eq!(client_params.num_inputs(), 2);
        let input_bitwidth_0 = client_params
            .input(0)
            .unwrap()
            .encryption_gate()
            .unwrap()
            .encoding()
            .precision();
        let input_bitwidth_1 = client_params
            .input(1)
            .unwrap()
            .encryption_gate()
            .unwrap()
            .encoding()
            .precision();

        assert_eq!(input_bitwidth_0, 5);
        assert_eq!(input_bitwidth_1, 5);

        assert_eq!(client_params.num_outputs(), 1);
        let output_bitwidth = client_params
            .output(0)
            .unwrap()
            .encryption_gate()
            .unwrap()
            .encoding()
            .precision();
        assert_eq!(output_bitwidth, 5);
    }

    #[test]
    fn test_compiler_compile_and_exec_scalar_args() {
        let module_to_compile = "
            func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
                    %0 = \"FHE.add_eint\"(%arg0, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
                    return %0 : !FHE.eint<5>
                }";
        let runtime_library_path = runtime_lib_path();
        let temp_dir = TempDir::new("concrete_compiler_test").unwrap();
        let lib_support =
            LibrarySupport::new(temp_dir.path().to_str().unwrap(), runtime_library_path).unwrap();
        // compile
        let result = lib_support.compile(module_to_compile, None).unwrap();
        // loading materials from compilation
        // - server_lambda: used for execution
        // - client_parameters: used for keygen, encryption, and evaluation keys
        let server_lambda = lib_support.load_server_lambda(&result).unwrap();
        let client_params = lib_support.load_client_parameters(&result).unwrap();
        let key_set = KeySet::new(&client_params, None, None, None).unwrap();
        let eval_keys = key_set.evaluation_keys().unwrap();
        // build lambda arguments from scalar and encrypt them
        let args = [
            LambdaArgument::from_scalar(4).unwrap(),
            LambdaArgument::from_scalar(2).unwrap(),
        ];
        let encrypted_args = key_set.encrypt_args(&args).unwrap();
        // execute the compiled function on the encrypted arguments
        let encrypted_result = lib_support
            .server_lambda_call(&server_lambda, &encrypted_args, &eval_keys)
            .unwrap();
        // decrypt the result of execution
        let result_arg = key_set.decrypt_result(&encrypted_result).unwrap();
        // get the scalar value from the result lambda argument
        let result = result_arg.scalar().unwrap();
        assert_eq!(result, 6);
    }

    #[test]
    fn test_compiler_compile_and_exec_with_serialization() {
        let module_to_compile = "
            func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
                    %0 = \"FHE.add_eint\"(%arg0, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
                    return %0 : !FHE.eint<5>
                }";
        let runtime_library_path = runtime_lib_path();
        let temp_dir = TempDir::new("concrete_compiler_test").unwrap();
        let lib_support =
            LibrarySupport::new(temp_dir.path().to_str().unwrap(), runtime_library_path).unwrap();
        // compile
        let result = lib_support.compile(module_to_compile, None).unwrap();
        // loading materials from compilation
        // - server_lambda: used for execution
        // - client_parameters: used for keygen, encryption, and evaluation keys
        let server_lambda = lib_support.load_server_lambda(&result).unwrap();
        let client_params = lib_support.load_client_parameters(&result).unwrap();
        // serialize client parameters
        let serialized_params = client_params.serialize().unwrap();
        let client_params = ClientParameters::unserialize(&serialized_params).unwrap();
        // generate keys
        let key_set = KeySet::new(&client_params, None, None, None).unwrap();
        let eval_keys = key_set.evaluation_keys().unwrap();
        // serialize eval keys
        let serialized_eval_keys = eval_keys.serialize().unwrap();
        let eval_keys = EvaluationKeys::unserialize(&serialized_eval_keys).unwrap();
        // build lambda arguments from scalar and encrypt them
        let args = [
            LambdaArgument::from_scalar(4).unwrap(),
            LambdaArgument::from_scalar(2).unwrap(),
        ];
        let encrypted_args = key_set.encrypt_args(&args).unwrap();
        // serialize args
        let serialized_encrypted_args = encrypted_args.serialize().unwrap();
        let encrypted_args =
            PublicArguments::unserialize(&serialized_encrypted_args, &client_params).unwrap();
        // execute the compiled function on the encrypted arguments
        let encrypted_result = lib_support
            .server_lambda_call(&server_lambda, &encrypted_args, &eval_keys)
            .unwrap();
        // serialize result
        let serialized_encrypted_result = encrypted_result.serialize().unwrap();
        let encrypted_result =
            PublicResult::unserialize(&serialized_encrypted_result, &client_params).unwrap();
        // decrypt the result of execution
        let result_arg = key_set.decrypt_result(&encrypted_result).unwrap();
        // get the scalar value from the result lambda argument
        let result = result_arg.scalar().unwrap();
        assert_eq!(result, 6);
    }

    #[test]
    fn test_tensor_lambda_argument() {
        let tensor_data = [1, 2, 3, 73u64];
        let tensor_dims = [2, 2i64];
        let tensor_arg = LambdaArgument::from_tensor_u64(&tensor_data, &tensor_dims).unwrap();
        assert!(!tensor_arg.is_null());
        assert!(!tensor_arg.is_scalar());
        assert!(tensor_arg.is_tensor());
        assert_eq!(tensor_arg.rank().unwrap(), 2);
        assert_eq!(tensor_arg.data_size().unwrap(), 4);
        assert_eq!(tensor_arg.dims().unwrap(), tensor_dims);
        assert_eq!(tensor_arg.data().unwrap(), tensor_data);
    }

    #[test]
    fn test_compiler_compile_and_exec_tensor_args() {
        let module_to_compile = "
            func.func @main(%arg0: tensor<2x3x!FHE.eint<5>>, %arg1: tensor<2x3x!FHE.eint<5>>) -> tensor<2x3x!FHE.eint<5>> {
                    %0 = \"FHELinalg.add_eint\"(%arg0, %arg1) : (tensor<2x3x!FHE.eint<5>>, tensor<2x3x!FHE.eint<5>>) -> tensor<2x3x!FHE.eint<5>>
                    return %0 : tensor<2x3x!FHE.eint<5>>
                }";
        let runtime_library_path = runtime_lib_path();
        let temp_dir = TempDir::new("concrete_compiler_test").unwrap();
        let lib_support =
            LibrarySupport::new(temp_dir.path().to_str().unwrap(), runtime_library_path).unwrap();
        // compile
        let result = lib_support.compile(module_to_compile, None).unwrap();
        // loading materials from compilation
        // - server_lambda: used for execution
        // - client_parameters: used for keygen, encryption, and evaluation keys
        let server_lambda = lib_support.load_server_lambda(&result).unwrap();
        let client_params = lib_support.load_client_parameters(&result).unwrap();
        let key_set = KeySet::new(&client_params, None, None, None).unwrap();
        let eval_keys = key_set.evaluation_keys().unwrap();
        // build lambda arguments from scalar and encrypt them
        let args = [
            LambdaArgument::from_tensor_u8(&[1, 2, 3, 4, 5, 6], &[2, 3]).unwrap(),
            LambdaArgument::from_tensor_u8(&[1, 4, 7, 4, 2, 9], &[2, 3]).unwrap(),
        ];
        let encrypted_args = key_set.encrypt_args(&args).unwrap();
        // execute the compiled function on the encrypted arguments
        let encrypted_result = lib_support
            .server_lambda_call(&server_lambda, &encrypted_args, &eval_keys)
            .unwrap();
        // decrypt the result of execution
        let result_arg = key_set.decrypt_result(&encrypted_result).unwrap();
        // check the tensor dims value from the result lambda argument
        assert_eq!(result_arg.rank().unwrap(), 2);
        assert_eq!(result_arg.data_size().unwrap(), 6);
        assert_eq!(result_arg.dims().unwrap(), [2, 3]);
        // check the tensor data from the result lambda argument
        assert_eq!(result_arg.data().unwrap(), [2, 6, 10, 8, 7, 15]);
    }
}
