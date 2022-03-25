#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt for license information.

"""Compiler submodule"""
from collections.abc import Iterable
import os
import atexit
from typing import List, Union

from mlir._mlir_libs._concretelang._compiler import terminate_parallelization as _terminate_parallelization

from mlir._mlir_libs._concretelang._compiler import round_trip as _round_trip

from mlir._mlir_libs._concretelang._compiler import ClientSupport as _ClientSupport

from mlir._mlir_libs._concretelang._compiler import ClientParameters

from mlir._mlir_libs._concretelang._compiler import KeySet
from mlir._mlir_libs._concretelang._compiler import KeySetCache

from mlir._mlir_libs._concretelang._compiler import PublicResult
from mlir._mlir_libs._concretelang._compiler import PublicArguments
from mlir._mlir_libs._concretelang._compiler import LambdaArgument as _LambdaArgument

from mlir._mlir_libs._concretelang._compiler import CompilationOptions

from mlir._mlir_libs._concretelang._compiler import JITLambdaSupport as _JITLambdaSupport
from mlir._mlir_libs._concretelang._compiler import JitCompilationResult
from mlir._mlir_libs._concretelang._compiler import JITLambda

from mlir._mlir_libs._concretelang._compiler import LibraryLambdaSupport as _LibraryLambdaSupport
from mlir._mlir_libs._concretelang._compiler import LibraryCompilationResult
from mlir._mlir_libs._concretelang._compiler import LibraryLambda
import numpy as np


ACCEPTED_NUMPY_UINTS = (np.uint8, np.uint16, np.uint32, np.uint64)
ACCEPTED_INTS = (int,) + ACCEPTED_NUMPY_UINTS
ACCEPTED_TYPES = (np.ndarray,) + ACCEPTED_INTS


# Terminate parallelization in the compiler (if init) during cleanup
atexit.register(_terminate_parallelization)


def _lookup_runtime_lib() -> str:
    """Try to find the absolute path to the runtime library.

    Returns:
        str: absolute path to the runtime library, or empty str if unsuccessful.
    """
    # Go up to site-packages level
    cwd = os.path.abspath(__file__)
    cwd = os.path.abspath(os.path.join(cwd, os.pardir))
    cwd = os.path.abspath(os.path.join(cwd, os.pardir))
    package_name = "concrete_compiler"
    libs_path = os.path.join(cwd, f"{package_name}.libs")
    # Can be because it's not a properly installed package
    if not os.path.exists(libs_path):
        return ""
    runtime_library_paths = [
        filename
        for filename in os.listdir(libs_path)
        if filename.startswith("libConcretelangRuntime")
    ]
    assert len(
        runtime_library_paths) == 1, "should be one and only one runtime library"
    return os.path.join(libs_path, runtime_library_paths[0])


def round_trip(mlir_str: str) -> str:
    """Parse the MLIR input, then return it back.

    Args:
        mlir_str (str): MLIR code to parse.

    Raises:
        TypeError: if the argument is not an str.

    Returns:
        str: parsed MLIR input.
    """
    if not isinstance(mlir_str, str):
        raise TypeError("input must be an `str`")
    return _round_trip(mlir_str)


class CompilerEngine:
    def __init__(self, mlir_str: str = None):
        self._engine = JITCompilerSupport()
        self._lambda = None
        if mlir_str is not None:
            self.compile_fhe(mlir_str)

    def compile_fhe(
        self,
        mlir_str: str,
        func_name: str = "main",
        unsecure_key_set_cache_path: str = None,
        auto_parallelize: bool = False,
        loop_parallelize: bool = False,
        df_parallelize: bool = False,
    ):
        """Compile the MLIR input.

        Args:
            mlir_str (str): MLIR to compile.
            func_name (str): name of the function to set as entrypoint (default: main).
            unsecure_key_set_cache_path (str): path to the activate keyset caching (default: None).
            auto_parallelize (bool): whether to activate auto-parallelization or not (default: False),
            loop_parallelize (bool): whether to activate loop-parallelization or not (default: False),
            df_parallelize (bool): whether to activate dataflow-parallelization or not (default: False),

        Raises:
            TypeError: if the argument is not an str.
        """
        if not all(
            isinstance(flag, bool)
            for flag in [auto_parallelize, loop_parallelize, df_parallelize]
        ):
            raise TypeError(
                "parallelization flags (auto_parallelize, loop_parallelize, df_parallelize), should be booleans"
            )
        unsecure_key_set_cache_path = unsecure_key_set_cache_path or ""
        if not isinstance(unsecure_key_set_cache_path, str):
            raise TypeError(
                "unsecure_key_set_cache_path must be a str"
            )
        options = CompilationOptions(func_name)
        options.auto_parallelize(auto_parallelize)
        options.loop_parallelize(loop_parallelize)
        options.dataflow_parallelize(df_parallelize)
        self._compilation_result = self._engine.compile(mlir_str, options)
        self._client_parameters = self._engine.load_client_parameters(
            self._compilation_result)
        keyset_cache = None
        if not unsecure_key_set_cache_path is None:
            keyset_cache = KeySetCache(unsecure_key_set_cache_path)
        self._key_set = ClientSupport.key_set(
            self._client_parameters, keyset_cache)

    def run(self, *args: List[Union[int, np.ndarray]]) -> Union[int, np.ndarray]:
        """Run the compiled code.

        Args:
            *args: list of arguments for execution. Each argument can be an int, or a numpy.array

        Raises:
            TypeError: if execution arguments can't be constructed
            RuntimeError: if the engine has not compiled any code yet
            RuntimeError: if the return type is unknown

        Returns:
            int or numpy.array: result of execution.
        """
        if self._compilation_result is None:
            raise RuntimeError("need to compile an MLIR code first")
        # Client
        public_arguments = ClientSupport.encrypt_arguments(self._client_parameters,
                                                           self._key_set, args)
        # Server
        server_lambda = self._engine.load_server_lambda(
            self._compilation_result)
        public_result = self._engine.server_call(
            server_lambda, public_arguments)
        # Client
        return ClientSupport.decrypt_result(self._key_set, public_result)


class ClientSupport:
    def key_set(client_parameters: ClientParameters, cache: KeySetCache = None) -> KeySet:
        """Generates a key set according to the given client parameters.
        If the cache is set the key set is loaded from it if exists, else the new generated key set is saved in the cache

        Args:
            client_parameters: A client parameters specification
            cache: An optional cache of key set.

        Returns:
            KeySet: the key set
        """
        return _ClientSupport.key_set(client_parameters, cache)

    def encrypt_arguments(client_parameters: ClientParameters, key_set: KeySet, args: List[Union[int, np.ndarray]]) -> PublicArguments:
        """Export clear arguments to public arguments.
        For each arguments this method encrypts the argument if it's declared as encrypted and pack to the public arguments object.

        Args:
            client_parameters: A client parameters specification
            key_set: A key set used to encrypt encrypted arguments

        Returns:
            PublicArguments: the public arguments
        """
        execution_arguments = [
            ClientSupport._create_execution_argument(arg) for arg in args]
        return _ClientSupport.encrypt_arguments(client_parameters, key_set, execution_arguments)

    def decrypt_result(key_set: KeySet, public_result: PublicResult) -> Union[int, np.ndarray]:
        """Decrypt a public result thanks the given key set.

        Args:
            key_set: The key set used to decrypt the result.
            public_result: The public result to descrypt.

        Returns:
            int or numpy.array: The result of decryption.
            """
        lambda_arg = _ClientSupport.decrypt_result(key_set, public_result)
        if lambda_arg.is_scalar():
            return lambda_arg.get_scalar()
        elif lambda_arg.is_tensor():
            shape = lambda_arg.get_tensor_shape()
            tensor = np.array(lambda_arg.get_tensor_data()).reshape(shape)
            return tensor
        else:
            raise RuntimeError("unknown return type")

    def _create_execution_argument(value: Union[int, np.ndarray]) -> _LambdaArgument:
        """Create an execution argument holding either an int or tensor value.

        Args:
            value (Union[int, numpy.array]): value of the argument, either an int, or a numpy array

        Raises:
            TypeError: if the values aren't in the expected range, or using a wrong type

        Returns:
            _LambdaArgument: lambda argument holding the appropriate value
        """
        if not isinstance(value, ACCEPTED_TYPES):
            raise TypeError(
                "value of execution argument must be either int, numpy.array or numpy.uint{8,16,32,64}")
        if isinstance(value, ACCEPTED_INTS):
            if isinstance(value, int) and not (0 <= value < np.iinfo(np.uint64).max):
                raise TypeError(
                    "single integer must be in the range [0, 2**64 - 1] (uint64)"
                )
            return _LambdaArgument.from_scalar(value)
        else:
            assert isinstance(value, np.ndarray)
            if value.shape == ():
                return _LambdaArgument.from_scalar(value)
            if value.dtype not in ACCEPTED_NUMPY_UINTS:
                raise TypeError(
                    "numpy.array must be of dtype uint{8,16,32,64}")
            return _LambdaArgument.from_tensor(value.flatten().tolist(), value.shape)


class JITCompilerSupport:
    def __init__(self, runtime_lib_path=None):
        if runtime_lib_path is None:
            runtime_lib_path = _lookup_runtime_lib()
        else:
            if not isinstance(runtime_lib_path, str):
                raise TypeError(
                    "runtime_lib_path must be an str representing the path to the runtime lib"
                )
        self._support = _JITLambdaSupport(runtime_lib_path)

    def compile(self, mlir_program: str, options: CompilationOptions = CompilationOptions("main")) -> JitCompilationResult:
        """JIT Compile a function define in the mlir_program to its homomorphic equivalent.

        Args:
            mlir_program: A textual representation of the mlir program to compile.
            func_name: The name of the function to compile.

        Returns:
            JITCompilationResult: the result of the JIT compilation.
        """
        if not isinstance(mlir_program, str):
            raise TypeError("mlir_program must be an `str`")
        return self._support.compile(mlir_program, options)

    def load_client_parameters(self, compilation_result: JitCompilationResult) -> ClientParameters:
        """Load the client parameters from the JIT compilation result"""
        return self._support.load_client_parameters(compilation_result)

    def load_server_lambda(self, compilation_result: JitCompilationResult) -> JITLambda:
        """Load the server lambda from the JIT compilation result"""
        return self._support.load_server_lambda(compilation_result)

    def server_call(self, server_lambda: JITLambda, public_arguments: PublicArguments):
        """Call the server lambda with public_arguments

        Args:
            server_lambda: A server lambda to call
            public_arguments: The arguments of the call

        Returns:
            PublicResult: the result of the call of the server lambda
        """
        return self._support.server_call(server_lambda, public_arguments)


class LibraryCompilerSupport:
    def __init__(self, outputPath="./out"):
        self._library_path = outputPath
        self._support = _LibraryLambdaSupport(outputPath)

    def compile(self, mlir_program: str, options: CompilationOptions = CompilationOptions("main")) -> LibraryCompilationResult:
        """Compile a function define in the mlir_program to its homomorphic equivalent and save as library.

        Args:
            mlir_program: A textual representation of the mlir program to compile.
            func_name: The name of the function to compile.

        Returns:
            LibraryCompilationResult: the result of the compilation.
        """
        if not isinstance(mlir_program, str):
            raise TypeError("mlir_program must be an `str`")
        if not isinstance(options, CompilationOptions):
            raise TypeError("mlir_program must be an `str`")
        return self._support.compile(mlir_program, options)

    def reload(self, func_name: str = "main") -> LibraryCompilationResult:
        """Reload the library compilation result from the outputPath.
        Args:
            library-path: The path of the compiled library.
            func_name: The name of the compiled function.

        Returns:
            LibraryCompilationResult: the result of a compilation.
        """
        if not isinstance(func_name, str):
            raise TypeError("func_name must be an `str`")
        return LibraryCompilationResult(self._library_path, func_name)

    def load_client_parameters(self, compilation_result: LibraryCompilationResult) -> ClientParameters:
        """Load the client parameters from the JIT compilation result"""
        if not isinstance(compilation_result, LibraryCompilationResult):
            raise TypeError(
                "compilation_result must be an `LibraryCompilationResult`")

        return self._support.load_client_parameters(compilation_result)

    def load_server_lambda(self, compilation_result: LibraryCompilationResult) -> LibraryLambda:
        """Load the server lambda from the JIT compilation result"""
        return self._support.load_server_lambda(compilation_result)

    def server_call(self, server_lambda: LibraryLambda, public_arguments: PublicArguments) -> PublicResult:
        """Call the server lambda with public_arguments

        Args:
            server_lambda: A server lambda to call
            public_arguments: The arguments of the call

        Returns:
            PublicResult: the result of the call of the server lambda
        """
        return self._support.server_call(server_lambda, public_arguments)
