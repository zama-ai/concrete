#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt for license information.

"""Compiler submodule"""
from collections.abc import Iterable
import os
from typing import List, Union

from mlir._mlir_libs._concretelang._compiler import JitCompilerEngine as _JitCompilerEngine
from mlir._mlir_libs._concretelang._compiler import LambdaArgument as _LambdaArgument
from mlir._mlir_libs._concretelang._compiler import round_trip as _round_trip
from mlir._mlir_libs._concretelang._compiler import library as _library
import numpy as np


ACCEPTED_NUMPY_UINTS = (np.uint8, np.uint16, np.uint32, np.uint64)
ACCEPTED_INTS = (int,) + ACCEPTED_NUMPY_UINTS
ACCEPTED_TYPES = (np.ndarray,) + ACCEPTED_INTS


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
    assert len(runtime_library_paths) == 1, "should be one and only one runtime library"
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

_MLIR_MODULES_TYPE = 'mlir_modules must be an `iterable` of `str` or a `str'

def library(library_path: str, mlir_modules: Union['Iterable[str]', str]) -> str:
    """Compile the MLIR inputs to a library.

    Args:
        library_path (str): destination path of the library
        mlir_modules (list[str]|str): code of MLIR modules

    Raises:
        TypeError: if arguments have incorrect types.

    Returns:
        str: parsed MLIR input.
    """
    if not isinstance(library_path, str):
        raise TypeError('library_path must be a `str`')
    if isinstance(mlir_modules, str):
        mlir_modules = [mlir_modules]
    elif isinstance(mlir_modules, list):
        pass
    elif isinstance(mlir_modules, Iterable):
        mlir_modules = list(mlir_modules)
    else:
        mlir_modules = [None]
        raise TypeError(_MLIR_MODULES_TYPE)

    if not all(isinstance(m, str) for m in mlir_modules):
        raise TypeError(_MLIR_MODULES_TYPE)

    return _library(library_path, mlir_modules)


def create_execution_argument(value: Union[int, np.ndarray]) -> "_LambdaArgument":
    """Create an execution argument holding either an int or tensor value.

    Args:
        value (Union[int, numpy.array]): value of the argument, either an int, or a numpy array

    Raises:
        TypeError: if the values aren't in the expected range, or using a wrong type

    Returns:
        _LambdaArgument: lambda argument holding the appropriate value
    """
    if not isinstance(value, ACCEPTED_TYPES):
        raise TypeError("value of execution argument must be either int, numpy.array or numpy.uint{8,16,32,64}")
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
            raise TypeError("numpy.array must be of dtype uint{8,16,32,64}")
        return _LambdaArgument.from_tensor(value.flatten().tolist(), value.shape)


class CompilerEngine:
    def __init__(self, mlir_str: str = None):
        self._engine = _JitCompilerEngine()
        self._lambda = None
        if mlir_str is not None:
            self.compile_fhe(mlir_str)

    def compile_fhe(
        self, mlir_str: str, func_name: str = "main", runtime_lib_path: str = None,
        unsecure_key_set_cache_path: str = None,
    ):
        """Compile the MLIR input.

        Args:
            mlir_str (str): MLIR to compile.
            func_name (str): name of the function to set as entrypoint (default: main).
            runtime_lib_path (str): path to the runtime lib (default: None).
            unsecure_key_set_cache_path (str): path to the activate keyset caching (default: None).

        Raises:
            TypeError: if the argument is not an str.
        """
        if not isinstance(mlir_str, str):
            raise TypeError("input must be an `str`")
        if runtime_lib_path is None:
            # Set to empty string if not found
            runtime_lib_path = _lookup_runtime_lib()
        else:
            if not isinstance(runtime_lib_path, str):
                raise TypeError(
                    "runtime_lib_path must be an str representing the path to the runtime lib"
                )
        unsecure_key_set_cache_path = unsecure_key_set_cache_path or ""
        if not isinstance(unsecure_key_set_cache_path, str):
            raise TypeError(
                "unsecure_key_set_cache_path must be a str"
            )
        self._lambda = self._engine.build_lambda(
            mlir_str, func_name, runtime_lib_path,
            unsecure_key_set_cache_path)

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
        if self._lambda is None:
            raise RuntimeError("need to compile an MLIR code first")
        execution_arguments = [create_execution_argument(arg) for arg in args]
        lambda_arg = self._lambda.invoke(execution_arguments)
        if lambda_arg.is_scalar():
            return lambda_arg.get_scalar()
        elif lambda_arg.is_tensor():
            shape = lambda_arg.get_tensor_shape()
            tensor = np.array(lambda_arg.get_tensor_data()).reshape(shape)
            return tensor
        else:
            raise RuntimeError("unknown return type")
