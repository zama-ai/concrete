"""Compiler submodule"""
from typing import List, Union
from mlir._mlir_libs._zamalang._compiler import JitCompilerEngine as _JitCompilerEngine
from mlir._mlir_libs._zamalang._compiler import ExecutionArgument as _ExecutionArgument
from mlir._mlir_libs._zamalang._compiler import round_trip as _round_trip
import numpy as np


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


def create_execution_argument(value: Union[int, List[int]]) -> "_ExecutionArgument":
    """Create an execution argument holding either an int or tensor value.

    Args:
        value (Union[int, List[int]]): value of the argument, either an int, or a list of int

    Raises:
        TypeError: if the values aren't in the expected range, or using a wrong type

    Returns:
        _ExecutionArgument: execution argument holding the appropriate value
    """
    if not isinstance(value, (int, list)):
        raise TypeError("value of execution argument must be either int or list[int]")
    if isinstance(value, int):
        if not (0 <= value < (2 ** 64 - 1)):
            raise TypeError("single integer must be in the range [0, 2**64 - 1] (uint64)")
    else:
        assert isinstance(value, list)
        for elem in value:
            if not (0 <= elem < (2 ** 8 - 1)):
                raise TypeError("values of the list must be in the range [0, 255] (uint8)")
    return _ExecutionArgument.create(value)


class CompilerEngine:
    def __init__(self, mlir_str: str = None):
        self._engine = _JitCompilerEngine()
        self._lambda = None
        if mlir_str is not None:
            self.compile_fhe(mlir_str)

    def compile_fhe(self, mlir_str: str, func_name: str = "main"):
        """Compile the MLIR input.

        Args:
            mlir_str (str): MLIR to compile.
            func_name (str): name of the function to set as entrypoint.

        Raises:
            TypeError: if the argument is not an str.
        """
        if not isinstance(mlir_str, str):
            raise TypeError("input must be an `str`")
        self._lambda = self._engine.build_lambda(mlir_str, func_name)

    def run(self, *args: List[Union[int, List[int]]]) -> Union[int, np.array]:
        """Run the compiled code.

        Args:
            *args: list of arguments for execution. Each argument can be an int, or a list of int

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
