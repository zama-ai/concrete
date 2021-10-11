"""Compiler submodule"""
from typing import List, Union
from mlir._mlir_libs._zamalang._compiler import CompilerEngine as _CompilerEngine
from mlir._mlir_libs._zamalang._compiler import ExecutionArgument as _ExecutionArgument
from mlir._mlir_libs._zamalang._compiler import round_trip as _round_trip


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
        self._engine = _CompilerEngine()
        if mlir_str is not None:
            self.compile_fhe(mlir_str)

    def compile_fhe(self, mlir_str: str) -> "CompilerEngine":
        """Compile the MLIR input and build a CompilerEngine.

        Args:
            mlir_str (str): MLIR to compile.

        Raises:
            TypeError: if the argument is not an str.

        Returns:
            CompilerEngine: engine used for execution.
        """
        if not isinstance(mlir_str, str):
            raise TypeError("input must be an `str`")
        return self._engine.compile_fhe(mlir_str)

    def run(self, *args: List[Union[int, List[int]]]) -> int:
        """Run the compiled code.

        Args:
            *args: list of arguments for execution. Each argument can be an int, or a list of int

        Raises:
            TypeError: if execution arguments can't be constructed

        Returns:
            int: result of execution.
        """
        execution_arguments = [create_execution_argument(arg) for arg in args]
        return self._engine.run(execution_arguments)

    def get_compiled_module(self) -> str:
        """Compiled module in printable form.

        Returns:
            str: Compiled module in printable form.
        """
        return self._engine.get_compiled_module()
