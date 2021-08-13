"""Compiler submodule"""
from typing import List
from _zamalang._compiler import CompilerEngine as _CompilerEngine
from _zamalang._compiler import round_trip as _round_trip


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

    def run(self, *args: List[int]) -> int:
        """Run the compiled code.

        Raises:
            TypeError: if arguments aren't of type int

        Returns:
            int: result of execution.
        """
        if not all(isinstance(arg, int) for arg in args):
            raise TypeError("arguments must be of type int")
        return self._engine.run(args)

    def get_compiled_module(self) -> str:
        """Compiled module in printable form.

        Returns:
            str: Compiled module in printable form.
        """
        return self._engine.get_compiled_module()
