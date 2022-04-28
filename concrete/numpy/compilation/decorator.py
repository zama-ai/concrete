"""
Declaration of `compiler` decorator.
"""

from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Union

from ..representation import Graph
from .artifacts import DebugArtifacts
from .circuit import Circuit
from .compiler import Compiler, EncryptionStatus
from .configuration import Configuration


def compiler(parameters: Mapping[str, EncryptionStatus]):
    """
    Provide an easy interface for compilation.

    Args:
        parameters (Dict[str, EncryptionStatus]):
            encryption statuses of the parameters of the function to compile
    """

    def decoration(function: Callable):
        class Compilable:
            """
            Compilable class, to wrap a function and provide methods to trace and compile it.
            """

            function: Callable
            compiler: Compiler

            def __init__(self, function: Callable):
                self.function = function  # type: ignore
                self.compiler = Compiler(self.function, dict(parameters))

            def __call__(self, *args, **kwargs) -> Any:
                self.compiler(*args, **kwargs)
                return self.function(*args, **kwargs)

            def trace(
                self,
                inputset: Optional[Union[Iterable[Any], Iterable[Tuple[Any, ...]]]] = None,
                configuration: Optional[Configuration] = None,
                artifacts: Optional[DebugArtifacts] = None,
                **kwargs,
            ) -> Graph:
                """
                Trace the function into computation graph.

                Args:
                    inputset (Optional[Union[Iterable[Any], Iterable[Tuple[Any, ...]]]]):
                        optional inputset to extend accumulated inputset before bounds measurement

                    configuration(Optional[Configuration], default = None):
                        configuration to use

                    artifacts (Optional[DebugArtifacts], default = None):
                        artifacts to store information about the process

                    kwargs (Dict[str, Any]):
                        configuration options to overwrite

                Returns:
                    Graph:
                        computation graph representing the function prior to MLIR conversion
                """

                return self.compiler.trace(inputset, configuration, artifacts, **kwargs)

            def compile(
                self,
                inputset: Optional[Union[Iterable[Any], Iterable[Tuple[Any, ...]]]] = None,
                configuration: Optional[Configuration] = None,
                artifacts: Optional[DebugArtifacts] = None,
                **kwargs,
            ) -> Circuit:
                """
                Compile the function into a circuit.

                Args:
                    inputset (Optional[Union[Iterable[Any], Iterable[Tuple[Any, ...]]]]):
                        optional inputset to extend accumulated inputset before bounds measurement

                    configuration(Optional[Configuration], default = None):
                        configuration to use

                    artifacts (Optional[DebugArtifacts], default = None):
                        artifacts to store information about the process

                    kwargs (Dict[str, Any]):
                        configuration options to overwrite

                Returns:
                    Circuit:
                        compiled circuit
                """

                return self.compiler.compile(inputset, configuration, artifacts, **kwargs)

        return Compilable(function)

    return decoration
