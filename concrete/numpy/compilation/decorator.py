"""
Declaration of `compiler` decorator.
"""

from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Union

from ..representation import Graph
from .artifacts import CompilationArtifacts
from .circuit import Circuit
from .compiler import Compiler, EncryptionStatus
from .configuration import Configuration


def compiler(
    parameters: Mapping[str, EncryptionStatus],
    configuration: Optional[Configuration] = None,
    artifacts: Optional[CompilationArtifacts] = None,
):
    """
    Provide an easy interface for compilation.

    Args:
        parameters (Dict[str, EncryptionStatus]):
            encryption statuses of the parameters of the function to compile

        configuration(Optional[Configuration], default = None):
            configuration to use for compilation

        artifacts (Optional[CompilationArtifacts], default = None):
            artifacts to store information about compilation
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
                self.compiler = Compiler(
                    self.function,
                    dict(parameters),
                    configuration,
                    artifacts,
                )

            def __call__(self, *args, **kwargs) -> Any:
                self.compiler(*args, **kwargs)
                return self.function(*args, **kwargs)

            def trace(
                self,
                inputset: Optional[Union[Iterable[Any], Iterable[Tuple[Any, ...]]]] = None,
                show_graph: bool = False,
            ) -> Graph:
                """
                Trace the function into computation graph.

                Args:
                    inputset (Optional[Union[Iterable[Any], Iterable[Tuple[Any, ...]]]]):
                        optional inputset to extend accumulated inputset before bounds measurement

                    show_graph (bool, default = False):
                        whether to print the computation graph

                Returns:
                    Graph:
                        computation graph representing the function prior to MLIR conversion
                """

                return self.compiler.trace(inputset, show_graph)

            def compile(
                self,
                inputset: Optional[Union[Iterable[Any], Iterable[Tuple[Any, ...]]]] = None,
                show_graph: bool = False,
                show_mlir: bool = False,
                virtual: bool = False,
            ) -> Circuit:
                """
                Compile the function into a circuit.

                Args:
                    inputset (Optional[Union[Iterable[Any], Iterable[Tuple[Any, ...]]]]):
                        optional inputset to extend accumulated inputset before bounds measurement

                    show_graph (bool, default = False):
                        whether to print the computation graph

                    show_mlir (bool, default = False):
                        whether to print the compiled mlir

                    virtual (bool, default = False):
                        whether to simulate the computation to allow large bit-widths

                Returns:
                    Circuit:
                        compiled circuit
                """

                return self.compiler.compile(inputset, show_graph, show_mlir, virtual)

        return Compilable(function)

    return decoration
