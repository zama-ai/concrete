"""
Declaration of `circuit` and `compiler` decorators.
"""

import inspect
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, Union

from ..representation import Graph
from ..tracing.typing import ScalarAnnotation
from ..values import Value
from .artifacts import DebugArtifacts
from .circuit import Circuit
from .compiler import Compiler, EncryptionStatus
from .configuration import Configuration


def circuit(
    parameters: Mapping[str, Union[str, EncryptionStatus]],
    configuration: Optional[Configuration] = None,
    artifacts: Optional[DebugArtifacts] = None,
    **kwargs,
):
    """
    Provide a direct interface for compilation.

    Args:
        parameters (Mapping[str, Union[str, EncryptionStatus]]):
            encryption statuses of the parameters of the function to compile

        configuration(Optional[Configuration], default = None):
            configuration to use

        artifacts (Optional[DebugArtifacts], default = None):
            artifacts to store information about the process

        kwargs (Dict[str, Any]):
            configuration options to overwrite
    """

    def decoration(function: Callable):
        signature = inspect.signature(function)

        parameter_values: Dict[str, Value] = {}
        for name, details in signature.parameters.items():
            if name not in parameters:
                continue

            annotation = details.annotation

            is_value = isinstance(annotation, Value)
            is_scalar_annotation = isinstance(annotation, type) and issubclass(
                annotation, ScalarAnnotation
            )

            if not (is_value or is_scalar_annotation):
                message = (
                    f"Annotation {annotation} for argument '{name}' is not valid "
                    f"(please use a cnp type such as "
                    f"`cnp.uint4` or 'cnp.tensor[cnp.uint4, 3, 2]')"
                )
                raise ValueError(message)

            parameter_values[name] = (
                annotation if is_value else Value(annotation.dtype, shape=(), is_encrypted=False)
            )

            status = EncryptionStatus(parameters[name].lower())
            parameter_values[name].is_encrypted = status == "encrypted"

        return Compiler.assemble(function, parameter_values, configuration, artifacts, **kwargs)

    return decoration


def compiler(parameters: Mapping[str, Union[str, EncryptionStatus]]):
    """
    Provide an easy interface for compilation.

    Args:
        parameters (Mapping[str, Union[str, EncryptionStatus]]):
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
