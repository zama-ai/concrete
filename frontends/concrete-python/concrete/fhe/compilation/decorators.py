"""
Declaration of `circuit` and `compiler` decorators.
"""

import functools
import inspect
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any, Callable, Optional, Union

from ..representation import Graph
from ..tracing.typing import ScalarAnnotation
from ..values import ValueDescription
from .artifacts import DebugArtifacts
from .circuit import Circuit
from .compiler import Compiler
from .configuration import Configuration
from .module_compiler import CompositionPolicy, FunctionDef, ModuleCompiler
from .status import EncryptionStatus
from .wiring import AllComposable


def circuit(
    parameters: Mapping[str, Union[str, EncryptionStatus]],
    configuration: Optional[Configuration] = None,
    artifacts: Optional[DebugArtifacts] = None,
    **kwargs,
):
    """
    Provide a direct interface for compilation of single circuit programs.

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

    def decoration(function_: Callable):
        signature = inspect.signature(function_)

        parameter_values: dict[str, ValueDescription] = {}
        for name, details in signature.parameters.items():
            if name not in parameters:
                continue

            annotation = details.annotation

            is_value = isinstance(annotation, ValueDescription)
            is_scalar_annotation = isinstance(annotation, type) and issubclass(
                annotation, ScalarAnnotation
            )

            if not (is_value or is_scalar_annotation):
                message = (
                    f"Annotation {annotation} for argument '{name}' is not valid "
                    f"(please use an fhe type such as "
                    f"`fhe.uint4` or 'fhe.tensor[fhe.uint4, 3, 2]')"
                )
                raise ValueError(message)

            parameter_values[name] = (
                annotation
                if is_value
                else ValueDescription(deepcopy(annotation.dtype), shape=(), is_encrypted=False)
            )

            status = EncryptionStatus(parameters[name].lower())
            parameter_values[name].is_encrypted = status == "encrypted"

        return Compiler.assemble(function_, parameter_values, configuration, artifacts, **kwargs)

    return decoration


class Compilable:
    """
    Compilable class, to wrap a function and provide methods to trace and compile it.
    """

    function: Callable
    compiler: Compiler

    def __init__(self, function_: Callable, parameters):
        self.function = function_  # type: ignore
        self.compiler = Compiler(self.function, dict(parameters))

    def __call__(self, *args, **kwargs) -> Any:
        self.compiler(*args, **kwargs)
        return self.function(*args, **kwargs)

    def trace(
        self,
        inputset: Optional[Union[Iterable[Any], Iterable[tuple[Any, ...]]]] = None,
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
        inputset: Optional[Union[Iterable[Any], Iterable[tuple[Any, ...]]]] = None,
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

        return self.compiler.compile(
            inputset if inputset is not None else [], configuration, artifacts, **kwargs
        )

    def reset(self):
        """
        Reset the compilable so that another compilation with another inputset can be performed.
        """

        self.compiler.reset()


def compiler(parameters: Mapping[str, Union[str, EncryptionStatus]]):
    """
    Provide an easy interface for the compilation of single-circuit programs.

    Args:
        parameters (Mapping[str, Union[str, EncryptionStatus]]):
            encryption statuses of the parameters of the function to compile
    """

    def decoration(function_: Callable):
        return Compilable(function_, parameters)

    return decoration


def module():
    """
    Provide an easy interface for the compilation of multi functions modules.
    """

    def decoration(class_):
        functions = inspect.getmembers(class_, lambda x: isinstance(x, FunctionDef))
        if not functions:
            error = "Tried to define an @fhe.module without any @fhe.function"
            raise RuntimeError(error)
        composition = getattr(class_, "composition", AllComposable())
        assert isinstance(composition, CompositionPolicy)
        return ModuleCompiler([f for (_, f) in functions], composition)

    return decoration


def function(parameters: dict[str, Union[str, EncryptionStatus]]):
    """
    Provide an easy interface to define a function within an fhe module.

    Args:
        parameters (Mapping[str, Union[str, EncryptionStatus]]):
            encryption statuses of the parameters of the function to compile
    """

    def decoration(function_: Callable):
        return functools.wraps(function_)(FunctionDef(function_, parameters))

    return decoration
