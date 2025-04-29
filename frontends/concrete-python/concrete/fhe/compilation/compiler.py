"""
Declaration of `Compiler` class.
"""

# pylint: disable=import-error,no-name-in-module


from collections.abc import Iterable
from typing import Any, Callable, Optional, Union

import numpy as np

from ..representation import Graph
from ..values import ValueDescription
from .artifacts import DebugArtifacts, FunctionDebugArtifacts, ModuleDebugArtifacts
from .circuit import Circuit
from .composition import CompositionPolicy
from .configuration import Configuration
from .module_compiler import FunctionDef, ModuleCompiler
from .status import EncryptionStatus
from .wiring import AllComposable, NotComposable, TracedOutput

# pylint: enable=import-error,no-name-in-module


class Compiler:
    """
    Compiler class, to glue the compilation pipeline.
    """

    _module_compiler: ModuleCompiler
    _function_name: str

    @staticmethod
    def assemble(
        function: Callable,
        parameter_values: dict[str, ValueDescription],
        configuration: Optional[Configuration] = None,
        artifacts: Optional[DebugArtifacts] = None,
        **kwargs,
    ) -> Circuit:
        """
        Assemble a circuit from the raw parameter values, used in direct circuit definition.

        Args:
            function (Callable):
                function to convert to a circuit

            parameter_values (Dict[str, ValueDescription]):
                parameter values of the function

            configuration(Optional[Configuration], default = None):
                configuration to use

            artifacts (Optional[DebugArtifacts], default = None):
                artifacts to store information about the process

            kwargs (Dict[str, Any]):
                configuration options to overwrite

        Returns:
            Circuit:
                assembled circuit
        """

        compiler = Compiler(
            function,
            {
                name: EncryptionStatus.ENCRYPTED if value.is_encrypted else EncryptionStatus.CLEAR
                for name, value in parameter_values.items()
            },
            composition=(
                AllComposable()
                if (configuration.composable if configuration is not None else False)
                else NotComposable()
            ),
        )

        # pylint: disable=protected-access
        compiler._func_def._is_direct = True
        compiler._func_def._parameter_values = parameter_values
        # pylint: enable=protected-access

        return compiler.compile([], configuration, artifacts, **kwargs)

    def __init__(
        self,
        function: Callable,
        parameter_encryption_statuses: dict[str, Union[str, EncryptionStatus]],
        composition: Optional[Union[NotComposable, AllComposable]] = None,
    ):
        if composition is None:
            composition = NotComposable()
        assert isinstance(composition, CompositionPolicy)
        func = FunctionDef(
            function=function, parameter_encryption_statuses=parameter_encryption_statuses
        )
        self._module_compiler = ModuleCompiler([func], composition)
        self._function_name = function.__name__

    @property
    def _func_def(self) -> FunctionDef:
        return getattr(self._module_compiler, self._function_name)

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[
        np.bool_,
        np.integer,
        np.floating,
        np.ndarray,
        TracedOutput,
        tuple[Union[np.bool_, np.integer, np.floating, np.ndarray, TracedOutput], ...],
    ]:
        return self._func_def(*args, **kwargs)

    def trace(
        self,
        inputset: Optional[Union[Iterable[Any], Iterable[tuple[Any, ...]]]] = None,
        configuration: Optional[Configuration] = None,
        artifacts: Optional[DebugArtifacts] = None,
        **kwargs,
    ) -> Graph:
        """
        Trace the function using an inputset.

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
        art = (
            artifacts.module_artifacts.functions.get(self._function_name, FunctionDebugArtifacts())
            if artifacts is not None
            else FunctionDebugArtifacts()
        )
        conf = (
            configuration
            if configuration is not None
            else self._module_compiler.default_configuration
        )
        if len(kwargs) != 0:
            conf = conf.fork(**kwargs)

        self._func_def.evaluate("Tracing", inputset, conf, art)
        assert self._func_def.graph is not None
        return self._func_def.graph

    # pylint: disable=too-many-branches,too-many-statements

    def compile(
        self,
        inputset: Optional[Union[Iterable[Any], Iterable[tuple[Any, ...]]]] = None,
        configuration: Optional[Configuration] = None,
        artifacts: Optional[DebugArtifacts] = None,
        **kwargs,
    ) -> Circuit:
        """
        Compile the function using an inputset.

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

        art = artifacts.module_artifacts if artifacts is not None else ModuleDebugArtifacts()
        conf = (
            configuration
            if configuration is not None
            else self._module_compiler.default_configuration
        )
        if len(kwargs) != 0:
            conf = conf.fork(**kwargs)

        if conf.composable:
            self._module_compiler.composition = AllComposable()
        fhe_module = self._module_compiler.compile(
            {self._function_name: inputset}, configuration=conf, module_artifacts=art
        )
        return Circuit(fhe_module)

    # pylint: enable=too-many-branches,too-many-statements

    def reset(self):
        """
        Reset the compiler so that another compilation with another inputset can be performed.
        """
        fdef = self._module_compiler.functions[self._function_name]
        fresh_compiler = Compiler(fdef.function, fdef.parameter_encryption_statuses)
        self.__dict__.update(fresh_compiler.__dict__)
