"""
Declaration of `Compiler` class.
"""

# pylint: disable=import-error,no-name-in-module

import inspect
import os
import traceback
from copy import deepcopy
from enum import Enum, unique
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from concrete.compiler import CompilationContext

from ..extensions import AutoRounder, AutoTruncator
from ..mlir import GraphConverter
from ..representation import Graph
from ..tracing import Tracer
from ..values import ValueDescription
from .artifacts import DebugArtifacts
from .circuit import Circuit
from .configuration import Configuration
from .utils import fuse, get_terminal_size

# pylint: enable=import-error,no-name-in-module


@unique
class EncryptionStatus(str, Enum):
    """
    EncryptionStatus enum, to represent encryption status of parameters.
    """

    CLEAR = "clear"
    ENCRYPTED = "encrypted"


class Compiler:
    """
    Compiler class, to glue the compilation pipeline.
    """

    function: Callable
    parameter_encryption_statuses: Dict[str, EncryptionStatus]

    configuration: Configuration
    artifacts: Optional[DebugArtifacts]

    inputset: List[Any]
    graph: Optional[Graph]

    compilation_context: CompilationContext

    _is_direct: bool
    _parameter_values: Dict[str, ValueDescription]

    @staticmethod
    def assemble(
        function: Callable,
        parameter_values: Dict[str, ValueDescription],
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
                name: "encrypted" if value.is_encrypted else "clear"
                for name, value in parameter_values.items()
            },
        )

        # pylint: disable=protected-access
        compiler._is_direct = True
        compiler._parameter_values = parameter_values
        # pylint: enable=protected-access

        return compiler.compile(None, configuration, artifacts, **kwargs)

    def __init__(
        self,
        function: Callable,
        parameter_encryption_statuses: Dict[str, Union[str, EncryptionStatus]],
    ):
        signature = inspect.signature(function)

        missing_args = list(signature.parameters)
        for arg in parameter_encryption_statuses.keys():
            if arg in signature.parameters:
                missing_args.remove(arg)

        if len(missing_args) != 0:
            parameter_str = repr(missing_args[0])
            for arg in missing_args[1:-1]:
                parameter_str += f", {repr(arg)}"
            if len(missing_args) != 1:
                parameter_str += f" and {repr(missing_args[-1])}"

            message = (
                f"Encryption status{'es' if len(missing_args) > 1 else ''} "
                f"of parameter{'s' if len(missing_args) > 1 else ''} "
                f"{parameter_str} of function '{function.__name__}' "
                f"{'are' if len(missing_args) > 1 else 'is'} not provided"
            )
            raise ValueError(message)

        additional_args = list(parameter_encryption_statuses)
        for arg in signature.parameters.keys():
            if arg in parameter_encryption_statuses:
                additional_args.remove(arg)

        if len(additional_args) != 0:
            parameter_str = repr(additional_args[0])
            for arg in additional_args[1:-1]:
                parameter_str += f", {repr(arg)}"
            if len(additional_args) != 1:
                parameter_str += f" and {repr(additional_args[-1])}"

            message = (
                f"Encryption status{'es' if len(additional_args) > 1 else ''} "
                f"of {parameter_str} {'are' if len(additional_args) > 1 else 'is'} provided but "
                f"{'they are' if len(additional_args) > 1 else 'it is'} not a parameter "
                f"of function '{function.__name__}'"
            )
            raise ValueError(message)

        self.function = function  # type: ignore
        self.parameter_encryption_statuses = {
            param: EncryptionStatus(status.lower())
            for param, status in parameter_encryption_statuses.items()
        }

        self.configuration = Configuration()
        self.artifacts = None

        self.inputset = []
        self.graph = None

        self.compilation_context = CompilationContext.new()

        self._is_direct = False
        self._parameter_values = {}

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[
        np.bool_,
        np.integer,
        np.floating,
        np.ndarray,
        Tuple[Union[np.bool_, np.integer, np.floating, np.ndarray], ...],
    ]:
        if len(kwargs) != 0:
            message = f"Calling function '{self.function.__name__}' with kwargs is not supported"
            raise RuntimeError(message)

        sample = args[0] if len(args) == 1 else args

        if self.graph is None:
            self._trace(sample)
            assert self.graph is not None

        self.inputset.append(sample)
        return self.graph(*args)

    def _trace(self, sample: Union[Any, Tuple[Any, ...]]):
        """
        Trace the function and fuse the resulting graph with a sample input.

        Args:
            sample (Union[Any, Tuple[Any, ...]]):
                sample to use for tracing
        """

        if self.artifacts is not None:
            self.artifacts.add_source_code(self.function)
            for param, encryption_status in self.parameter_encryption_statuses.items():
                self.artifacts.add_parameter_encryption_status(param, encryption_status)

        parameters = {
            param: ValueDescription.of(arg, is_encrypted=(status == EncryptionStatus.ENCRYPTED))
            for arg, (param, status) in zip(
                (
                    sample
                    if len(self.parameter_encryption_statuses) > 1 or isinstance(sample, tuple)
                    else (sample,)
                ),
                self.parameter_encryption_statuses.items(),
            )
        }

        self.graph = Tracer.trace(self.function, parameters)
        if self.artifacts is not None:
            self.artifacts.add_graph("initial", self.graph)

        fuse(self.graph, self.artifacts)

    def _evaluate(
        self,
        action: str,
        inputset: Optional[Union[Iterable[Any], Iterable[Tuple[Any, ...]]]],
    ):
        """
        Trace, fuse, measure bounds, and update values in the resulting graph in one go.

        Args:
            action (str):
                action being performed (e.g., "trace", "compile")

            inputset (Optional[Union[Iterable[Any], Iterable[Tuple[Any, ...]]]]):
                optional inputset to extend accumulated inputset before bounds measurement
        """

        if self._is_direct:
            self.graph = Tracer.trace(self.function, self._parameter_values, is_direct=True)
            if self.artifacts is not None:
                self.artifacts.add_graph("initial", self.graph)  # pragma: no cover

            fuse(self.graph, self.artifacts)
            if self.artifacts is not None:
                self.artifacts.add_graph("final", self.graph)  # pragma: no cover

            return

        if inputset is not None:
            previous_inputset_length = len(self.inputset)
            for index, sample in enumerate(iter(inputset)):
                self.inputset.append(sample)

                if not isinstance(sample, tuple):
                    sample = (sample,)

                if len(sample) != len(self.parameter_encryption_statuses):
                    self.inputset = self.inputset[:previous_inputset_length]

                    expected = (
                        "a single value"
                        if len(self.parameter_encryption_statuses) == 1
                        else f"a tuple of {len(self.parameter_encryption_statuses)} values"
                    )
                    actual = (
                        "a single value" if len(sample) == 1 else f"a tuple of {len(sample)} values"
                    )

                    message = (
                        f"Input #{index} of your inputset is not well formed "
                        f"(expected {expected} got {actual})"
                    )
                    raise ValueError(message)

        if self.configuration.auto_adjust_rounders:
            AutoRounder.adjust(self.function, self.inputset)

        if self.configuration.auto_adjust_truncators:
            AutoTruncator.adjust(self.function, self.inputset)

        if self.graph is None:
            try:
                first_sample = next(iter(self.inputset))
            except StopIteration as error:
                message = (
                    f"{action} function '{self.function.__name__}' "
                    f"without an inputset is not supported"
                )
                raise RuntimeError(message) from error

            self._trace(first_sample)
            assert self.graph is not None

        bounds = self.graph.measure_bounds(self.inputset)
        self.graph.update_with_bounds(bounds)

        if self.artifacts is not None:
            self.artifacts.add_graph("final", self.graph)

    def trace(
        self,
        inputset: Optional[Union[Iterable[Any], Iterable[Tuple[Any, ...]]]] = None,
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

        old_configuration = deepcopy(self.configuration)
        old_artifacts = deepcopy(self.artifacts)

        if configuration is not None:
            self.configuration = configuration

        if len(kwargs) != 0:
            self.configuration = self.configuration.fork(**kwargs)

        self.artifacts = (
            artifacts
            if artifacts is not None
            else DebugArtifacts()
            if self.configuration.dump_artifacts_on_unexpected_failures
            else None
        )

        try:
            self._evaluate("Tracing", inputset)
            assert self.graph is not None

            if self.configuration.verbose or self.configuration.show_graph:
                graph = self.graph.format()
                longest_line = max(len(line) for line in graph.split("\n"))

                try:  # pragma: no cover
                    # this branch cannot be covered
                    # because `os.get_terminal_size()`
                    # raises an exception during tests

                    columns, _ = os.get_terminal_size()
                    if columns == 0:  # noqa: SIM108
                        columns = min(longest_line, 80)
                    else:
                        columns = min(longest_line, columns)
                except OSError:  # pragma: no cover
                    columns = min(longest_line, 80)

                print()

                print("Computation Graph")
                print("-" * columns)
                print(graph)
                print("-" * columns)

                print()

            return self.graph

        except Exception:  # pragma: no cover
            # this branch is reserved for unexpected issues and hence it shouldn't be tested
            # if it could be tested, we would have fixed the underlying issue

            # if the user desires so,
            # we need to export all the information we have about the compilation

            if self.configuration.dump_artifacts_on_unexpected_failures:
                assert self.artifacts is not None
                self.artifacts.export()

                traceback_path = self.artifacts.output_directory.joinpath("traceback.txt")
                with open(traceback_path, "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())

            raise

        finally:
            self.configuration = old_configuration
            self.artifacts = old_artifacts

    # pylint: disable=too-many-branches,too-many-statements

    def compile(
        self,
        inputset: Optional[Union[Iterable[Any], Iterable[Tuple[Any, ...]]]] = None,
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

        old_configuration = deepcopy(self.configuration)
        old_artifacts = deepcopy(self.artifacts)

        if configuration is not None:
            self.configuration = configuration

        if len(kwargs) != 0:
            self.configuration = self.configuration.fork(**kwargs)

        self.artifacts = (
            artifacts
            if artifacts is not None
            else DebugArtifacts()
            if self.configuration.dump_artifacts_on_unexpected_failures
            else None
        )

        try:
            self._evaluate("Compiling", inputset)
            assert self.graph is not None

            show_graph = (
                self.configuration.show_graph
                if self.configuration.show_graph is not None
                else self.configuration.verbose
            )
            show_bit_width_constraints = (
                self.configuration.show_bit_width_constraints
                if self.configuration.show_bit_width_constraints is not None
                else self.configuration.verbose
            )
            show_bit_width_assignments = (
                self.configuration.show_bit_width_assignments
                if self.configuration.show_bit_width_assignments is not None
                else self.configuration.verbose
            )
            show_assigned_graph = (
                self.configuration.show_assigned_graph
                if self.configuration.show_assigned_graph is not None
                else self.configuration.verbose
            )
            show_mlir = (
                self.configuration.show_mlir
                if self.configuration.show_mlir is not None
                else self.configuration.verbose
            )
            show_optimizer = (
                self.configuration.show_optimizer
                if self.configuration.show_optimizer is not None
                else self.configuration.verbose
            )
            show_statistics = (
                self.configuration.show_statistics
                if self.configuration.show_statistics is not None
                else self.configuration.verbose
            )

            columns = get_terminal_size()
            is_first = True

            if (
                show_graph
                or show_bit_width_constraints
                or show_bit_width_assignments
                or show_assigned_graph
                or show_mlir
                or show_optimizer
                or show_statistics
            ):
                if show_graph:
                    if is_first:  # pragma: no cover
                        print()
                        is_first = False

                    print("Computation Graph")
                    print("-" * columns)
                    print(self.graph.format())
                    print("-" * columns)

                    print()

            # in-memory MLIR module
            mlir_context = self.compilation_context.mlir_context()
            mlir_module = GraphConverter().convert(self.graph, self.configuration, mlir_context)
            # textual representation of the MLIR module
            mlir_str = str(mlir_module).strip()
            if self.artifacts is not None:
                self.artifacts.add_mlir_to_compile(mlir_str)

            if show_bit_width_constraints:
                if is_first:  # pragma: no cover
                    print()
                    is_first = False

                print("Bit-Width Constraints")
                print("-" * columns)
                print(self.graph.format_bit_width_constraints())
                print("-" * columns)

                print()

            if show_bit_width_assignments:
                if is_first:  # pragma: no cover
                    print()
                    is_first = False

                print("Bit-Width Assignments")
                print("-" * columns)
                print(self.graph.format_bit_width_assignments())
                print("-" * columns)

                print()

            if show_assigned_graph:
                if is_first:  # pragma: no cover
                    print()
                    is_first = False

                print("Bit-Width Assigned Computation Graph")
                print("-" * columns)
                print(self.graph.format(show_assigned_bit_widths=True))
                print("-" * columns)

                print()

            if show_mlir:
                if is_first:  # pragma: no cover
                    print()
                    is_first = False

                print("MLIR")
                print("-" * columns)
                print(mlir_str)
                print("-" * columns)

                print()

            if show_optimizer:
                if is_first:  # pragma: no cover
                    print()
                    is_first = False

                print("Optimizer")
                print("-" * columns)

            circuit = Circuit(
                self.graph,
                mlir_module,
                self.compilation_context,
                self.configuration,
            )

            if hasattr(circuit, "client"):
                client_parameters = circuit.client.specs.client_parameters
                if self.artifacts is not None:
                    self.artifacts.add_client_parameters(client_parameters.serialize())

            if show_optimizer:
                print("-" * columns)
                print()

            if show_statistics:
                if is_first:  # pragma: no cover
                    print()

                print("Statistics")
                print("-" * columns)

                def pretty(d, indent=0):  # pragma: no cover
                    if indent > 0:
                        print("{")

                    for key, value in d.items():
                        if isinstance(value, dict) and len(value) == 0:
                            continue

                        print("    " * indent + str(key) + ": ", end="")

                        if isinstance(value, dict):
                            pretty(value, indent + 1)
                        else:
                            print(value)

                    if indent > 0:
                        print("    " * (indent - 1) + "}")

                pretty(circuit.statistics)

                print("-" * columns)

                print()

        except Exception:  # pragma: no cover
            # this branch is reserved for unexpected issues and hence it shouldn't be tested
            # if it could be tested, we would have fixed the underlying issue

            # if the user desires so,
            # we need to export all the information we have about the compilation

            if self.configuration.dump_artifacts_on_unexpected_failures:
                assert self.artifacts is not None
                self.artifacts.export()

                traceback_path = self.artifacts.output_directory.joinpath("traceback.txt")
                with open(traceback_path, "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())

            raise

        finally:
            self.configuration = old_configuration
            self.artifacts = old_artifacts

        return circuit

    # pylint: enable=too-many-branches,too-many-statements
