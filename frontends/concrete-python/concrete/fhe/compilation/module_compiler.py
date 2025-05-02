"""
Declaration of `MultiCompiler` class.
"""

# pylint: disable=import-error,no-name-in-module

import inspect
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
from concrete.compiler import CompilationContext

from ..extensions import AutoRounder, AutoTruncator
from ..mlir import GraphConverter
from ..representation import Graph
from ..tracing import Tracer
from ..values import ValueDescription
from .artifacts import DebugManager, FunctionDebugArtifacts, ModuleDebugArtifacts
from .composition import CompositionPolicy
from .configuration import Configuration
from .module import FheModule
from .status import EncryptionStatus
from .utils import fuse
from .wiring import Input, Output, TracedOutput, Wire, Wired, WireTracingContextManager

DEFAULT_OUTPUT_DIRECTORY: Path = Path(".artifacts")

# pylint: enable=import-error,no-name-in-module


class FunctionDef:
    """
    An object representing the definition of a function as used in an fhe module.
    """

    function: Callable
    parameter_encryption_statuses: dict[str, EncryptionStatus]
    inputset: list[Any]
    graph: Optional[Graph]
    location: str

    _is_direct: bool
    _parameter_values: dict[str, ValueDescription]
    _trace_wires: Optional[set["Wire"]]

    def __init__(
        self,
        function: Callable,
        parameter_encryption_statuses: dict[str, Union[str, EncryptionStatus]],
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
        self.inputset = []
        self.graph = None
        self._is_direct = False
        self._parameter_values = {}
        self.location = (
            f"{self.function.__code__.co_filename}:{self.function.__code__.co_firstlineno}"
        )
        self._trace_wires = None

    @property
    def name(self) -> str:
        """Return the name of the function."""
        return self.function.__name__

    def trace(
        self,
        sample: Union[Any, tuple[Any, ...]],
        artifacts: Optional[FunctionDebugArtifacts] = None,
    ):
        """
        Trace the function and fuse the resulting graph with a sample input.

        Args:
            sample (Union[Any, Tuple[Any, ...]]):
                sample to use for tracing
            artifacts: Optiona[FunctionDebugArtifacts]:
                the object to store artifacts in
        """

        if artifacts is not None:
            artifacts.add_source_code(self.function)
            for param, encryption_status in self.parameter_encryption_statuses.items():
                artifacts.add_parameter_encryption_status(param, encryption_status)

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

        self.graph = Tracer.trace(self.function, parameters, location=self.location)
        if artifacts is not None:
            artifacts.add_graph("initial", self.graph)

        fuse(self.graph, artifacts)

    def evaluate(
        self,
        action: str,
        inputset: Optional[Union[Iterable[Any], Iterable[tuple[Any, ...]]]],
        configuration: Configuration,
        artifacts: FunctionDebugArtifacts,
    ):
        """
        Trace, fuse, measure bounds, and update values in the resulting graph in one go.

        Args:
            action (str):
                action being performed (e.g., "trace", "compile")

            inputset (Optional[Union[Iterable[Any], Iterable[Tuple[Any, ...]]]]):
                optional inputset to extend accumulated inputset before bounds measurement

            configuration (Configuration):
                configuration to be used

            artifacts (FunctionDebugArtifacts):
                artifact object to store informations in
        """

        if self._is_direct:
            self.graph = Tracer.trace(
                self.function,
                self._parameter_values,
                is_direct=True,
                location=self.location,
            )
            artifacts.add_graph("initial", self.graph)  # pragma: no cover
            fuse(
                self.graph,
                artifacts,
            )
            artifacts.add_graph("final", self.graph)  # pragma: no cover
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

        if configuration.auto_adjust_rounders:
            AutoRounder.adjust(self.function, self.inputset)

        if configuration.auto_adjust_truncators:
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

            self.trace(first_sample, artifacts)
            assert self.graph is not None

        bounds = self.graph.measure_bounds(self.inputset)
        self.graph.update_with_bounds(bounds)

        artifacts.add_graph("final", self.graph)

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[
        np.bool_,
        np.integer,
        np.floating,
        np.ndarray,
        "TracedOutput",
        tuple[Union[np.bool_, np.integer, np.floating, np.ndarray, "TracedOutput"], ...],
    ]:
        if len(kwargs) != 0:
            message = f"Calling function '{self.function.__name__}' with kwargs is not supported"
            raise RuntimeError(message)

        # The actual call to the function graph object gets wrapped between two calls to methods
        # that allows to trace the wiring.
        #
        # When activated:
        #    + `_trace_wire_outputs` method wraps ciphered outputs into a `TracedOutput` object,
        #      along with its origin information.
        #    + `_trace_wire_inputs` method unwraps the `TracedOutput`, records the wiring, and
        #      returns unwrapped values for execution.
        traced_inputs = self._trace_wires_inputs(*args)

        if self.graph is None:
            # Note that the tracing must be executed on the `traced_inputs` which are unwrapped
            # from the potential `TracedOutput` added by wire tracing.
            self.trace(traced_inputs)
            assert self.graph is not None

        self.inputset.append(traced_inputs)

        if isinstance(traced_inputs, tuple):
            raw_outputs = self.graph(*traced_inputs)
        else:
            raw_outputs = self.graph(traced_inputs)

        if isinstance(raw_outputs, tuple):
            traced_output = self._trace_wires_outputs(*raw_outputs)
        else:
            traced_output = self._trace_wires_outputs(raw_outputs)

        return traced_output

    def _trace_wires_inputs(
        self,
        *args: Any,
    ) -> Union[
        np.bool_,
        np.integer,
        np.floating,
        np.ndarray,
        tuple[Union[np.bool_, np.integer, np.floating, np.ndarray], ...],
    ]:
        # If the _trace_wires property points to a wire list, we use wire tracing.
        if self._trace_wires is None:
            return args[0] if len(args) == 1 else args

        for i, arg in enumerate(args):
            if isinstance(arg, TracedOutput):
                # Wire gets added to the wire list
                self._trace_wires.add(Wire(arg.output_info, Input(self, i)))

        output = tuple(arg.returned_value if isinstance(arg, TracedOutput) else arg for arg in args)

        return output[0] if len(output) == 1 else output

    def _trace_wires_outputs(
        self,
        *args: Any,
    ) -> Union[
        np.bool_,
        np.integer,
        np.floating,
        np.ndarray,
        "TracedOutput",
        tuple[Union[np.bool_, np.integer, np.floating, np.ndarray, "TracedOutput"], ...],
    ]:
        # If the _trace_wires property points to a wire list, we use wire tracing.
        if self._trace_wires is None:
            return args[0] if len(args) == 1 else args

        output = tuple(TracedOutput(Output(self, i), arg) for (i, arg) in enumerate(args))

        return output[0] if len(output) == 1 else output


class ModuleCompiler:
    """
    Compiler class for multiple functions, to glue the compilation pipeline.
    """

    default_configuration: Configuration
    functions: dict[str, FunctionDef]
    compilation_context: CompilationContext
    composition: CompositionPolicy

    def __init__(self, functions: list[FunctionDef], composition: CompositionPolicy):
        self.default_configuration = Configuration(
            p_error=0.00001,
            parameter_selection_strategy="multi",
        )
        self.functions = {function.name: function for function in functions}
        self.compilation_context = CompilationContext()
        self.composition = composition

    def wire_pipeline(self, inputset: Union[Iterable[Any], Iterable[tuple[Any, ...]]]):
        """
        Return a context manager that traces wires automatically.
        """
        self.composition = Wired(set())
        return WireTracingContextManager(self, inputset)

    def compile(
        self,
        inputsets: Optional[
            dict[str, Optional[Union[Iterable[Any], Iterable[tuple[Any, ...]]]]]
        ] = None,
        configuration: Optional[Configuration] = None,
        module_artifacts: Optional[ModuleDebugArtifacts] = None,
        **kwargs,
    ) -> FheModule:
        """
        Compile the module using an ensemble of inputsets.

        Args:
            inputsets (Optional[Dict[str, Union[Iterable[Any], Iterable[Tuple[Any, ...]]]]]):
                optional inputsets to extend accumulated inputsets before bounds measurement

            configuration(Optional[Configuration], default = None):
                configuration to use

            artifacts (Optional[ModuleDebugArtifacts], default = None):
                artifacts to store information about the process

            kwargs (Dict[str, Any]):
                configuration options to overwrite

        Returns:
            FheModule:
                compiled module
        """

        configuration = configuration if configuration is not None else self.default_configuration
        if len(kwargs) != 0:
            configuration = configuration.fork(**kwargs)

        module_artifacts = (
            module_artifacts if module_artifacts is not None else ModuleDebugArtifacts()
        )
        if not module_artifacts.functions:
            module_artifacts.functions = {
                f: FunctionDebugArtifacts() for f in self.functions.keys()
            }

        dbg = DebugManager(configuration)

        try:
            # Trace and fuse the functions
            for name, function in self.functions.items():
                inputset = inputsets[name] if inputsets is not None else None
                function_artifacts = module_artifacts.functions[name]
                function.evaluate("Compiling", inputset, configuration, function_artifacts)
                assert function.graph is not None
                dbg.debug_computation_graph(name, function.graph)

            # Convert the graphs to an mlir module
            mlir_context = self.compilation_context.mlir_context()
            graphs = {}

            for name, function in self.functions.items():
                assert function.graph is not None
                graphs[name] = function.graph

            # pylint: disable=protected-access
            mlir_module = GraphConverter(
                configuration,
                self.composition.get_rules_iter(
                    list(filter(None, [f.graph for f in self.functions.values()]))
                ),
            ).convert_many(graphs, mlir_context)
            mlir_str = str(mlir_module).strip()
            dbg.debug_mlir(mlir_str)
            module_artifacts.add_mlir_to_compile(mlir_str)

            # Debug some function informations
            for name, function in self.functions.items():
                dbg.debug_bit_width_constaints(name, function.graph)
                dbg.debug_bit_width_assignments(name, function.graph)
                dbg.debug_assigned_graph(name, function.graph)

            # Compile to a module!
            with dbg.debug_table("Optimizer", activate=dbg.show_optimizer()):
                # pylint: disable=protected-access
                output = FheModule(
                    graphs,
                    mlir_module,
                    self.compilation_context,
                    configuration,
                    self.composition.get_rules_iter(
                        list(filter(None, [f.graph for f in self.functions.values()]))
                    ),
                )
                module_artifacts.add_execution_runtime(output.execution_runtime)

            dbg.debug_statistics(output)

        except Exception:  # pragma: no cover
            # this branch is reserved for unexpected issues and hence it shouldn't be tested
            # if it could be tested, we would have fixed the underlying issue

            # if the user desires so,
            # we need to export all the information we have about the compilation

            if configuration.dump_artifacts_on_unexpected_failures:
                module_artifacts.export()

                traceback_path = module_artifacts.output_directory.joinpath("traceback.txt")
                with open(traceback_path, "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())

            raise

        return output

    # pylint: enable=too-many-branches,too-many-statements

    def __getattr__(self, item) -> FunctionDef:
        if item not in list(self.functions.keys()):
            error = f"No attribute {item}"
            raise AttributeError(error)
        return self.functions[item]
